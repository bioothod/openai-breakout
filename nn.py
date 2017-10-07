import tensorflow as tf

import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_param_name(s):
    return s.split('/', 1)[1].replace('/', 'X').split(':')[0]
def get_scope_name(s):
    return s.split('/')[0].split(':')[0]
def get_transform_placeholder_name(s):
    return get_param_name(s) + '_ext'

class nn(object):
    def __init__(self, scope, config):
        self.train_num = 0

        self.clip_value = config.get('clip_gradient_norm')
        self.learning_rate_start = config.get('learning_rate_start')
        self.learning_rate_end = config.get('learning_rate_end')
        self.learning_rate_decay_steps = config.get('learning_rate_decay_steps')
        self.learning_rate = config.get('learning_rate')
        self.xentropy_reg_beta = config.get('xentropy_reg_beta')
        self.policy_reg_beta = config.get('policy_reg_beta')
        self.summary_flush_num = config.get('summary_flush_num')
        self.device = config.get('device', '/cpu:0')

        logger.info("going to initialize scope %s" % scope)
        self.summary_writer = config.get('summary_writer')
        self.scope = scope
        with tf.variable_scope(scope) as vscope:
            self.vscope = vscope
            with tf.device(self.device):
                self.do_init(config)

            self.session_init(config)
            logger.info("scope %s has been initialized" % scope)

    def session_init(self, config):
        gpu_config = tf.GPUOptions(
                per_process_gpu_memory_fraction = config.get('per_process_gpu_memory_fraction', 1.0)
            )

        tf_config = tf.ConfigProto(
                intra_op_parallelism_threads = 8,
                inter_op_parallelism_threads = 8,
                allow_soft_placement=True,
                gpu_options = gpu_config,
            )
        self.sess = config.get('session', tf.Session(config=tf_config))

    def init_model(self, config):
        state_steps = config.get("state_steps")
        config_input_shape = config.get('input_shape')
        input_shape = (config_input_shape[0], config_input_shape[1], config_input_shape[2] * state_steps)

        output_size = config.get('output_size')

        dense_layer_units = config.get('dense_layer_units')

        x = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], input_shape[2]], name='x')
        action = tf.placeholder(tf.int32, [None, 1], name='action')
        reward = tf.placeholder(tf.float32, [None, 1], name='reward')

        self.add_summary(tf.summary.histogram('action', action))
        self.add_summary(tf.summary.histogram('reward', reward))

        input_layer = tf.reshape(x, [-1, input_shape[0], input_shape[1], input_shape[2]])

        prelu_alpha = 0.0001

        c1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=8, strides=4, padding='same',
                activation=tf.contrib.keras.layers.PReLU(alpha_initializer=tf.constant_initializer(prelu_alpha)))
        c2 = tf.layers.conv2d(inputs=c1, filters=32, kernel_size=4, strides=2, padding='same',
                activation=tf.contrib.keras.layers.PReLU(alpha_initializer=tf.constant_initializer(prelu_alpha)))

        flat = tf.reshape(c2, [-1, np.prod(c2.get_shape().as_list()[1:])])

        kinit = tf.contrib.layers.xavier_initializer()

        self.dense = tf.layers.dense(inputs=flat, units=dense_layer_units,
                activation=tf.contrib.keras.layers.PReLU(alpha_initializer=tf.constant_initializer(prelu_alpha)),
                use_bias=True, name='dense_layer',
                kernel_initializer=kinit,
                bias_initializer=tf.random_normal_initializer(0, 0.1))

        policy = tf.layers.dense(inputs=self.dense, units=output_size, use_bias=True, name='policy_layer',
                kernel_initializer=kinit,
                bias_initializer=tf.random_normal_initializer(0, 0.1))

        self.policy = tf.nn.softmax(policy)

        self.value = tf.layers.dense(inputs=self.dense, units=1, use_bias=True, name='value_layer',
                kernel_initializer=kinit,
                bias_initializer=tf.random_normal_initializer(0, 0.1))

        self.clip_names = ['{0}/{1}'.format(self.scope, name) for name in ['dense_layer', 'policy_layer', 'value_layer']]

        actions = tf.one_hot(action, output_size)
        actions = tf.squeeze(actions, 1)

        for i in range(output_size):
            x = tf.one_hot(i, output_size)
            pi = policy * x
            spi = tf.reduce_sum(pi, axis=-1)
            self.add_summary(tf.summary.scalar("policy_{0}".format(i), tf.reduce_mean(spi)))

        log_softmax = tf.nn.log_softmax(policy)
        self.add_summary(tf.summary.scalar("log_softmax", tf.reduce_mean(log_softmax)))

        log_softmax_logexp = tf.log(tf.reduce_sum(tf.exp(policy)))
        self.add_summary(tf.summary.scalar("log_softmax_logexp_mean", tf.reduce_mean(log_softmax_logexp)))

        log_probability_per_action = tf.reduce_sum(log_softmax * actions, axis=-1, keep_dims=True)
        self.add_summary(tf.summary.scalar("log_probability_mean", tf.reduce_mean(log_probability_per_action)))

        advantage = reward - tf.stop_gradient(self.value)
        self.add_summary(tf.summary.scalar("advantage_mean", tf.reduce_mean(advantage)))

        self.cost_policy = -advantage * log_probability_per_action
        tf.losses.add_loss(self.cost_policy)
        self.add_summary(tf.summary.scalar("cost_policy_mean", tf.reduce_mean(self.cost_policy)))


        self.cost_value = tf.reduce_mean(tf.square(reward - self.value), axis=-1, keep_dims=True)
        tf.losses.add_loss(self.cost_value)
        self.add_summary(tf.summary.scalar("cost_value_mean", tf.reduce_mean(self.cost_value)))


        xentropy = tf.reduce_sum(self.policy * log_softmax, axis=-1, keep_dims=True)
        self.add_summary(tf.summary.scalar("xentropy_mean", tf.reduce_mean(xentropy)))

        xentropy_loss = xentropy * self.xentropy_reg_beta
        tf.losses.add_loss(xentropy_loss)
        #self.add_summary(tf.summary.scalar("xentropy_loss_mean", tf.reduce_mean(xentropy_loss)))

        policy_l2_loss = tf.reduce_sum(tf.square(policy), axis=-1, keep_dims=True)
        self.add_summary(tf.summary.scalar("policy_l2_loss", tf.reduce_mean(policy_l2_loss)))
        tf.losses.add_loss(policy_l2_loss * self.policy_reg_beta)

        self.add_summary(tf.summary.scalar("input_reward_mean", tf.reduce_mean(reward)))
        self.add_summary(tf.summary.scalar("value_mean", tf.reduce_mean(self.value)))

        self.losses = tf.losses.get_total_loss()
        self.add_summary(tf.summary.scalar("loss_mean", tf.reduce_mean(self.losses)))

    def add_summary(self, s):
        self.summary_all.append(s)

    def setup_gradient_stats(self, opt):
        grads = opt.compute_gradients(self.losses)
        gradients, variables = zip(*grads)
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
        grads = zip(gradients, variables)

        for name in self.clip_names:
            reduced_max = []
            reduced_min = []
            reduced_mean = []
            for grad, var in grads:
                if name in var.name:
                    #print "{0} -> {1}".format(grad, var)
                    reduced_max.append(tf.reduce_max(grad))
                    reduced_min.append(tf.reduce_min(grad))
                    reduced_mean.append(tf.reduce_mean(grad))

            n = name.split('/')[-1]

            max_grad = tf.reduce_max(reduced_max)
            self.add_summary(tf.summary.scalar("{0}.grad_max".format(n), max_grad))

            min_grad = tf.reduce_min(reduced_min)
            self.add_summary(tf.summary.scalar("{0}.grad_min".format(n), min_grad))

            mean_grad = tf.reduce_mean(reduced_mean)
            self.add_summary(tf.summary.scalar("{0}.grad_mean".format(n), mean_grad))

    def want_clip(self, name):
        for n in self.clip_names:
            if n in name:
                return True

        return False

    def setup_clipped_train(self, opt):
        gradients, variables = zip(*opt.compute_gradients(self.losses))
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
        return opt.apply_gradients(zip(gradients, variables), global_step=self.global_step)

    def do_init(self, config):
        self.summary_all = []
        self.episode_stats_update = []
        self.summary_apply_gradients = []

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        if self.learning_rate == 0:
            self.learning_rate = tf.train.polynomial_decay(self.learning_rate_start, self.global_step,
                self.learning_rate_decay_steps, self.learning_rate_end)

        self.add_summary(tf.summary.scalar('learning_rate', self.learning_rate))

        rewards = tf.placeholder(tf.float32, [None], name='episode_rewards')
        rewards_summary = []
        rewards_summary.append(tf.summary.scalar("episode_rewards_mean", tf.reduce_mean(rewards)))
        rewards_summary.append(tf.summary.scalar("episode_rewards_max", tf.reduce_max(rewards)))
        rewards_summary.append(tf.summary.scalar("episode_rewards_min", tf.reduce_min(rewards)))
        self.update_rewards_ops = tf.summary.merge(rewards_summary)

        self.init_model(config)

        opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, use_locking=True, name='optimizer')

        self.train_clipped_step = self.setup_clipped_train(opt)
        self.setup_gradient_stats(opt)

        self.assign_ops = []
        self.transform_variables = []
        for v in tf.trainable_variables():
            if self.scope != get_scope_name(v.name):
                continue

            ev = tf.placeholder(tf.float32, None, name=get_transform_placeholder_name(v.name))
            self.assign_ops.append(tf.assign(v, ev, validate_shape=False))

            self.transform_variables.append(v)
            #print "{0}: transform variable: {1}".format(self.scope, v)

        self.summary_merged = tf.summary.merge(self.summary_all)

    def predict(self, states):
        p, v = self.sess.run([self.policy, self.value], feed_dict={
                self.scope + '/x:0': states,
            })
        return p, v
    
    def predict_actions(self, states):
        p = self.sess.run([self.policy], feed_dict={
                self.scope + '/x:0': states,
            })
        return p[0]
    
    def predict_values(self, states):
        v = self.sess.run([self.value], feed_dict={
                self.scope + '/x:0': states,
            })
        return v[0]

    def train(self, states, action, reward):
        self.train_num += 1
        feed_dict = {
            self.scope + '/x:0': states,
            self.scope + '/action:0': action,
            self.scope + '/reward:0': reward,
        }

        if self.train_num % self.summary_flush_num == 0:
            ops = [self.summary_merged, self.train_clipped_step, self.global_step]
            summary = self.sess.run(ops, feed_dict)
            self.summary_writer.add_summary(summary[0], summary[-1])
        else:
            self.sess.run(self.train_clipped_step, feed_dict)

    def export_params(self):
        res = self.sess.run(self.transform_variables)
        d = {}
        for k, v in zip(self.transform_variables, res):
            #print "export: scope: {0}, key: {1}".format(self.scope, k)
            d[k.name] = v
        return d

    def import_params(self, d, self_rate):
        def name(name):
            return self.scope + '/' + get_transform_placeholder_name(name) + ':0'

        ext_d = {}
        for k, ext_v in d.iteritems():
            ext_d[name(k)] = ext_v

            #if 'policy_layer/kernel' in k:
            #    print "exported {0}: name: {1}, value: {2}".format(k, name(k), ext_v)

        import_d = {}
        for k, self_v in self.export_params().iteritems():
            tn = name(k)

            ext_var = ext_d.get(tn, self_v)

            import_d[tn] = self_v * self_rate + ext_var * (1. - self_rate)

            #if 'policy_layer/kernel' in k:
            #    print "import: scope: {0}, name: {1}, self_rate: {2}, self_v: {3}, ext_var: {4}, saving: {5}".format(
            #            self.scope, tn, self_rate, self_v, ext_var, import_d[tn])

        #print("{0}: imported params: {1}, total params: {2}".format(self.scope, len(d), len(d1)))
        self.sess.run(self.assign_ops, feed_dict=import_d)

    def update_rewards(self, rewards):
        feed_dict = {
            self.scope + '/episode_rewards:0': rewards,
        }

        s = self.sess.run([self.update_rewards_ops, self.global_step], feed_dict)
        self.summary_writer.add_summary(s[0], s[1])

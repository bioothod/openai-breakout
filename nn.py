import tensorflow as tf

import numpy as np

RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def get_param_name(s):
    return s.split('/', 1)[1].replace('/', 'X').split(':')[0]
def get_scope_name(s):
    return s.split('/')[0].split(':')[0]
def get_transform_placeholder_name(s):
    return get_param_name(s) + '_ext'

class nn(object):
    def __init__(self, scope, input_shape, output_size, summary_writer):
        self.reward_mean = 0.0
        self.reward_mean_alpha = 0.9

        print "going to initialize scope %s" % scope
        self.summary_writer = summary_writer
        self.scope = scope
        with tf.variable_scope(scope) as vscope:
            self.vscope = vscope
            self.do_init(input_shape, output_size)
            print "scope %s has been initialized" % scope
        
        self.saver = tf.train.Saver()

    def init_model(self, input_shape, output_size):
        print "init_model scope: %s" % (tf.get_variable_scope().name)

        x = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], input_shape[2]], name='x')
        action = tf.placeholder(tf.int32, [None, 1], name='action')
        reward = tf.placeholder(tf.float32, [None, 1], name='reward')

        self.add_summary(tf.summary.histogram('action', action))
        self.add_summary(tf.summary.histogram('reward', reward))

        input_layer = tf.reshape(x, [-1, input_shape[0], input_shape[1], input_shape[2]])

        c1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
        p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=2, strides=2, padding='same')
        
        c2 = tf.layers.conv2d(inputs=p1, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
        p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=2, strides=2, padding='same')
        
        c3 = tf.layers.conv2d(inputs=p2, filters=64, kernel_size=4, padding='same', activation=tf.nn.relu)
        p3 = tf.layers.max_pooling2d(inputs=c3, pool_size=2, strides=2, padding='same')
        
        c4 = tf.layers.conv2d(inputs=p3, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
        p4 = tf.layers.max_pooling2d(inputs=c4, pool_size=2, strides=2, padding='same')

        flat = tf.reshape(p4, [-1, np.prod(p4.get_shape().as_list()[1:])])

        dense = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)

        policy = tf.layers.dense(inputs=dense, units=output_size)
        self.policy = tf.nn.softmax(policy)
        self.value = tf.layers.dense(inputs=dense, units=1)

        actions = tf.one_hot(action, output_size)
        actions = tf.squeeze(actions, 1)

        log_softmax = tf.nn.log_softmax(policy)
        self.add_summary(tf.summary.scalar("log_softmax", tf.reduce_mean(log_softmax)))
        self.add_summary(tf.summary.scalar("policy_softmax", tf.reduce_mean(self.policy)))

        log_softmax_logexp = tf.log(tf.reduce_sum(tf.exp(policy)))
        self.add_summary(tf.summary.scalar("log_softmax_logexp_mean", tf.reduce_mean(log_softmax_logexp)))

        log_probability_per_action = tf.reduce_sum(log_softmax * actions, axis=-1, keep_dims=True)
        self.add_summary(tf.summary.scalar("log_probability_mean", tf.reduce_mean(log_probability_per_action)))

        advantage = (reward - tf.stop_gradient(self.value))
        self.add_summary(tf.summary.scalar("advantage_mean", tf.reduce_mean(advantage)))

        self.cost_policy = -advantage * log_probability_per_action
        tf.losses.add_loss(self.cost_policy)
        self.add_summary(tf.summary.scalar("cost_policy_mean", tf.reduce_mean(self.cost_policy)))


        self.cost_value = tf.reduce_mean(tf.square(self.value - reward), axis=-1, keep_dims=True)
        tf.losses.add_loss(self.cost_value)
        self.add_summary(tf.summary.scalar("cost_value_mean", tf.reduce_mean(self.cost_value)))


        xentropy = tf.reduce_sum(self.policy * log_softmax, axis=-1, keep_dims=True)
        self.add_summary(tf.summary.scalar("xentropy_mean", tf.reduce_mean(xentropy)))

        xentropy_loss = xentropy * self.reg_beta
        tf.losses.add_loss(xentropy_loss)
        self.add_summary(tf.summary.scalar("xentropy_loss_mean", tf.reduce_mean(xentropy_loss)))


        self.add_summary(tf.summary.scalar("input_reward_mean", tf.reduce_mean(reward)))
        self.add_summary(tf.summary.scalar("value_mean", tf.reduce_mean(self.value)))
        self.add_summary(tf.summary.scalar("policy_mean", tf.reduce_mean(policy)))

        self.losses = tf.losses.get_total_loss()
        self.add_summary(tf.summary.scalar("loss_mean", tf.reduce_mean(self.losses)))


    def add_summary(self, s):
        self.summary_all.append(s)

    def setup_gradients(self, prefix, opt, cost):
        grads = opt.compute_gradients(cost)
        ret_grads = []
        ret_names = []
        ret_apply = []

        for e in grads:
            grad, var = e

            if grad is None or var is None:
                continue

            #print "var: %s, gradient: %s" % (var, grad)
            if self.scope != get_scope_name(var.name):
                continue

            gname = get_param_name(grad.name)
            print "gradient %s -> %s" % (var, gname)

            # get all gradients
            ret_grads.append(grad)
            ret_names.append(gname)

            pl = tf.placeholder(tf.float32, shape=var.get_shape(), name=gname)
            clip = tf.clip_by_average_norm(pl, 0.01)
            ret_apply.append((clip, var))

            ag = tf.summary.histogram('%s/apply_%s'% (prefix, gname), clip)
            self.summary_apply_gradients.append(ag)

        return ret_grads, ret_names, ret_apply

    def do_init(self, input_shape, output_size):
        self.learning_rate_start = 0.0003
        self.reg_beta_start = 0.01
        self.transform_rate_start = 1.0

        self.train_num = 0

        self.summary_all = []
        self.episode_stats_update = []
        self.summary_apply_gradients = []

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        #self.transform_lr = 0.00001 + tf.train.exponential_decay(self.transform_lr_start, self.global_step, 100000, 0.6, staircase=True)
        #self.learning_rate = 0.0003 + tf.train.exponential_decay(self.learning_rate_start, self.global_step, 100000, 0.9, staircase=True)
        self.learning_rate = 0.0005
        self.reg_beta = 0.01
        self.transform_rate = 0.9
        #self.reg_beta = 0.0001 + tf.train.exponential_decay(self.reg_beta_start, self.global_step, 100000, 1.5, staircase=True)

        self.add_summary(tf.summary.scalar('reg_beta', self.reg_beta))
        #self.add_summary(tf.summary.scalar('transform_lr', self.transform_lr))
        self.add_summary(tf.summary.scalar('learning_rate', self.learning_rate))

        #episodes_passed_p = tf.placeholder(tf.int32, [], name='episodes_passed')
        #episode_reward_p = tf.placeholder(tf.float32, [], name='episode_reward')
        #total_actions_p = tf.placeholder(tf.float32, [], name='total_actions')
        #random_alpha_p = tf.placeholder(tf.float32, [], name='random_alpha')

        reward_mean_p = tf.placeholder(tf.float32, [], name='reward_mean')
        self.add_summary(tf.summary.scalar("reward_mean", reward_mean_p))

        self.init_model(input_shape, output_size)

        opt = tf.train.RMSPropOptimizer(self.learning_rate,
                RMSPROP_DECAY,
                momentum=RMSPROP_MOMENTUM,
                epsilon=RMSPROP_EPSILON, name='optimizer')

        self.train_step = opt.minimize(self.losses, global_step=self.global_step)

        self.gradient_names_policy = []
        self.apply_grads_policy = []

        self.gradient_names_value = []
        self.apply_grads_value = []

        self.compute_gradients_step_policy, self.gradient_names_policy, self.apply_grads_policy = self.setup_gradients("policy", opt, self.cost_policy)

        self.compute_gradients_step_value, self.gradient_names_value, self.apply_grads_value = self.setup_gradients("value", opt, self.cost_value)

        apply_gradients = self.apply_grads_policy + self.apply_grads_value

        self.apply_gradients_step = opt.apply_gradients(apply_gradients, global_step=self.global_step)

        self.assign_ops = []
        self.transform_variables = []
        for v in tf.trainable_variables():
            if self.scope != get_scope_name(v.name):
                continue

            ev = tf.placeholder(tf.float32, None, name=get_transform_placeholder_name(v.name))
            self.assign_ops.append(tf.assign(v, ev, validate_shape=False))

            self.transform_variables.append(v)
            print "{0}: transform variable: {1}".format(self.scope, v)

        config=tf.ConfigProto(
                intra_op_parallelism_threads = 8,
                inter_op_parallelism_threads = 8,
            )
        #self.sess = tf.Session(config=config)
        self.sess = tf.Session()
        self.summary_merged = tf.summary.merge(self.summary_all)
        self.summary_apply_gradients_merged = tf.summary.merge(self.summary_apply_gradients)

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.sess.run(init)


    def update_gradients(self, states, dret, names, grads):
        for gname, grad in zip(names, grads):
            if np.isnan(grad).any():
                continue

            #value = np.sum(grad) / float(len(states))
            #value = grad / float(len(states))
            value = grad

            g = dret.get(gname)
            if g:
                g.update(value)
            else:
                dret[gname] = value
            #print "computed gradients %s, shape: %s" % (gname, grad.shape)
            #print grad

    def compute_gradients(self, states, action, reward):
        self.train_num += 1

        ops = [self.summary_merged, self.compute_gradients_step_policy, self.compute_gradients_step_value]
        summary, grads_policy, grads_value = self.sess.run(ops, feed_dict={
                self.scope + '/x:0': states,
                self.scope + '/action:0': action,
                self.scope + '/reward:0': reward,

                self.scope + '/reward_mean:0': self.reward_mean,
            })
        self.summary_writer.add_summary(summary, self.train_num)

        dret = {}
        self.update_gradients(states, dret, self.gradient_names_policy, grads_policy)
        self.update_gradients(states, dret, self.gradient_names_value, grads_value)
        return dret


    def apply_gradients(self, grads):
        if len(grads) == 0:
            print "empty gradients to apply"
            return

        feed_dict = {}
        #print "apply: %s" % grads
        for n, g in grads.iteritems():
            gname = self.scope + '/' + n + ':0'
            #print "apply gradients to %s, shape: %s" % (gname, g.shape)
            #print g
            feed_dict[gname] = g

        ops = [self.apply_gradients_step, self.summary_apply_gradients_merged]
        grads, apply_summary = self.sess.run(ops, feed_dict=feed_dict)
        self.summary_writer.add_summary(apply_summary, self.train_num)

    def predict(self, states):
        p = self.sess.run([self.policy, self.value], feed_dict={
                self.scope + '/x:0': states,
            })
        return p

    def train(self, states, action, reward):
        self.train_num += 1

        ops = [self.summary_merged, self.train_step]
        summary = self.sess.run(ops, feed_dict={
                self.scope + '/x:0': states,
                self.scope + '/action:0': action,
                self.scope + '/reward:0': reward,

                self.scope + '/reward_mean:0': self.reward_mean,
            })
        self.summary_writer.add_summary(summary[0], self.train_num)

    def export_params(self):
        res = self.sess.run(self.transform_variables)
        d = {}
        for k, v in zip(self.transform_variables, res):
            #print "export: {0}: {1}".format(self.scope, k)
            d[k] = v
        return d

    def import_params(self, d, rate):
        self.train_num += 1

        def name(v):
            return self.scope + '/' + get_transform_placeholder_name(v.name) + ':0'

        d1 = {}
        for k, v in d.iteritems():
            d1[name(k)] = v

        for k, v in self.export_params().iteritems():
            var = d1.get(name(k), v)

            d1[name(k)] = v * rate + var * (1. - rate)

        print "{0}: imported params: {1}, total params: {2}".format(self.scope, len(d), len(d1))
        self.sess.run(self.assign_ops, feed_dict=d1)

    def update_reward(self, r):
        self.reward_mean = self.reward_mean_alpha * self.reward_mean + (1. - self.reward_mean_alpha) * r

    def save(self, path):
        if self.saver:
            self.saver.save(self.sess, path, global_step=self.global_step)

    def restore(self, path):
        if self.saver:
            self.saver.restore(self.sess, path)

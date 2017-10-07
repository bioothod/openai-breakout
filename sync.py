import tensorflow as tf
import numpy as np

import logging
import threading
import time

import env
import nn
import runner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class sync(object):
    def __init__(self, config):
        self.save_per_total_steps = None
        self.save_per_minutes = None

        self.thread_num = config.get('thread_num')
        self.env_num = config.get('env_num')
        self.lock = threading.Lock()
        self.episode_rewards = [[] for i in range(self.thread_num)]

        output_path = config.get("output_path")
        if output_path:
            config.put('summary_writer', tf.summary.FileWriter(output_path))

        self.coord = tf.train.Coordinator()

        self.env_sets = []

        dummy_env = env.env_holder('dummy', config)
        config.put('output_size', dummy_env.osize)

        self.saved_total_steps = 0
        self.saved_time = 0
        save_path = config.get('save_path')
        if save_path:
            self.save_path = save_path

            self.save_per_total_steps = config.get('save_per_total_steps', 10000)
            self.save_per_minutes = config.get('save_per_minutes')

        self.master = nn.nn('master', config)
        config.put('session', self.master.sess)

        self.runners = []
        for r in range(self.thread_num):
            n = nn.nn('r{0}'.format(r), config)
            e = env.env_set(r, config)
            self.runners.append(runner.runner(r, self.master, n, e, config))

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.master.sess.run(init)

        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='master')

        self.saver = tf.train.Saver(var_list=save_vars, max_to_keep=config.get('save_max_to_keep', 5))
        load_path = config.get('load_path')
        if load_path:
            self.restore(load_path)

            if config.get('global_step_reset', False):
                self.master.sess.run([tf.assign(self.master.global_step, 0)])

        cmp_dict = {}
        for k, v in self.master.export_params().iteritems():
            cmp_dict[nn.get_transform_placeholder_name(k)] = v
        for r in self.runners:
            r.network.import_params(self.master.export_params(), 0)
            for k, v in r.network.export_params().iteritems():
                master_v = cmp_dict[nn.get_transform_placeholder_name(k)]
                assert((master_v == v).all())

    def start(self):
        threads = [threading.Thread(target=r.run, args=(self.coord, self.check_save)) for r in self.runners]
        for t in threads:
            t.start()

        try:
            self.coord.join(threads)
        except:
            self.coord.request_stop()
            self.coord.join(threads)

    def save(self):
        if self.saver:
            self.saver.save(self.master.sess, self.save_path, global_step=self.master.global_step)
            logger.info("Network params have been saved to {0}".format(self.save_path))

    def restore(self, path):
        if self.saver:
            self.saver.restore(self.master.sess, path)
            logger.info("Network params have been loaded from {0}".format(path))

    def check_save(self, total_steps, rid, episode_rewards):
        self.lock.acquire()
        try:
            self.episode_rewards[rid] = episode_rewards

            rewards = [r for runner_rewards in self.episode_rewards for r in runner_rewards]
            self.master.update_rewards(rewards)

            if not self.save_path:
                return

            if self.save_per_total_steps or self.save_per_minutes:
                if total_steps >= self.saved_total_steps + self.save_per_total_steps or time.time() > self.saved_time + self.save_per_minutes * 60:
                    self.saved_time = time.time()
                    self.saved_total_steps = total_steps
                    self.save()
        finally:
            self.lock.release()

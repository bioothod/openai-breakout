import tensorflow as tf
import numpy as np

import threading
import time

import env
import nn
import runner

class sync(object):
    def __init__(self, config):
        self.save_per_total_steps = None
        self.save_per_minutes = None

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

        self.runners = []

        config.put('session', self.master.sess)

        for r in range(config.get('thread_num')):
            n = nn.nn('r{0}'.format(r), config)
            e = env.env_set(r, config)
            self.runners.append(runner.runner(self.master, n, e, config))

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.master.sess.run(init)

        for r in self.runners:
            r.network.import_params(self.master.export_params(), 0)

    def start(self):
        threads = [threading.Thread(target=r.run, args=(self.coord, self.check_save)) for r in self.runners]
        for t in threads:
            t.start()

        try:
            self.coord.join(threads)
        except:
            self.coord.request_stop()
            self.coord.join(threads)

    def check_save(self, total_steps):
        if not self.save_path:
            return

        if self.save_per_total_steps:
            if total_steps >= self.saved_total_steps + self.save_per_total_steps:
                self.master.save(self.save_path)
                self.saved_time = time.time()
                self.saved_total_steps = total_steps
                return

        if self.save_per_minutes:
            if time.time() > self.saved_time + self.save_per_minutes * 60:
                self.master.save(self.save_path)
                self.saved_time = time.time()
                self.saved_total_steps = total_steps
                return

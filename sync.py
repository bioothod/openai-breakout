import tensorflow as tf
import numpy as np

import threading

import env
import nn
import runner

class sync(object):
    def __init__(self, nr_runners, config):
        self.swriter = None
        self.save_per_total_steps = None
        self.save_per_minutes = None

        output_path = config.get("output_path")
        if output_path:
            self.swriter = tf.summary.FileWriter(output_path)

        self.coord = tf.train.Coordinator()

        self.env_sets = [env.env_set(r, config) for r in range(nr_runners)]

        self.network = self.init_network(config)

        self.saved_total_steps = 0
        self.saved_time = 0
        save_path = config.get('save_path')
        if save_path:
            self.save_path = save_path

            self.save_per_total_steps = config.get('save_per_total_steps', 10000)
            self.save_per_minutes = config.get('save_per_minutes')

        self.runners = [runner.runner(self.network, config) for r in range(nr_runners)]

    def init_network(self, config):
        state_steps = config.get("state_steps")
        config_input_shape = config.get("input_shape")

        input_shape = (config_input_shape[0], config_input_shape[1], config_input_shape[2] * state_steps)
        osize = config.get('output_size')

        return nn.nn("main", input_shape, osize, self.swriter)

    def start(self):
        threads = [threading.Thread(target=r.run, args=(es.envs, self.coord, self.check_save)) for r, es in zip(self.runners, self.env_sets)]
        for t in threads:
            t.start()

        try:
            self.coord.join(threads)
        except:
            self.coord.request_stop()
            self.coord.join(threads)

    def check_save(self):
        if not self.save_path:
            return

        #if self.save_per_total_steps:
        #    if self.total_steps >= self.saved_total_steps + self.save_per_total_save:
        #        self.network.save(self.save_path)
        #        self.saved_time = time.time()
        #        self.saved_total_steps = self.total_steps
        #        return

        if self.save_per_minutes:
            if time.time() > self.saved_time + self.save_per_minute * 60:
                self.network.save(self.save_path)
                self.saved_time = time.time()
                #self.saved_total_steps = self.total_steps
                return

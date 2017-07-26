import tensorflow as tf
import numpy as np

import math
import threading

import env
import gradient
import nn

class runner(object):
    def __init__(self, network, config):
        self.config = config

        self.grads = {}
        self.gradient_update_step = config.get("gradient_update_step")

        self.gamma = config.get("gamma")

        self.update_reward_steps = config.get("update_reward_steps")

        self.total_steps = 0
        self.total_actions = 0

        self.ra_range_begin = config.get("ra_range_begin")
        self.ra_alpha_cap = config.get("ra_alpha_cap")
        self.ra_alpha = config.get("ra_alpha")

        self.batch = []
        self.batch_size = config.get("batch_size")
 
        self.state_steps = config.get("state_steps")
        self.input_shape = config.get("input_shape")

        self.save_per_total_steps = config.get('save_per_total_steps')
        self.save_per_minutes = config.get('save_per_minutes')
        self.save_timer = time.time()

        self.network = network

    def get_actions(self, states):
        input = [s.read() for s in states]

        def random_choice(p):
            ra_alpha = 0.1
            random_choice = np.random.choice([True, False], p=[ra_alpha, 1.-ra_alpha])

            if random_choice:
                return np.random.randint(0, len(p))

            return np.random.choice(len(p), p=p)

        action_probs, values = self.network.predict(input)
        #actions = [random_choice(p) for p in action_probs]
        actions = [np.random.choice(len(p), p=p) for p in action_probs]

        return actions, values

    def calc_grads(self, states, action, reward, done):
        grads = self.network.compute_gradients(states, action, reward)
        if grads:
            for k, v in grads.iteritems():
                e = self.grads.get(k)
                if e:
                    e.update(v)
                else:
                    self.grads[k] = gradient.gradient(v)

        self.total_steps += 1

        if self.total_steps % self.gradient_update_step == 0 or done:
            grads = {}
            for n, g in self.grads.iteritems():
                grads[n] = g.read()

            self.network.apply_gradients(grads)

            for n, g in self.grads.iteritems():
                g.clear()

    def update_episode_stats(self, episodes, reward):
        self.network.update_episode_stats(episodes, reward, self.total_actions, self.ra_alpha)

    def run_sample(self, batch):
        states_shape = (len(batch), self.input_shape[0], self.input_shape[1], self.input_shape[2] * self.state_steps)
        states = np.zeros(shape=states_shape)
        new_states = np.zeros(shape=states_shape)

        rashape = (len(batch), 1)
        reward = np.zeros(shape=rashape)
        action = np.zeros(shape=rashape)

        idx = 0
        for e in batch:
            s, a, r, sn, done = e

            states[idx] = s.read()
            new_states[idx] = sn.read()
            action[idx] = a
            reward[idx] = r
            idx += 1

        self.network.train(states, action, reward)
        #self.calc_grads(states, action, reward, True)

    def run_batch(self, h):
        if len(h) == 0:
            return

        self.run_sample(h)

    def update_reward(self, e, done):
        rev = 0.0
        if not done:
            rev = e.last_value

        h = []
        for elm in reversed(e.history.history):
            s, a, r, sn, done = elm
            rev = r + self.gamma * rev

            h.append((s, a, rev, sn, done))

        self.batch += h
        if len(self.batch) >= self.batch_size:
            self.run_batch(self.batch)
            self.batch = []

    def run(self, envs, coord):
        states = [e.reset() for e in envs]

        while not coord.should_stop():
            actions, values = self.get_actions(states)
            new_states = []
            for e, s, a, v in zip(envs, states, actions, values):
                sn, reward, done = e.step(s, a)

                if done or e.total_steps % self.update_reward_steps == 0:
                    e.last_value = v

                    self.update_reward(e, done)

                    e.clear()

                    if done:
                        self.network.update_reward(e.creward)

                        e.clear_stats()
                        sn = e.reset()

                new_states.append(sn)

                self.check_save()

            states = new_states

        coord.request_stop()

class sync(object):
    def __init__(self, nr_runners, config):
        self.swriter = None
        self.saver = None
        self.save_per_total_steps = None
        self.save_per_minutes = None

        output_path = config.get("output_path")
        if output_path:
            self.swriter = tf.summary.FileWriter(output_path)

        self.saved_total_steps = 0
        self.saved_time = 0
        save_path = config.get('save_path')
        if save_path:
            self.saver = tf.train.Saver()
            self.save_path = save_path

            self.save_per_total_steps = config.get('save_per_total_steps', 10000)
            self.save_per_minutes = config.get('save_per_minutes')

        self.coord = tf.train.Coordinator()

        self.env_sets = [env.env_set(r, config) for r in range(nr_runners)]

        self.network = self.init_network(config)

        self.runners = [runner(self.network, config) for r in range(nr_runners)]

    def init_network(self, config):
        state_steps = config.get("state_steps")
        config_input_shape = config.get("input_shape")

        input_shape = (config_input_shape[0], config_input_shape[1], config_input_shape[2] * state_steps)
        osize = config.get('output_size')

        return nn.nn("main", input_shape, osize, self.swriter)

    def start(self):
        threads = [threading.Thread(target=r.run, args=(es.envs, self.coord,)) for r, es in zip(self.runners, self.env_sets)]
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

        if self.save_per_total_steps:
            if self.total_steps >= self.saved_total_steps + self.save_per_total_save:
                self.network.save(self.save_path)
                self.saved_time = time.time()
                self.saved_total_steps = self.total_steps
                return

        if self.save_per_minutes:
            if time.time() > self.saved_time + self.save_per_minute * 60:
                self.network.save(self.save_path)
                self.saved_time = time.time()
                self.saved_total_steps = self.total_steps
                return

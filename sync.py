import tensorflow as tf
import numpy as np

import math
import threading

import env
import gradient
import nn

class runner(object):
    def __init__(self, rid, config, train_mode):
        self.swriter = None
        output_path = config.get("output_path")
        if output_path:
            output_path += '.%s' % (rid)
            self.swriter = tf.summary.FileWriter(output_path)
       
        self.rid = rid
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

        self.osize = config.get("output_size")

        input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2] * self.state_steps)

        self.network = nn.nn(rid, input_shape, self.osize, self.swriter, train_mode)

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

        have_done = False

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

    def run(self, envs):
        states = [e.reset() for e in envs]

        while True:
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

            states = new_states


class sync(object):
    def __init__(self, nr_runners, config, train_mode):
        self.envs = []
        for i in range(nr_runners):
            rid = 'runner%02d' % i
            
            e = env.env_holder(rid, config)
            self.envs.append(e)
        
        config.put('output_size', self.envs[0].osize)
        self.network = runner('main', config, train_mode)

    def start(self):
        self.network.run(self.envs)

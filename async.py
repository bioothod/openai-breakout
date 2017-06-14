import tensorflow as tf
import numpy as np

from collections import deque
from copy import deepcopy

import gym
import math
import threading

import gradient
import history
import nn
import state

class runner(object):
    def __init__(self, rid, config):
        self.swriter = None
        output_path = config.get("output_path")
        if output_path:
            output_path += '.%s' % (rid)
            self.swriter = tf.summary.FileWriter(output_path)
       
        self.rid = rid
        self.config = config

        self.env = gym.make(config.get("game"))

        self.grads = {}
        self.gradient_update_step = config.get("gradient_update_step", 10)

        self.gamma = config.get("gamma", 0.99)

        self.update_reward_steps = config.get("update_reward_steps", 128)

        self.total_steps = 0
        self.total_actions = 0

        self.ra_range_begin = config.get("ra_range_begin", 0.1)
        self.ra_alpha_cap = config.get("ra_alpha_cap", 0.5)
        self.ra_alpha = config.get("ra_alpha", 0.1)

        self.batch_size = config.get("batch_size", 512)

        self.preprocess = config.get("preprocess")

        self.state_steps = config.get("state_steps", 2)
        self.state_size = config.get("state_size", 1600)
        self.current_state = state.state(self.state_size, self.state_steps)

        self.isize = self.state_size * self.state_steps
        self.osize = self.env.action_space.n

        self.history_size = config.get("history_size", 100)
        self.history = history.history(self.history_size)

        self.network = nn.nn(rid, self.isize, self.osize, self.swriter)
        self.target = config.get("target")
        if self.target:
            self.network.import_params(self.target.export_params())

    def get_action(self, s):
        self.total_actions += 1
        self.ra_alpha = self.ra_range_begin + self.ra_alpha_cap * math.exp(-0.00001 * self.total_actions)
        random_choice = np.random.choice([True, False], p=[self.ra_alpha, 1-self.ra_alpha])

        if random_choice:
            return np.random.randint(0, self.osize)

        q = self.network.predict_policy(s.vector())
        return np.argmax(q[0])

    def new_state(self, obs):
        I = obs
        if self.preprocess:
            I = self.preprocess(obs)
        I = np.concatenate(I)
        self.current_state.push_array(I)
        return deepcopy(self.current_state)

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

            if self.target:
                self.target.apply_gradients(grads)
                self.network.import_params(self.target.export_params())
            else:
                self.network.apply_gradients(grads)

            for n, g in self.grads.iteritems():
                g.clear()

    def update_episode_stats(self, episodes, reward):
        self.network.update_episode_stats(episodes, reward, self.total_actions, self.ra_alpha)

    def run_sample(self, batch):
        states_shape = (len(batch), self.isize)
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

        P = self.network.predict_policy(states)
        V = self.network.predict_value(new_states)

        have_done = False

        self.calc_grads(states, action, reward, True)

    def run_batch(self):
        if self.history.size() == 0:
            return

        batch = self.history.sample(self.batch_size)
        self.run_sample(batch)

    def update_reward(self, e, cr):
        rev = cr
        for i in range(e.size() - 1, 0, -1):
            s, a, r, sn, done = e.get(i)
            rev = r + self.gamma * rev

            self.history.append((s, a, rev, sn, done))

    def run_episode(self):
        observation = self.env.reset()
        s = self.new_state(observation)

        e = history.history(10000)

        done = False
        cr = 0
        steps = 0
        steps_after_update = 0
        while not done:
            a = self.get_action(s)
            new_observation, reward, done, info = self.env.step(a)
            #self.env.render()

            sn = self.new_state(new_observation)

            e.append((s, a, reward, sn, done))

            if reward != 0 or steps_after_update % self.update_reward_steps == 0:
                h = history.history(500)
                h.load(e.last(300))
                self.update_reward(h, reward)
                e.clear()
                self.run_batch()

                steps_after_update = 0

            if done:
                if cr == 0:
                    cr = -1
                    self.update_reward(e, cr)

            cr += reward
            steps += 1
            steps_after_update += 1

            s = sn

        self.run_batch()

        return steps, cr

    def run(self, coord):
        last_rewards = deque()
        last_rewards_size = 100

        total_steps = 0

        episodes = 0

        while not coord.should_stop():
            steps, cr = self.run_episode()

            self.update_episode_stats(episodes, cr)
            total_steps += steps

            if len(last_rewards) >= last_rewards_size:
                last_rewards.popleft()

            last_rewards.append(cr)
            mean = np.mean(last_rewards)
            std = np.std(last_rewards)

            print "%s: %4d: reward: %4d, total steps: %7d, mean reward over last %3d episodes: %.1f, std: %.1f" % (
                    self.rid, episodes, cr, total_steps, len(last_rewards), mean, std)

            episodes += 1

class async(object):
    def __init__(self, nr_runners, config):
        self.target_runner = runner('dummy', config)
        self.coord = tf.train.Coordinator()

        self.runners = []

        config.put('target', self.target_runner.network)
        for i in range(nr_runners):
            rid = 'runner%02d' % i
            
            r = runner(rid, config)
            self.runners.append(r)

    def start(self):
        threads = [threading.Thread(target=r.run, args=(self.coord,)) for r in self.runners]
        for t in threads:
            t.start()

        self.coord.join(threads)

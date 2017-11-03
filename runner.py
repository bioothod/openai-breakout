import numpy as np

from collections import deque

import logging

import gradient
import nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class runner(object):
    def __init__(self, rid, master, network, env_set, config):
        self.config = config
        self.rid = rid

        self.grads = {}

        self.gamma = config.get("gamma")

        self.update_reward_steps = config.get("update_reward_steps")

        self.total_steps = 0
        self.total_actions = 0

        self.import_self_weight = config.get('import_self_weight')

        self.will_train = config.get("will_train")

        self.batch = []
 
        self.state_steps = config.get("state_steps")
        self.input_shape = config.get("input_shape")

        self.envs = env_set.envs
        self.master = master
        self.network = network

    def get_actions(self, states):
        input = [s.read() for s in states]

        action_probs = self.network.predict_actions(input)
        if self.will_train:
            actions = [np.random.choice(len(p), p=p) for p in action_probs]
        else:
            actions = np.argmax(action_probs, axis=1)

        return actions
    
    def get_values(self, states):
        input = [s.read() for s in states]

        return self.network.predict_values(input)

    def run_sample(self, batch):
        states_shape = (len(batch), self.input_shape[0], self.input_shape[1], self.input_shape[2] * self.state_steps)
        states = np.zeros(shape=states_shape)

        rashape = (len(batch), 1)
        reward = np.zeros(shape=rashape)
        action = np.zeros(shape=rashape)

        idx = 0
        for e in batch:
            s, a, r, sn, done = e

            states[idx] = s.read()
            action[idx] = a
            reward[idx] = r
            idx += 1

        self.master.train(states, action, reward)
        self.network.import_params(self.master.export_params(), self.import_self_weight)

    def run_batch(self, h):
        if len(h) == 0:
            return

        self.run_sample(h)

    def update_reward(self, e, rev):
        if not self.will_train:
            return

        h = []
        for elm in reversed(e.history()):
            s, a, r, sn, done = elm
            rev = r + self.gamma * rev

            h.append((s, a, rev, sn, done))

        self.batch += h

    def run(self, coord, check_save):
        running_envs = self.envs
        states = [e.reset() for e in running_envs]

        total_steps = 0
        while not coord.should_stop():
            actions = self.get_actions(states)
            new_states = []
            new_running_envs = []
            done_envs = []
            for e, s, a in zip(running_envs, states, actions):
                sn, reward, done = e.step(s, a)

                if done:
                    logger.info("%s: %3d reward: %3d, total steps: %6d/%4d" % (
                            e.eid, e.episodes, e.creward, total_steps, e.total_steps_diff()))

                    self.update_reward(e, 0)

                    e.clear()
                    e.clear_stats()

                    done_envs.append(e)
                else:
                    new_states.append(sn)
                    new_running_envs.append(e)

            total_steps += 1
            if total_steps % self.update_reward_steps == 0 and self.will_train:
                if len(new_states) > 0:
                    estimated_values = self.get_values(new_states)
                    for e, rev in zip(new_running_envs, estimated_values):
                        self.update_reward(e, rev[0])
                        e.clear()

                self.run_batch(self.batch)
                self.batch = []

                check_save(total_steps, self.rid, [e.last_creward for e in self.envs])

            states = new_states + [e.reset() for e in done_envs]
            running_envs = new_running_envs + done_envs

        coord.request_stop()


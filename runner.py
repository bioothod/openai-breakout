import numpy as np

from collections import deque

import gradient
import nn

class runner(object):
    def __init__(self, master, network, env_set, config):
        self.config = config

        self.grads = {}

        self.gamma = config.get("gamma")

        self.update_reward_steps = config.get("update_reward_steps")

        self.total_steps = 0
        self.total_actions = 0

        self.last_rewards = deque()
        self.last_rewards_size = 100

        self.max_reward = 0

        self.import_self_weight = config.get('import_self_weight')

        self.ra_range_begin = config.get("ra_range_begin")
        self.ra_alpha_cap = config.get("ra_alpha_cap")
        self.ra_alpha = config.get("ra_alpha")

        self.batch = []
 
        self.state_steps = config.get("state_steps")
        self.input_shape = config.get("input_shape")

        self.envs = env_set.envs
        self.master = master
        self.network = network

    def get_actions(self, states):
        input = [s.read() for s in states]

        action_probs = self.network.predict_actions(input)
        actions = [np.random.choice(len(p), p=p) for p in action_probs]

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
        h = []
        for elm in reversed(e.history()):
            s, a, r, sn, done = elm
            rev = r + self.gamma * rev

            h.append((s, a, rev, sn, done))

        self.batch += h

    def run(self, coord, check_save):
        states = []
        running_envs = []
        episode_rewards = []
        total_steps = 0
        while not coord.should_stop():
            if len(running_envs) == 0:
                check_save(total_steps, episode_rewards)

                episode_rewards = []
                states = []
                running_envs = self.envs
                for e in running_envs:
                    states.append(e.reset())


            actions = self.get_actions(states)
            new_states = []
            new_running_envs = []
            for e, s, a in zip(running_envs, states, actions):
                sn, reward, done = e.step(s, a)

                if done:
                    if len(self.last_rewards) >= self.last_rewards_size:
                        self.last_rewards.popleft()

                    self.last_rewards.append(e.creward)

                    mean = np.mean(self.last_rewards)
                    max_last = np.max(self.last_rewards)

                    if e.creward > self.max_reward:
                        self.max_reward = e.creward

                    episode_rewards.append(e.creward)

                    print("%s: %3d %2d/%d reward: %3d/%3d/%3d, total steps: %6d/%4d, mean reward over last %3d episodes: %.1f, per episode: %.1f, min/max: %d/%d" % (
                            e.eid, e.episodes, len(running_envs), len(self.envs),
                            e.creward, max_last, self.max_reward, e.total_steps, e.total_steps_diff(),
                            len(self.last_rewards), mean,
                            np.mean(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)))
                    self.update_reward(e, 0)

                    e.clear()
                    e.clear_stats()
                else:
                    new_states.append(sn)
                    new_running_envs.append(e)

            total_steps += 1
            if total_steps % self.update_reward_steps == 0:
                if len(new_states) > 0:
                    estimated_values = self.get_values(new_states)
                    for e, rev in zip(new_running_envs, estimated_values):
                        self.update_reward(e, rev[0])
                        e.clear()

                self.run_batch(self.batch)
                self.batch = []

            states = new_states
            running_envs = new_running_envs

        coord.request_stop()


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
        self.batch_size = config.get("batch_size")
 
        self.state_steps = config.get("state_steps")
        self.input_shape = config.get("input_shape")

        self.envs = env_set.envs
        self.master = master
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

        self.master.train(states, action, reward)
        self.network.import_params(self.master.export_params(), self.import_self_weight)

    def run_batch(self, h):
        if len(h) == 0:
            return

        self.run_sample(h)

    def update_reward(self, e, done):
        local_batch = e.last(self.batch_size)

        rev = 0.0
        if not done:
            s, a, r, sn, done = local_batch[-1]

            _, rev = self.get_actions([sn])

        h = []
        for elm in reversed(local_batch):
            s, a, r, sn, done = elm
            rev = r + self.gamma * rev

            h.append((s, a, rev, sn, done))

        self.batch += h
        if len(self.batch) >= self.batch_size:
            self.run_batch(self.batch)
            self.batch = []

    def run(self, coord, check_save):
        states = []
        running_envs = []
        episode_rewards = []
        while not coord.should_stop():
            if len(running_envs) == 0:
                check_save(self.envs[0].total_steps, episode_rewards)

                running_envs = self.envs
                states = [e.reset() for e in running_envs]
                episode_rewards = []

            actions, values = self.get_actions(states)
            new_states = []
            new_running_envs = []
            for e, s, a, v in zip(running_envs, states, actions, values):
                sn, reward, done = e.step(s, a)

                if e.total_steps % self.update_reward_steps == 0 or done:
                    self.update_reward(e, done)

                    e.clear()

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

                    e.clear_stats()
                else:
                    new_states.append(sn)
                    new_running_envs.append(e)

            states = new_states
            running_envs = new_running_envs

        coord.request_stop()


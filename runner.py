import numpy as np

from collections import deque

import gradient

class runner(object):
    def __init__(self, network, follower, config):
        self.config = config

        self.grads = {}
        self.gradient_update_step = config.get("gradient_update_step")

        self.gamma = config.get("gamma")

        self.update_reward_steps = config.get("update_reward_steps")

        self.total_steps = 0
        self.total_actions = 0

        self.last_rewards = deque()
        self.last_rewards_size = 100

        self.max_reward = 0

        self.ra_range_begin = config.get("ra_range_begin")
        self.ra_alpha_cap = config.get("ra_alpha_cap")
        self.ra_alpha = config.get("ra_alpha")

        self.batch = []
        self.batch_size = config.get("batch_size")
 
        self.state_steps = config.get("state_steps")
        self.input_shape = config.get("input_shape")

        self.follower_update_steps = config.get('follower_update_steps')

        self.network = network
        self.follower = follower

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

        self.network.train_clipped(states, action, reward)
        #self.network.train(states, action, reward)
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

    def run(self, envs, coord, check_save):
        states = []
        running_envs = []
        while not coord.should_stop():
            if len(running_envs) == 0:
                running_envs = envs
                states = [e.reset() for e in running_envs]

            actions, values = self.get_actions(states)
            new_states = []
            new_running_envs = []
            for e, s, a, v in zip(running_envs, states, actions, values):
                e.last_value = v
                sn, reward, done = e.step(s, a)

                if e.total_steps % self.follower_update_steps == 0:
                    sync_follower = True

                if e.total_steps % self.update_reward_steps == 0:
                    self.update_reward(e, done)

                    e.clear()

                if done:
                    self.network.update_reward(e.creward)

                    if len(self.last_rewards) >= self.last_rewards_size:
                        self.last_rewards.popleft()

                    self.last_rewards.append(e.creward)

                    mean = np.mean(self.last_rewards)
                    std = np.std(self.last_rewards)
                    max_last = np.max(self.last_rewards)

                    if e.creward > self.max_reward:
                        self.max_reward = e.creward

                    print "%s: %2d/%2d %4d: reward: %4d/%4d/%4d, total steps: %7d, mean reward over last %3d episodes: %.1f, std: %.1f" % (
                            e.eid, len(running_envs), len(envs),
                            e.episodes, e.creward, max_last, self.max_reward, e.total_steps,
                            len(self.last_rewards), mean, std)


                    e.clear_stats()
                else:
                    new_states.append(sn)
                    new_running_envs.append(e)

                check_save()

            states = new_states
            running_envs = new_running_envs

        coord.request_stop()


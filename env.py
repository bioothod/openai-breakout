import numpy as np

from copy import deepcopy

import cv2
import gym

import state

class env_holder(object):
    def __init__(self, eid, config):
        self.env = gym.make(config.get("game"))
        monitor_dir = config.get('monitor_dir')
        if monitor_dir:
            self.env = gym.wrappers.Monitor(self.env, directory=monitor_dir, video_callable=False, write_upon_reset=True)

        self.osize = self.env.action_space.n
        self.eid = eid

        self.input_shape = config.get('input_shape')

        self.state_steps = config.get("state_steps")
        self.current_state = state.state(self.input_shape, self.state_steps)

        self.history_buffer = []

        self.creward = 0
        self.last_creward = 0

        self.episodes = 0
        self.total_steps = 0
        self.prev_total_steps = 0

    def new_state(self, state):
        #state = obs[35:195]
        #state = state[::, ::, 0]
        state = 0.2126 * state[:, :, 0] + 0.7152 * state[:, :, 1] + 0.0722 * state[:, :, 2]

        state = state.astype(np.float32)
        res = cv2.resize(state, (self.input_shape[0], self.input_shape[1]))
        #res /= 255.

        res = np.reshape(res, self.input_shape)

        self.current_state.push_tensor(res)
        return deepcopy(self.current_state)

    def reset(self):
        self.current_state = state.state(self.input_shape, self.state_steps)
        obs = self.env.reset()
        return self.new_state(obs)

    def step(self, s, action):
        obs, reward, done, info = self.env.step(action)
        sn = self.new_state(obs)
        self.history_buffer.append((s, action, reward, sn, done))
        self.creward += reward
        self.total_steps += 1

        if done:
            self.episodes += 1

        #print "%s: %4d: reward: %4d, total steps: %7d, action: %d" % (
        #            self.eid, self.episodes, self.creward, self.total_steps, action)
        return sn, reward, done

    def clear_stats(self):
        self.last_creward = self.creward
        self.creward = 0
        self.last_value = 0.0
        self.prev_total_steps = self.total_steps

    def total_steps_diff(self):
        return self.total_steps - self.prev_total_steps

    def clear(self):
        self.history_buffer = []

    def history(self):
        return self.history_buffer

class env_set(object):
    def __init__(self, rid, config):
        self.envs = []

        env_num = config.get('env_num')

        for i in range(env_num):
            eid = 'r%02d.%02d' % (rid, i)

            e = env_holder(eid, config)

            self.envs.append(e)

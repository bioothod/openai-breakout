import time

import async
import config

def preprocess(obs):
    I = obs[35:195]
    I = I[::2,::2,0]
    return I

class breakout(object):
    def __init__(self):
        c = config.config()
        c.put('game', 'Breakout-v0')
        c.put('gradient_update_step', 10)
        c.put('gamma', 0.99)
        c.put('update_reward_steps', 128)
        c.put('ra_range_begin', 0.1)
        c.put('ra_alpha_cap', 0.5)
        c.put('ra_alpha', 0.1)
        c.put('batch_size', 512)
        c.put('preprocess', preprocess)
        c.put('state_steps', 2)
        c.put('state_size', 80*80)
        c.put('history_size', 100)
        c.put('output_path', 'output/%s.%d' % (c.get("game"), time.time()))

        self.async = async.async(3, c)

    def start(self):
        self.async.start()

if __name__ == '__main__':
    game = breakout()
    game.start()

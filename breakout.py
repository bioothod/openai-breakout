import time

import sync
import config

class breakout(object):
    def __init__(self):
        c = config.config()
        c.put('game', 'Breakout-v0')
        c.put('gradient_update_step', 40000)
        c.put('gamma', 0.99)
        c.put('update_reward_steps', 10)
        c.put('batch_size', 128)
        c.put('input_shape', (80, 80, 1))
        c.put('state_steps', 2)
        c.put('history_size', 100)
        c.put('env_num', 50)
        c.put('output_path', 'output/%s.%d' % (c.get("game"), time.time()))

        self.ac = sync.sync(2, c)

    def start(self):
        self.ac.start()

if __name__ == '__main__':
    game = breakout()
    game.start()

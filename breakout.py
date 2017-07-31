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

        c.put('follower_update_steps', 300)

        name = '%s.%d' % (c.get("game"), time.time())
        c.put('output_path', 'output/' + name)

        c.put('save_path', 'save/' + name)
        c.put('save_per_total_steps', 10000)
        c.put('save_per_minutes', 60)

        self.ac = sync.sync(3, c)

    def start(self):
        self.ac.start()

if __name__ == '__main__':
    game = breakout()
    game.start()

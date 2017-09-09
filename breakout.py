import argparse
import time

import sync
import config

class breakout(object):
    def __init__(self, args):
        c = config.config()
        c.put('game', 'Breakout-v0')
        c.put('gradient_update_step', 40000)
        c.put('gamma', 0.99)
        c.put('update_reward_steps', 5)
        c.put('batch_size', 128)
        c.put('input_shape', (80, 80, 1))
        c.put('state_steps', 2)
        c.put('env_num', args.env_num)

        c.put('learning_rate_start', 24e-5)
        c.put('learning_rate_end', 2.5e-5)
        c.put('learning_rate_decay_steps', 500000)
        c.put('learning_rate', args.learning_rate)

        c.put('follower_update_steps', 300)

        name = '%s.%d' % (c.get("game"), time.time())
        c.put('output_path', 'output/' + name)

        c.put('save_path', 'save/' + name)
        c.put('save_per_total_steps', 100000)
        c.put('save_per_minutes', 60)
        c.put('save_max_to_keep', 5)
        if args.load:
            c.put('load_path', args.load)
        c.put('global_step_reset', True)

        c.put('import_self_weight', 0.1)

        c.put('reward_mean_alpha', 0.9)
        c.put('clip_gradient_norm', 0.1)
        c.put('xentropy_reg_beta', 0.01)
        c.put('policy_reg_beta', 0.001)

        c.put('dense_layer_units', 512)

        c.put('thread_num', args.thread_num)

        self.ac = sync.sync(c)

    def start(self):
        self.ac.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0, help='Set learning rate to this fixed value')
    parser.add_argument('--thread_num', type=int, default=3, help='Number of runner threads')
    parser.add_argument('--env_num', type=int, default=65, help='Number of environments in each runner thread')
    parser.add_argument('--load', action='store', help='Load previously saved model')
    args=parser.parse_args()

    game = breakout(args)
    game.start()

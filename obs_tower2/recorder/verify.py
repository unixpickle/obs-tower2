"""
Make sure a recording plays back properly.

A lot of recordings do not play back deterministically,
and so will fail this script.
This is due to some randomness in the environment.
"""

import os
import random
import sys

import numpy as np
from obstacle_tower_env import ObstacleTowerEnv

from obs_tower2.recording import Recording


def main():
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: record_improve.py <recording_path>\n')
        os.exit(1)
    rec = Recording(sys.argv[1])
    env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'], worker_id=random.randrange(11, 20))
    try:
        env.seed(rec.seed)
        if rec.floor:
            env.floor(rec.floor)
        env.reset()
        i = 0
        for i, (action, rew) in enumerate(zip(rec.actions, rec.rewards)):
            _, real_rew, done, _ = env.step(action)
            if not np.allclose(real_rew, rew):
                print('mismatching result at step %d' % i)
                sys.exit(1)
            if done != (i == rec.num_steps - 1):
                print('invalid done result at step %d' % i)
                sys.exit(1)
        print('match succeeded')
    finally:
        env.close()


if __name__ == '__main__':
    main()

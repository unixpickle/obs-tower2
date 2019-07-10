"""
Record part of a demonstration starting at the floor where
an agent tends to get stuck.

I used an early version of this script to generate a lot
of recordings before (or at) the 10th floor.
It is unclear if these recordings actually helped in the
end.
"""

import os
import random
import sys

from obstacle_tower_env import ObstacleTowerEnv

from obs_tower2.recorder.env_interactor import EnvInteractor
from obs_tower2.recorder.record import record_episode, select_seed

MAX_STEPS = 300


def main():
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: python record_tail.py <start_floor>\n')
        sys.exit(1)
    start_floor = int(sys.argv[1])
    viewer = EnvInteractor()
    env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'], worker_id=random.randrange(11, 20))
    while True:
        seed = select_seed(floor=start_floor)
        env.seed(seed)
        env.floor(start_floor)
        obs = env.reset()
        viewer.reset()
        record_episode(seed, env, viewer, obs, max_steps=MAX_STEPS)


if __name__ == '__main__':
    main()

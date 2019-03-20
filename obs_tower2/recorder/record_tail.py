"""
Record part of a demonstration starting at the floor where
an agent tends to get stuck.
"""

import os
import random
import sys
import time

from obstacle_tower_env import ObstacleTowerEnv

from obs_tower2.recorder.record import EnvInteractor, record_episode, select_seed

RES_DIR = os.environ['OBS_TOWER_TAIL_RECORDINGS']
MAX_STEPS = 300


def main():
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: python record_tail.py <floor_evals.csv>\n')
        sys.exit(1)
    min_floors, max_floors = read_floors(sys.argv[1])
    viewer = EnvInteractor()
    env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'], worker_id=random.randrange(11, 20))
    while True:
        seed = select_seed(res_dir=RES_DIR)
        env.seed(seed)
        if random.random() < 0.5:
            env.floor(min_floors.get(seed, 0))
        else:
            env.floor(max_floors.get(seed, 0))
        obs = env.reset()
        record_episode(seed, env, viewer, obs, res_dir=RES_DIR, max_steps=MAX_STEPS)
        time.sleep(2)


def read_floors(path):
    mins = {}
    maxes = {}
    with open(path, 'r') as in_file:
        pairs = [x.strip().split(',') for x in in_file.readlines() if x.strip()]
        for seed, floor in [(int(x), int(y)) for x, y in pairs]:
            if seed not in mins or floor < mins[seed]:
                mins[seed] = floor
            if seed not in maxes or floor > maxes[seed]:
                maxes[seed] = floor
    return mins, maxes


if __name__ == '__main__':
    main()

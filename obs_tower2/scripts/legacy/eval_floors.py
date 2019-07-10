"""
Check which floor+seed pairs a model is able to solve.

I used an older version of this script to generate some
targeted recordings when my model wasn't able to solve the
first ten floors reliably. However, it is unclear if these
recordings had any real effect in the end.
"""

import random
import sys

import numpy as np
import torch

from obs_tower2.util import create_single_env
from obs_tower2.model import ACModel
from obs_tower2.states import StateEnv


def main():
    if len(sys.argv) != 4:
        sys.stderr.write('Usage: eval_floors.py <model> <start_floor> <end_floor>\n')
        sys.exit(1)

    model_path = sys.argv[1]
    start_floor = int(sys.argv[2])
    end_floor = int(sys.argv[3])

    env = StateEnv(create_single_env(random.randrange(20, 25)))
    model = ACModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(torch.device('cuda'))
    for seed in range(100):
        env.unwrapped.seed(seed)
        for floor in range(start_floor, end_floor):
            env.unwrapped.floor(floor)
            print('%d,%d,%d' % (seed, floor, run_episode(env, model)))


def run_episode(env, model):
    state, obs = env.reset()
    start_floor = env.unwrapped._floor
    while True:
        output = model.step(np.array([state]), np.array([obs]))
        (state, obs), rew, done, info = env.step(output['actions'][0])
        if done:
            return 0
        if info['current_floor'] != start_floor:
            return 1


if __name__ == '__main__':
    main()

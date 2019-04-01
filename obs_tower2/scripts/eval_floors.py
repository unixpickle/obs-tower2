import random

import numpy as np
import torch

from obs_tower2.util import create_single_env
from obs_tower2.model import ACModel
from obs_tower2.states import StateEnv


def main():
    env = StateEnv(create_single_env(random.randrange(20, 25)))
    model = ACModel()
    state = torch.load('save.pkl', map_location='cpu')
    model.load_state_dict(state)
    model.to(torch.device('cuda'))
    for seed in range(100):
        env.seed(seed)
        for _ in range(5):
            print('%d,%d' % (seed, run_episode(env, model)))


def run_episode(env, model):
    state, obs = env.reset()
    floor = 0
    while True:
        output = model.step(np.array([state]), np.array([obs]))
        (state, obs), rew, done, info = env.step(output['actions'][0])
        if rew == 1.0:
            floor += 1
        if done:
            break
    return floor


if __name__ == '__main__':
    main()

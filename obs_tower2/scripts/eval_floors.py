import random

import numpy as np
import torch

from obs_tower2.constants import IMAGE_SIZE, IMAGE_DEPTH, NUM_ACTIONS
from obs_tower2.util import create_single_env
from obs_tower2.model import ACModel


def main():
    env = create_single_env(random.randrange(20, 25))
    model = ACModel(NUM_ACTIONS, IMAGE_SIZE, IMAGE_DEPTH)
    state = torch.load('save.pkl', map_location='cpu')
    model.load_state_dict(state)
    model.to(torch.device('cuda'))
    for seed in range(100):
        env.seed(seed)
        for _ in range(5):
            print('%d,%d' % (seed, run_episode(env, model)))


def run_episode(env, model):
    obs = env.reset()
    state = np.zeros([1, model.state_size], dtype=np.float32)
    floor = 0
    while True:
        output = model.step(state, np.array([obs]))
        obs, rew, done, info = env.step(output['actions'][0])
        if rew == 1.0:
            floor += 1
        state = output['states']
        if done:
            break
    return floor


if __name__ == '__main__':
    main()

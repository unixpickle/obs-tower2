import itertools
import json
import random

import numpy as np
import torch

from obs_tower2.constants import HUMAN_ACTIONS
from obs_tower2.model import ACModel
from obs_tower2.states import StateEnv
from obs_tower2.util import create_single_env


def main():
    model = ACModel()
    model.load_state_dict(torch.load('save_tail.pkl', map_location='cpu'))
    model.to(torch.device('cuda'))
    for i in itertools.count():
        env = StateEnv(create_single_env(i % 16, clear=False))
        try:
            seeds = [random.randrange(100) for _ in range(3)]
            floors = [random.randrange(10, 15) for _ in range(3)]
            act_seqs = [[], [], []]
            for seed, floor, actions in zip(seeds, floors, act_seqs):
                env.unwrapped.seed(seed)
                env.unwrapped.floor(floor)
                state, obs = env.reset()
                while True:
                    output = model.step(np.array([state]), np.array([obs]))
                    action = output['actions'][0]
                    actions.append(HUMAN_ACTIONS[action])
                    with open('hang.json', 'w+') as out_file:
                        json.dump({'seeds': seeds, 'floors': floors, 'actions': act_seqs}, out_file)
                    (state, obs), rew, done, info = env.step(action)
                    if done:
                        break
        finally:
            env.close()


if __name__ == '__main__':
    main()

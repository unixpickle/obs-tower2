import json
import random

import numpy as np
import torch

from obs_tower2.constants import HUMAN_ACTIONS
from obs_tower2.model import ACModel
from obs_tower2.states import StateEnv
from obs_tower2.util import create_single_env


def main():
    env = StateEnv(create_single_env(random.randrange(15, 20), clear=False))
    try:
        model = ACModel()
        model.load_state_dict(torch.load('save.pkl', map_location='cpu'))
        model.to(torch.device('cuda'))
        while True:
            seed = random.randrange(100)
            floor = random.randrange(5, 10)
            actions = []
            env.unwrapped.seed(seed)
            env.unwrapped.floor(floor)
            state, obs = env.reset()
            while True:
                output = model.step(np.array([state]), np.array([obs]))
                action = output['actions'][0]
                actions.append(HUMAN_ACTIONS[action])
                with open('hang.json', 'w+') as out_file:
                    json.dump({'actions': actions, 'seed': seed, 'floor': floor}, out_file)
                (state, obs), rew, done, info = env.step(action)
                if done or rew == 1.0:
                    break
    finally:
        env.close()


if __name__ == '__main__':
    main()

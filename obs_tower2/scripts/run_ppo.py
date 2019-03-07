import os

import torch

from obs_tower2.constants import IMAGE_SIZE, IMAGE_DEPTH
from obs_tower2.model import ACModel
from obs_tower2.ppo import PPO
from obs_tower2.roller import Roller
from obs_tower2.util import create_batched_env

NUM_ENVS = 2
HORIZON = 512
BATCH_SIZE = NUM_ENVS * HORIZON / 8
LR = 1e-4
ITERS = 24
ENTROPY_REG = 0.001
GAE_LAM = 0.95
GAE_GAMMA = 0.9975


def main():
    env = create_batched_env(NUM_ENVS)
    model = ACModel(54, IMAGE_SIZE, IMAGE_DEPTH)
    if os.path.exists('save.pkl'):
        model.load_state_dict(torch.load('save.pkl'))
    model.to(torch.device('cuda'))
    roller = Roller(env, model, HORIZON)
    ppo = PPO(model, gamma=GAE_GAMMA, lam=GAE_LAM, lr=LR, ent_reg=ENTROPY_REG)
    ppo.outer_loop(roller, num_steps=ITERS, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()

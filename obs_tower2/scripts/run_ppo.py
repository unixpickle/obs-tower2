"""
Train an agent with raw PPO.

The hyper-parameters have not been tuned very well, and
the agent does not end up being very good on its own.
"""

import os

import torch

from obs_tower2.model import ACModel
from obs_tower2.ppo import PPO
from obs_tower2.roller import Roller
from obs_tower2.states import BatchedStateEnv
from obs_tower2.util import create_batched_env

NUM_ENVS = 8
HORIZON = 512
BATCH_SIZE = NUM_ENVS * HORIZON // 8
LR = 1e-4
ITERS = 24
ENTROPY_REG = 0.01
GAE_LAM = 0.95
GAE_GAMMA = 0.9975


def main():
    env = BatchedStateEnv(create_batched_env(NUM_ENVS))
    model = ACModel()
    if os.path.exists('save.pkl'):
        model.load_state_dict(torch.load('save.pkl'))
    model.to(torch.device('cuda'))
    roller = Roller(env, model, HORIZON)
    ppo = PPO(model, gamma=GAE_GAMMA, lam=GAE_LAM, lr=LR, ent_reg=ENTROPY_REG)
    ppo.outer_loop(roller, num_steps=ITERS, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()

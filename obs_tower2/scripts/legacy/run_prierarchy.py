"""
A legacy script for running prierarchy starting at the
first floor. This alone was not enough to learn very well,
since the agent had to get very good at reaching the 10th
floor reliably before it started to master the higher
floors.
"""

import os

import torch

from obs_tower2.model import ACModel
from obs_tower2.prierarchy import Prierarchy
from obs_tower2.states import BatchedStateEnv
from obs_tower2.util import LogRoller, create_batched_env

NUM_ENVS = 8
HORIZON = 512
BATCH_SIZE = NUM_ENVS * HORIZON // 8
LR = 3e-5
ITERS = 24
PRIOR_REG = 0.003
GAE_LAM = 0.95
GAE_GAMMA = 0.9975


def main():
    env = BatchedStateEnv(create_batched_env(NUM_ENVS))
    model = ACModel()
    prior = ACModel()
    if os.path.exists('save.pkl'):
        model.load_state_dict(torch.load('save.pkl'))
    if os.path.exists('save_prior.pkl'):
        prior.load_state_dict(torch.load('save_prior.pkl'))
    model.to(torch.device('cuda'))
    prior.to(torch.device('cuda'))
    roller = LogRoller(env, model, HORIZON)
    ppo = Prierarchy(prior, model, gamma=GAE_GAMMA, lam=GAE_LAM, lr=LR, ent_reg=PRIOR_REG)
    ppo.outer_loop(roller, num_steps=ITERS, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()

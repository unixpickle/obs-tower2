"""
Train an agent with generative adversarial imitation
learning (GAIL).

GAIL did not end up helping very much, and prierarchy
ended up being enough on its own.
"""

import os

import torch

from obs_tower2.gail import GAIL
from obs_tower2.model import ACModel, DiscriminatorModel
from obs_tower2.prierarchy import Prierarchy
from obs_tower2.recording import load_data
from obs_tower2.states import BatchedStateEnv, StateFeatures
from obs_tower2.util import LogRoller, create_batched_env

NUM_ENVS = 8
HORIZON = 512
BATCH_SIZE = NUM_ENVS * HORIZON // 8
LR = 1e-4
ITERS = 24
PRIOR_REG = 0.003
GAE_LAM = 0.95
GAE_GAMMA = 0.9975
REWARD_SCALE = 1.0
GAIL_REWARD_SCALE = 0.01
GAIL_HORIZON = 256
GAIL_NUM_ENVS = (HORIZON * NUM_ENVS) // GAIL_HORIZON


def main():
    state_features = StateFeatures()
    env = BatchedStateEnv(create_batched_env(NUM_ENVS, augment=True),
                          state_features=state_features)
    model = ACModel()
    prior = ACModel()
    discriminator = DiscriminatorModel()
    if os.path.exists('save.pkl'):
        model.load_state_dict(torch.load('save.pkl'))
    if os.path.exists('save_disc.pkl'):
        discriminator.load_state_dict(torch.load('save_disc.pkl'))
    else:
        discriminator.cnn.load_state_dict(model.cnn.state_dict())
    if os.path.exists('save_prior.pkl'):
        prior.load_state_dict(torch.load('save_prior.pkl'))
    model.to(torch.device('cuda'))
    discriminator.to(torch.device('cuda'))
    prior.to(torch.device('cuda'))
    train, test = load_data(augment=True)
    recordings = train + test
    roller = LogRoller(env, model, HORIZON)
    ppo = Prierarchy(prior, model, gamma=GAE_GAMMA, lam=GAE_LAM, lr=LR, ent_reg=PRIOR_REG)
    gail = GAIL(discriminator, lr=LR)
    gail.outer_loop(ppo,
                    roller,
                    recordings,
                    state_features,
                    rew_scale=GAIL_REWARD_SCALE,
                    real_rew_scale=REWARD_SCALE,
                    disc_num_steps=HORIZON * NUM_ENVS // BATCH_SIZE,
                    disc_batch_size=BATCH_SIZE,
                    expert_batch=GAIL_NUM_ENVS,
                    expert_horizon=GAIL_HORIZON,
                    num_steps=ITERS,
                    batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()

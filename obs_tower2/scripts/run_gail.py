import os

import torch

from obs_tower2.constants import IMAGE_SIZE, IMAGE_DEPTH, NUM_ACTIONS
from obs_tower2.gail import GAIL
from obs_tower2.model import ACModel, DiscriminatorModel
from obs_tower2.prierarchy import Prierarchy
from obs_tower2.recording import load_data
from obs_tower2.util import LogRoller, create_batched_env

NUM_ENVS = 8
HORIZON = 512
BATCH_SIZE = NUM_ENVS * HORIZON // 8
LR = 1e-4
ITERS = 24
PRIOR_REG = 0.01
GAE_LAM = 0.95
GAE_GAMMA = 0.9975
REWARD_SCALE = 0.01


def main():
    env = create_batched_env(NUM_ENVS, key_reward=True)
    model = ACModel(NUM_ACTIONS, IMAGE_SIZE, IMAGE_DEPTH)
    prior = ACModel(NUM_ACTIONS, IMAGE_SIZE, IMAGE_DEPTH)
    discriminator = DiscriminatorModel(IMAGE_SIZE, IMAGE_DEPTH)
    if os.path.exists('save.pkl'):
        model.load_state_dict(torch.load('save.pkl'))
    if os.path.exists('save_disc.pkl'):
        discriminator.load_state_dict(torch.load('save_disc.pkl'))
    if os.path.exists('save_prior.pkl'):
        prior.load_state_dict(torch.load('save_prior.pkl'))
    model.to(torch.device('cuda'))
    discriminator.to(torch.device('cuda'))
    train, test = load_data()
    recordings = train + test
    roller = LogRoller(env, model, HORIZON)
    ppo = Prierarchy(prior, model, gamma=GAE_GAMMA, lam=GAE_LAM, lr=LR, ent_reg=PRIOR_REG)
    gail = GAIL(discriminator, lr=LR)
    gail.outer_loop(ppo,
                    roller,
                    recordings,
                    rew_scale=REWARD_SCALE,
                    disc_num_steps=ITERS,
                    disc_batch_size=BATCH_SIZE,
                    num_steps=ITERS,
                    batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()

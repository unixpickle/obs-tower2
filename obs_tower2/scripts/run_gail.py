import os

import torch

from obs_tower2.constants import IMAGE_SIZE, IMAGE_DEPTH, NUM_ACTIONS
from obs_tower2.gail import GAIL
from obs_tower2.model import ACModel, DiscriminatorModel
from obs_tower2.ppo import PPO
from obs_tower2.recording import load_data
from obs_tower2.util import LogRoller, create_batched_env

NUM_ENVS = 8
HORIZON = 512
BATCH_SIZE = NUM_ENVS * HORIZON // 8
LR = 3e-5
ITERS = 24
ENT_REG = 0.01
GAE_LAM = 0.95
GAE_GAMMA = 0.9975
REWARD_SCALE = 0.01
GAIL_HORIZON = 256
GAIL_NUM_ENVS = (HORIZON * NUM_ENVS) // GAIL_HORIZON


def main():
    env = create_batched_env(NUM_ENVS, augment=True)
    model = ACModel(NUM_ACTIONS, IMAGE_SIZE, IMAGE_DEPTH)
    discriminator = DiscriminatorModel(IMAGE_SIZE, IMAGE_DEPTH)
    if os.path.exists('save.pkl'):
        model.load_state_dict(torch.load('save.pkl'))
    if os.path.exists('save_disc.pkl'):
        discriminator.load_state_dict(torch.load('save_disc.pkl'))
    model.to(torch.device('cuda'))
    discriminator.to(torch.device('cuda'))
    train, test = load_data(augment=True)
    recordings = train + test
    roller = LogRoller(env, model, HORIZON)
    ppo = PPO(model, gamma=GAE_GAMMA, lam=GAE_LAM, lr=LR, ent_reg=ENT_REG)
    gail = GAIL(discriminator, lr=LR)
    gail.outer_loop(ppo,
                    roller,
                    recordings,
                    rew_scale=REWARD_SCALE,
                    real_rew_scale=1.0,
                    disc_num_steps=HORIZON * NUM_ENVS // BATCH_SIZE,
                    disc_batch_size=BATCH_SIZE,
                    expert_batch=GAIL_NUM_ENVS,
                    expert_horizon=GAIL_HORIZON,
                    num_steps=ITERS,
                    batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()

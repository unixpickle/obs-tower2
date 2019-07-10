"""
Train an agent with prierarchy from an arbitrary range of
starting floors.

This is called training a "tail" because, by default, it
starts training from the 10th floor, i.e. at the tail-end
of an episode.

This is the main agent training script that you should
use. The hyper-parameters have been fairly well selected.
To train a model for floors 0 through 10, simply pass
arguments --min 0 --max 1.
"""

import argparse
import os

import torch

from obs_tower2.model import ACModel
from obs_tower2.prierarchy import Prierarchy
from obs_tower2.states import BatchedStateEnv
from obs_tower2.util import LogRoller, create_batched_env

NUM_ENVS = 8
HORIZON = 512
BATCH_SIZE = NUM_ENVS * HORIZON // 16
LR = 1e-5
ITERS = 48
PRIOR_REG = 0.003
GAE_LAM = 0.95
GAE_GAMMA = 0.9975


def main():
    args = arg_parser().parse_args()
    env = BatchedStateEnv(create_batched_env(NUM_ENVS,
                                             augment=True,
                                             start=args.worker_idx,
                                             rand_floors=(args.min, args.max)))
    model = ACModel()
    prior = ACModel()
    if os.path.exists(args.path):
        model.load_state_dict(torch.load(args.path))
    if os.path.exists('save_prior.pkl'):
        prior.load_state_dict(torch.load('save_prior.pkl'))
    model.to(torch.device('cuda'))
    prior.to(torch.device('cuda'))
    roller = LogRoller(env, model, HORIZON)
    ppo = Prierarchy(prior, model, gamma=GAE_GAMMA, lam=GAE_LAM, lr=LR, ent_reg=PRIOR_REG)
    ppo.outer_loop(roller, num_steps=ITERS, batch_size=BATCH_SIZE, save_path=args.path)


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--min', default=10, type=int)
    parser.add_argument('--max', default=15, type=int)
    parser.add_argument('--worker-idx', default=8, type=int)
    parser.add_argument('--path', default='save_tail.pkl')
    return parser


if __name__ == '__main__':
    main()

"""
Train a prior using behavior cloning.

This produces a file called "save_clone.pkl".
To use it as a prior, rename or copy the file to be called
"save_prior.pkl".

On one GPU, behavior cloning may take somewhere between 3
to 12 hours. You should kill the program once the training
loss is much lower than the test loss on average, since
this indicates overfitting. To track when this happens,
you will likely want to average the training and test
outputs over a few hundred iterations. I used my sh-avg
tool (https://github.com/unixpickle/statushub) to do this,
but that is likely overkill.
"""

import itertools
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from obs_tower2.constants import HUMAN_ACTIONS, NUM_ACTIONS
from obs_tower2.model import ACModel
from obs_tower2.recording import load_data, recording_rollout, truncate_recordings
from obs_tower2.states import StateFeatures
from obs_tower2.util import atomic_save

LR = 1e-4
BATCH = 4
HORIZON = 64
EPSILON = 0.0
TRUNCATE = None


def main():
    model = ACModel()
    if os.path.exists('save_clone.pkl'):
        model.load_state_dict(torch.load('save_clone.pkl'))
    model.to(torch.device('cuda'))
    state_features = StateFeatures()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train, test = load_data(augment=True)
    if TRUNCATE is not None:
        train = truncate_recordings(train, TRUNCATE)
        test = truncate_recordings(test, TRUNCATE)
    for i in itertools.count():
        train_rollout = recording_rollout(train, BATCH, HORIZON, state_features)
        test_rollout = recording_rollout(test, BATCH, HORIZON, state_features)
        test_loss = cloning_loss(model, test_rollout).item()
        train_loss = cloning_loss(model, train_rollout)
        print('step %d: test=%f train=%f' % (i, test_loss, train_loss.item()))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if not i % 100:
            atomic_save(model.state_dict(), 'save_clone.pkl')


def cloning_loss(model, rollout):
    model_outs = apply_model(model, rollout)
    actions = np.array([epsilon_greedy(HUMAN_ACTIONS.index(a))
                        for m in rollout.model_outs[:-1]
                        for a in m['actions']])
    return F.cross_entropy(model_outs['actor'], model.tensor(actions).long())


def apply_model(model, rollout, no_time=True):
    states = rollout.states[:-1].reshape(-1, *rollout.states.shape[2:])
    obses = rollout.obses[:-1].reshape(-1, *rollout.obses.shape[2:])
    if no_time:
        obses[:, 6:10] = np.random.uniform(high=255, size=obses[:, 6:10].shape).astype(np.uint8)
    return model(model.tensor(states), model.tensor(obses))


def epsilon_greedy(action):
    if random.random() < EPSILON:
        return random.randrange(NUM_ACTIONS)
    return action


if __name__ == '__main__':
    main()

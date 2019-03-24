import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from obs_tower2.constants import IMAGE_SIZE, IMAGE_DEPTH, NUM_ACTIONS, HUMAN_ACTIONS
from obs_tower2.model import ACModel
from obs_tower2.recording import load_data, recording_rollout

LR = 1e-4
BATCH = 4
HORIZON = 64
EPSILON = 0.1


def main():
    model = ACModel(NUM_ACTIONS, IMAGE_SIZE, IMAGE_DEPTH)
    if os.path.exists('save_clone.pkl'):
        model.load_state_dict(torch.load('save_clone.pkl'))
    model.to(torch.device('cuda'))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train, test = load_data(augment=True)
    for i in itertools.count():
        train_rollout = recording_rollout(train, BATCH, HORIZON)
        test_rollout = recording_rollout(test, BATCH, HORIZON)
        test_loss = cloning_loss(model, test_rollout).item()
        train_loss = cloning_loss(model, train_rollout)
        print('step %d: test=%f train=%f' % (i, test_loss, train_loss.item()))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if not i % 100:
            torch.save(model.state_dict(), 'save_clone.pkl')


def cloning_loss(model, rollout):
    model_outs = apply_model(model, rollout)
    actions = np.array([HUMAN_ACTIONS.index(a)
                        for m in rollout.model_outs[:-1]
                        for epsilon_greedy(a) in m['actions']])
    return F.cross_entropy(model_outs['actor'], model.tensor(actions).long())


def apply_model(model, rollout, no_time=True):
    model_rollout = model.run_for_rollout(rollout)
    states = model_rollout.states[:-1].reshape(-1, model.state_size)
    obses = model_rollout.obses[:-1].reshape(-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH)
    if no_time:
        obses[:, 6:10] = np.random.uniform(high=255, size=obses[:, 6:10].shape).astype(np.uint8)
    return model(model.tensor(states), model.tensor(obses))


def epsilon_greedy(action):
    if random.random() < EPSILON:
        return random.randrange(NUM_ACTIONS)
    return action


if __name__ == '__main__':
    main()

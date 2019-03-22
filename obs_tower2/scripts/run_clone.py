import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from obs_tower2.constants import IMAGE_SIZE, IMAGE_DEPTH, NUM_ACTIONS, HUMAN_ACTIONS
from obs_tower2.model import ACModel
from obs_tower2.recording import load_data, recording_rollout
from obs_tower2.util import LogRoller, create_batched_env

LR = 1e-4
BATCH = 4
HORIZON = 64


def main():
    env = create_batched_env(BATCH, augment=True, rand_floor=True)
    model = ACModel(NUM_ACTIONS, IMAGE_SIZE, IMAGE_DEPTH)
    if os.path.exists('save_clone.pkl'):
        model.load_state_dict(torch.load('save_clone.pkl'))
    model.to(torch.device('cuda'))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train, test = load_data(augment=True)
    roller = LogRoller(env, model, HORIZON)
    for i in itertools.count():
        train_rollout = recording_rollout(train, BATCH, HORIZON)
        test_rollout = recording_rollout(test, BATCH, HORIZON)
        ent_rollout = roller.rollout()
        test_loss = cloning_loss(model, test_rollout).item()
        train_loss = cloning_loss(model, train_rollout)
        entropy = action_entropy(model, ent_rollout)
        print('step %d: test=%f train=%f entropy=%f' % (i, test_loss, train_loss.item(),
                                                        entropy.item()))
        loss = train_loss - entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not i % 100:
            torch.save(model.state_dict(), 'save_clone.pkl')


def cloning_loss(model, rollout):
    model_outs = apply_model(model, rollout)
    actions = np.array([HUMAN_ACTIONS.index(a)
                        for m in rollout.model_outs[:-1]
                        for a in m['actions']])
    return F.cross_entropy(model_outs['actor'], model.tensor(actions).long())


def action_entropy(model, rollout):
    model_outs = apply_model(model, rollout)
    all_probs = torch.log_softmax(model_outs['actor'], dim=-1)
    return -torch.mean(torch.sum(torch.exp(all_probs) * all_probs, dim=-1))


def apply_model(model, rollout, no_time=True):
    model_rollout = model.run_for_rollout(rollout)
    states = model_rollout.states[:-1].reshape(-1, model.state_size)
    obses = model_rollout.obses[:-1].reshape(-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH)
    if no_time:
        obses[:, 6:10] = np.random.uniform(high=255, size=obses[:, 6:10].shape).astype(np.uint8)
    return model(model.tensor(states), model.tensor(obses))


if __name__ == '__main__':
    main()

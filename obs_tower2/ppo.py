import itertools
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


class PPO:
    def __init__(self, model, epsilon=0.2, gamma=0.99, lam=0.95, lr=1e-4, ent_reg=0.001):
        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.ent_reg = ent_reg

    def outer_loop(self, roller, save_path='save.pkl', **kwargs):
        for i in itertools.count():
            terms = self.inner_loop(roller.rollout(), **kwargs)
            print('step %d: clipped=%f entropy=%f explained=%f' %
                  (i, terms['clip_frac'], terms['entropy'], terms['explained']))
            torch.save(self.model.state_dict(), save_path)

    def inner_loop(self, rollout, num_steps=12, batch_size=None):
        if batch_size is None:
            batch_size = rollout.num_steps * rollout.batch_size
        advs = rollout.advantages(self.gamma, self.lam)
        targets = advs + rollout.value_predictions()
        actions = rollout.actions()
        log_probs = rollout.log_probs()
        i = 0
        first_terms = None
        for entries in rollout_batches(rollout, batch_size):
            def choose(values):
                return self.model.tensor(np.array([values[t, b] for t, b in entries]))
            model_outs = self.model(choose(rollout.states), choose(rollout.obses))
            terms = self._terms(model_outs,
                                choose(advs),
                                choose(targets),
                                choose(actions),
                                choose(log_probs))
            self.optimizer.zero_grad()
            terms['loss'].backward()
            self.optimizer.step()
            if not i:
                first_terms = {k: v.item() for k, v in terms.items()}
            i += 1
            if i == num_steps:
                break
        return first_terms

    def _terms(self, model_outs, advs, targets, actions, log_probs):
        vf_loss = torch.mean(torch.pow(model_outs['critic'] - targets, 2))
        variance = torch.var(targets)
        explained = 1 - vf_loss / variance

        new_log_probs = F.cross_entropy(model_outs['actor'], actions)
        ratio = torch.exp(new_log_probs - log_probs)
        clip_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        pi_loss = torch.mean(torch.min(ratio * advs, clip_ratio * advs))
        clip_frac = torch.mean(torch.gt(ratio * advs, clip_ratio * advs).float())

        all_probs = torch.log_softmax(model_outs['actor'], dim=-1)
        neg_entropy = torch.mean(torch.sum(torch.exp(all_probs) * all_probs, dim=-1))
        ent_loss = self.ent_reg * neg_entropy

        return {
            'explained': explained,
            'clip_frac': clip_frac,
            'entropy': -neg_entropy,
            'loss': vf_loss + pi_loss + ent_loss,
        }

    def _tensor(self, x):
        return torch.from_numpy(x).to(self.model.device)


def rollout_batches(rollout, batch_size):
    batch = []
    for entry in rollout_entries(rollout):
        batch.append(entry)
        if len(batch) == batch_size:
            yield batch
            batch = []


def rollout_entries(rollout):
    entries = []
    for t in range(rollout.num_steps):
        for b in range(rollout.batch_size):
            entries.append((t, b))
    while True:
        random.shuffle(entries)
        for entry in entries:
            yield entry
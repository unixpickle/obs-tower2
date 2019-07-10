import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .util import atomic_save


class PPO:
    """
    A base implementation of Proximal Policy Optimization.
    See: https://arxiv.org/abs/1707.06347.
    """

    def __init__(self, model, epsilon=0.2, gamma=0.99, lam=0.95, lr=1e-4, ent_reg=0.001):
        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.ent_reg = ent_reg

    def outer_loop(self, roller, save_path='save.pkl', **kwargs):
        """
        Run training indefinitely, saving periodically.
        """
        for i in itertools.count():
            terms, last_terms = self.inner_loop(roller.rollout(), **kwargs)
            self.print_outer_loop(i, terms, last_terms)
            atomic_save(self.model.state_dict(), save_path)

    def print_outer_loop(self, i, terms, last_terms):
        print('step %d: clipped=%f entropy=%f explained=%f' %
              (i, last_terms['clip_frac'], terms['entropy'], terms['explained']))

    def inner_loop(self, rollout, num_steps=12, batch_size=None):
        if batch_size is None:
            batch_size = rollout.num_steps * rollout.batch_size
        advs = rollout.advantages(self.gamma, self.lam)
        targets = advs + rollout.value_predictions()[:-1]
        advs = (advs - np.mean(advs)) / (1e-8 + np.std(advs))
        actions = rollout.actions()
        log_probs = rollout.log_probs()
        firstterms = None
        lastterms = None
        for entries in rollout.batches(batch_size, num_steps):
            def choose(values):
                return self.model.tensor(np.array([values[t, b] for t, b in entries]))
            terms = self.terms(choose(rollout.states),
                               choose(rollout.obses),
                               choose(advs),
                               choose(targets),
                               choose(actions),
                               choose(log_probs))
            self.optimizer.zero_grad()
            terms['loss'].backward()
            self.optimizer.step()
            lastterms = {k: v.item() for k, v in terms.items() if k != 'model_outs'}
            if firstterms is None:
                firstterms = lastterms
            del terms
        return firstterms, lastterms

    def terms(self, states, obses, advs, targets, actions, log_probs):
        model_outs = self.model(states, obses)

        vf_loss = torch.mean(torch.pow(model_outs['critic'] - targets, 2))
        variance = torch.var(targets)
        explained = 1 - vf_loss / variance

        new_log_probs = -F.cross_entropy(model_outs['actor'], actions.long(), reduction='none')
        ratio = torch.exp(new_log_probs - log_probs)
        clip_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        pi_loss = -torch.mean(torch.min(ratio * advs, clip_ratio * advs))
        clip_frac = torch.mean(torch.gt(ratio * advs, clip_ratio * advs).float())

        all_probs = torch.log_softmax(model_outs['actor'], dim=-1)
        neg_entropy = torch.mean(torch.sum(torch.exp(all_probs) * all_probs, dim=-1))
        ent_loss = self.ent_reg * neg_entropy

        return {
            'explained': explained,
            'clip_frac': clip_frac,
            'entropy': -neg_entropy,
            'vf_loss': vf_loss,
            'pi_loss': pi_loss,
            'ent_loss': ent_loss,
            'loss': vf_loss + pi_loss + ent_loss,
            'model_outs': model_outs,
        }

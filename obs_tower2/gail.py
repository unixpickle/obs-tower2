"""
An implementation of Generative Adversarial Imitation
Learning (GAIL): https://arxiv.org/abs/1606.03476.

This was helpful in my initial experiments before
prierarchy, but prierarchy alone ended up working better.
"""

import itertools

import numpy as np
import torch
import torch.optim as optim

from .recording import recording_rollout
from .util import atomic_save


class GAIL:
    def __init__(self, discriminator, lr=1e-4):
        self.discriminator = discriminator
        self.optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    def outer_loop(self,
                   ppo,
                   roller,
                   recordings,
                   state_features,
                   save_path='save.pkl',
                   rew_scale=0.01,
                   real_rew_scale=1.0,
                   disc_save_path='save_disc.pkl',
                   disc_num_steps=12,
                   disc_batch_size=None,
                   expert_batch=None,
                   expert_horizon=None,
                   **ppo_kwargs):
        if expert_batch is None:
            expert_batch = roller.batched_env.num_envs_per_sub_batch
        if expert_horizon is None:
            expert_horizon = roller.num_steps
        for i in itertools.count():
            rollout_pi = roller.rollout()
            rollout_expert = recording_rollout(recordings=recordings,
                                               batch=expert_batch,
                                               horizon=expert_horizon,
                                               state_features=state_features)
            terms, last_terms = ppo.inner_loop(self.add_rewards(rollout_pi, rew_scale,
                                                                real_rew_scale),
                                               **ppo_kwargs)
            disc_loss = self.inner_loop(rollout_pi,
                                        rollout_expert,
                                        num_steps=disc_num_steps,
                                        batch_size=disc_batch_size)
            print('step %d: clipped=%f entropy=%f explained=%f %sloss=%f' %
                  (i, last_terms['clip_frac'], terms['entropy'], terms['explained'],
                   ('' if 'kl' not in terms else 'kl=%f ' % terms['kl']), disc_loss))
            atomic_save(ppo.model.state_dict(), save_path)
            atomic_save(self.discriminator.state_dict(), disc_save_path)

    def add_rewards(self, rollout_pi, rew_scale, real_rew_scale):
        result = rollout_pi.copy()
        result.rews = result.rews.copy() * real_rew_scale
        applied = self.discriminator.run_for_rollout(result)
        for t, model_outs in enumerate(applied.model_outs[1:]):
            result.rews[t - 1] -= (1 - applied.dones[t]) * model_outs['prob_pi'] * rew_scale
        return result

    def inner_loop(self, rollout_pi, rollout_expert, num_steps=12, batch_size=None):
        if batch_size is None:
            batch_size = rollout_pi.num_steps * rollout_pi.batch_size
        batches_pi = rollout_pi.batches(batch_size, num_steps)
        batches_expert = rollout_expert.batches(batch_size, num_steps)
        first_loss = None
        for batch_pi, batch_expert in zip(batches_pi, batches_expert):
            def run_disc(rollout, batch):
                states = np.array([rollout.states[t, b] for t, b in batch])
                obses = np.array([rollout.obses[t, b] for t, b in batch])
                return self.discriminator(self.discriminator.tensor(states),
                                          self.discriminator.tensor(obses))
            disc_pi = run_disc(rollout_pi, batch_pi)
            disc_expert = run_disc(rollout_expert, batch_expert)
            loss = -torch.mean(disc_pi['prob_pi'] + disc_expert['prob_expert'])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if first_loss is None:
                first_loss = loss.item()
            del loss
            del disc_pi
            del disc_expert
        return first_loss

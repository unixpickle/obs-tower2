import numpy as np
import torch
import torch.optim as optim


class GAIL:
    def __init__(self, discriminator, lr=1e-4):
        self.discriminator = discriminator
        self.optimizer = optim.Adam(lr=lr)

    def inner_loop(self, rollout_pi, rollout_expert, num_steps=12, batch_size=None):
        if batch_size is None:
            batch_size = rollout_pi.num_steps * rollout_pi.batch_size
        pi_batches = rollout_pi.batches(batch_size, num_steps)
        expert_batches = rollout_expert.batches(batch_size, num_steps)
        for batch_pi, batch_expert in zip(pi_batches, expert_batches):
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

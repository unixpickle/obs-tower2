import numpy as np
import torch
import torch.nn.functional as F

from .ppo import PPO


class Prierarchy(PPO):
    """
    An implementation of "prierarchy".

    This is an algorithm that builds on top of PPO.
    It replaces the entropy regularization term with a KL
    term that pulls the policy close to a prior policy.

    I also tried a variant of Prierarchy that adds the KL
    bonus to the rewards, but it did not help much. To use
    this option, set kl_coeff to some non-zero value.
    """

    def __init__(self, prior, *args, kl_coeff=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior = prior
        self.kl_coeff = kl_coeff

    def print_outer_loop(self, i, terms, last_terms):
        print('step %d: clipped=%f entropy=%f explained=%f kl=%f' %
              (i, last_terms['clip_frac'], last_terms['entropy'], terms['explained'],
               terms['kl']))

    def inner_loop(self, rollout, num_steps=12, batch_size=None):
        if batch_size is None:
            batch_size = rollout.num_steps * rollout.batch_size
        prior_rollout = self.prior.run_for_rollout(rollout)
        prior_logits = prior_rollout.logits()
        rollout = self.add_rewards(rollout, prior_rollout)
        advs = rollout.advantages(self.gamma, self.lam)
        targets = advs + rollout.value_predictions()[:-1]
        actions = rollout.actions()
        log_probs = rollout.log_probs()
        firstterms = None
        lastterms = None
        for entries in rollout.batches(batch_size, num_steps):
            def choose(values):
                return self.model.tensor(np.array([values[t, b] for t, b in entries]))
            terms = self.extended_terms(choose(prior_logits),
                                        choose(rollout.states),
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

    def extended_terms(self, prior_logits, states, obses, advs, targets, actions, log_probs):
        super_out = self.terms(states, obses, advs, targets, actions, log_probs)
        log_prior = F.log_softmax(prior_logits, dim=-1)
        log_posterior = F.log_softmax(super_out['model_outs']['actor'], dim=-1)
        kl = torch.mean(torch.sum(torch.exp(log_posterior) * (log_posterior - log_prior), dim=-1))
        kl_loss = kl * self.ent_reg
        super_out['kl'] = kl
        super_out['kl_loss'] = kl_loss
        super_out['loss'] = super_out['vf_loss'] + super_out['pi_loss'] + kl_loss
        return super_out

    def add_rewards(self, rollout, prior_rollout):
        rollout = rollout.copy()
        rollout.rews = rollout.rews.copy()

        def log_probs(r):
            return F.log_softmax(torch.from_numpy(np.array([m['actor'] for m in r.model_outs])),
                                 dim=-1)

        q = log_probs(prior_rollout)
        p = log_probs(rollout)
        kls = torch.sum(torch.exp(p) * (p - q), dim=-1).numpy()

        rollout.rews -= kls[:-1] * self.kl_coeff

        return rollout

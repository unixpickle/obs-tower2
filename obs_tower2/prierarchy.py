import numpy as np
import torch
import torch.nn.functional as F

from .ppo import PPO

class Prierarchy(PPO):
    def __init__(self, prior, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior = prior

    def print_outer_loop(self, i, terms, last_terms):
        print('step %d: clipped=%f entropy=%f explained=%f kl=%f' %
              (i, last_terms['clip_frac'], last_terms['entropy'], terms['explained'],
               terms['kl']))

    def inner_loop(self, rollout, num_steps=12, batch_size=None):
        if batch_size is None:
            batch_size = rollout.num_steps * rollout.batch_size
        prior_rollout = self.prior.run_for_rollout(rollout)
        advs = rollout.advantages(self.gamma, self.lam)
        targets = advs + rollout.value_predictions()[:-1]
        actions = rollout.actions()
        log_probs = rollout.log_probs()
        firstterms = None
        lastterms = None
        for entries in rollout.batches(batch_size, num_steps):
            def choose(values):
                return self.model.tensor(np.array([values[t, b] for t, b in entries]))
            terms = self.extended_terms(choose(prior_rollout.states),
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
        return firstterms, lastterms

    def extended_terms(self, prior_states, states, obses, advs, targets, actions, log_probs):
        super_out = self.terms(states, obses, advs, targets, actions, log_probs)
        prior_outs = self.prior(prior_states, obses)
        actor = prior_outs['actor'].detach()
        log_prior = F.log_softmax(actor, dim=-1)
        log_posterior = F.log_softmax(super_out['model_outs']['actor'], dim=-1)
        kl = torch.mean(torch.sum(torch.exp(log_posterior) * (log_posterior - log_prior), dim=-1))
        kl_loss = kl * self.ent_reg
        super_out['kl'] = kl
        super_out['kl_loss'] = kl_loss * self.ent_reg
        super_out['loss'] = super_out['vf_loss'] + super_out['pi_loss'] + kl_loss
        return super_out

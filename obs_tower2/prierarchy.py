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

    def terms(self, states, obses, advs, targets, actions, log_probs):
        super_out = super().terms(states, obses, advs, targets, actions, log_probs)
        prior_outs = self.prior(states, obses)
        actor = prior_outs['actor'].detach()
        log_prior = F.log_softmax(actor, dim=-1)
        log_posterior = F.log_softmax(super_out['model_outs']['actor'], dim=-1)
        kl = torch.mean(torch.sum(torch.exp(log_prior) * (log_prior - log_posterior), dim=-1))
        kl_loss = kl * self.ent_reg
        super_out['kl'] = kl
        super_out['kl_loss'] = kl_loss * self.ent_reg
        super_out['loss'] = super_out['vf_loss'] + super_out['pi_loss'] + kl_loss
        return super_out

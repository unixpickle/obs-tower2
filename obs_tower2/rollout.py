"""
Storing and manipulating trajectories from an agent.
"""

import numpy as np


class Rollout:
    """
    A batch of trajectory segments.

    The dones, obses, states, and model_outs are one
    timestep longer than the other sequences, for
    bootstrapping off the value function.

    The model_outs will vary by the type of model that
    generated the rollouts. Typically it will have these
    keys:

        values: outputs from the value function.
        actions: actions from the policy.
        states: output states from the network.

    Members have shape [num_steps x batch_size x ...] or
    [(num_steps + 1) x batch_size x ...].
    """

    def __init__(self, states, obses, rews, dones, infos, model_outs):
        self.states = np.array(states, dtype=np.float32)
        self.obses = np.array(obses)
        self.rews = np.array(rews, dtype=np.float32)
        self.dones = np.array(dones, dtype=np.float32)
        self.infos = infos
        self.model_outs = model_outs

    def copy(self):
        return Rollout(self.states, self.obses, self.rews, self.dones, self.infos, self.model_outs)

    @property
    def num_steps(self):
        return len(self.rews)

    @property
    def batch_size(self):
        return len(self.rews[0])

    def value_predictions(self):
        """
        Get the value predictions from the model at each
        timestep.
        """
        return np.array([m['values'] for m in self.model_outs], dtype=np.float32)

    def actions(self):
        """
        Get the integer actions from the model at each
        timestep.
        """
        return np.array([m['actions'] for m in self.model_outs], dtype=np.int32)

    def log_probs(self):
        """
        Get the initial log probabilities from the model
        at each timestep.
        """
        return np.array([m['log_probs'] for m in self.model_outs], dtype=np.float32)

    def advantages(self, gamma, lam):
        """
        Generate a [num_steps x batch_size] array of
        generalized advantages using GAE.
        """
        values = self.value_predictions()
        result = np.zeros([self.num_steps, self.batch_size], dtype=np.float32)
        current = np.zeros([self.batch_size], dtype=np.float32)
        for t in range(self.num_steps - 1, -1, -1):
            delta = self.rews[t] - values[t]
            delta += (1 - self.dones[t + 1]) * gamma * values[t + 1]
            current *= gamma * lam
            current += delta
            result[t] = current
            current *= (1 - self.dones[t])
        return result

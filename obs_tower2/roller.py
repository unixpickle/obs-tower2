"""
Gathering trajectories from environments.
"""

import numpy as np

from .rollout import Rollout


class Roller:
    """
    A Roller runs a policy on a batched environment and
    produces rollouts containing the results.

    Args:
        batched_env: a BatchedEnv implementation.
        model: a Model with 'actions' in its output dicts.
        num_steps: the number of timesteps to run per
          batch of rollouts that are generated.
    """

    def __init__(self, batched_env, model, num_steps):
        self.batched_env = batched_env
        self.model = model
        self.num_steps = num_steps
        self._prev_obs = None
        self._prev_states = None
        self._prev_dones = None

    def reset(self):
        self._prev_states, self._prev_obs = self.batched_env.reset()
        self._prev_states = np.array(self._prev_states)
        self._prev_obs = np.array(self._prev_obs)
        self._prev_dones = np.zeros([self.batched_env.num_envs], dtype=np.bool)

    def rollout(self):
        if self._prev_obs is None:
            self.reset()
        batch = self.batched_env.num_envs
        states = np.zeros((self.num_steps + 1,) + self._prev_states.shape, dtype=np.float32)
        obses = np.zeros((self.num_steps + 1,) + self._prev_obs.shape, dtype=self._prev_obs.dtype)
        rews = np.zeros([self.num_steps, batch], dtype=np.float32)
        dones = np.zeros([self.num_steps + 1, batch], dtype=np.bool)
        infos = []
        model_outs = []
        for t in range(self.num_steps):
            states[t] = self._prev_states
            obses[t] = self._prev_obs
            dones[t] = self._prev_dones
            model_out = self.model.step(self._prev_states, self._prev_obs)
            step_result = self.batched_env.step(model_out['actions'])
            (step_states, step_obs), step_rews, step_dones, step_infos = step_result
            self._prev_states = np.array(step_states)
            self._prev_obs = np.array(step_obs)
            self._prev_dones = np.array(step_dones)
            rews[t] = np.array(step_rews)
            infos.append(step_infos)
            model_outs.append(model_out)
        states[-1] = self._prev_states
        obses[-1] = self._prev_obs
        dones[-1] = self._prev_dones
        model_outs.append(self.model.step(self._prev_states, self._prev_obs))
        return Rollout(states, obses, rews, dones, infos, model_outs)

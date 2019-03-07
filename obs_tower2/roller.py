"""
Gathering trajectories from environments.
"""

import numpy as np

from .rollout import Rollout


class Roller:
    def __init__(self, batched_env, model, num_steps):
        assert batched_env.num_sub_batches == 1, 'multiple sub-batches not supported'
        self.batched_env = batched_env
        self.model = model
        self.num_steps = num_steps
        self._prev_obs = None
        self._prev_states = None
        self._prev_dones = None

    def reset(self):
        self.batched_env.reset_start()
        self._prev_obs = np.array(self.batched_env.reset_wait())
        self._prev_states = np.zeros([self.batched_env.num_envs_per_sub_batch,
                                      self.model.state_size],
                                     dtype=np.float32)
        self._prev_dones = np.zeros([self.batched_env.num_envs_per_sub_batch], dtype=np.bool)

    def rollout(self):
        if self._prev_obs is None:
            self.reset()
        batch = self.batched_env.num_envs_per_sub_batch
        states = np.zeros([self.num_steps + 1, batch, self.model.state_size], dtype=np.float32)
        obses = np.zeros((self.num_steps + 1, batch) + self.batched_env.observation_space.shape,
                         dtype=self.batched_env.observation_space.dtype)
        rews = np.zeros([self.num_steps, batch], dtype=np.float32)
        dones = np.zeros([self.num_steps + 1, batch], dtype=np.bool)
        infos = []
        model_outs = []
        for t in range(self.num_steps):
            states[t] = self._prev_states
            obses[t] = self._prev_obs
            dones[t] = self._prev_dones
            model_out = self.model.step(self._prev_states, self._prev_obs)
            self.batched_env.step_start(model_out['actions'])
            step_obs, step_rews, step_dones, step_infos = self.batched_env.step_wait()
            self._prev_obs = np.array(step_obs)
            self._prev_dones = np.array(step_dones)
            self._prev_states = model_out['states'] * (1 - self._prev_dones[:, None])
            rews[t] = np.array(step_rews)
            infos.append(step_infos)
            model_outs.append(model_out)
        states[-1] = self._prev_states
        obses[-1] = self._prev_obs
        dones[-1] = self._prev_dones
        model_outs.append(self.model.step(self._prev_states, self._prev_obs))
        return Rollout(states, obses, rews, dones, infos, model_outs)

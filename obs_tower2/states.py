from anyrl.envs.wrappers import BatchedWrapper
import gym
import numpy as np
import torch

from .constants import STATE_SIZE, STATE_STACK, NUM_ACTIONS
from .model import StateClassifier


class StateEnv(gym.Wrapper):
    def __init__(self, env, state_features=None):
        super().__init__(env)
        self.state_features = state_features or StateFeatures()
        self.prev_states = np.zeros([STATE_STACK, STATE_SIZE], dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_states.fill(0)
        self.prev_states[-1, NUM_ACTIONS + 1:] = self.state_features.features(np.array([obs]))
        return (self.prev_states.copy(), obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.prev_states[:-1] = self.prev_states[1:]
        self.prev_states[-1] = np.zeros_like(self.prev_states[-1])
        self.prev_states[-1, action] = 1
        self.prev_states[-1, NUM_ACTIONS] = rew
        self.prev_states[-1, NUM_ACTIONS + 1:] = self.state_features.features(np.array([obs]))
        return (self.prev_states.copy(), obs), rew, done, info


class BatchedStateEnv(BatchedWrapper):
    def __init__(self, env, state_features=None):
        super().__init__(env)
        self.state_features = state_features or StateFeatures()
        self.prev_states = np.zeros([env.num_sub_batches, env.num_envs_per_sub_batch,
                                     STATE_STACK, STATE_SIZE], dtype=np.float32)
        self.prev_actions = [[None] * env.num_envs_per_sub_batch
                             for _ in range(env.num_sub_batches)]

    def reset_wait(self, sub_batch=0):
        obses = self.env.reset_wait(sub_batch=sub_batch)
        feats = self.state_features.features(np.array(obses))
        self.prev_states[sub_batch].fill(0)
        self.prev_states[sub_batch, :, -1, NUM_ACTIONS + 1:] = feats
        return (self.prev_states[sub_batch].copy(), obses)

    def step_start(self, actions, sub_batch=0):
        self.prev_actions[sub_batch] = actions
        return self.env.step_start(actions, sub_batch=sub_batch)

    def step_wait(self, sub_batch=0):
        obses, rews, dones, infos = self.env.step_wait(sub_batch=sub_batch)
        self.prev_states[sub_batch, :, :-1] = self.prev_states[sub_batch, :, 1:]
        features = self.state_features.features(np.array(obses)[..., -3:])
        for i, done in enumerate(dones):
            if done:
                self.prev_states[sub_batch, i].fill(0.0)
            else:
                self.prev_states[sub_batch, i, -1].fill(0.0)
                self.prev_states[sub_batch, i, -1, self.prev_actions[sub_batch][i]] = 1.0
                self.prev_states[sub_batch, i, -1, NUM_ACTIONS] = rews[i]
        self.prev_states[sub_batch, :, -1, NUM_ACTIONS + 1:] = features
        return (self.prev_states[sub_batch].copy(), obses), rews, dones, infos


class StateFeatures:
    def __init__(self, path='save_class.pkl'):
        self.classifier = StateClassifier()
        self.classifier.load_state_dict(torch.load(path))
        self.classifier.to(torch.device('cuda'))

    def features(self, obses):
        res = []
        for obs in obses:
            # Check if we have a key.
            if (obs[3] != 0).any():
                res.append([1.0])
            else:
                res.append([0.0])
        device = next(self.classifier.parameters()).device
        obs_tensor = torch.from_numpy(obses).to(device)
        class_out = torch.sigmoid(self.classifier(obs_tensor)).detach().cpu().numpy()
        return np.concatenate([np.array(res), class_out], axis=-1)

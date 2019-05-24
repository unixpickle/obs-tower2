import gym
import numpy as np
import torch

from .batched_env import BatchedWrapper
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
        feats = self.state_features.features(np.array([obs])[..., -3:])
        self.prev_states[-1, NUM_ACTIONS + 1:] = feats
        return (self.prev_states.copy(), obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.prev_states[:-1] = self.prev_states[1:]
        self.prev_states[-1] = np.zeros_like(self.prev_states[-1])
        self.prev_states[-1, action] = 1
        self.prev_states[-1, NUM_ACTIONS] = rew
        feats = self.state_features.features(np.array([obs])[..., -3:])
        self.prev_states[-1, NUM_ACTIONS + 1:] = feats
        if 'extra_reward' in info:
            rew += info['extra_reward']
        return (self.prev_states.copy(), obs), rew, done, info


class BatchedStateEnv(BatchedWrapper):
    def __init__(self, env, state_features=None):
        super().__init__(env)
        self.state_features = state_features or StateFeatures()
        self.prev_states = np.zeros([env.num_envs, STATE_STACK, STATE_SIZE], dtype=np.float32)

    def reset(self):
        obses = self.env.reset()
        feats = self.state_features.features(np.array(obses)[..., -3:])
        self.prev_states.fill(0)
        self.prev_states[:, -1, NUM_ACTIONS + 1:] = feats
        return (self.prev_states.copy(), obses)

    def step(self, actions):
        obses, rews, dones, infos = self.env.step(actions)
        self.prev_states[:, :-1] = self.prev_states[:, 1:]
        features = self.state_features.features(np.array(obses)[..., -3:])
        for i, done in enumerate(dones):
            if done:
                self.prev_states[i].fill(0.0)
            else:
                self.prev_states[i, -1].fill(0.0)
                self.prev_states[i, -1, actions[i]] = 1.0
                self.prev_states[i, -1, NUM_ACTIONS] = rews[i]
        self.prev_states[:, -1, NUM_ACTIONS + 1:] = features
        for i, info in enumerate(infos):
            if 'extra_reward' in info:
                rews[i] += info['extra_reward']
        return (self.prev_states.copy(), obses), rews, dones, infos


class StateFeatures:
    def __init__(self, path='save_classifier.pkl'):
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
        class_out = self.classifier.features(obs_tensor).detach().cpu().numpy()
        return np.concatenate([np.array(res), class_out], axis=-1)

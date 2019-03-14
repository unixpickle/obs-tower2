from PIL import Image
from anyrl.envs import batched_gym_env
from anyrl.envs.wrappers import FrameStackEnv
import gym
import numpy as np
import os

from .constants import HUMAN_ACTIONS
from .roller import Roller


def big_obs(obs, info):
    res = (info['brain_info'].visual_observations[0][0, :, :, :] * 255).astype(np.uint8)
    res[:20] = np.array(Image.fromarray(obs).resize((168, 168)))[:20]
    return res


def create_batched_env(num_envs, key_reward=False):
    return batched_gym_env([lambda i=i: create_single_env(i, key_reward=key_reward)
                            for i in range(num_envs)])


def create_single_env(idx, clear=True, key_reward=False):
    from obstacle_tower_env import ObstacleTowerEnv
    env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'], worker_id=idx)
    env = FrameStackEnv(env)
    env = HumanActionEnv(env)
    if clear:
        env = ClearInfoEnv(env)
    if key_reward:
        env = KeyRewardEnv(env)
    return env


def log_floors(rollout):
    for t in range(rollout.num_steps):
        for b in range(rollout.batch_size):
            if rollout.dones[t, b]:
                print('floor=%d' % rollout.infos[t][b]['floor'])


class LogRoller(Roller):
    def rollout(self):
        result = super().rollout()
        log_floors(result)
        return result


class ClearInfoEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, _ = self.env.step(action)
        return obs, rew, done, {}


class HumanActionEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = HUMAN_ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, act):
        return self.actions[act]


class KeyRewardEnv(gym.Wrapper):
    def __init__(self, env, reward=10.0):
        super().__init__(env)
        self.key_reward = reward
        self.top_line = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.top_line = obs[3]
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        top_line = obs[3]
        if not (top_line == self.top_line).all():
            info['extra_reward'] = self.key_reward
        else:
            info['extra_reward'] = 0.0
        rew += info['extra_reward']
        self.top_line = top_line
        return obs, rew, done, info


class FloorTrackEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.floor = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.floor = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if rew == 1.0:
            self.floor += 1.0
        info['floor'] = self.floor
        return obs, rew, done, info

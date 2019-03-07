from PIL import Image
from anyrl.envs import batched_gym_env
from anyrl.envs.wrappers import FrameStackEnv
import gym
import numpy as np
import os


def big_obs(obs, info):
    res = (info['brain_info'].visual_observations[0][0, :, :, :] * 255).astype(np.uint8)
    res[:20] = np.array(Image.fromarray(obs).resize((168, 168)))[:20]
    return res


def create_batched_env(num_envs):
    return batched_gym_env([lambda i=i: create_single_env(i) for i in range(num_envs)])


def create_single_env(idx, clear=True):
    from obstacle_tower_env import ObstacleTowerEnv
    env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'], worker_id=idx)
    env = FrameStackEnv(env)
    if clear:
        env = ClearInfoEnv(env)
    return env


class ClearInfoEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, _ = self.env.step(action)
        return obs, rew, done, {}

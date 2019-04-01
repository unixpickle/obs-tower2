import os
import random

from PIL import Image
from anyrl.envs import batched_gym_env
import gym
import numpy as np
import torchvision.transforms.functional as TF

from .constants import HUMAN_ACTIONS
from .roller import Roller


def big_obs(obs, info):
    res = (info['brain_info'].visual_observations[0][0, :, :, :] * 255).astype(np.uint8)
    res[:20] = np.array(Image.fromarray(obs).resize((168, 168)))[:20]
    return res


def create_batched_env(num_envs, **kwargs):
    return batched_gym_env([lambda i=i: create_single_env(i, **kwargs)
                            for i in range(num_envs)])


def create_single_env(idx, clear=True, key_reward=False, augment=False, rand_floor=False):
    from obstacle_tower_env import ObstacleTowerEnv
    env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'], worker_id=idx)
    if rand_floor:
        env = RandomFloorEnv(env)
    if augment:
        env = AugmentEnv(env)
    env = FrameStackEnv(env)
    env = HumanActionEnv(env)
    if clear:
        env = ClearInfoEnv(env)
    if key_reward:
        env = KeyRewardEnv(env)
    env = FloorTrackEnv(env)
    env = TimeRewardEnv(env)
    return env


def log_floors(rollout):
    for t in range(1, rollout.num_steps):
        for b in range(rollout.batch_size):
            if rollout.dones[t, b]:
                print('floor=%d' % rollout.infos[t - 1][b]['floor'])


def mirror_obs(obs):
    obs = obs.copy()
    obs[10:] = obs[10:, ::-1]
    return obs


def mirror_action(act):
    direction = (act % 18) // 6
    act -= direction * 6
    if direction == 1:
        direction = 2
    elif direction == 2:
        direction = 1
    act += direction * 6
    return act


class Augmentation:
    def __init__(self):
        self.brightness = random.random() * 0.1 + 0.95
        self.contrast = random.random() * 0.1 + 0.95
        self.gamma = random.random() * 0.1 + 0.95
        self.hue = random.random() * 0.1 - 0.05
        self.saturation = random.random() * 0.1 + 0.95
        self.translation = (random.randrange(-2, 3), random.randrange(-2, 3))

    def apply(self, image):
        return Image.fromarray(self.apply_np(np.array(image)))

    def apply_np(self, np_image):
        content = Image.fromarray(np_image[10:])
        content = TF.adjust_brightness(content, self.brightness)
        content = TF.adjust_contrast(content, self.contrast)
        content = TF.adjust_gamma(content, self.gamma)
        content = TF.adjust_hue(content, self.hue)
        content = TF.adjust_saturation(content, self.saturation)
        content = TF.affine(content, 0, self.translation, 1.0, 0)
        result = np.array(np_image)
        result[10:] = np.array(content)
        return result


class LogRoller(Roller):
    def rollout(self):
        result = super().rollout()
        log_floors(result)
        return result


class AugmentEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.augmentation = None

    def reset(self, **kwargs):
        self.augmentation = Augmentation()
        obs = self.env.reset(**kwargs)
        return self.augmentation.apply_np(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self.augmentation.apply_np(obs)
        return obs, rew, done, info


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


class TimeRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.mid_line = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.mid_line = obs[7]
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        mid_line = obs[7]
        if rew != 1.0 and np.sum(mid_line == 0) < np.sum(self.mid_line == 0):
            rew += 0.1
        self.mid_line = mid_line
        return obs, rew, done, info


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
        self.floor = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if rew == 1.0:
            self.floor += 1.0
        info['floor'] = self.floor
        return obs, rew, done, info


class RandomFloorEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        self.env.floor(random.randrange(25))
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class FrameStackEnv(gym.Wrapper):
    """
    An environment that stacks images.
    The stacking is ordered from oldest to newest.
    At the beginning of an episode, the first observation
    is repeated in order to complete the stack.
    """

    def __init__(self, env, num_images=2):
        """
        Create a frame stacking environment.
        Args:
          env: the environment to wrap.
          num_images: the number of images to stack.
            This includes the current observation.
        """
        super().__init__(env)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(np.repeat(old_space.low, num_images, axis=-1),
                                                np.repeat(old_space.high, num_images, axis=-1),
                                                dtype=old_space.dtype)
        self._num_images = num_images
        self._history = []

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._history = [obs] * self._num_images
        return self._cur_obs()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._history.append(obs)
        self._history = self._history[1:]
        return self._cur_obs(), rew, done, info

    def _cur_obs(self):
        return np.concatenate(self._history, axis=-1)

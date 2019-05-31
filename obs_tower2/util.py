import os
import random

from PIL import Image
import gym
import gym.spaces
import numpy as np
from obstacle_tower_env import ObstacleTowerEnv
import torch
import torchvision.transforms.functional as TF

from .batched_env import BatchedGymEnv
from .constants import HUMAN_ACTIONS, IMAGE_DEPTH, IMAGE_SIZE, NUM_ACTIONS
from .roller import Roller


def big_obs(obs, info):
    res = (info['brain_info'].visual_observations[0][0, :, :, :] * 255).astype(np.uint8)
    res[:20] = np.array(Image.fromarray(obs).resize((168, 168)))[:20]
    return res


def create_batched_env(num_envs, start=0, **kwargs):
    env_fns = [lambda i=i: create_single_env(i + start, **kwargs) for i in range(num_envs)]
    return BatchedGymEnv(gym.spaces.Discrete(NUM_ACTIONS),
                         gym.spaces.Box(low=0, high=0xff,
                                        shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH),
                                        dtype=np.uint8),
                         env_fns)


def create_single_env(idx, clear=True, augment=False, rand_floors=None):
    env = TimeRewardEnv(os.environ['OBS_TOWER_PATH'], worker_id=idx)
    if clear:
        env = ClearInfoEnv(env)
    if rand_floors is not None:
        env = RandomFloorEnv(env, rand_floors)
    if augment:
        env = AugmentEnv(env)
    env = FrameStackEnv(env)
    env = HumanActionEnv(env)
    env = FloorTrackEnv(env)
    return env


def log_floors(rollout):
    for t in range(1, rollout.num_steps):
        for b in range(rollout.batch_size):
            if rollout.dones[t, b]:
                info = rollout.infos[t - 1][b]
                if 'start_floor' in info:
                    print('start=%d floor=%d' % (info['start_floor'], info['floor']))
                else:
                    print('floor=%d' % info['floor'])


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


def atomic_save(obj, path):
    torch.save(obj, path + '.tmp')
    os.rename(path + '.tmp', path)


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


class TimeRewardEnv(ObstacleTowerEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_time = None

    def reset(self):
        config = {"total-floors": 25}
        self.last_time = None
        obs = super().reset(config)
        return obs

    def _single_step(self, info):
        obs, rew, done, final_info = super()._single_step(info)
        extra_reward = 0.0
        if self.last_time is not None:
            if rew != 1.0 and info.vector_observations[0][6] > self.last_time:
                extra_reward = 0.1
        self.last_time = info.vector_observations[0][6]
        final_info['extra_reward'] = extra_reward
        return obs, rew, done, final_info


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
        obs, rew, done, info = self.env.step(action)
        new_info = {}
        if 'extra_reward' in info:
            new_info['extra_reward'] = info['extra_reward']
        return obs, rew, done, new_info


class HumanActionEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = HUMAN_ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, act):
        return self.actions[act]


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
    def __init__(self, env, floors):
        super().__init__(env)
        self.floors = floors

    def reset(self, **kwargs):
        self.unwrapped.floor(random.randrange(*self.floors))
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info['start_floor'] = self.unwrapped._floor
        return obs, rew, done, info


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

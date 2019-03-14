import functools
import json
import os
import random

from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

from .constants import IMAGE_SIZE, IMAGE_DEPTH
from .rollout import Rollout


def load_data(dirpath=None, augment=False):
    if dirpath is None:
        dirpath = os.environ['OBS_TOWER_RECORDINGS']
    training = []
    testing = []
    for item in os.listdir(dirpath):
        path = os.path.join(dirpath, item)
        if not os.path.isdir(path):
            continue
        if not os.path.exists(os.path.join(path, 'actions.json')):
            continue
        recording = Recording(path, augment=augment)
        if recording.seed < 25:
            testing.append(recording)
        else:
            training.append(recording)
    return training, testing


def recording_rollout(recordings, batch, horizon):
    """
    Create a rollout of segments from recordings.
    """
    rollout = Rollout(states=np.zeros([horizon + 1, batch, 0], dtype=np.float32),
                      obses=np.zeros([horizon + 1, batch, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH],
                                     dtype=np.uint8),
                      rews=np.zeros([horizon, batch], dtype=np.float32),
                      dones=np.zeros([horizon + 1, batch], dtype=np.float32),
                      infos=[[{} for _ in range(batch)] for _ in range(horizon)],
                      model_outs=[{'actions': [None] * batch} for _ in range(horizon + 1)])
    for b in range(batch):
        recording = random.choice(recordings)
        recording.sample_augmentation()
        t0 = random.randrange(recording.num_steps - horizon - 1)
        for t in range(t0, t0 + horizon):
            rollout.obses[t - t0, b] = recording.observation(t)
            rollout.rews[t - t0, b] = recording.rewards[t]
            rollout.model_outs[t - t0]['actions'][b] = recording.actions[t]
        rollout.obses[-1, b] = recording.observation(t0 + horizon)
        rollout.model_outs[-1]['actions'][b] = recording.actions[t0 + horizon]

    return rollout


class Recording:
    def __init__(self, path, augment=False):
        self.path = path
        self.augment = augment
        self.augmentation = None
        self.seed = int(os.path.basename(path).split('_')[0])
        self.actions = self._load_json('actions.json')
        self.rewards = self._load_json('rewards.json')

    def sample_augmentation(self):
        self.augmentation = Augmentation()

    @property
    def num_steps(self):
        return len(self.actions)

    def observation(self, timestep, stack=2):
        history = []
        for i in range(timestep - stack + 1, timestep + 1):
            img = self._load_image(max(0, i))
            history.append(img)
        return np.concatenate(history, axis=-1)

    @functools.lru_cache(maxsize=4)
    def _load_image(self, idx):
        img = Image.open(os.path.join(self.path, '%d.png' % idx))
        if self.augment and self.augmentation is not None:
            img = self.augmentation.apply(img)
        return np.array(img)

    def _load_json(self, name):
        path = os.path.join(self.path, name)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as in_file:
            return json.load(in_file)


class Augmentation:
    def __init__(self):
        self.brightness = random.random() * 0.1 + 0.95
        self.contrast = random.random() * 0.1 + 0.95
        self.gamma = random.random() * 0.1 + 0.95
        self.hue = random.random() * 0.05
        self.saturation = random.random() * 0.1 + 0.95
        self.translation = (random.randrange(-2, 3), random.randrange(-2, 3))

    def apply(self, image):
        arr = np.array(image)
        content = Image.fromarray(arr[10:])
        content = TF.adjust_brightness(content, self.brightness)
        content = TF.adjust_contrast(content, self.contrast)
        content = TF.adjust_gamma(content, self.gamma)
        content = TF.adjust_hue(content, self.hue)
        content = TF.adjust_saturation(content, self.saturation)
        content = TF.affine(content, 0, self.translation, 1.0, 0)
        arr[10:] = np.array(content)
        return Image.fromarray(arr)

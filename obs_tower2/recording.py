import functools
import json
import os

from PIL import Image
import numpy as np


def load_data(dirpath=None, require_door_open=False):
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
        recording = Recording(path)
        if require_door_open and recording.door_open is None:
            continue
        if recording.seed < 25:
            testing.append(recording)
        else:
            training.append(recording)
    return training, testing


class Recording:
    def __init__(self, path):
        self.path = path
        self.seed = int(os.path.basename(path).split('_')[0])
        self.actions = self._load_json('actions.json')
        self.rewards = self._load_json('rewards.json')

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
        return np.array(Image.open(os.path.join(self.path, '%d.png' % idx)))

    def _load_json(self, name):
        path = os.path.join(self.path, name)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as in_file:
            return json.load(in_file)

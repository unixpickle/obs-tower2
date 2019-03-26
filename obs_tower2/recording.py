import functools
import json
import os
import random

from PIL import Image
import numpy as np

from .constants import HUMAN_ACTIONS, IMAGE_DEPTH, IMAGE_SIZE, NUM_ACTIONS, STATE_SIZE, STATE_STACK
from .rollout import Rollout
from .states import StateFeatures
from .util import Augmentation


def load_all_data(**kwargs):
    train, test = load_data(**kwargs)
    return train + test


def load_data(dirpaths=(os.environ['OBS_TOWER_RECORDINGS'],),
              augment=False):
    training = []
    testing = []
    for dirpath in dirpaths:
        for item in os.listdir(dirpath):
            path = os.path.join(dirpath, item)
            if not os.path.isdir(path):
                continue
            if not os.path.exists(os.path.join(path, 'actions.json')):
                continue
            recording = Recording(path, augment=augment)
            if recording.uid < 1e8:
                testing.append(recording)
            else:
                training.append(recording)
    return training, testing


def recording_rollout(recordings, batch, horizon):
    """
    Create a rollout of segments from recordings.
    """
    assert all([rec.num_steps > horizon for rec in recordings])
    rollout = Rollout(states=np.zeros([horizon + 1, batch, STATE_STACK, STATE_SIZE],
                                      dtype=np.float32),
                      obses=np.zeros([horizon + 1, batch, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH],
                                     dtype=np.uint8),
                      rews=np.zeros([horizon, batch], dtype=np.float32),
                      dones=np.zeros([horizon + 1, batch], dtype=np.float32),
                      infos=[[{} for _ in range(batch)] for _ in range(horizon)],
                      model_outs=[{'actions': [None] * batch} for _ in range(horizon + 1)])
    weights = np.array([rec.num_steps for rec in recordings], dtype=np.float)
    weights /= np.sum(weights)
    for b in range(batch):
        recording = recordings[np.random.choice(len(recordings), p=weights)]
        recording.sample_augmentation()
        t0 = random.randrange(recording.num_steps - horizon - 1)
        for t in range(t0, t0 + horizon):
            rollout.states[t - t0, b] = recording.state(t)
            rollout.obses[t - t0, b] = recording.observation(t)
            rollout.rews[t - t0, b] = recording.rewards[t]
            rollout.model_outs[t - t0]['actions'][b] = recording.actions[t]
        rollout.states[-1, b] = recording.state(t0 + horizon)
        rollout.obses[-1, b] = recording.observation(t0 + horizon)
        rollout.model_outs[-1]['actions'][b] = recording.actions[t0 + horizon]

    return rollout


class Recording:
    def __init__(self, path, augment=False):
        self.path = path
        self.augment = augment
        self.augmentation = None
        comps = os.path.basename(path).split('_')
        self.seed = int(comps[0])
        self.uid = int(comps[1])
        self.actions = self._load_json('actions.json')
        self.rewards = self._load_json('rewards.json')
        # TODO: load this from JSON, where it should be cached
        # before use.
        self.current_state = [None] * (self.num_steps + 1)

    def sample_augmentation(self):
        self.augmentation = Augmentation()

    @property
    def num_steps(self):
        return len(self.actions)

    def observation(self, timestep, stack=2):
        history = []
        for i in range(timestep - stack + 1, timestep + 1):
            img = self.load_frame(max(0, i))
            history.append(img)
        return np.concatenate(history, axis=-1)

    def state(self, timestep):
        result = []
        for i in range(timestep - STATE_STACK + 1, timestep + 1):
            result.append(self._current_state(i))
        return np.array(result, dtype=np.float32)

    def _current_state(self, timestep):
        if timestep < 0:
            return [0.0] * STATE_SIZE
        if self.current_state[timestep] is None:
            feats = StateFeatures().features(np.array([self.load_frame(timestep)]))[0]
            self.current_state[timestep] = [0.0] * (STATE_SIZE - len(feats)) + list(feats)
            if timestep > 0:
                self.current_state[timestep][HUMAN_ACTIONS.index(self.actions[timestep - 1])] = 1.0
                self.current_state[timestep][NUM_ACTIONS] = self.rewards[timestep - 1]
        return self.current_state[timestep]

    @functools.lru_cache(maxsize=4)
    def load_frame(self, idx):
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

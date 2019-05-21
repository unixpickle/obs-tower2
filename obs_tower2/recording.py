import json
import os
import random

from PIL import Image
import numpy as np

from .constants import (FRAME_STACK, HUMAN_ACTIONS, IMAGE_DEPTH, IMAGE_SIZE, NUM_ACTIONS,
                        STATE_SIZE, STATE_STACK)
from .rollout import Rollout
from .util import Augmentation, mirror_obs, mirror_action


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
                if augment:
                    testing.append(recording.mirror())
            else:
                training.append(recording)
                if augment:
                    training.append(recording.mirror())
    return training, testing


def sample_recordings(recordings, count):
    weights = np.array([rec.num_steps for rec in recordings], dtype=np.float)
    weights /= np.sum(weights)
    return [recordings[np.random.choice(len(recordings), p=weights)] for _ in range(count)]


def recording_rollout(recordings, batch, horizon, state_features):
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
    for b, recording in enumerate(sample_recordings(recordings, batch)):
        recording.sample_augmentation()
        t0 = random.randrange(recording.num_steps - horizon - 1)
        rollout.obses[:, b], rollout.states[:, b] = recording.obses_and_states(t0, horizon + 1,
                                                                               state_features)
        for t in range(t0, t0 + horizon):
            rollout.rews[t - t0, b] = recording.rewards[t]
            rollout.model_outs[t - t0]['actions'][b] = recording.actions[t]
        rollout.model_outs[-1]['actions'][b] = recording.actions[t0 + horizon]
    return rollout


class Recording:
    def __init__(self, path, augment=False, mirrored=False):
        if path.endswith('/'):
            path = path[:-1]
        self.path = path
        self.augment = augment
        self.augmentation = None
        self.mirrored = mirrored
        comps = os.path.basename(path).split('_')
        self.seed = int(comps[0])
        self.uid = int(comps[1])
        self.floor = int(comps[2])
        self.version = comps[3]
        self.actions = self._load_json('actions.json')
        self.rewards = self._load_json('rewards.json')
        if mirrored:
            self.actions = [mirror_action(a) for a in self.actions]

    def sample_augmentation(self):
        self.augmentation = Augmentation()

    def mirror(self):
        return Recording(self.path, augment=self.augment, mirrored=not self.mirrored)

    @property
    def num_steps(self):
        return len(self.actions)

    @property
    def num_floors(self):
        return sum(x == 1 for x in self.rewards)

    def obses_and_states(self, t0, count, state_features):
        start_time = min(t0 - FRAME_STACK + 1, t0 - STATE_STACK + 1)
        frames = np.array([self.load_frame(i) for i in range(start_time, t0 + count)])
        features = state_features.features(frames)
        raw_states = [self.raw_state(i + start_time, f) for i, f in enumerate(features)]
        obses = []
        states = []
        for t in range(t0, t0 + count):
            offset = t - start_time
            obses.append(np.concatenate(frames[offset - FRAME_STACK + 1:offset + 1], axis=-1))
            states.append(np.array(raw_states[offset - STATE_STACK + 1:offset + 1]))
        return np.array(obses), np.array(states)

    def raw_state(self, timestep, features):
        if timestep < 0:
            return [0.0] * STATE_SIZE
        res = [0.0] * (STATE_SIZE - len(features)) + list(features)
        if timestep > 0:
            res[HUMAN_ACTIONS.index(self.actions[timestep - 1])] = 1.0
            res[NUM_ACTIONS] = self.rewards[timestep - 1]
        return res

    def load_frame(self, idx):
        if idx < 0:
            idx = 0
        img = Image.open(os.path.join(self.path, '%d.png' % idx))
        if self.mirrored:
            img = Image.fromarray(mirror_obs(np.array(img)))
        if self.augment and self.augmentation is not None:
            img = self.augmentation.apply(img)
        return np.array(img)

    def _load_json(self, name):
        path = os.path.join(self.path, name)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as in_file:
            return json.load(in_file)

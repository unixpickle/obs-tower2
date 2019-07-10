"""
Tools for processing recording data from human
demonstrators.
"""

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
    """
    Load training and test recordings from disk.

    Args:
        dirpaths: a sequence of directories where
          recordings are stored.
        augment: if True, the resulting recordings will
          include mirrored versions of every recording on
          disk, and will have their augment flags enabled.
          This flag affects how the recordings sample
          observations.
    """
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


def truncate_recordings(recordings, max_floor, min_length=300):
    """
    Truncate the recordings so that they never exceed a
    given floor.

    This can be used, for example, to train an agent to
    solve the beginning of every level.

    Args:
        recordings: a list of Recordings.
        max_floor: the floor which no recording should get
          up to.
        min_length: the minimum length of a truncated
          recording for it to be included in the result.

    Returns:
        A list of truncated Recordings.
    """
    res = []
    for rec in recordings:
        if rec.floor < 0:
            continue
        trunc = rec.truncate(max_floor)
        if trunc is not None and trunc.num_steps >= min_length:
            res.append(trunc)
    return res


def sample_recordings(recordings, count):
    """
    Sample recordings such that recordings are weighted in
    proportion to their number of frames.
    """
    weights = np.array([rec.num_steps for rec in recordings], dtype=np.float)
    weights /= np.sum(weights)
    return [recordings[np.random.choice(len(recordings), p=weights)] for _ in range(count)]


def recording_rollout(recordings, batch, horizon, state_features):
    """
    Create a rollout of segments from recordings.

    Args:
        recordings: a sequence of recordings.
        batch: the number of segments to generate.
        horizon: the number of timesteps per segment.
        state_features: a StateFeatures instance to
          generate states for all of the observations.
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
    """
    An object that represents a single recording of an
    agent playing obstacle tower.

    Recordings are stored on disk as a directory of frame
    images and several metadata JSON files.

    Args:
        path: the directory path of the recording.
        augment: if True, frames will be augmented so long
          as sample_augmentation() has been called.
        mirrored: if True, all observations are actions
          are flipped from left to right.
    """

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
        """
        Change the augmentation settings that are used for
        loaded frames.

        This is used to ensure that augmentation settings
        are the same throughout a synthetic rollout, but
        re-sampled across synthetic rollouts.

        This only has an effect if the augment flag was
        True when creating this recording.
        """
        self.augmentation = Augmentation()

    def mirror(self):
        """
        Copy this recording, but flip it left-to-right.
        """
        return Recording(self.path, augment=self.augment, mirrored=not self.mirrored)

    @property
    def num_steps(self):
        """
        Get the number of timesteps in the recording.
        """
        return len(self.actions)

    @property
    def num_floors(self):
        """
        Get the number of floors the agent passed in the
        recording.
        """
        return sum(x > 0.99 for x in self.rewards)

    def obses_and_states(self, t0, count, state_features):
        """
        Generate a batch of observations and state stacks
        for a range of timesteps in this recording.

        Args:
            t0: the first timestep to look at.
            count: the number of timesteps to look at.
            state_features: a StateFeatures instance.

        Returns:
            A tuple (observations, states):
              observations: a [count x H x W x D] array.
              states: a [count x stack x size] array of
                state stacks.
        """
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
        """
        Compute the instantaneous state vector for a given
        timestep, given existing classifier outputs.
        """
        if timestep < 0:
            return [0.0] * STATE_SIZE
        res = [0.0] * (STATE_SIZE - len(features)) + list(features)
        if timestep > 0:
            res[HUMAN_ACTIONS.index(self.actions[timestep - 1])] = 1.0
            res[NUM_ACTIONS] = self.rewards[timestep - 1]
        return res

    def load_frame(self, idx):
        """
        Load an array for the frame image at the given
        timestep index.
        """
        if idx < 0:
            idx = 0
        img = Image.open(os.path.join(self.path, '%d.png' % idx))
        if self.mirrored:
            img = Image.fromarray(mirror_obs(np.array(img)))
        if self.augment and self.augmentation is not None:
            img = self.augmentation.apply(img)
        return np.array(img)

    def truncate(self, max_floor):
        """
        Truncate the recording to a maximum floor number.
        """
        num_steps = 0
        floor = self.floor
        for rew in self.rewards:
            if rew > 0.99:
                floor += 1
            if floor == max_floor:
                break
            num_steps += 1
        res = Recording(self.path, mirrored=self.mirrored)
        res.augmentation = self.augmentation
        res.actions = self.actions[:num_steps]
        res.rewards = self.rewards[:num_steps]
        return res

    def _load_json(self, name):
        path = os.path.join(self.path, name)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as in_file:
            return json.load(in_file)

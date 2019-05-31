"""
Record a human demonstration with a simple GUI.

Use the keyboard to control the player. The arrow keys and
space bar are all you need.

To pause, press 'p'. To resume, press 'r'. To cancel the
recording, press escape.
"""

import json
import os
import random
import shutil
import time

from PIL import Image
from gym.envs.classic_control.rendering import SimpleImageViewer
import numpy as np
from obstacle_tower_env import ObstacleTowerEnv
import pyglet.window

from obs_tower2.recording import load_all_data
from obs_tower2.util import big_obs

RES_DIR = os.environ['OBS_TOWER_RECORDINGS']
TMP_DIR = '/tmp/obs_tower_recordings'


class EnvInteractor(SimpleImageViewer):
    def __init__(self):
        super().__init__(maxwidth=800)
        self.keys = pyglet.window.key.KeyStateHandler()
        self._paused = False
        self._jump = False
        self._finish_early = False
        self.imshow(np.zeros([168, 168, 3], dtype=np.uint8))

    def imshow(self, image):
        was_none = self.window is None
        image = Image.fromarray(image)
        image = image.resize((800, 800))
        image = np.array(image)
        super().imshow(image)
        if was_none:
            self.window.event(self.on_key_press)
            self.window.push_handlers(self.keys)

    def get_action(self):
        event = 0
        if self.keys[pyglet.window.key.UP]:
            event += 18
        if self.keys[pyglet.window.key.LEFT]:
            event += 6
        elif self.keys[pyglet.window.key.RIGHT]:
            event += 12
        if self.keys[pyglet.window.key.SPACE] or self._jump:
            event += 3
            self._jump = False
        if self.keys[pyglet.window.key.ESCAPE]:
            self._finish_early = True
        return event

    def pause(self):
        self._paused = True

    def paused(self):
        if self.keys[pyglet.window.key.P]:
            self._paused = True
        elif self.keys[pyglet.window.key.R]:
            self._paused = False
        return self._paused

    def finish_early(self):
        return self._finish_early

    def on_key_press(self, x, y):
        if x == pyglet.window.key.SPACE:
            self._jump = True
        return True

    def reset(self):
        self._jump = False
        self._paused = False
        self._finish_early = False


def main():
    viewer = EnvInteractor()
    env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'], worker_id=random.randrange(11, 20))
    run_episode(env, viewer)


def run_episode(env, viewer):
    seed = select_seed()
    env.seed(seed)
    obs = env.reset()
    record_episode(seed, env, viewer, obs)


def record_episode(seed, env, viewer, obs, tmp_dir=TMP_DIR, res_dir=RES_DIR, max_steps=None,
                   min_floors=1):
    for p in [tmp_dir, res_dir]:
        if not os.path.exists(p):
            os.mkdir(p)

    dirname = '%d_%d_%d_%s' % (seed, int(random.random() * 1e9), env.unwrapped._floor or 0,
                               env.unwrapped.version)
    tmp_dir = os.path.join(tmp_dir, dirname)
    os.mkdir(tmp_dir)

    done = False
    floors = 0
    action_log = []
    reward_log = []
    Image.fromarray(obs).save(os.path.join(tmp_dir, '0.png'))
    i = 1
    last_time = time.time()
    while not done and not viewer.finish_early():
        if not viewer.paused():
            action = viewer.get_action()
            action_log.append(action)
            obs, rew, done, info = env.step(action)
            if rew == 1.0:
                floors += 1
                print('solved %d floors' % floors)
            reward_log.append(rew)
            Image.fromarray(obs).save(os.path.join(tmp_dir, '%d.png' % i))
            if max_steps is not None and i >= max_steps and floors > 0:
                break
            i += 1
        viewer.imshow(big_obs(obs, info))
        pyglet.clock.tick()
        delta = time.time() - last_time
        time.sleep(max(0, 1 / 10 - delta))
        last_time = time.time()

    if floors < min_floors:
        print('Not saving recording.')
        shutil.rmtree(tmp_dir)
        return

    with open(os.path.join(tmp_dir, 'actions.json'), 'w+') as out:
        json.dump(action_log, out)
    with open(os.path.join(tmp_dir, 'rewards.json'), 'w+') as out:
        json.dump(reward_log, out)

    os.rename(tmp_dir, os.path.join(res_dir, dirname))


def select_seed(res_dir=RES_DIR, floor=0):
    if not os.path.exists(res_dir):
        return random.randrange(100)
    counts = {k: 0 for k in range(100)}
    for rec in load_all_data(dirpaths=(res_dir,)):
        if rec.floor == floor:
            counts[rec.seed] += 1
    pairs = list(counts.items())
    min_count = min([x[1] for x in pairs])
    min_seeds = [x[0] for x in pairs if x[1] == min_count]
    return random.choice(min_seeds)


if __name__ == '__main__':
    main()

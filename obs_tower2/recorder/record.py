"""
Record a human demonstration with a simple GUI.

Use the keyboard to control the player. The arrow keys and
space bar are all you need.

To pause, press 'p'. To resume, press 'r'. To finish the
recording before the end of the episode, press escape.
"""

import json
import os
import random
import shutil

from PIL import Image
from obstacle_tower_env import ObstacleTowerEnv

from obs_tower2.recording import load_all_data
from obs_tower2.util import big_obs
from obs_tower2.recorder.env_interactor import EnvInteractor

RES_DIR = os.environ['OBS_TOWER_RECORDINGS']
TMP_DIR = '/tmp/obs_tower_recordings'


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

    start_floor = env.unwrapped._floor or 0
    dirname = '%d_%d_%d_%s' % (seed, int(random.random() * 1e9), start_floor, env.unwrapped.version)
    tmp_dir = os.path.join(tmp_dir, dirname)
    os.mkdir(tmp_dir)

    Image.fromarray(obs).save(os.path.join(tmp_dir, '0.png'))

    floors = [0]
    action_log = []
    reward_log = []
    i = [1]

    def step(action):
        action_log.append(action)
        obs, rew, done, info = env.step(action)
        new_floors = int(info['current_floor']) - start_floor
        if new_floors != floors[0]:
            floors[0] = new_floors
            print('solved %d floor%s' % (floors[0], '' if floors[0] == 1 else 's'))
        reward_log.append(rew)
        Image.fromarray(obs).save(os.path.join(tmp_dir, '%d.png' % i[0]))
        if done or (max_steps is not None and i[0] >= max_steps and floors[0] > 0):
            return None
        i[0] += 1
        return big_obs(obs, info)

    viewer.run_loop(step)

    if floors[0] < min_floors or (max_steps is not None and i[0] < max_steps):
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

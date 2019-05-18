"""
Record at the last floor reached in an existing
demonstration.
"""

import json
import os
import random
import sys
import tempfile
import time

from PIL import Image
from obstacle_tower_env import ObstacleTowerEnv
import pyglet.window

from obs_tower2.recording import Recording
from obs_tower2.recorder.record import EnvInteractor
from obs_tower2.util import big_obs


def main():
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: record_improve.py <recording_path>\n')
        os.exit(1)
    rec = Recording(sys.argv[1])
    env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'], worker_id=random.randrange(11, 20))
    try:
        floors = sum([x == 1.0 for x in rec.rewards])
        if rec.floor:
            env.floor(rec.floor)
        env.seed(rec.seed)
        env.reset()
        reached_floors = 0
        taken_steps = 0
        for action, rew in zip(rec.actions, rec.rewards):
            taken_steps += 1
            _, new_rew, done, _ = env.step(action)
            if rew != new_rew:
                sys.stderr.write('mismatching reward at step %d\n' % taken_steps)
                os.exit(1)
            if rew == 1.0:
                reached_floors += 1
            if reached_floors == floors:
                break
        print('Starting at timestep %d of %d' % (taken_steps, len(rec.actions)))
        record_tail(env, rec, taken_steps)
    finally:
        env.close()


def record_tail(obs, env, rec, timestep):
    obs = rec.load_frame(timestep)
    viewer = EnvInteractor()
    with tempfile.TemporaryDirectory() as tmp_dir:
        last_time = time.time()
        done = False
        action_log = rec.actions[:timestep]
        reward_log = rec.rewards[:timestep]
        floors = 0
        i = timestep + 1
        while not done:
            if not viewer.paused():
                action = viewer.get_action()
                action_log.append(action)
                obs, rew, done, info = env.step(action)
                if rew == 1.0:
                    floors += 1
                    print('solved %d floors' % floors)
                reward_log.append(rew)
                Image.fromarray(obs).save(os.path.join(tmp_dir, '%d.png' % i))
                i += 1
            viewer.imshow(big_obs(obs, info))
            pyglet.clock.tick()
            delta = time.time() - last_time
            time.sleep(max(0, 1 / 10 - delta))
            last_time = time.time()
        if not floors:
            print('did not solve any floors, aborting.')
            return

        for i in range(timestep + 1, len(rec.actions) + 1):
            os.remove(os.path.join(rec.path, '%d.png' % i))
        for i in range(timestep + 1, len(action_log) + 1):
            os.rename(os.path.join(tmp_dir, '%d.png' % i), os.path.join(rec.path, '%d.png' % i))

        with open(os.path.join(rec.path, 'actions.json'), 'w+') as out:
            json.dump(action_log, out)
        with open(os.path.join(rec.path, 'rewards.json'), 'w+') as out:
            json.dump(reward_log, out)


if __name__ == '__main__':
    main()

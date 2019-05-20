"""
Fix seed numbers due to the seed() function not working as
intended.
"""

import os
import random
import sys

from obstacle_tower_env import ObstacleTowerEnv

from obs_tower2.recording import load_all_data


def main():
    recordings = {}
    for recording in load_all_data():
        if not os.path.exists(cookie_path(recording)):
            img_hash = str(recording.load_frame(0).flatten().tolist())
            if img_hash not in recordings:
                recordings[img_hash] = []
            recordings[img_hash].append(recording)
    for img_hash, seed in seed_hashes():
        if not len(recordings):
            return
        if img_hash in recordings:
            print('got %d results for seed %d' % (len(recordings[img_hash]), seed))
            for rec in recordings[img_hash]:
                rename_recording(rec, seed)
            del recordings[img_hash]
        else:
            print('NO RECORDINGS')


def cookie_path(rec):
    return os.path.join(rec.path, 'fixed_seed.txt')


def rename_recording(rec, seed):
    with open(cookie_path(rec), 'w+') as f:
        f.write('fixed from %d\n' % rec.seed)
    base = os.path.basename(rec.path)
    new_base = str(seed) + '_' + base.split('_', 1)[1]
    new_path = os.path.join(os.path.dirname(rec.path), new_base)
    print('%s -> %s' % (rec.path, new_path))
    os.rename(rec.path, new_path)


def seed_hashes():
    mapping = {}
    while len(mapping) < 100:
        if os.path.exists('UnitySDK.log'):
            os.remove('UnitySDK.log')
        while True:
            try:
                env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'],
                                       worker_id=random.randrange(1000))
                break
            except KeyboardInterrupt:
                sys.exit(1)
            except:
                pass
        env.seed(25) # random argument
        obs = env.reset()
        env.close()
        with open('UnitySDK.log') as f:
            contents = next(l for l in f.readlines() if 'seed:' in l)
        seed = int(contents.split(': ')[-1])
        yield str(obs.flatten().tolist()), seed
    return mapping


if __name__ == '__main__':
    main()

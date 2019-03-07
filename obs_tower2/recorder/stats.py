"""
Print statistics about the recordings dataset.
"""

from collections import Counter
import math

import numpy as np

from obs_tower2.recording import load_data


def main():
    train, _ = load_data()
    action_counter = Counter()
    total_actions = 0
    floors = []
    rews = []
    for rec in train:
        for action in rec.actions:
            action_counter.update({action: 1})
            total_actions += 1
        floors.append(sum([x for x in rec.rewards if x == 1.0]))
        rews.append(sum(rec.rewards))
    entropy = 0.0
    for count in action_counter.values():
        prob = count / total_actions
        entropy -= math.log(prob) * prob
    print('action entropy: %f' % entropy)
    print('mean floor: %f' % np.mean(floors))
    print('max floor: %d' % max(floors))
    print('mean reward: %f' % np.mean(rews))
    print('total timesteps: %d' % total_actions)
    print('total episodes: %d' % len(train))


if __name__ == '__main__':
    main()

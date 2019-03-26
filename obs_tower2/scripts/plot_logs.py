import os
import sys

import matplotlib.pyplot as plt

KEYS = ['entropy', 'floor']


def main():
    paths = sys.argv[1:]
    logs = [read_log(path) for path in paths]
    names = [os.path.basename(path) for path in paths]
    for key in KEYS:
        plt.figure()
        for log in logs:
            values = log[key]
            plt.plot([i * 4096 / 1e6 for i, _ in enumerate(values)], smooth(values))
        plt.legend(names)
        plt.xlabel('timesteps (1e6)')
        plt.ylabel(key)
        plt.savefig(key + '.png')


def read_log(log_path):
    with open(log_path, 'r') as in_file:
        lines = [l for l in in_file.readlines() if 'step' in l or 'floor=' in l]
    mean_floor = None
    entropies = []
    floors = []
    for l in lines:
        if 'floor=' in l:
            floor_num = float(l.split('floor=')[-1])
            if mean_floor is None:
                mean_floor = floor_num
            else:
                mean_floor += 0.1 * (floor_num - mean_floor)
        elif 'step' in l:
            entropies.append(float(l.split('entropy=')[1].split(' ')[0]))
            floors.append(mean_floor if mean_floor is not None else 0.0)
    return {'entropy': entropies, 'floor': floors}


def smooth(values):
    value = values[0]
    for i, x in enumerate(values.copy()):
        value += 0.1 * (x - value)
        values[i] = value
    return values


if __name__ == '__main__':
    main()

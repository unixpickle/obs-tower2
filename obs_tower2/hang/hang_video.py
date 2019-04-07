import json
import os

from anyrl.utils.ffmpeg import export_video
import numpy as np
from obstacle_tower_env import ObstacleTowerEnv

from obs_tower2.util import big_obs

with open('hang.json', 'r') as in_file:
    data = json.load(in_file)

env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'], worker_id=4)
env.reset()


def f():
    env.seed(data['seed'])
    env.floor(data['floor'])
    env.reset()
    for act in data['actions'][:-1]:
        obs, _, _, info = env.step(act)
        yield big_obs(obs, info)


export_video('hang.mp4', 168, 168, 10, f())

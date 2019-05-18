import json
import os

from anyrl.utils.ffmpeg import export_video
import numpy as np
from obstacle_tower_env import ObstacleTowerEnv

from obs_tower2.util import big_obs

with open('stuck_box.json', 'r') as in_file:
    data = json.load(in_file)

env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'], worker_id=1)
env.seed(56)
env.reset()


def f():
    for i, act in enumerate(data):
        obs, _, _, info = env.step(act)
        if i > 5275:
            yield big_obs(obs, info)


export_video('stuck_box.mp4', 168, 168, 10, f())

import os
import random

from obstacle_tower_env import ObstacleTowerEnv

counter = {}
env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'], worker_id=2)
while True:
    env.seed(random.randrange(100))
    env.reset()
    for _ in range(50):
        obs, _, _, _ = env.step(0)
    key = str(obs.flatten().tolist())
    counter[key] = True
    print('got %d start states' % len(counter))

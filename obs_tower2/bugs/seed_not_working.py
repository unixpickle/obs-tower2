import os

from obstacle_tower_env import ObstacleTowerEnv

counter = {}
for i in range(0, 25):
    env = ObstacleTowerEnv(os.environ['OBS_TOWER_PATH'], worker_id=i)
    env.seed(25)
    env.reset()
    for _ in range(50):
        obs, _, _, _ = env.step(0)
    key = str(obs.flatten().tolist())
    counter[key] = True
    print('got %d start states' % len(counter))
    env.close()

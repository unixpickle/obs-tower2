import random

from anyrl.utils.ffmpeg import export_video
import numpy as np
import torch

from obs_tower2.model import ACModel
from obs_tower2.states import StateEnv
from obs_tower2.util import big_obs, create_single_env


def main():
    def image_fn():
        env = StateEnv(create_single_env(random.randrange(15, 20), clear=False))
        model = ACModel()
        state = torch.load('save.pkl', map_location='cpu')
        model.load_state_dict(state)
        model.to(torch.device('cuda'))
        state, obs = env.reset()
        while True:
            output = model.step(state, np.array([obs]))
            (state, obs), rew, done, info = env.step(output['actions'][0])
            yield big_obs(obs[..., -3:], info)
            if done:
                break
        env.close()
    export_video('export.mp4', 168, 168, 10, image_fn())


if __name__ == '__main__':
    main()

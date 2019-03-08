import random

from anyrl.utils.ffmpeg import export_video
import numpy as np
import torch

from obs_tower2.constants import IMAGE_SIZE, IMAGE_DEPTH
from obs_tower2.util import create_single_env
from obs_tower2.model import ACModel
from obs_tower2.util import big_obs


def main():
    def image_fn():
        env = create_single_env(random.randrange(15, 20), clear=False)
        model = ACModel(54, IMAGE_SIZE, IMAGE_DEPTH)
        state = torch.load('save.pkl', map_location='cuda')
        model.load_state_dict(state)
        obs = env.reset()
        state = np.zeros([1, model.state_size], dtype=np.float32)
        while True:
            output = model.step(state, np.array([obs]))
            obs, rew, done, info = env.step(output['actions'][0])
            yield big_obs(obs[..., -3:], info)
            state = output['states']
            if done:
                break
        env.close()
    export_video('export.mp4', 168, 168, 10, image_fn())


if __name__ == '__main__':
    main()

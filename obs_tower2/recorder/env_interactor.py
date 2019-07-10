import time

from PIL import Image
from gym.envs.classic_control.rendering import SimpleImageViewer
import numpy as np
import pyglet.window


class EnvInteractor(SimpleImageViewer):
    """
    This class manages the user interface for playing
    obstacle tower, pausing the game, etc.
    """

    def __init__(self):
        super().__init__(maxwidth=800)
        self.keys = pyglet.window.key.KeyStateHandler()
        self._paused = False
        self._jump = False
        self._finish_early = False
        self._last_image = None
        self.imshow(np.zeros([168, 168, 3], dtype=np.uint8))

    def imshow(self, image):
        self._last_image = image
        was_none = self.window is None
        image = Image.fromarray(image)
        image = image.resize((800, 800))
        image = np.array(image)
        super().imshow(image)
        if was_none:
            self.window.event(self.on_key_press)
            self.window.push_handlers(self.keys)

    def get_action(self):
        event = 0
        if self.keys[pyglet.window.key.UP]:
            event += 18
        if self.keys[pyglet.window.key.LEFT]:
            event += 6
        elif self.keys[pyglet.window.key.RIGHT]:
            event += 12
        if self.keys[pyglet.window.key.SPACE] or self._jump:
            event += 3
            self._jump = False
        if self.keys[pyglet.window.key.ESCAPE]:
            self._finish_early = True
        return event

    def pause(self):
        self._paused = True

    def paused(self):
        if self.keys[pyglet.window.key.P]:
            self._paused = True
        elif self.keys[pyglet.window.key.R]:
            self._paused = False
        return self._paused

    def finish_early(self):
        return self._finish_early

    def on_key_press(self, x, y):
        if x == pyglet.window.key.SPACE:
            self._jump = True
        return True

    def reset(self):
        self._jump = False
        self._paused = False
        self._finish_early = False

    def run_loop(self, step_fn):
        """
        Run an environment interaction loop.

        The step_fn will be continually called with
        actions, and it should return observations.

        When step_fn returns None, the loop is done.
        """
        last_time = time.time()
        while not self.finish_early():
            if not self.paused():
                obs = step_fn(self.get_action())
                if obs is None:
                    return
                self.imshow(obs)
            else:
                # Needed to run the event loop
                self.imshow(self._last_image)
            pyglet.clock.tick()
            delta = time.time() - last_time
            time.sleep(max(0, 1 / 10 - delta))
            last_time = time.time()

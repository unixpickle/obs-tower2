"""
Labels for supervised classification.
"""

import json
import os

from PIL import Image


def load_all_labeled_images(dir_path=os.environ['OBS_TOWER_IMAGE_LABELS']):
    pngs = [x for x in os.listdir(dir_path) if x.endswith('.png')]
    names = [png[:-len('.png')] for png in pngs]
    return [LabeledImage.load(dir_path, name) for name in names]


def load_labeled_images(**kwargs):
    images = load_all_labeled_images(**kwargs)
    return ([x for x in images if x.uid >= 1e8],
            [x for x in images if x.uid < 1e8])


class LabeledImage:
    def __init__(self,
                 dir_path,
                 name,
                 closed_door,
                 locked_door,
                 boxed_door,
                 open_door,
                 key,
                 box,
                 hurtle,
                 orb,
                 goal):
        self.dir_path = dir_path
        self.name = name
        self.uid = int(name.split('_')[1])
        self.closed_door = closed_door
        self.locked_door = locked_door
        self.boxed_door = boxed_door
        self.open_door = open_door
        self.key = key
        self.box = box
        self.hurtle = hurtle
        self.orb = orb
        self.goal = goal
        self.dir_path = dir_path or os.environ['OBS_TOWER_IMAGE_LABELS']

    @classmethod
    def load(cls, dir_path, name):
        with open(os.path.join(dir_path, name + '.json'), 'r') as in_file:
            labels = json.load(in_file)
            return cls(dir_path, name, *labels)

    def image(self):
        return Image.open(self._image_path())

    def save(self, image):
        image.save(self._image_path())
        with open(os.path.join(self.dir_path, self.name + '.json'), 'w+') as out_file:
            json.dump(self.pack_labels(), out_file)

    def pack_labels(self):
        return [self.closed_door, self.locked_door, self.boxed_door, self.open_door, self.key,
                self.box, self.hurtle, self.orb, self.goal]

    def _image_path(self):
        return os.path.join(self.dir_path, self.name + '.png')

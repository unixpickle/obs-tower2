"""
Calculate the test performance of a classifier for every
label separately.
"""

import numpy as np
import torch

from obs_tower2.constants import LABELS, NUM_LABELS
from obs_tower2.labels import load_labeled_images
from obs_tower2.model import StateClassifier


def main():
    device = torch.device('cuda')
    model = StateClassifier()
    model.load_state_dict(torch.load('save_classifier.pkl', map_location='cpu'))
    model.to(device)
    _, dataset = load_labeled_images()
    counts = [0] * NUM_LABELS
    correct = [0] * NUM_LABELS
    for datum in dataset:
        inputs = torch.from_numpy(np.array(datum.image())[None]).to(device)
        outputs = model(inputs).detach().cpu().numpy()[0]
        for i, value in enumerate(datum.pack_labels()):
            if value:
                counts[i] += 1
            if value == (outputs[i] > 0):
                correct[i] += 1
    for i, label in enumerate(LABELS):
        correct_frac = correct[i] / len(dataset)
        baseline = 1 - counts[i] / len(dataset)
        normalized = (correct_frac - baseline) / (1 - baseline)
        print('%s: %f%% (%f%% from baseline %f%%)' %
              (label, 100 * normalized, 100 * correct_frac, 100 * baseline))


if __name__ == '__main__':
    main()

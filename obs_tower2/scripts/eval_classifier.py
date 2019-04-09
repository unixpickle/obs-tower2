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
        outputs = model(inputs).detach().cpu().numpy()
        for i, value in enumerate(datum.pack_labels()):
            if value:
                counts[i] += 1
            if value == (outputs[i] > 0):
                correct[i] += 1
    for i, label in enumerate(LABELS):
        print('%s: %f%% (baseline %f%%)' % (label,
                                            100 * (correct[i] / len(dataset)),
                                            100 * (1 - counts[i] / len(dataset))))


if __name__ == '__main__':
    main()

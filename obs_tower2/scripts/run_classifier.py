import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from obs_tower2.labels import load_labeled_images
from obs_tower2.model import ACModel, StateClassifier
from obs_tower2.util import Augmentation, mirror_obs

LR = 1e-4
BATCH = 128


def main():
    model = StateClassifier()
    if os.path.exists('save_classifier.pkl'):
        model.load_state_dict(torch.load('save_classifier.pkl'))
    model.to(torch.device('cuda'))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train, test = load_labeled_images()
    for i in itertools.count():
        test_loss = classification_loss(model, test).item()
        train_loss = classification_loss(model, train)
        print('step %d: test=%f train=%f' % (i, test_loss, train_loss.item()))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if not i % 100:
            torch.save(model.state_dict(), 'save_classifier.pkl')


def classification_loss(model, dataset):
    images = []
    labels = []
    for _ in range(BATCH):
        aug = Augmentation()
        sample = random.choice(dataset)
        img = np.array(aug.apply(sample.image()))
        if random.random() < 0.5:
            img = mirror_obs(img)
        images.append(img)
        labels.append(sample.pack_labels())
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.float32)
    device = next(model.parameters()).device
    image_tensor = torch.from_numpy(images).to(device)
    label_tensor = torch.from_numpy(labels).to(device)
    logits = model(image_tensor)
    loss = nn.BCEWithLogitsLoss()
    return loss(logits, label_tensor)


if __name__ == '__main__':
    main()

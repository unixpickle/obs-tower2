"""
Train a classifier to feed into the agent's memory.

By default, this uses the MixMatch semi-supervised
learning algorithm to improve data efficiency, since there
are many more frames in the demonstrations than there are
in the labeled dataset.
"""

import itertools
from multiprocessing.dummy import Pool
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from obs_tower2.constants import NUM_LABELS
from obs_tower2.labels import load_labeled_images
from obs_tower2.recording import load_data
from obs_tower2.model import StateClassifier
from obs_tower2.util import Augmentation, atomic_save, mirror_obs

LR = 1e-4
BATCH = 128
USE_MIXMATCH = True
NUM_AUGMENTATIONS = 2
UNLABELED_WEIGHT = 1
MIXUP_ALPHA = 0.75
TEMPERATURE = 0.5


def main():
    model = StateClassifier()
    if os.path.exists('save_classifier.pkl'):
        model.load_state_dict(torch.load('save_classifier.pkl'))
    model.to(torch.device('cuda'))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train, test = load_labeled_images()
    recordings, _ = load_data()
    thread_pool = Pool(8)
    for i in itertools.count():
        test_loss = classification_loss(thread_pool, model, test).item()
        if USE_MIXMATCH:
            loss = mixmatch_loss(model,
                                 *labeled_data(thread_pool, model, train),
                                 *unlabeled_data(thread_pool, model, recordings))
            print('step %d: test=%f mixmatch=%f' % (i, test_loss, loss.item()))
        else:
            loss = classification_loss(thread_pool, model, train)
            print('step %d: test=%f train=%f' % (i, test_loss, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not i % 100:
            atomic_save(model.state_dict(), 'save_classifier.pkl')


def classification_loss(pool, model, dataset):
    image_tensor, label_tensor = labeled_data(pool, model, dataset)
    logits = model(image_tensor)
    loss = nn.BCEWithLogitsLoss()
    return loss(logits, label_tensor)


def mixmatch_loss(model, real_images, real_labels, other_images, other_labels):
    real_images, real_labels, other_images, other_labels = mixmatch(real_images, real_labels,
                                                                    other_images, other_labels)
    model_out = model(torch.cat([real_images, other_images]))
    real_out = model_out[:real_images.shape[0]]
    other_out = model_out[real_images.shape[0]:]

    bce = nn.BCEWithLogitsLoss()
    real_loss = bce(real_out, real_labels)
    other_loss = torch.mean(torch.pow(torch.sigmoid(other_out) - other_labels, 2))
    return real_loss + UNLABELED_WEIGHT * other_loss


def mixmatch(real_images, real_labels, other_images, other_labels):
    all_images = torch.cat([real_images, other_images])
    all_labels = torch.cat([real_labels, other_labels])
    indices = list(range(all_images.shape[0]))
    random.shuffle(indices)
    all_images, all_labels = mixup(all_images, all_labels,
                                   all_images[indices], all_labels[indices])
    return (all_images[:real_images.shape[0]], all_labels[:real_labels.shape[0]],
            all_images[real_images.shape[0]:], all_labels[real_labels.shape[0]:])


def mixup(real_images, real_labels, other_images, other_labels):
    probs = []
    for _ in range(real_images.shape[0]):
        p = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
        probs.append(min(p, 1 - p))
    prob_tensor = torch.from_numpy(np.array(probs, dtype=np.float32)).to(real_images.device)
    interp_images = (real_images.float() + prob_tensor.view(-1, 1, 1, 1)
                     * (other_images - real_images).float()).byte()
    interp_labels = real_labels + prob_tensor.view(-1, 1) * (other_labels - real_labels)
    return interp_images, interp_labels


def labeled_data(pool, model, dataset):
    def load_image(sample):
        aug = Augmentation()
        img = np.array(aug.apply(sample.image()))
        if random.random() < 0.5:
            img = mirror_obs(img)
        return img
    samples = [random.choice(dataset) for _ in range(BATCH)]
    images = pool.map(load_image, samples)
    labels = [sample.pack_labels() for sample in samples]
    image_tensor = model_tensor(model, np.array(images, dtype=np.uint8))
    label_tensor = model_tensor(model, np.array(labels, dtype=np.float32))
    return image_tensor, label_tensor


def unlabeled_data(pool, model, recordings):
    def load_image_set(_):
        img_set = []
        rec = random.choice(recordings)
        img = rec.load_frame(random.randrange(rec.num_steps))
        for _ in range(NUM_AUGMENTATIONS):
            aug = Augmentation()
            img1 = np.array(aug.apply(img))
            if random.random() < 0.5:
                img1 = mirror_obs(img1)
            img_set.append(img1)
        return img_set
    images = [x for y in pool.map(load_image_set, range(BATCH)) for x in y]
    image_tensor = model_tensor(model, np.array(images, dtype=np.uint8))
    preds = torch.sigmoid(model(image_tensor)).detach()
    preds = preds.view(BATCH, NUM_AUGMENTATIONS, NUM_LABELS)
    mixed = torch.mean(preds, dim=1, keepdim=True)
    sharpened = sharpen_predictions(mixed)
    broadcasted = (sharpened + torch.zeros_like(preds)).view(-1, NUM_LABELS)
    return image_tensor, broadcasted


def sharpen_predictions(preds):
    pow1 = torch.pow(preds, 1 / TEMPERATURE)
    pow2 = torch.pow(1 - preds, 1 / TEMPERATURE)
    return pow1 / (pow1 + pow2)


def model_tensor(model, nparray):
    device = next(model.parameters()).device
    return torch.from_numpy(nparray).to(device)


if __name__ == '__main__':
    main()

"""
Get statistics about the classifier dataset.

Counts the instances for each label, etc.
"""

from obs_tower2.constants import LABELS, NUM_LABELS
from obs_tower2.labels import load_all_labeled_images


def main():
    images = load_all_labeled_images()
    counts = [0] * NUM_LABELS
    for img in images:
        for i, label in enumerate(img.pack_labels()):
            if label:
                counts[i] += 1
    print('total of %d images' % len(images))
    for name, count in zip(LABELS, counts):
        print('%s: %d (%d%%)' % (name, count, round(count / len(images) * 100)))


if __name__ == '__main__':
    main()

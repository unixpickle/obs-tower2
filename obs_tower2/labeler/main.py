import io
import json
import os
import random

from PIL import Image
from flask import Flask, send_file, send_from_directory
import numpy as np
import torch

from obs_tower2.labels import LabeledImage, load_all_labeled_images
from obs_tower2.model import StateClassifier
from obs_tower2.recording import load_all_data, sample_recordings


app = Flask(__name__, static_url_path='')
labelled = load_all_labeled_images()
recordings = load_all_data()

CLASSIFIER_PATH = '../scripts/save_classifier.pkl'
if os.path.exists(CLASSIFIER_PATH):
    classifier = StateClassifier()
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location='cpu'))
else:
    classifier = None


@app.route('/assets/<path:path>')
def handle_asset(path):
    return send_from_directory('assets', path)


@app.route('/')
def handle_root():
    return send_from_directory('.', 'index.html')


@app.route('/sample')
def handle_sample():
    return sample_new_name()


@app.route('/frame/<name>')
def handle_frame(name):
    buffer = io.BytesIO()
    load_frame(name).save(buffer, 'PNG')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')


@app.route('/key/<name>')
def handle_key(name):
    return json.dumps(check_key(name))


@app.route('/classify/<name>')
def handle_classify(name):
    if classifier is None:
        return 'null'
    img = np.array(load_frame(name))
    inputs = torch.from_numpy(img[None])
    outputs = torch.sigmoid(classifier(inputs)).detach().numpy()[0]
    return json.dumps([float(x) for x in outputs])


@app.route('/save/<name>/<labels>')
def handle_save(name, labels):
    frame = load_frame(name)
    labels = [x == '1' for x in labels.split(',')]
    img = LabeledImage(os.environ['OBS_TOWER_IMAGE_LABELS'], name, *labels)
    img.save(frame)
    labelled.append(img)
    return 'success'


def sample_new_name():
    while True:
        rec = sample_recordings(recordings, 1)[0]
        frame = random.randrange(rec.num_steps)
        name = '%d_%d_%d' % (rec.seed, rec.uid, frame)
        if any([x for x in labelled if x.name == name]):
            continue
        return name


def load_frame(name):
    rec, frame = find_rec_frame(name)
    return Image.fromarray(rec.load_frame(frame))


def check_key(name):
    rec, frame = find_rec_frame(name)
    for i in range(frame + 10, min(frame + 50, rec.num_steps), 5):
        img = rec.load_frame(i)
        if not (img[2] == 0).all():
            return True
    return False


def find_rec_frame(name):
    parts = name.split('_')
    seed = int(parts[0])
    uid = int(parts[1])
    frame = int(parts[2])
    rec = next(x for x in recordings if x.seed == seed and x.uid == uid)
    return rec, frame


if __name__ == '__main__':
    app.run()

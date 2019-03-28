import io
import json
import os
import random

from flask import Flask, send_file, send_from_directory

from obs_tower2.labels import LabeledImageV2, load_all_labeled_images


app = Flask(__name__, static_url_path='')
labelled = load_all_labeled_images()


@app.route('/assets/<path:path>')
def handle_asset(path):
    return send_from_directory('assets', path)


@app.route('/')
def handle_root():
    return send_from_directory('.', 'index.html')


@app.route('/sample')
def handle_sample():
    return sample_new_name()


@app.route('/classes/<name>')
def handle_classes(name):
    return json.dumps(next(x for x in labelled if x.name == name).pack_labels())


@app.route('/frame/<name>')
def handle_frame(name):
    buffer = io.BytesIO()
    load_frame(name).save(buffer, 'PNG')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')


@app.route('/save/<name>/<labels>')
def handle_save(name, labels):
    global labelled
    frame = load_frame(name)
    labels = [x == '1' for x in labels.split(',')]
    img = LabeledImageV2(os.environ['OBS_TOWER_IMAGE_LABELS'], name, *labels)
    img.save(frame)
    labelled = [x for x in labelled if x.name != name]
    print('%d samples remaining' % len(labelled))
    return 'success'


def sample_new_name():
    return random.choice(labelled).name


def load_frame(name):
    return next(l for l in labelled if l.name == name).image()


if __name__ == '__main__':
    app.run()

# obs-tower2

This is my solution to the [Unity Obstacle Tower Challenge](https://www.aicrowd.com/challenges/unity-obstacle-tower-challenge). Almost all of the code was freshly written for this contest, including a simple implementation of [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347).

# Overview

The final agent has the following components. This is what is included in my contest submissions:

 * A classifier to tell what objects (e.g. box, door, key) are in an image
 * A state-augmented environment, providing a history of previous actions, rewards, and classifier outputs to the agent
 * Two feedforward policies: one for floors 0 through 9, and one for floors 10 and upwards

The agents are pre-trained with behavior cloning on roughly 2 million frames (~2.3 days) of human demonstration data. These pre-trained agents themselves do not perform well (they achieve an average floor of ~6, and solve floors >9 with a fairly low probability). The pre-trained agents are then fine-tuned using [prierarchy](https://blog.aqnichol.com/2019/04/03/prierarchy-implicit-hierarchies/), where the prior is the original pre-trained agent. This can be seen as fine-tuning the agent while keeping its behavior close to "human" behavior.

Two different agents are fine-tuned: one for floors 0-9, and one for 10 and onwards. This is because, while the 0-9 agent makes lots of progress on the lower floors very quickly, it seems to ignore the higher floors. Only after the 0-9 agent reaches an average of ~9.8 floors does it slowly start to conquer higher floors. My hypothesis is that the 0-9 agent has much more signal coming from the lower floors than from the higher floors, since lower floors are a much larger source of rewards. Thus, the lower floors drown out any learning that might take place on the higher floors.

The "10 and onwards" agent starts out solving floors with a fairly low probability (between 1% and 5%). Since this agent never sees easier (i.e. lower) floors, it has no choice but to focus on the difficult Sokoban puzzle and the other difficulties of the higher floors. Because of the human-based prior, the agent succeeds at these challenges with a non-negligible probability, giving it enough signal to learn from.

The agent itself is a feedforward model; it contains no recurrent connections. To help the agent remember the past, I feed it a stack of *state vectors* for the past 50 frames. Each state vector contains:

 * The action taken at that timestep
 * The reward received
 * Whether or not the agent has a key
 * The probability outputs from a hand-crafted classifier

During behavior cloning and fine-tuning, the agent has little control over what features it can remember from the past. All it has access to is what I thought would be important (e.g. whether or not a box was on the screen). This has obvious drawbacks, but it also has the advantage that the agent will definitely have access to important information. In practice, I found that using an RNN model was not nearly as effective as hand-crafting the agent's memory.

## Codebase overview

This codebase has several components:

 * [obs_tower2](obs_tower2) - a library of learning algorithms and ML models
 * [scripts](obs_tower2/scripts) - a set of scripts for training classifiers and agents
 * [recorder](obs_tower2/recorder) - an application for recording human demonstrations
 * [labeler](obs_tower2/labeler) - a web application for labeling images with various classes

# Running the code

First, install the `obs_tower2` package using `pip`:

```
pip install -e .
```

Next, configure your environment variables. The scripts depend on a few environment variables to locate training data and the obstacle tower binary. Here are the environment variables you should set:

 * `OBS_TOWER_PATH` - the path to the obstacle tower binary.
 * `OBS_TOWER_RECORDINGS` - the path to a directory where demonstrations are stored.
 * `OBS_TOWER_IMAGE_LABELS` - the path to the directory of labeled images.

## Getting data

If you don't have a directory of labeled images or recordings, you can create an empty directory. However, the training scripts require that you have some data, and the agent will not learn well unless you give it a lot of hand-labeled and human-recorded data. You can either hand-generate the data yourself, or [download all of the data I created myself](http://obstower.aqnichol.com/).

To record data yourself, see the scripts [recorder/record.py](obs_tower2/recorder/record.py) and [labeler/main.py](obs_tower2/labeler/main.py), which help you record demonstrations and label images, respectively. The recorder uses a pyglet UI to record demonstrations. The labeler is a web application that loads images from the recordings and lets you check off which classes they contain. The labeler also supports keyboard inputs, making it possible for an expert labeler to hit rates of anywhere from 20 to 40 labels per minute.

## Training the models

Once you have labeled data and recordings, you are ready to train the classifier used for the agent's memory:

```
cd obs_tower2/scripts
python run_classifier.py
```

This script saves its result to `save_classifier.pkl` periodically. You will want to run the script until the model starts to overfit. With my dataset, this takes a couple of hours on a single GPU.

Next, you can use behavior cloning to train a prior agent:

```
python run_clone.py
```

This script saves its result to `save_clone.pkl` periodically. This may take up to a day to run, and with my dataset it will not overfit very much no matter how long you run it for. Once this is done, you can copy the classifier to be used as a prior:

```
cp save_clone.pkl save_prior.pkl
```

Next, you can train an agent that solves the first 10 floors:

```
cp save_prior.pkl save.pkl
python run_tail.py --min 0 --max 1 --path save.pkl
```

This script saves its result to whatever is passed as the `--path`, in this case `save.pkl`. Notice how we start out by copying `save_prior.pkl` as `save.pkl`. This means that the agent is initialized out as the human-based prior. You will likely want to run this script for a couple of weeks. You can run this at the same time as training an agent that solves the 10th floor and greater.

To train an agent that solves floors above the 10th floor, you can use the same `run_tail.py` script with different arguments:

```
cp save_prior.pkl save_tail.pkl
python run_tail.py --min 10 --max 15 --path save_tail.pkl
```

If you want to run two `run_tail.py` instances simultaneously, you should pass `--worker-idx 0` to one of them. This ensures that one script uses worker IDs 0-7, while the other uses IDs 8-16.

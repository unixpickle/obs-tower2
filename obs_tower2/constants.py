# The observation image size is fixed throughout the
# entire project.
IMAGE_SIZE = 84

# Feed the agent the current and the previous frame.
FRAME_STACK = 2

# There are FRAME_STACK number of RGB images.
IMAGE_DEPTH = FRAME_STACK * 3

HUMAN_ACTIONS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33)
NUM_ACTIONS = len(HUMAN_ACTIONS)

# Human-readable names of all the classifier labels.
LABELS = ('closed door', 'locked door', 'boxed door', 'open door', 'key', 'box', 'hurtle', 'orb',
          'goal', 'box target', 'box undo')

NUM_LABELS = len(LABELS)

# Size of the state vector that is fed for a single frame.
# Includes: one-hot action, label probabilities, reward,
# a boolean indicating if the agent has a key.
STATE_SIZE = NUM_ACTIONS + 2 + NUM_LABELS

# The number of previous states to feed the agent.
STATE_STACK = 50

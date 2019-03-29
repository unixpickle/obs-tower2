IMAGE_SIZE = 84
FRAME_STACK = 2
IMAGE_DEPTH = FRAME_STACK * 3
HUMAN_ACTIONS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33)
NUM_ACTIONS = len(HUMAN_ACTIONS)
LABELS = ('closed door', 'locked door', 'boxed door', 'open door', 'key', 'box', 'hurtle', 'orb',
          'goal', 'box target', 'box undo')
NUM_LABELS = len(LABELS)
STATE_SIZE = NUM_ACTIONS + 2 + NUM_LABELS
STATE_STACK = 50

"""
Neural network models for classification, RL, GAIL
discrimination, etc.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import IMAGE_DEPTH, IMAGE_SIZE, NUM_ACTIONS, NUM_LABELS, STATE_SIZE, STATE_STACK


class Model(nn.Module):
    """
    An abstract RL model or a GAIL discriminator.
    """
    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def state_size(self):
        raise NotImplementedError

    def forward(self, states, observations):
        """
        Run the model for one timestep and return a dict
        of outputs.

        Args:
            states: a Tensor of previous states.
            observations: a Tensor of observations.

        Returns:
            A dict of Tensors.
        """
        raise NotImplementedError

    def step(self, states, observations):
        """
        Like forward() but with numpy arrays.
        """
        res = self.forward(self.tensor(states), self.tensor(observations))
        return model_outs_to_cpu(res)

    def run_for_rollout(self, rollout):
        """
        Run the model on the rollout and create a new
        rollout with the filled-in model_outs.

        This may be more efficient than using step
        manually in a loop, since it may be able to batch
        certain parts of the model across all timesteps.
        """
        raise NotImplementedError

    def tensor(self, x):
        return torch.from_numpy(x).to(self.device)


class BaseModel(Model):
    """
    A Model that computes the base outputs for a CNN that
    also takes state history into account.
    """

    def __init__(self, cnn_class=None):
        super().__init__()
        self.cnn = (cnn_class or FixupCNN)(IMAGE_SIZE, IMAGE_DEPTH)
        self.state_mlp = nn.Sequential(
            nn.Linear(STATE_STACK * STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.state_mixer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

    def forward(self, states, observations):
        float_obs = observations.float() / 255.0
        cnn_out = self.cnn(float_obs)
        flat_states = states.view(states.shape[0], -1)
        states_out = self.state_mlp(flat_states)
        concatenated = torch.cat([cnn_out, states_out], dim=-1)
        mixed = self.state_mixer(concatenated)
        output = {'base': mixed}
        self.add_fields(output)
        return output

    def add_fields(self, output):
        pass

    def run_for_rollout(self, rollout):
        mixed = self._base_outs(rollout)
        result = rollout.copy()
        result.model_outs = []
        for t in range(rollout.num_steps + 1):
            model_out = {'base': self.tensor(mixed[t])}
            self.add_fields(model_out)
            result.model_outs.append(model_outs_to_cpu(model_out))
        return result

    def _base_outs(self, rollout):
        # Even though we could run the entire batch of
        # rollouts in one forward pass, doing so may use
        # too much memory, so we split the rollout up into
        # mini-batches.
        batch_size = 128

        def index_samples():
            for t in range(rollout.num_steps + 1):
                for b in range(rollout.batch_size):
                    yield (t, b)

        def index_batches():
            batch = []
            for x in index_samples():
                batch.append(x)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if len(batch):
                yield batch

        result = np.zeros([rollout.num_steps + 1, rollout.batch_size, 256], dtype=np.float32)
        for batch in index_batches():
            images = np.array([rollout.obses[t, b] for t, b in batch])
            float_obs = self.tensor(images).float() / 255.0
            cnn_out = self.cnn(float_obs)
            states = self.tensor(np.array([rollout.states[t, b] for t, b in batch]))
            flat_states = states.view(states.shape[0], -1)
            states_out = self.state_mlp(flat_states)
            concatenated = torch.cat([cnn_out, states_out], dim=-1)
            mixed = self.state_mixer(concatenated).detach().cpu().numpy()
            for (t, b), base_out in zip(batch, mixed):
                result[t, b] = base_out

        return result


class ACModel(BaseModel):
    """
    An actor-critic model, which produces actions and
    value predictions on top of the features from a base
    model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_actions = NUM_ACTIONS
        self.actor = nn.Linear(256, self.num_actions)
        self.critic = nn.Linear(256, 1)
        for parameter in list(self.actor.parameters()) + list(self.critic.parameters()):
            parameter.data.zero_()

    def add_fields(self, output):
        output['actor'] = self.actor(output['base'])
        output['critic'] = self.critic(output['base']).view(-1)
        log_probs = F.log_softmax(output['actor'], dim=-1)
        probs = np.exp(log_probs.detach().cpu().numpy())
        actions = [np.random.choice(self.num_actions, p=p) for p in probs]
        output['actions'] = self.tensor(np.array(actions))
        output['log_probs'] = torch.stack([log_probs[i, a] for i, a in enumerate(actions)])


class DiscriminatorModel(BaseModel):
    """
    A discriminator for GAIL.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, cnn_class=MaskedCNN)
        self.discriminator = nn.Linear(256, 1)
        for parameter in self.discriminator.parameters():
            parameter.data.zero_()

    def add_fields(self, output):
        output['logits'] = self.discriminator(output['base']).view(-1)
        log_disc = F.logsigmoid(output['logits'])
        log_neg_disc = F.logsigmoid(-output['logits'])
        output['prob_pi'] = torch.mean(log_disc)
        output['prob_expert'] = torch.mean(log_neg_disc)


class StateClassifier(nn.Module):
    """
    A classifier that generates labels to put into the
    state history.

    The outputs of this classifier are fed as part of the
    input to the policy.
    """

    def __init__(self):
        super().__init__()
        self.cnn = FixupCNN(IMAGE_SIZE, 3)
        self.final_layer = nn.Linear(256, NUM_LABELS)

    def forward(self, x):
        mask = np.ones(x.shape[1:], dtype=np.uint8)
        mask[0:10] = 0.0
        float_obs = (torch.from_numpy(mask).to(x.device) * x).float() / 255.0
        features = self.cnn(float_obs)
        logits = self.final_layer(features)
        return logits


class ImpalaCNN(nn.Module):
    """
    The CNN architecture used in the IMPALA paper.

    See https://arxiv.org/abs/1802.01561.
    """

    def __init__(self, image_size, depth_in):
        super().__init__()
        layers = []
        for depth_out in [16, 32, 32]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                ImpalaResidual(depth_out),
                ImpalaResidual(depth_out),
            ])
            depth_in = depth_out
        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(math.ceil(image_size / 8) ** 2 * depth_in, 256)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv_layers(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)
        return x


class ImpalaResidual(nn.Module):
    """
    A residual block for an IMPALA CNN.
    """

    def __init__(self, depth):
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x


class FixupCNN(nn.Module):
    """
    A larger version of the IMPALA CNN with Fixup init.
    See Fixup: https://arxiv.org/abs/1901.09321.
    """

    def __init__(self, image_size, depth_in):
        super().__init__()
        layers = []
        for depth_out in [32, 64, 64]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                FixupResidual(depth_out, 8),
                FixupResidual(depth_out, 8),
            ])
            depth_in = depth_out
        layers.extend([
            FixupResidual(depth_in, 8),
            FixupResidual(depth_in, 8),
        ])
        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(math.ceil(image_size / 8) ** 2 * depth_in, 256)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv_layers(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)
        return x


class FixupResidual(nn.Module):
    def __init__(self, depth, num_residual):
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        for p in self.conv1.parameters():
            p.data.mul_(1 / math.sqrt(num_residual))
        for p in self.conv2.parameters():
            p.data.zero_()
        self.bias1 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias2 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias3 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias4 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.scale = nn.Parameter(torch.ones([depth, 1, 1]))

    def forward(self, x):
        x = F.relu(x)
        out = x + self.bias1
        out = self.conv1(out)
        out = out + self.bias2
        out = F.relu(out)
        out = out + self.bias3
        out = self.conv2(out)
        out = out * self.scale
        out = out + self.bias4
        return out + x


class MaskedCNN(FixupCNN):
    """
    A CNN that ignores the part of the screen that
    indicates how much time the agent has left.

    This is useful for behavior cloning, where the time
    remaining should have little influence on behavior.
    This is especially important because the agent tends
    to be slower than a real human player, so the agent
    might get confused when it sees that it's running out
    of time, since it never saw such a thing happen to a
    human.
    """

    def forward(self, x):
        mask = np.ones(x.shape[1:], dtype=np.float32)
        mask[6:10] = 0.0
        return super().forward(torch.from_numpy(mask).to(x.device) * x)


def model_outs_to_cpu(model_outs):
    return {k: v.detach().cpu().numpy() for k, v in model_outs.items()}

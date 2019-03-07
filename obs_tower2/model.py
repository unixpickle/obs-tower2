import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


class Model(nn.Module):
    """
    An RL model or a discriminator.
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

    def add_states(self, rollout):
        """
        Run the model on the rollout and create a new
        rollout with the filled-in states.

        This may be more efficient than using step
        manually in a loop, since it may be able to batch
        certain parts of the model across all timesteps.
        """
        raise NotImplementedError


class BaseModel(Model):
    """
    A Model that computes the base outputs for a recurrent,
    IMPALA-based echo-state network.
    """

    def __init__(self, image_size, depth_in, state_size=256):
        super().__init__()
        self._state_size = state_size
        self.impala_cnn = ImpalaCNN(image_size, depth_in)
        self.state_transition = nn.Linear(state_size + 256, state_size)
        self.state_norm = nn.LayerNorm((state_size,))
        self.state_mixer = nn.Linear(state_size + 256, 256)

    @property
    def state_size(self):
        return self._state_size

    def forward(self, states, observations):
        float_obs = observations.float() / 255.0
        impala_out = self.impala_cnn(float_obs)
        concatenated = torch.cat([impala_out, states], dim=-1)
        new_state = F.relu(self.state_norm(self.state_transition(concatenated)))
        mixed = F.relu(self.state_mixer(concatenated))
        return {
            'base': mixed,
            'states': new_state,
        }

    def add_states(self, rollout):
        impala_outs = self._impala_outs(rollout)
        states = torch.zeros(rollout.batch_size, self.state_size).to(self.device)
        result = rollout.copy()
        result.states = np.zeros([rollout.num_steps, rollout.batch_size, self.state_size],
                                 dtype=np.float32)
        for t in range(rollout.num_steps):
            impala_batch = torch.from_numpy(impala_outs[t]).to(self.device)
            concatenated = torch.cat([impala_batch, states], dim=-1)
            states = F.relu(self.state_norm(self.state_transition(concatenated)))
            result.states[t] = states.detach().cpu().numpy()
        return result

    def _impala_outs(self, rollout):
        batch_size = 128

        def image_samples():
            for t in range(rollout.num_steps):
                for b in range(rollout.batch_size):
                    yield (t, b)

        def image_batches():
            batch = []
            for x in image_samples():
                batch.append(x)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if len(batch):
                yield batch

        result = np.zeros([rollout.num_steps, rollout.batch_size, 256], dtype=np.float32)
        for batch in image_batches():
            images = np.array([rollout.obses[t, b] for t, b in batch])
            float_obs = torch.from_numpy(images).float() / 255.0
            outputs = self.impala_cnn(float_obs.to(self.device))
            for (t, b), output in zip(batch, outputs):
                result[t, b] = output.detach().cpu().numpy()

        return result


class ACModel(BaseModel):
    def __init__(self, num_actions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)
        for parameter in list(self.actor.parameters()) + list(self.critic.parameters()):
            parameter.zero_()

    def forward(self, states, observations):
        output = super().forward(states, observations)
        output['actor'] = self.actor(output['base'])
        output['critic'] = self.critic(output['base'])
        # TODO: sample action here.
        return output


class ImpalaCNN(nn.Module):
    def __init__(self, image_size, depth_in):
        layers = []
        for depth_out in [16, 32, 32]:
            layers.append(
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                ImpalaResidual(depth_out),
                ImpalaResidual(depth_out),
            )
            depth_in = depth_out
        layers.append(nn.Flatten)
        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear((image_size // 8) ** 2 * depth_in, 256)

    def forward(self, x):
        x = self.conv_layers(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)
        return x


class ImpalaResidual(nn.Module):
    def __init__(self, depth):
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x

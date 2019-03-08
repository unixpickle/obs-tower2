import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def step(self, states, observations):
        """
        Like forward() but with numpy arrays.
        """
        res = self.forward(self.tensor(states), self.tensor(observations))
        return {k: v.detach().cpu().numpy() for k, v in res.items()}

    def run_for_rollout(self, rollout):
        """
        Run the model on the rollout and create a new
        rollout with the filled-in states and model_outs.

        This may be more efficient than using step
        manually in a loop, since it may be able to batch
        certain parts of the model across all timesteps.
        """
        raise NotImplementedError

    def tensor(self, x):
        return torch.from_numpy(x).to(self.device)


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
        return self._forward_with_impala(states, impala_out)

    def _forward_with_impala(self, states, impala_out):
        concatenated = torch.cat([impala_out, states], dim=-1)
        new_state = F.relu(self.state_norm(self.state_transition(concatenated)))
        mixed = F.relu(self.state_mixer(concatenated))
        res = {
            'base': mixed,
            'states': new_state,
        }
        self.add_fields(res)
        return res

    def add_fields(self, output):
        pass

    def run_for_rollout(self, rollout):
        impala_outs = self._impala_outs(rollout)
        states = torch.zeros(rollout.batch_size, self.state_size).to(self.device)
        result = rollout.copy()
        result.states = np.zeros([rollout.num_steps + 1, rollout.batch_size, self.state_size],
                                 dtype=np.float32)
        result.model_outs = []
        for t in range(rollout.num_steps):
            impala_batch = self.tensor(impala_outs[t])
            model_outs = self._forward_with_impala(states, impala_batch)
            states = model_outs['states'] * (1 - result.dones[t + 1])
            result.states[t + 1] = states.detach().cpu().numpy()
        result.model_outs.append(self._forward_with_impala(states, self.tensor(impala_outs[-1])))
        return result

    def _impala_outs(self, rollout):
        batch_size = 128

        def image_samples():
            for t in range(rollout.num_steps + 1):
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
        self.num_actions = num_actions
        self.actor = nn.Linear(256, num_actions)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = nn.Linear(256, 1)
        for parameter in self.discriminator.parameters():
            parameter.data.zero_()

    def add_fields(self, output):
        output['logits'] = self.discriminator(output['base']).view(-1)
        log_disc = F.logsigmoid(output['logits'])
        log_neg_disc = F.logsigmoid(output['logits'])
        output['prob_pi'] = torch.mean(log_disc)
        output['prob_expert'] = torch.mean(log_neg_disc)


class ImpalaCNN(nn.Module):
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

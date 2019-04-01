from abc import ABC, abstractmethod, abstractproperty
from multiprocessing import Pipe, Process

import cloudpickle
import numpy as np


class BatchedEnv(ABC):
    def __init__(self, action_space, obs_space):
        self.action_space = action_space
        self.observation_space = obs_space

    @abstractproperty
    def num_envs(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    def close(self):
        pass


class BatchedGymEnv(BatchedEnv):
    def __init__(self, action_space, obs_space, env_fns):
        super().__init__(action_space, obs_space)
        self._procs = []
        self._pipes = []
        for env_fn in env_fns:
            pipe, child_pipe = Pipe()
            proc = Process(target=self._worker, args=(child_pipe, cloudpickle.dumps(env_fn),))
            proc.start()
            child_pipe.close()
            self._procs.append(proc)
            self._pipes.append(pipe)
        for pipe in self._pipes:
            self._recv_pipe(pipe)

    @property
    def num_envs(self):
        return len(self._procs)

    def reset(self):
        for pipe in self._pipes:
            pipe.send(('reset', None))
        return np.array([self._recv_pipe(pipe) for pipe in self._pipes])

    def step(self, actions):
        for pipe, action in zip(self._pipes, actions):
            pipe.send(('step', action))
        obses = []
        rews = []
        dones = []
        infos = []
        for pipe in self._pipes:
            obs, rew, done, info = self._recv_pipe(pipe)
            obses.append(obs)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
        return np.array(obses), np.array(rews), np.array(dones), infos

    def close(self):
        for pipe in self._pipes:
            pipe.send(('close', None))
        for proc in self._procs:
            proc.join()

    @staticmethod
    def _worker(pipe, env_str):
        try:
            env = cloudpickle.loads(env_str)()
            pipe.send((None, None))
            try:
                while True:
                    cmd, arg = pipe.recv()
                    if cmd == 'reset':
                        pipe.send((env.reset(), None))
                    elif cmd == 'step':
                        pipe.send((env.step(arg), None))
                    elif cmd == 'close':
                        return
            finally:
                env.close()
        except Exception as exc:
            pipe.send((None, exc))

    @staticmethod
    def _recv_pipe(pipe):
        try:
            value, exc = pipe.recv()
        except EOFError:
            raise RuntimeError('worker has died')
        if exc is not None:
            raise exc
        return value


class BatchedWrapper(BatchedEnv):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    @property
    def num_envs(self):
        return self.env.num_envs

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def close(self):
        self.env.close()

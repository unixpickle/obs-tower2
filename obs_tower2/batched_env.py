from abc import ABC, abstractmethod, abstractproperty
from multiprocessing import Process, Queue
import os
from queue import Empty
import sys

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
        self._queues = []
        self._env_fns = env_fns
        for env_fn in env_fns:
            q = Queue()
            proc = Process(target=self._worker, args=(q, cloudpickle.dumps(env_fn),))
            proc.start()
            self._procs.append(proc)
            self._queues.append(q)
        for queue in self._queues:
            self._queue_get(queue)

    @property
    def num_envs(self):
        return len(self._procs)

    def reset(self):
        for q in self._queues:
            q.put(('reset', None))
        return np.array([self._queue_get(q) for q in self._queues])

    def step(self, actions):
        for q, action in zip(self._queues, actions):
            q.put(('step', action))
        obses = []
        rews = []
        dones = []
        infos = []
        for i, q in enumerate(self._queues.copy()):
            try:
                obs, rew, done, info = self._queue_get(q, timeout=10)
            except Empty:
                sys.stderr.write('restarting worker %d due to hang.\n' % i)
                self._restart_worker(i)
                obs, rew, done, info = self._queue_get(self._queues[i])
                done = True
            obses.append(obs)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
        return np.array(obses), np.array(rews), np.array(dones), infos

    def close(self):
        for q in self._queues:
            q.put(('close', None))
        for proc in self._procs:
            proc.join()

    def _restart_worker(self, idx):
        os.system('kill -9 $(ps -o pid= --ppid %d)' % self._procs[idx].pid)
        self._procs[idx].kill()
        self._procs[idx].join()
        q = Queue()
        proc = Process(target=self._worker, args=(q, cloudpickle.dumps(self._env_fns[idx]),))
        proc.start()
        self._procs[idx] = proc
        self._queues[idx] = q
        self._queue_get(q)

    @staticmethod
    def _worker(pipe, env_str):
        try:
            env = cloudpickle.loads(env_str)()
            pipe.put((None, None))
            try:
                while True:
                    cmd, arg = pipe.get()
                    if cmd == 'reset':
                        pipe.put((env.reset(), None))
                    elif cmd == 'step':
                        obs, rew, done, info = env.step(arg)
                        if done:
                            obs = env.reset()
                        pipe.put(((obs, rew, done, info), None))
                    elif cmd == 'close':
                        return
            finally:
                env.close()
        except Exception as exc:
            pipe.put((None, exc))

    @staticmethod
    def _queue_get(queue, timeout=None):
        value, exc = queue.get(timeout=timeout)
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

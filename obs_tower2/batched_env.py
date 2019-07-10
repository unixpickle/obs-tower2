"""
Tools for running multiple environments at once.

This code is influenced by openai/baselines and
unixpickle/anyrl-py, but it was rewritten specifically for
this contest.

One feature of this code is that it automatically deals
with environments that hang due to bugs. When this occurs,
the environment is killed and restarted automatically.
"""

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
        self._command_queues = []
        self._result_queues = []
        self._env_fns = env_fns
        for env_fn in env_fns:
            cmd_queue = Queue()
            res_queue = Queue()
            proc = Process(target=self._worker,
                           args=(cmd_queue, res_queue, cloudpickle.dumps(env_fn)))
            proc.start()
            self._procs.append(proc)
            self._command_queues.append(cmd_queue)
            self._result_queues.append(res_queue)
        for q in self._result_queues:
            self._queue_get(q)

    @property
    def num_envs(self):
        return len(self._procs)

    def reset(self):
        for q in self._command_queues:
            q.put(('reset', None))
        return np.array([self._queue_get(q) for q in self._result_queues])

    def step(self, actions):
        for q, action in zip(self._command_queues, actions):
            q.put(('step', action))
        obses = []
        rews = []
        dones = []
        infos = []
        for i, q in enumerate(self._result_queues.copy()):
            try:
                obs, rew, done, info = self._queue_get(q)
            except Empty:
                sys.stderr.write('restarting worker %d due to hang.\n' % i)
                self._restart_worker(i)
                q = self._result_queues[i]
                self._command_queues[i].put(('reset', None))
                self._queue_get(q)
                self._command_queues[i].put(('step', actions[i]))
                obs, rew, done, info = self._queue_get(q)
                done = True
            obses.append(obs)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
        return np.array(obses), np.array(rews), np.array(dones), infos

    def close(self):
        for q in self._command_queues:
            q.put(('close', None))
        for proc in self._procs:
            proc.join()

    def _restart_worker(self, idx):
        os.system('kill -9 $(ps -o pid= --ppid %d)' % self._procs[idx].pid)
        self._procs[idx].terminate()
        self._procs[idx].join()
        cmd_queue = Queue()
        res_queue = Queue()
        proc = Process(target=self._worker,
                       args=(cmd_queue, res_queue, cloudpickle.dumps(self._env_fns[idx]),))
        proc.start()
        self._procs[idx] = proc
        self._command_queues[idx] = cmd_queue
        self._result_queues[idx] = res_queue
        self._queue_get(res_queue)

    @staticmethod
    def _worker(cmd_queue, res_queue, env_str):
        try:
            env = cloudpickle.loads(env_str)()
            res_queue.put((None, None))
            try:
                while True:
                    cmd, arg = cmd_queue.get()
                    if cmd == 'reset':
                        res_queue.put((env.reset(), None))
                    elif cmd == 'step':
                        obs, rew, done, info = env.step(arg)
                        if done:
                            obs = env.reset()
                        res_queue.put(((obs, rew, done, info), None))
                    elif cmd == 'close':
                        return
            finally:
                env.close()
        except Exception as exc:
            res_queue.put((None, exc))

    @staticmethod
    def _queue_get(queue):
        value, exc = queue.get(timeout=20)
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

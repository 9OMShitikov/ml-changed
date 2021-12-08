from sumTree import SumTree

import numpy as np
import random


class PriorityReplayBuffer(object):
    def __init__(self, size, alpha, eps):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            Determines how much prioritization is used. alpha==0 - uniform randomized buffer
        """
        self._storage = SumTree(size)
        self._alpha = alpha  # alpha determines how much prioritization is used
        self._eps = eps  # epsilon smooths priority, priority = (TD_error + eps) ** alpha

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, priority):
        data = (obs_t, action, reward, obs_tp1, done)
        self._storage.insert(data, priority)

    def _encode_sample(self, records):
        ids, obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], [], []
        for data in records:
            idx, (obs_t, action, reward, obs_tp1, done) = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            ids.append(idx)
        return np.array(obses_t), np.array(actions),\
               np.array(rewards), np.array(obses_tp1),\
               np.array(dones), np.array(ids)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        idx: np.array
            indexes of observations
        """
        segment = self._storage.total_sum() / batch_size
        records = [self._storage.get(np.random.uniform(i * segment, (i+1) * segment)) for i in range(batch_size)]
        return self._encode_sample(records)

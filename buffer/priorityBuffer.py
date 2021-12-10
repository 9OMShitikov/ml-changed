from sumTree import SumTree

import numpy as np
import random


class PriorityReplayBuffer(object):
    def __init__(self, size, alpha=0.6, eps=1e-5):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            Determines how much prioritization is used. alpha==0 - uniform randomized buffer
        """
        self._tree = SumTree(size)
        self._storage = self._tree._data[:]
        self._alpha = alpha  # alpha determines how much prioritization is used
        self.beta = 0.4 # beta, set up during training
        self._eps = eps  # epsilon smooths priority, priority = (TD_error + eps) ** alpha

    def __len__(self):
        return len(self._tree)

    def add(self, obs_t, action, reward, obs_tp1, done, td_error):
        priority = (td_error + self._eps) ** self._alpha
        self._storage = self._tree._data[: len(self._tree)]
        data = (obs_t, action, reward, obs_tp1, done)
        self._tree.insert(data, priority)

    def update(self, idxs, td_error):
        priorities = (td_error + self._eps) ** self._alpha
        for idx, priority in zip(idxs, priorities):
          self._tree.update(idx, priority)

    def _encode_sample(self, records):
        ids, obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], [], []
        probas = []
        for data in records:
            idx, proba, (obs_t, action, reward, obs_tp1, done) = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            ids.append(idx)
            probas.append(proba)
        
        w = (len(records) * np.array(probas) /\
                            self._tree.total_sum()) ** -self.beta
        w /= np.max(w)

        return np.array(obses_t), np.array(actions),\
               np.array(rewards), np.array(obses_tp1),\
               np.array(dones), np.array(ids), w

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
        segment = self._tree.total_sum() / batch_size
        records = [self._tree.get(np.random.uniform(i * segment, (i+1) * segment)) for i in range(batch_size)]
        return self._encode_sample(records)

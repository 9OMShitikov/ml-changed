import numpy as np


class SumTree:
    def __init__(self, size):
        self.is_full = False
        self._size = 0
        self._maxsize = size
        self._next_idx = 0
        self._tree = np.zeros(2 * size - 1)
        self._data = np.zeros(size, dtype=object)

    def __len__(self):
        return self._size

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self._tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        # current node is leaf node
        if left >= len(self._tree):
            return idx

        if s <= self._tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self._tree[left])

    def total_sum(self):
        return self._tree[0]

    def insert(self, data, priority):
        self._data[self._next_idx] = data

        idx = self._next_idx + self._maxsize - 1
        self.update(idx, priority)

        self._next_idx += 1
        if self._next_idx >= self._maxsize:
            self._next_idx = 0
            self._size = self._maxsize
        if self._size != self._maxsize:
            self._size += 1

    def update(self, idx, priority):
        change = priority - self._tree[idx]

        self._tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self._maxsize + 1

        # returns tuple: (tree_idx, data)
        return idx, self._tree[idx], self._data[data_idx]

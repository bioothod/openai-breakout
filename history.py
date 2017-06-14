from collections import deque
import numpy as np

class history(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.history = deque()

    def load(self, d):
        self.history = d

    def clear(self):
        self.history = deque()

    def last(self, n):
        if n <= 0:
            return deque()

        return deque(self.history, n)

    def size(self):
        return len(self.history)

    def full(self):
        return self.size() >= self.max_size

    def append(self, e):
        qlen = len(self.history) + 1
        if qlen > self.max_size:
            for i in range(qlen - self.max_size):
                self.history.popleft()

        self.history.append(e)

    def get(self, idx):
        return self.history[idx]

    def sample(self, size):
        if self.size() == 0:
            return deque()

        idx = range(self.size())

        ch = np.random.choice(idx, min(size, self.size()))

        ret = deque()
        for i in ch:
            ret.append(self.history[i])
        
        return ret

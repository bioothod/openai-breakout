import numpy as np

from collections import deque

class state(object):
    def __init__(self, shape, size):
        self.steps = deque()
        self.shape = shape
        self.size = size
        self.value = None

        for i in range(size):
            self.push_zeroes()

    def push_zeroes(self):
        self.push_tensor(np.zeros(self.shape))

    def push_tensor(self, st):
        if self.shape != st.shape:
            print("self.shape: %s, tensor.shape: %s" % (self.shape, st.shape))
            assert self.shape == st.shape

        if len(self.steps) == self.size:
            self.steps.popleft()

        self.steps.append(st)
        self.merge()

    def merge(self):
        self.value = np.concatenate(self.steps, axis=len(self.shape)-1)

    def read(self):
        return self.value

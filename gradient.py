class gradient(object):
    def __init__(self, grad):
        self.grad = grad

    def update(self, v):
        self.grad += v

    def clear(self):
        self.grad = 0

    def read(self):
        return self.grad


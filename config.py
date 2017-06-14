class config(object):
    def __init__(self):
        self.conf = {}

    def get(self, name, default=None):
        return self.conf.get(name, default)

    def put(self, name, value):
        self.conf[name] = value


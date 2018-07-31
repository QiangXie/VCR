import time

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.start_time = 0.

    def tic(self):
        self.start_time = time.time()
        self.total_time = 0.

    def toc(self):
        self.total_time += time.time() - self.start_time

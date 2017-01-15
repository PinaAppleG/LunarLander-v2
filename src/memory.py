import random

class Memory(object):
    """ Memory buffer for storing past experiences of the agent """
    def __init__(self, maxsize=100000):
        # Create lists for storing a set of observations
        # Components of each observations are stored in separate lists
        self.maxsize = maxsize
        self.s  = []
        self.a  = []
        self.r  = []
        self.s1 = []
        self.done = []

    def save(self, s_, a_, r_, s1_, done_):
        """ Saves a single experience tuple to buffer """
        # Add input observation to the component lists
        self.s.append(s_)
        self.a.append(a_)
        self.r.append(r_)
        self.s1.append(s1_)
        self.done.append(done_)
        # If the buffer is overflowing, remove the oldest added observation
        if len(self.s) > self.maxsize:
            self.s.pop(0)
            self.a.pop(0)
            self.r.pop(0)
            self.s1.pop(0)
            self.done.pop(0)

    def getsize(self):
        """ returns the current size of the buffer """
        return len(self.s)

    def sample(self, n):
        """ Returns n random experience tuples from the buffer """
        # Current number of observations stored
        N    = len(self.s)
        n    = min(n, N)
        # Sample a list of n random indices from the range [0, N)
        # without replacement
        ids  = random.sample(xrange(N), n)
        # Create slices of each observation component using
        # this list of random indices
        s    = [self.s[ii] for ii in ids]
        a    = [self.a[ii] for ii in ids]
        r    = [self.r[ii] for ii in ids]
        s1   = [self.s1[ii] for ii in ids]
        done = [self.done[ii] for ii in ids]
        return (s, a, r, s1, done)

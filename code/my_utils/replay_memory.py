from collections import namedtuple
import random
import itertools
import numpy as np

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward', 'latent_code', 'latent_archive_id'))
Transition.__new__.__defaults__ = tuple([None for _ in range(len(Transition._fields)-1)] + [-1])

class Memory(object):
    def __init__(self, capacity=None):
        self.capacity = capacity
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))
        if self.capacity is not None and len(self.memory) > self.capacity:
            self.memory = self.memory[1:]

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            # index = random.sample(range(len(self.memory)), batch_size)
            # random_batch = [self.memory[i] for i in index]    # super slow 
            # random_batch = random.sample(self.memory, batch_size)   # still slow

            random_batch = random.choices(self.memory, k=batch_size)    # the fastest somehow
            return Transition(*zip(*random_batch))

    ## For N-step returns update. 
    ## This function return a list (sequence) length N of Transition tuple. 
    def sample_n_step(self, batch_size=None, N=1):
        if N == 1:
            return [self.sample(batch_size)]    # return a list always

        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            index = random.sample(range(len(self.memory)-N), batch_size)    # cannot select the last N element, since we have not observe future N step yet
            out = []
            for n in range(0, N):
                random_batch = [self.memory[i+n] for i in index]
                out += [Transition(*zip(*random_batch))]
            return out

    def append(self, new_memory):
        self.memory += new_memory.memory
        if self.capacity is not None and len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity:]

    def reset(self):
        self.memory = []

    def size(self):
        return len(self.memory)

    def __len__(self):
        return len(self.memory)

class RamFusionDistr(object):
    def __init__(self, buf_size, subsample_ratio=0.5):
        self.buf_size = buf_size
        self.buffer = []
        self.subsample_ratio = subsample_ratio

    def subsample_append(self, batch, subsample=True):
        transitions = list(zip(*batch))
        if subsample:
            transitions = transitions[:int(len(transitions)*self.subsample_ratio)]
        self.buffer.extend(transitions)
        overflow = len(self.buffer)-self.buf_size
        while overflow > 0:
            N = len(self.buffer)
            probs = np.arange(N)+1
            probs = probs/float(np.sum(probs))
            pidx = np.random.choice(np.arange(N), p=probs)
            self.buffer.pop(pidx)
            overflow -= 1

    def sample_concat(self, batch):
        if len(self.buffer) == 0:
            return batch
        else:
            n = len(batch[0])
            pidxs = np.random.randint(0, len(self.buffer), size=(n))
            sampled = [self.buffer[pidx] for pidx in pidxs]
            original = zip(*batch)
            return Transition(*zip(*itertools.chain(sampled, original)))

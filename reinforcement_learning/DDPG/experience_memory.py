from collections import deque
from random import sample
import numpy as np


class ReplayMemory(deque):
    """ Experience replay memory
    Args:
        size: maximum size of memory
    """

    def __init__(self, size):
        super(ReplayMemory, self).__init__(maxlen=size)

    def get_batch(self, batch_size):
        """ Get random batch arrays for training
        Args:
            batch_size: number of replay sample
        Returns:
            tuple of random batch arrays
        """
        sampled = sample(self, batch_size)
        X_batch, action_batch, reward_batch, X_next_batch, done_batch = list(), list(), list(), list(), list()
        for X, action, reward, X_next, done in sampled:
            X_batch.append(X)
            action_batch.append(action)
            reward_batch.append(reward)
            X_next_batch.append(X_next)
            done_batch.append(done)

        X_batch = np.vstack(X_batch).astype(np.float32)
        action_batch = np.vstack(action_batch).astype(np.float32)
        reward_batch = np.vstack(reward_batch).astype(np.float32)
        X_next_batch = np.vstack(X_next_batch).astype(np.float32)
        done_batch = np.vstack(done_batch).astype(np.float32)

        return X_batch, action_batch, reward_batch, X_next_batch, done_batch


from collections import deque

import numpy as np
import random

class ReplayBuffer:
    def __init__(self, cap):
        self.buffer = deque(maxlen=cap)

    def push(self, state, action, reward, next, done):
        self.buffer.append((state, action, reward, next, done))

    def sample(self, batch_size):
        states, actions, rewards, nexts, dones = zip(*random.sample(self.buffer, batch_size))

        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(nexts),
            np.array(dones, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)
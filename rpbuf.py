from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np
import random


class ReplayBuffer:
    """A simple experience replay buffer storing transitions as numpy arrays."""

    def __init__(self, cap: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=cap)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: float) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))

        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self) -> int:
        return len(self.buffer)
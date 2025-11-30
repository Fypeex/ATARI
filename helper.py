from typing import Deque

import cv2 as cv
import numpy as np
from collections import deque


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """Convert raw Atari RGB frame to 84x84 grayscale uint8.

    Args:
        obs: Array of shape (210,160,3) in RGB.
    Returns:
        Grayscale 84x84 uint8 image.
    """
    img = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
    img = img[34:194, :]
    img = cv.resize(img, (84, 84), interpolation=cv.INTER_AREA)
    img = img.astype(np.uint8)  # keep as bytes 0–255
    return img  # (84, 84), uint8


def init_state_stack(obs: np.ndarray) -> Deque[np.ndarray]:
    """Initialize a deque with 4 preprocessed frames from the first observation."""
    frame = preprocess_obs(obs)
    stack: Deque[np.ndarray] = deque(maxlen=4)
    for _ in range(4):
        stack.append(frame)
    return stack


def get_state_from_stack(stack: Deque[np.ndarray]) -> np.ndarray:
    """Stack the last 4 frames into a (4,84,84) uint8 tensor-like numpy array."""
    return np.stack(stack, axis=0)  # (4,84,84), uint8

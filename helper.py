import cv2 as cv
import numpy as np
from collections import deque

def preprocess_obs(obs):
    # obs: (210,160,3) RGB
    img = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
    img = img[34:194, :]
    img = cv.resize(img, (84, 84), interpolation=cv.INTER_AREA)
    img = img.astype(np.uint8)   # keep as bytes 0–255
    return img  # (84, 84), uint8


def init_state_stack(obs):
    frame = preprocess_obs(obs)
    stack = deque(maxlen=4)
    for _ in range(4):
        stack.append(frame)
    return stack

def get_state_from_stack(stack):
    # stack of 4 (84,84) uint8
    return np.stack(stack, axis=0)   # (4,84,84), uint8

"""Shared configuration for the ATARI DQN project.

Centralizes constants that are used across training and evaluation scripts
so we avoid duplication and keep behavior consistent.
"""
from __future__ import annotations

import os
import torch

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
CHECKPOINT_DIR = os.path.join("models")
OBS_SAVE_DIR = os.path.join("observations")

# Action mapping used by both training and evaluation
# 0 -> NOOP, 2/3 -> paddle up/down (depending on ALE mapping)
VALID_ACTIONS = [0, 2, 3]

"""Checkpoint utilities for saving and loading DQN models.

Provides a single place for reading/writing checkpoints so training and
evaluation scripts do not duplicate logic.
"""
from __future__ import annotations

import glob
import os
from typing import Optional, Tuple

import torch

from config import CHECKPOINT_DIR


def save_checkpoint(policy_net: torch.nn.Module,
                    target_net: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    frame_idx: int,
                    directory: str = CHECKPOINT_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    ckpt_path = os.path.join(directory, f"dqn_pong_model_{frame_idx}.pth")
    checkpoint = {
        "frame_idx": frame_idx,
        "policy_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, ckpt_path)
    print(f"[Checkpoint] Saved model at '{ckpt_path}' (frame {frame_idx})")
    return ckpt_path


def load_checkpoint(policy_net: torch.nn.Module,
                    device: torch.device,
                    checkpoint_option: str,
                    *,
                    target_net: Optional[torch.nn.Module] = None,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    directory: str = CHECKPOINT_DIR) -> int:
    """
    Load a checkpoint into the provided modules.

    checkpoint_option:
      - "none": do not load anything
      - "latest": load newest checkpoint in directory
      - path: load the specific path

    Returns frame_idx if present, else 0.
    """
    if checkpoint_option == "none":
        print("[Checkpoint] No checkpoint specified, starting fresh.")
        return 0

    if checkpoint_option == "latest":
        pattern = os.path.join(directory, "dqn_pong_model_*.pth")
        ckpts = glob.glob(pattern)
        if not ckpts:
            print(f"[Checkpoint] No checkpoints found in '{directory}', starting fresh.")
            return 0
        ckpt_path = max(ckpts, key=os.path.getmtime)
    else:
        ckpt_path = checkpoint_option
        if not os.path.isfile(ckpt_path):
            print(f"[Checkpoint] Specified checkpoint '{ckpt_path}' not found, starting fresh.")
            return 0

    print(f"[Checkpoint] Loading checkpoint from '{ckpt_path}'")
    checkpoint = torch.load(ckpt_path, map_location=device)

    policy_net.load_state_dict(checkpoint["policy_state_dict"])

    if target_net is not None:
        if "target_state_dict" in checkpoint:
            target_net.load_state_dict(checkpoint["target_state_dict"])
        else:
            target_net.load_state_dict(checkpoint["policy_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    frame_idx = int(checkpoint.get("frame_idx", 0))
    print(f"[Checkpoint] Loaded frame {frame_idx}")
    return frame_idx

import os
import glob
import time

import gymnasium as gym
import torch
import numpy as np
from collections import deque

import ale_py
import cv2 as cv
import win32gui
from mss import mss

from helper import preprocess_obs, get_state_from_stack, init_state_stack
from nn import DQN
from rpbuf import ReplayBuffer

# =========================
# Checkpoint configuration
# =========================
# Options:
#   "none"    -> start from scratch
#   "latest"  -> load the most recent checkpoint in models/
#   "<path>"  -> load a specific checkpoint file
CHECKPOINT_OPTION = "latest"          # e.g. "none", "latest", "models/dqn_pong_model_100000.pth"
CHECKPOINT_DIR = "models"

# ==============
# NN Parameters
# ==============
GAMMA = 0.99
BATCH_SIZE = 32
LR = 1e-4
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_FRAMES = 1_000_000
TARGET_UPDATE = 10_000
REPLAY_INIT = 50_000
REPLAY_CAP = 400_000
MAX_FRAMES = 10_000_000
MOVE_PENALTY = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================
# Game Parameters
# ================
VALID_ACTIONS = [0, 2, 3]  # 0 -> NOOP, 2/3 -> paddle up/down (depending on mapping)
env = gym.make("ALE/Pong-v5", render_mode=None)
obs, info = env.reset()
stack = init_state_stack(obs)
state = get_state_from_stack(stack)   # (4,84,84)

# ===============
# CNN Setup
# ===============
policy_net = DQN(in_channels=4, num_actions=len(VALID_ACTIONS)).to(device)
target_net = DQN(in_channels=4, num_actions=len(VALID_ACTIONS)).to(device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# ===============
# Replay Buffer
# ===============
replay = ReplayBuffer(cap=REPLAY_CAP)

episode_reward = 0.0
episode_move_count = 0


# =====================
# Checkpoint utilities
# =====================
def save_checkpoint(frame_idx: int):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"dqn_pong_model_{frame_idx}.pth")
    checkpoint = {
        "frame_idx": frame_idx,
        "policy_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, ckpt_path)
    print(f"[Checkpoint] Saved model at '{ckpt_path}' (frame {frame_idx})")


def load_checkpoint(option: str) -> int:
    """
    Returns the starting frame index (0 if none loaded).
    """
    if option == "none":
        print("[Checkpoint] Starting from scratch (no checkpoint).")
        return 0

    if option == "latest":
        pattern = os.path.join(CHECKPOINT_DIR, "dqn_pong_model_*.pth")
        ckpts = glob.glob(pattern)
        if not ckpts:
            print("[Checkpoint] No checkpoints found, starting from scratch.")
            return 0
        # Pick latest by modification time
        ckpt_path = max(ckpts, key=os.path.getmtime)
    else:
        ckpt_path = option
        if not os.path.isfile(ckpt_path):
            print(f"[Checkpoint] Specified checkpoint '{ckpt_path}' not found, starting from scratch.")
            return 0

    print(f"[Checkpoint] Loading checkpoint from '{ckpt_path}'")
    checkpoint = torch.load(ckpt_path, map_location=device)

    policy_net.load_state_dict(checkpoint["policy_state_dict"])
    if "target_state_dict" in checkpoint:
        target_net.load_state_dict(checkpoint["target_state_dict"])
    else:
        target_net.load_state_dict(checkpoint["policy_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_frame = int(checkpoint.get("frame_idx", 0))
    print(f"[Checkpoint] Loaded frame {start_frame}")
    return start_frame


# ========================
# Action selection (ε-greedy)
# ========================
def select_action(state, frame_idx):
    eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * frame_idx / EPS_DECAY_FRAMES)
    if np.random.rand() < eps:
        # Random INDEX into VALID_ACTIONS (0,1,2)
        a_idx = np.random.randint(len(VALID_ACTIONS))
    else:
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(device)  # (1,4,84,84), uint8
            s = s.float() / 255.0
            q_values = policy_net(s)
            a_idx = int(torch.argmax(q_values, dim=1).item())
    return a_idx, eps


# ========================
# Optimization step
# ========================
def optimize_model():
    if len(replay) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)

    states = torch.from_numpy(states).float().to(device) / 255.0
    next_states = torch.from_numpy(next_states).float().to(device) / 255.0
    actions     = torch.from_numpy(actions).long().to(device)     # (B,)
    rewards     = torch.from_numpy(rewards).to(device)            # (B,)
    dones       = torch.from_numpy(dones).to(device)              # (B,)

    # Q(s,a)
    q_values = policy_net(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Q_target(s',a')
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        expected_q = rewards + GAMMA * next_q_values * (1.0 - dones)

    loss = torch.nn.functional.mse_loss(q_value, expected_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()


# ==================================
# Load checkpoint if requested
# ==================================
start_frame = load_checkpoint(CHECKPOINT_OPTION)

# =====================
# Main training loop
# =====================
for frame_idx in range(start_frame + 1, MAX_FRAMES + 1):
    # select action index for our reduced action set
    a_idx, epsilon = select_action(state, frame_idx)

    if frame_idx % 1000 == 0:
        print(f"Frame {frame_idx}, epsilon: {epsilon:.3f}")

    # Save checkpoint periodically
    if frame_idx % 10000 == 0:
        save_checkpoint(frame_idx)

    # Map index -> actual ALE action
    env_action = VALID_ACTIONS[a_idx]

    next_obs, reward, terminated, truncated, info = env.step(env_action)
    done = terminated or truncated
    episode_reward += reward

    is_movement = (env_action != 0)
    movement_reward = MOVE_PENALTY if is_movement else 0.0
    episode_move_count += is_movement
    shaped_reward = reward + movement_reward


    frame = preprocess_obs(next_obs)
    stack.append(frame)
    next_state = get_state_from_stack(stack)

    # Store index (0,1,2) in replay buffer
    replay.push(state, a_idx, shaped_reward, next_state, float(done))
    state = next_state

    # train after initial replay fill
    if frame_idx > REPLAY_INIT:
        optimize_model()

    # periodically update target network
    if frame_idx % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if done:
        print(f"Frame {frame_idx}, episode reward: {episode_reward:.1f}, episode move count: {episode_move_count}, epsilon: {epsilon:.3f}")
        obs, info = env.reset()
        stack = init_state_stack(obs)
        state = get_state_from_stack(stack)
        episode_reward = 0.0
        episode_move_count = 0

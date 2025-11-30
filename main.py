import gymnasium as gym
import torch
import numpy as np
import ale_py  # noqa: F401 (ensure ALE environments are registered)

from helper import preprocess_obs, get_state_from_stack, init_state_stack
from nn import DQN
from rpbuf import ReplayBuffer
from config import DEVICE, VALID_ACTIONS, CHECKPOINT_DIR
from checkpoint import save_checkpoint as save_ckpt_util, load_checkpoint as load_ckpt_util

# =========================
# Checkpoint configuration
# =========================
# Options:
#   "none"    -> start from scratch
#   "latest"  -> load the most recent checkpoint in models/
#   "<path>"  -> load a specific checkpoint file
CHECKPOINT_OPTION = "latest"  # e.g. "none", "latest", "models/dqn_pong_model_100000.pth"

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

device = DEVICE
print("Using device:", device)

# ================
# Environment setup
# ================
env = gym.make("ALE/Pong-v5", render_mode=None)
obs, info = env.reset()
stack = init_state_stack(obs)
state = get_state_from_stack(stack)  # (4,84,84)

# ===============
# Networks & Optimizer
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


if __name__ == "__main__":
    # ==================================
    # Load checkpoint if requested
    # ==================================
    start_frame = load_ckpt_util(
        policy_net,
        device,
        CHECKPOINT_OPTION,
        target_net=target_net,
        optimizer=optimizer,
        directory=CHECKPOINT_DIR,
    )

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
            save_ckpt_util(policy_net, target_net, optimizer, frame_idx, directory=CHECKPOINT_DIR)

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
            print(
                f"Frame {frame_idx}, episode reward: {episode_reward:.1f}, "
                f"episode move count: {episode_move_count}, epsilon: {epsilon:.3f}"
            )
            obs, info = env.reset()
            stack = init_state_stack(obs)
            state = get_state_from_stack(stack)
            episode_reward = 0.0
            episode_move_count = 0

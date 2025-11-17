import os
import glob
import argparse

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from nn import DQN
from helper import preprocess_obs, init_state_stack, get_state_from_stack

import ale_py

# Same action mapping as in your training script
VALID_ACTIONS = [0, 2, 3]  # 0 -> NOOP, 2/3 -> paddle up/down
CHECKPOINT_DIR = "models"
OBS_SAVE_DIR = "observations"

def load_checkpoint(policy_net, device, checkpoint_option):
    """
    checkpoint_option:
      - "latest": load most recent checkpoint in CHECKPOINT_DIR
      - "<path>.pth": load specific file
      - "none": do not load anything (random weights)

    Returns frame_idx if present in checkpoint, else 0.
    """
    if checkpoint_option == "none":
        print("[Eval] No checkpoint specified, using random weights.")
        return 0

    if checkpoint_option == "latest":
        pattern = os.path.join(CHECKPOINT_DIR, "dqn_pong_model_*.pth")
        ckpts = glob.glob(pattern)
        if not ckpts:
            print(f"[Eval] No checkpoints found in '{CHECKPOINT_DIR}', using random weights.")
            return 0
        ckpt_path = max(ckpts, key=os.path.getmtime)
    else:
        ckpt_path = checkpoint_option
        if not os.path.isfile(ckpt_path):
            print(f"[Eval] Specified checkpoint '{ckpt_path}' not found, using random weights.")
            return 0

    print(f"[Eval] Loading checkpoint from '{ckpt_path}'")
    checkpoint = torch.load(ckpt_path, map_location=device)

    policy_net.load_state_dict(checkpoint["policy_state_dict"])
    frame_idx = int(checkpoint.get("frame_idx", 0))
    print(f"[Eval] Loaded frame {frame_idx}")
    return frame_idx


def evaluate(checkpoint_option: str, episodes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Eval] Using device:", device)

    # Create env with human rendering so you can watch
    env = gym.make("ALE/Pong-v5", render_mode="human")

    # Build network (same arch as training)
    policy_net = DQN(in_channels=4, num_actions=len(VALID_ACTIONS)).to(device)
    policy_net.eval()

    # Load checkpoint if requested
    _ = load_checkpoint(policy_net, device, checkpoint_option)

    all_rewards = []

    observations = []

    for ep in range(episodes):
        obs, info = env.reset()
        stack = init_state_stack(obs)
        state = get_state_from_stack(stack)  # (4,84,84), likely uint8

        done = False
        ep_reward = 0.0

        while not done:
            # Prepare state for network
            with torch.no_grad():
                s = torch.from_numpy(state).unsqueeze(0).to(device)  # (1,4,84,84)
                # if stored as uint8, convert to float in [0,1]
                if s.dtype != torch.float32:
                    s = s.float() / 255.0
                q_values = policy_net(s)
                a_idx = int(torch.argmax(q_values, dim=1).item())

            env_action = VALID_ACTIONS[a_idx]
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            ep_reward += reward
            observations.append(next_obs)
            frame = preprocess_obs(next_obs)
            stack.append(frame)
            state = get_state_from_stack(stack)

        #Save all observations as images
        for i, obs_arr in enumerate(observations):
            # Ensure uint8 HxWxC
            arr = obs_arr
            if arr.dtype != np.uint8:
                # simple heuristic: assume [0,1] float, scale to [0,255]
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)

            img = Image.fromarray(arr)
            img.save(os.path.join(
                OBS_SAVE_DIR,
                f"episode_{ep + 1:03d}_frame_{i:05d}.png"
            ))

        all_rewards.append(ep_reward)
        print(f"[Eval] Episode {ep + 1}/{episodes} reward: {ep_reward:.1f}")

    env.close()

    if all_rewards:
        mean_r = np.mean(all_rewards)
        std_r = np.std(all_rewards)
        print(f"[Eval] Mean reward over {episodes} episodes: {mean_r:.2f} ± {std_r:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN Pong policy from checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help=(
            "Which checkpoint to load: "
            "'latest' (default) for newest in models/, "
            "'none' for random weights, or a specific path like "
            "'models/dqn_pong_model_100000.pth'."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to play for evaluation."
    )
    args = parser.parse_args()

    evaluate(args.checkpoint, args.episodes)

import os
import argparse

import gymnasium as gym
import numpy as np
import torch
import ale_py  # noqa: F401 (ensure ALE environments are registered)
from PIL import Image

from nn import DQN
from helper import preprocess_obs, init_state_stack, get_state_from_stack
from config import DEVICE, VALID_ACTIONS, CHECKPOINT_DIR, OBS_SAVE_DIR
from checkpoint import load_checkpoint as load_ckpt_util



def evaluate(checkpoint_option: str, episodes: int):
    device = DEVICE
    print("[Eval] Using device:", device)

    # Ensure output directory exists
    os.makedirs(OBS_SAVE_DIR, exist_ok=True)

    # Create env with human rendering so you can watch
    env = gym.make("ALE/Pong-v5", render_mode="human")

    # Build network (same arch as training)
    policy_net = DQN(in_channels=4, num_actions=len(VALID_ACTIONS)).to(device)
    policy_net.eval()

    # Load checkpoint if requested
    _ = load_ckpt_util(policy_net, device, checkpoint_option, directory=CHECKPOINT_DIR)

    all_rewards = []

    for ep in range(episodes):
        obs, info = env.reset()
        stack = init_state_stack(obs)
        state = get_state_from_stack(stack)  # (4,84,84), likely uint8

        done = False
        ep_reward = 0.0
        frame_i = 0

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

            frame_i += 1

            # Update state stack
            frame = preprocess_obs(next_obs)
            stack.append(frame)
            state = get_state_from_stack(stack)

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
        default="none",
        help=(
            "Which checkpoint to load: "
            "'latest' for newest in models/, "
            "'none' (default) for random weights, or a specific path like "
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

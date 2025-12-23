#!/usr/bin/env python3
"""
Evaluation script for trained LSTM-PPO agent on CarRacing-v3.

Usage:
    python evaluate.py --checkpoint checkpoints/model_latest.pt
    python evaluate.py --checkpoint checkpoints/model_latest.pt --episodes 10
"""

import argparse

import numpy as np
import torch

from agents import LSTMPPOAgent
from utils import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LSTM-PPO agent on CarRacing-v3")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to evaluate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    return parser.parse_args()


def evaluate(agent, env, num_episodes):
    """
    Evaluate agent for multiple episodes.

    Args:
        agent: LSTMPPOAgent
        env: Gymnasium environment
        num_episodes: Number of episodes to run

    Returns:
        Tuple of (episode_rewards, episode_lengths)
    """
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        hidden = agent.get_initial_hidden()

        total_reward = 0
        steps = 0
        done = False

        while not done:
            action, _, _, hidden = agent.get_action(obs, hidden, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward = {total_reward:.1f}, Steps = {steps}")

    return episode_rewards, episode_lengths


def main():
    args = parse_args()

    # Setup device
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    # Load agent
    print(f"Loading checkpoint from {args.checkpoint}")
    agent = LSTMPPOAgent.from_checkpoint(args.checkpoint, device=device)
    agent.network.eval()  # Set to evaluation mode

    # Create environment (headless)
    env = make_env(render_mode="rgb_array")

    print(f"\nEvaluating for {args.episodes} episodes...")
    print("-" * 50)

    # Run evaluation
    rewards, lengths = evaluate(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
    )

    # Print statistics
    if rewards:
        print("-" * 50)
        print(f"Results over {len(rewards)} episodes:")
        print(f"  Mean reward:   {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
        print(f"  Min reward:    {np.min(rewards):.1f}")
        print(f"  Max reward:    {np.max(rewards):.1f}")
        print(f"  Mean length:   {np.mean(lengths):.0f}")

    env.close()


if __name__ == "__main__":
    main()

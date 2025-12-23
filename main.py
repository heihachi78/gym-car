#!/usr/bin/env python3
"""
Play CarRacing-v3 using trained LSTM-PPO model.

Usage:
    python main.py                                    # Use latest checkpoint
    python main.py --checkpoint checkpoints/model.pt  # Use specific checkpoint
    python main.py --random                           # Use random actions (no model)
"""

import argparse

from agents import LSTMPPOAgent
from utils import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="Play CarRacing-v3 with LSTM-PPO agent")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_latest.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--random", action="store_true",
                        help="Use random actions instead of model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create wrapped environment for agent's processed observations
    # We use render_mode="human" to visualize the game window while the agent gets processed obs.
    agent_env = make_env(render_mode="human")

    if args.random:
        print("Playing with random actions...")
        agent = None
        hidden = None
    else:
        print(f"Loading model from {args.checkpoint}")
        try:
            agent = LSTMPPOAgent.from_checkpoint(args.checkpoint, device=args.device)
            agent.network.eval()
            hidden = agent.get_initial_hidden()
            print("Model loaded successfully!")
        except FileNotFoundError:
            print(f"Checkpoint not found: {args.checkpoint}")
            print("Run 'python train.py' first to train a model, or use --random for random actions")
            return

    # Reset both environments with the same seed for synchronized state
    #seed = 42
    #observation, _ = agent_env.reset(seed=seed)
    observation, _ = agent_env.reset()
    print(f"Observation shape: {observation.shape}")

    episode_over = False
    total_reward = 0.0
    steps = 0

    while not episode_over:
        if agent is None or hidden is None:
            # Random action
            action = agent_env.action_space.sample()
        else:
            # Model action
            action, _, _, hidden = agent.get_action(observation, hidden, deterministic=True)

        # Step environment
        observation, reward, terminated, truncated, info = agent_env.step(action)
        total_reward += float(reward)
        steps += 1
        episode_over = terminated or truncated

        if steps <= 10:
            print(f"Step {steps}: action={action}, reward={reward:.2f}, term={terminated}, trunc={truncated}")

        # render_mode="human" renders automatically, render() returns None (not an error)
        agent_env.render()

    print(f"Episode finished! Total reward: {total_reward:.1f}, Steps: {steps}")
    agent_env.close()


if __name__ == "__main__":
    main()

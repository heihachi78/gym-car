#!/usr/bin/env python3
"""
Batch evaluation script for all checkpoints in a directory.

Evaluates each model and prints a comparison table showing:
- Mean reward per model
- Median reward per model
- Which model achieved the highest mean/median reward

Usage:
    python evaluate_all.py
    python evaluate_all.py --episodes 10
    python evaluate_all.py --checkpoints-dir checkpoints --episodes 5
"""

import argparse
import os
import re
import statistics

import numpy as np
import torch

from agents import LSTMPPOAgent
from utils import make_env, get_normalize_wrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints and compare results"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes per model (default: 5)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default="checkpoints",
        help="Directory containing checkpoint files"
    )
    return parser.parse_args()


def find_checkpoints(directory):
    """
    Find all model checkpoint files, sorted by timestep.

    Args:
        directory: Path to checkpoints directory

    Returns:
        List of (timestep, filepath) tuples sorted by timestep
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Checkpoints directory not found: {directory}")

    checkpoints = []
    pattern = re.compile(r"model_(\d+)\.pt$")

    for filename in os.listdir(directory):
        # Skip model_latest.pt to avoid duplicate evaluation
        if filename == "model_latest.pt":
            continue

        match = pattern.match(filename)
        if match:
            timestep = int(match.group(1))
            filepath = os.path.join(directory, filename)
            checkpoints.append((timestep, filepath))

    # Sort by timestep
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def evaluate_model(checkpoint_path, num_episodes, device):
    """
    Load and evaluate a single model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        num_episodes: Number of episodes to run
        device: Device to use for inference

    Returns:
        Dictionary with evaluation results
    """
    # Load agent
    agent, obs_rms = LSTMPPOAgent.from_checkpoint(checkpoint_path, device=device)
    agent.network.eval()

    # Create environment
    env = make_env(render_mode="rgb_array")

    # Restore normalization statistics
    if obs_rms is not None:
        norm_wrapper = get_normalize_wrapper(env)
        if norm_wrapper is not None:
            norm_wrapper.obs_rms.mean = obs_rms['mean']
            norm_wrapper.obs_rms.var = obs_rms['var']
            norm_wrapper.obs_rms.count = obs_rms['count']
            norm_wrapper.update_running_mean = False

    # Run evaluation episodes
    rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        hidden = agent.get_initial_hidden()

        total_reward = 0
        done = False
        steps = 0

        while not done:
            action, _, _, hidden = agent.get_action(obs, hidden, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        print(f"  Episode {episode + 1}/{num_episodes}: Reward = {total_reward:.1f}, Steps = {steps}")

    env.close()

    # Compute statistics
    return {
        'checkpoint': os.path.basename(checkpoint_path),
        'rewards': rewards,
        'mean': np.mean(rewards),
        'median': statistics.median(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
    }


def print_results_table(results):
    """
    Print formatted comparison table with results.

    Args:
        results: List of result dictionaries from evaluate_model()
    """
    if not results:
        print("No results to display.")
        return

    # Find best for each metric (higher is better for all)
    best_mean_idx = max(range(len(results)), key=lambda i: results[i]['mean'])
    best_median_idx = max(range(len(results)), key=lambda i: results[i]['median'])
    best_min_idx = max(range(len(results)), key=lambda i: results[i]['min'])
    best_max_idx = max(range(len(results)), key=lambda i: results[i]['max'])

    # Print table header
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'Model':<20} | {'Mean':>10} | {'Median':>10} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 80)

    # Print each row
    for i, r in enumerate(results):
        model_name = r['checkpoint'].replace('.pt', '')
        mean_str = f"{r['mean']:>10.1f}"
        median_str = f"{r['median']:>10.1f}"
        min_str = f"{r['min']:>8.1f}"
        max_str = f"{r['max']:>8.1f}"

        # Mark best with asterisk
        if i == best_mean_idx:
            mean_str = f"{r['mean']:>9.1f}*"
        if i == best_median_idx:
            median_str = f"{r['median']:>9.1f}*"
        if i == best_min_idx:
            min_str = f"{r['min']:>7.1f}*"
        if i == best_max_idx:
            max_str = f"{r['max']:>7.1f}*"

        print(f"{model_name:<20} | {mean_str} | {median_str} | {r['std']:>8.1f} | {min_str} | {max_str}")

    print("=" * 80)

    # Print summary
    best_mean = results[best_mean_idx]
    best_median = results[best_median_idx]
    best_min = results[best_min_idx]
    best_max = results[best_max_idx]

    print(f"\nBest Mean Reward:   {best_mean['checkpoint']} ({best_mean['mean']:.1f})")
    print(f"Best Median Reward: {best_median['checkpoint']} ({best_median['median']:.1f})")
    print(f"Best Min Reward:    {best_min['checkpoint']} ({best_min['min']:.1f})")
    print(f"Best Max Reward:    {best_max['checkpoint']} ({best_max['max']:.1f})")


def main():
    args = parse_args()

    # Setup device
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    # Find all checkpoints
    checkpoints = find_checkpoints(args.checkpoints_dir)

    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoints_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints to evaluate")
    print(f"Episodes per model: {args.episodes}")
    print("=" * 80)

    # Evaluate each checkpoint
    results = []
    for idx, (timestep, filepath) in enumerate(checkpoints):
        print(f"\n[{idx + 1}/{len(checkpoints)}] {os.path.basename(filepath)}")

        result = evaluate_model(filepath, args.episodes, device)
        results.append(result)

        print(f"  Mean: {result['mean']:.1f}, Median: {result['median']:.1f}")

    # Print final comparison table
    print_results_table(results)


if __name__ == "__main__":
    main()

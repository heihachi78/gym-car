#!/usr/bin/env python3
"""
Training script for LSTM-PPO agent on CarRacing-v3.

Usage:
    python train.py
    python train.py --total-timesteps 1000000
    python train.py --checkpoint checkpoints/model_latest.pt
"""

import argparse
from pathlib import Path

import torch

import numpy as np

from models import ActorCriticLSTM, LSTMState
from training import RolloutBuffer, PPOTrainer
from utils import make_env, make_vec_env, get_obs_shape, get_num_actions
from config import ConfigLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM-PPO on CarRacing-v3")

    # Configuration
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom YAML/JSON config file")

    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total timesteps to train")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=2048,
                        help="Steps per rollout per environment")
    parser.add_argument("--num-epochs", type=int, default=4,
                        help="PPO epochs per update")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (number of sequences)")
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Sequence length for LSTM")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                        help="PPO clip range")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.03,
                        help="Entropy bonus coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum gradient norm")
    parser.add_argument("--adam-eps", type=float, default=1e-5,
                        help="Adam optimizer epsilon")

    # Network parameters
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="LSTM hidden size")
    parser.add_argument("--num-lstm-layers", type=int, default=1,
                        help="Number of LSTM layers")

    # Logging and checkpointing
    parser.add_argument("--log-dir", type=str, default="runs",
                        help="TensorBoard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-interval", type=int, default=250000,
                        help="Steps between checkpoints")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Episodes between logging")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    return parser.parse_args()


def collect_rollout(env, network, buffer, num_steps, num_envs, device,
                    obs=None, hidden=None, current_episode_rewards=None):
    """
    Collect rollout experience from vectorized environments.

    Args:
        env: Vectorized environment (SyncVectorEnv)
        network: ActorCriticLSTM network
        buffer: RolloutBuffer for storing transitions
        num_steps: Number of steps to collect per environment
        num_envs: Number of parallel environments
        device: Torch device
        obs: Current observations (persisted from previous rollout)
        hidden: Current hidden state (persisted from previous rollout)
        current_episode_rewards: Accumulated rewards for ongoing episodes

    Returns:
        episode_rewards: List of completed episode rewards
        last_obs: Last observations (num_envs, H, W)
        last_done: Whether last states were terminal (num_envs,)
        hidden: Current hidden state
        last_value: Value estimates for last states (num_envs,)
        current_episode_rewards: Accumulated rewards for ongoing episodes
    """
    episode_rewards = []

    # Initialize state only on first call
    if obs is None:
        obs, _ = env.reset()  # (num_envs, H, W)
    if hidden is None:
        hidden = network.get_initial_hidden(batch_size=num_envs, device=device)
    if current_episode_rewards is None:
        current_episode_rewards = np.zeros(num_envs, dtype=np.float32)

    for step in range(num_steps):
        # Store hidden state BEFORE action
        hidden_before = LSTMState(
            h=hidden.h.clone(),
            c=hidden.c.clone()
        )

        # Get action: obs (num_envs, H, W, C) -> (num_envs, 1, C, H, W)
        obs_tensor = torch.from_numpy(obs).float().permute(0, 3, 1, 2).unsqueeze(1).to(device)
        with torch.no_grad():
            action, log_prob, _, value, hidden = network.get_action_and_value(
                obs_tensor, hidden
            )

        # Extract from tensors: (num_envs, 1) -> (num_envs,)
        actions_np = action.squeeze(1).cpu().numpy()
        log_probs_np = log_prob.squeeze(1).cpu().numpy()
        values_np = value.squeeze(1).cpu().numpy()

        # Environment step (vectorized)
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions_np)
        dones = np.logical_or(terminateds, truncateds)

        # Track episode rewards
        current_episode_rewards += rewards
        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(current_episode_rewards[i])
                current_episode_rewards[i] = 0

        # Store transition (all vectorized)
        buffer.add(
            obs=obs,
            action=actions_np,
            reward=rewards,
            done=dones,
            log_prob=log_probs_np,
            value=values_np,
            hidden_state=hidden_before
        )

        obs = next_obs

        # Reset hidden states for done environments
        for i, done in enumerate(dones):
            if done:
                hidden.h[:, i, :] = 0
                hidden.c[:, i, :] = 0

    # Get value for last states (for GAE computation)
    obs_tensor = torch.from_numpy(obs).float().permute(0, 3, 1, 2).unsqueeze(1).to(device)
    with torch.no_grad():
        _, last_value, _ = network.forward(obs_tensor, hidden)
        last_value = last_value.squeeze(1).squeeze(-1).cpu().numpy()  # (num_envs,)

    return episode_rewards, obs, dones, hidden, last_value, current_episode_rewards


def main():
    args = parse_args()

    # Load configuration
    config = ConfigLoader.load(args.config) if args.config else ConfigLoader.load()

    # Apply CLI overrides
    config = ConfigLoader.update_from_cli_args(config, args)

    # Setup device
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    # Extract config values
    num_envs = config['training']['num_envs']
    num_steps = config['training']['num_steps']
    total_timesteps = config['training']['total_timesteps']

    # Create vectorized environment
    env = make_vec_env(num_envs=num_envs, render_mode="rgb_array")
    obs_shape = get_obs_shape()
    num_actions = get_num_actions()

    print(f"Number of environments: {num_envs}")
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")

    # Create network
    network = ActorCriticLSTM(
        obs_shape=obs_shape,
        num_actions=num_actions,
        config=config
    ).to(device)

    # Create trainer
    trainer = PPOTrainer(
        network=network,
        ppo_config=config['ppo'],
        device=device,
        log_dir=args.log_dir
    )

    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=num_steps,
        num_envs=num_envs,
        obs_shape=obs_shape,
        hidden_size=config['lstm']['hidden_size'],
        num_lstm_layers=config['lstm']['num_layers'],
        device=device,
        gamma=config['ppo']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        normalize_epsilon=config['ppo']['normalize_epsilon']
    )

    # Load checkpoint if provided
    start_step = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        start_step = trainer.load_checkpoint(args.checkpoint)
        print(f"Resuming from step {start_step}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    all_episode_rewards = []
    global_step = start_step
    num_updates = 0

    steps_per_update = num_steps * num_envs
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"Steps per rollout: {num_steps} x {num_envs} envs = {steps_per_update}")
    print(f"Sequence length: {config['ppo']['seq_len']}")
    print(f"Batch size: {config['ppo']['batch_size']}")
    print("-" * 50)

    # Persistent state across rollouts
    current_obs = None
    current_hidden = None
    current_ep_rewards = None

    while global_step < total_timesteps:
        # Collect rollout (state persists across calls)
        episode_rewards, current_obs, last_done, current_hidden, last_value, current_ep_rewards = collect_rollout(
            env, network, buffer, num_steps, num_envs, device,
            obs=current_obs, hidden=current_hidden, current_episode_rewards=current_ep_rewards
        )
        all_episode_rewards.extend(episode_rewards)

        # Compute advantages
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Update learning rate with linear decay
        progress = global_step / total_timesteps
        trainer.update_learning_rate(progress)

        # PPO update
        stats = trainer.update(buffer)
        num_updates += 1

        # Logging
        global_step += steps_per_update
        trainer.log_training_stats(stats, episode_rewards, global_step)

        if episode_rewards:
            reward_window = config['training']['reward_window_size']
            recent_rewards = all_episode_rewards[-reward_window:]
            print(
                f"Step {global_step:>8} | "
                f"Episodes: {len(all_episode_rewards):>5} | "
                f"Mean reward ({reward_window}): {sum(recent_rewards)/len(recent_rewards):>7.1f} | "
                f"Policy loss: {stats['policy_loss']:.4f} | "
                f"Value loss: {stats['value_loss']:.4f}"
            )

        # Save checkpoint
        if global_step % args.save_interval < steps_per_update:
            checkpoint_path = checkpoint_dir / f"model_{global_step}.pt"
            trainer.save_checkpoint(str(checkpoint_path), global_step, config, all_episode_rewards)
            print(f"Saved checkpoint to {checkpoint_path}")

            # Also save as latest
            latest_path = checkpoint_dir / "model_latest.pt"
            trainer.save_checkpoint(str(latest_path), global_step, config, all_episode_rewards)

        # Reset buffer
        buffer.reset()

    # Final save
    final_path = checkpoint_dir / "model_final.pt"
    trainer.save_checkpoint(str(final_path), global_step, config, all_episode_rewards)
    print(f"\nTraining complete! Final model saved to {final_path}")

    # Cleanup
    trainer.close()
    env.close()


if __name__ == "__main__":
    main()

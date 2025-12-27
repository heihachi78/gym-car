import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from models import ActorCriticLSTM, LSTMState
from .rollout_buffer import RolloutBuffer


class PPOTrainer:
    """
    PPO trainer for LSTM-based actor-critic network.

    Implements Proximal Policy Optimization with:
    - Clipped surrogate objective
    - Value function clipping (optional)
    - Entropy bonus
    - Generalized Advantage Estimation (GAE)
    """

    def __init__(
        self,
        network: ActorCriticLSTM,
        ppo_config: dict,
        device: torch.device = None,
        log_dir: str = "runs"
    ):
        """
        Args:
            network: ActorCriticLSTM network
            ppo_config: PPO configuration dict with all hyperparameters
            device: Torch device
            log_dir: Directory for TensorBoard logs
        """
        self.network = network
        self.ppo_config = ppo_config

        # Extract config values
        self.clip_epsilon = ppo_config['clip_epsilon']
        self.value_coef = ppo_config['value_coef']
        self.entropy_coef = ppo_config['entropy_coef']
        self.max_grad_norm = ppo_config['max_grad_norm']
        self.num_epochs = ppo_config['num_epochs']
        self.batch_size = ppo_config['batch_size']
        self.seq_len = ppo_config['seq_len']
        self.gamma = ppo_config['gamma']
        self.gae_lambda = ppo_config['gae_lambda']
        self.loss_epsilon = ppo_config['loss_epsilon']

        if device is None:
            device = next(network.parameters()).device
        self.device = device

        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=ppo_config['learning_rate'],
            eps=ppo_config['adam_eps']
        )
        self.initial_lr = ppo_config['learning_rate']

        # TensorBoard logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def update(self, buffer: RolloutBuffer) -> dict:
        """
        Perform PPO update using collected rollout data.

        Args:
            buffer: RolloutBuffer with collected experience

        Returns:
            Dictionary with training statistics
        """
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        num_updates = 0

        for epoch in range(self.num_epochs):
            for batch in buffer.get_batches(self.batch_size, self.seq_len):
                obs = batch['observations']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                old_values = batch.get('values')
                hidden_state = batch['hidden_state']
                masks = batch['masks']

                # Forward pass
                _, new_log_probs, entropy, values, _ = self.network.get_action_and_value(
                    obs, hidden_state, actions
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                policy_loss_unclipped = -advantages * ratio
                policy_loss_clipped = -advantages * clipped_ratio
                policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped)

                # Apply mask to exclude steps after episode end
                policy_loss = (policy_loss * masks).sum() / (masks.sum() + self.loss_epsilon)

                # Value loss
                value_loss = (values - returns) ** 2
                if old_values is not None:
                    values_clipped = old_values + torch.clamp(values - old_values, -self.clip_epsilon, self.clip_epsilon)
                    value_loss_clipped = (values_clipped - returns) ** 2
                    value_loss = torch.max(value_loss, value_loss_clipped)
                value_loss = (value_loss * masks).sum() / (masks.sum() + self.loss_epsilon)

                # Entropy loss (we want to maximize entropy)
                entropy_loss = -(entropy * masks).sum() / (masks.sum() + self.loss_epsilon)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean()
                    total_approx_kl += approx_kl.item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

        # Average statistics
        stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'approx_kl': total_approx_kl / num_updates,
        }

        return stats

    def update_learning_rate(self, progress: float):
        """
        Update learning rate with linear decay.

        Args:
            progress: Training progress from 0.0 to 1.0
        """
        lr = self.initial_lr * (1.0 - progress)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def log_training_stats(self, stats: dict, episode_rewards: list, step: int):
        """Log training statistics to TensorBoard."""
        self.writer.add_scalar('Loss/policy', stats['policy_loss'], step)
        self.writer.add_scalar('Loss/value', stats['value_loss'], step)
        self.writer.add_scalar('Loss/entropy', stats['entropy'], step)
        self.writer.add_scalar('Loss/approx_kl', stats['approx_kl'], step)
        # Log current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Train/learning_rate', current_lr, step)

        if episode_rewards:
            self.writer.add_scalar('Reward/mean', np.mean(episode_rewards), step)
            self.writer.add_scalar('Reward/max', np.max(episode_rewards), step)
            self.writer.add_scalar('Reward/min', np.min(episode_rewards), step)

    def save_checkpoint(
        self,
        path: str,
        step: int,
        config: dict,
        episode_rewards: list = None,
    ):
        """Save training checkpoint."""
        checkpoint = {
            'step': step,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_shape': self.network.obs_shape,
            'num_actions': self.network.num_actions,
            'hidden_size': self.network.hidden_size,
            'num_lstm_layers': self.network.num_lstm_layers,
            'config': config,  # Save full config
        }
        if episode_rewards:
            num_recent = self.ppo_config.get('num_recent_rewards', 100)
            checkpoint['recent_rewards'] = episode_rewards[-num_recent:]

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint. Returns the step number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['step']

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()

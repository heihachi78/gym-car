import numpy as np
import torch
from typing import Generator

from models import LSTMState


class RolloutBuffer:
    """
    Rollout buffer for storing experience during PPO training with LSTM.

    Supports vectorized environments: stores (buffer_size, num_envs, ...) arrays
    and provides batched sequences for training.
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_shape: tuple[int, int],
        hidden_size: int,
        num_lstm_layers: int,
        device: torch.device,
        gamma: float,
        gae_lambda: float,
        normalize_epsilon: float
    ):
        """
        Args:
            buffer_size: Number of steps to store per environment
            num_envs: Number of parallel environments
            obs_shape: (height, width) of observations
            hidden_size: LSTM hidden size
            num_lstm_layers: Number of LSTM layers
            device: Torch device
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            normalize_epsilon: Epsilon for advantage normalization
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_epsilon = normalize_epsilon

        # Storage arrays: (buffer_size, num_envs, ...)
        self.observations = np.zeros((buffer_size, num_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_envs), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)

        # Hidden states at each step: (buffer_size, num_lstm_layers, num_envs, hidden_size)
        self.hidden_h = np.zeros(
            (buffer_size, num_lstm_layers, num_envs, hidden_size), dtype=np.float32
        )
        self.hidden_c = np.zeros(
            (buffer_size, num_lstm_layers, num_envs, hidden_size), dtype=np.float32
        )

        # Computed after rollout: (buffer_size, num_envs)
        self.advantages = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_envs), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        hidden_state: LSTMState
    ):
        """
        Add a step from all environments to the buffer.

        Args:
            obs: Observations (num_envs, H, W)
            action: Actions (num_envs,)
            reward: Rewards (num_envs,)
            done: Done flags (num_envs,)
            log_prob: Log probabilities (num_envs,)
            value: Value estimates (num_envs,)
            hidden_state: LSTM hidden state (num_layers, num_envs, hidden_size)
        """
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done.astype(np.float32)
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value

        # Store hidden state
        self.hidden_h[self.pos] = hidden_state.h.cpu().numpy()
        self.hidden_c[self.pos] = hidden_state.c.cpu().numpy()

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,
        last_done: np.ndarray
    ):
        """
        Compute GAE advantages and returns for all environments after rollout.

        Args:
            last_value: Value estimates for last states (num_envs,)
            last_done: Whether last states were terminal (num_envs,)
        """
        last_gae = np.zeros(self.num_envs, dtype=np.float32)

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_done.astype(np.float32)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]

            delta = (
                self.rewards[step]
                + self.gamma * next_value * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(
        self,
        batch_size: int,
        seq_len: int
    ) -> Generator[dict, None, None]:
        """
        Generate batches of sequences for training.

        Data is flattened across environments before creating sequences.
        Total sequences = (buffer_size * num_envs) // seq_len

        Args:
            batch_size: Number of sequences per batch
            seq_len: Length of each sequence

        Yields:
            Dictionary containing batch tensors
        """
        # Flatten env dimension: (buffer_size, num_envs, ...) -> (buffer_size * num_envs, ...)
        flat_obs = self.observations.reshape(-1, *self.obs_shape)
        flat_actions = self.actions.reshape(-1)
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_dones = self.dones.reshape(-1)

        # Hidden states: (buffer_size, num_layers, num_envs, hidden) -> (buffer_size * num_envs, num_layers, hidden)
        flat_hidden_h = self.hidden_h.transpose(0, 2, 1, 3).reshape(-1, self.num_lstm_layers, self.hidden_size)
        flat_hidden_c = self.hidden_c.transpose(0, 2, 1, 3).reshape(-1, self.num_lstm_layers, self.hidden_size)

        total_steps = self.buffer_size * self.num_envs
        num_sequences = total_steps // seq_len
        indices = np.arange(num_sequences) * seq_len

        np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]

            # Collect sequences
            batch_obs = []
            batch_actions = []
            batch_log_probs = []
            batch_advantages = []
            batch_returns = []
            batch_hidden_h = []
            batch_hidden_c = []
            batch_masks = []

            for idx in batch_indices:
                end_idx = idx + seq_len

                batch_obs.append(flat_obs[idx:end_idx])
                batch_actions.append(flat_actions[idx:end_idx])
                batch_log_probs.append(flat_log_probs[idx:end_idx])
                batch_advantages.append(flat_advantages[idx:end_idx])
                batch_returns.append(flat_returns[idx:end_idx])

                # Hidden state at sequence start
                batch_hidden_h.append(flat_hidden_h[idx])
                batch_hidden_c.append(flat_hidden_c[idx])

                # Create mask (0 after episode end within sequence)
                mask = np.ones(seq_len, dtype=np.float32)
                for i in range(seq_len - 1):
                    if flat_dones[idx + i]:
                        mask[i + 1:] = 0
                        break
                batch_masks.append(mask)

            # Stack and convert to tensors
            batch_obs = torch.from_numpy(np.stack(batch_obs)).to(self.device)
            batch_actions = torch.from_numpy(np.stack(batch_actions)).to(self.device)
            batch_log_probs = torch.from_numpy(np.stack(batch_log_probs)).to(self.device)
            batch_advantages = torch.from_numpy(np.stack(batch_advantages)).to(self.device)
            batch_returns = torch.from_numpy(np.stack(batch_returns)).to(self.device)
            batch_masks = torch.from_numpy(np.stack(batch_masks)).to(self.device)

            # Hidden states: (batch, layers, hidden) -> (layers, batch, hidden)
            batch_hidden_h = np.stack(batch_hidden_h).transpose(1, 0, 2)
            batch_hidden_c = np.stack(batch_hidden_c).transpose(1, 0, 2)

            hidden_state = LSTMState(
                h=torch.from_numpy(batch_hidden_h).to(self.device).contiguous(),
                c=torch.from_numpy(batch_hidden_c).to(self.device).contiguous()
            )

            # Normalize advantages
            adv_mean = batch_advantages.mean()
            adv_std = batch_advantages.std() + self.normalize_epsilon
            batch_advantages = (batch_advantages - adv_mean) / adv_std

            yield {
                'observations': batch_obs,
                'actions': batch_actions,
                'old_log_probs': batch_log_probs,
                'advantages': batch_advantages,
                'returns': batch_returns,
                'hidden_state': hidden_state,
                'masks': batch_masks
            }

    def reset(self):
        """Reset buffer for next rollout."""
        self.pos = 0
        self.full = False

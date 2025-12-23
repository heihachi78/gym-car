import numpy as np
import torch

from models import ActorCriticLSTM, LSTMState


class LSTMPPOAgent:
    """
    PPO agent with LSTM-based actor-critic network.

    Handles action selection and hidden state management during rollouts.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int],
        num_actions: int,
        hidden_size: int = 256,
        num_lstm_layers: int = 1,
        device: str = "cuda"
    ):
        """
        Args:
            obs_shape: (height, width) of observation
            num_actions: Number of discrete actions
            hidden_size: LSTM hidden size
            num_lstm_layers: Number of LSTM layers
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        self.network = ActorCriticLSTM(
            obs_shape=obs_shape,
            num_actions=num_actions,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)

    @torch.no_grad()
    def get_action(
        self,
        obs: np.ndarray,
        hidden_state: LSTMState,
        deterministic: bool = False
    ) -> tuple[int, float, float, LSTMState]:
        """
        Select action given observation and hidden state.

        Args:
            obs: numpy array of shape (H, W)
            hidden_state: LSTMState

        Returns:
            action: Selected action (int)
            log_prob: Log probability of action (float)
            value: State value estimate (float)
            new_hidden: Updated LSTMState
        """
        # Prepare observation: (1, 1, H, W) - batch=1, seq_len=1
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)
        obs_tensor = obs_tensor.to(self.device)

        # Forward pass
        action, log_prob, _, value, new_hidden = self.network.get_action_and_value(
            obs_tensor, hidden_state, deterministic=deterministic
        )

        return (
            action.item(),
            log_prob.item(),
            value.item(),
            new_hidden
        )

    @torch.no_grad()
    def get_value(
        self,
        obs: np.ndarray,
        hidden_state: LSTMState
    ) -> float:
        """
        Get value estimate for observation.

        Args:
            obs: numpy array of shape (H, W)
            hidden_state: LSTMState

        Returns:
            value: State value estimate (float)
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)
        obs_tensor = obs_tensor.to(self.device)

        _, values, _ = self.network.forward(obs_tensor, hidden_state)

        return values.item()

    def get_initial_hidden(self) -> LSTMState:
        """Return zero-initialized hidden state for single environment."""
        return self.network.get_initial_hidden(batch_size=1, device=self.device)

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'obs_shape': self.network.obs_shape,
            'num_actions': self.network.num_actions,
            'hidden_size': self.hidden_size,
            'num_lstm_layers': self.num_lstm_layers,
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str = "cuda"
    ) -> "LSTMPPOAgent":
        """
        Load agent from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model on

        Returns:
            agent: Loaded LSTMPPOAgent
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        agent = cls(
            obs_shape=checkpoint['obs_shape'],
            num_actions=checkpoint['num_actions'],
            hidden_size=checkpoint['hidden_size'],
            num_lstm_layers=checkpoint['num_lstm_layers'],
            device=device
        )
        agent.network.load_state_dict(checkpoint['network_state_dict'])

        return agent

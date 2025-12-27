import torch
import torch.nn as nn

from .cnn_feature_extractor import CNNFeatureExtractor
from .lstm_state import LSTMState


class ActorCriticLSTM(nn.Module):
    """
    Combined actor-critic network with CNN feature extractor and LSTM.

    Architecture:
        Input (batch, seq_len, H, W)
            -> CNN Feature Extractor (per frame)
            -> LSTM (temporal processing)
            -> Actor head (policy logits)
            -> Critic head (value estimate)
    """

    def __init__(
        self,
        obs_shape: tuple[int, int] | tuple[int, int, int],
        num_actions: int,
        config: dict
    ):
        """
        Args:
            obs_shape: (height, width) or (height, width, channels) of observation
            num_actions: Number of discrete actions
            config: Full configuration dict with cnn, lstm, actor_critic sections
        """
        super().__init__()

        # Extract configs
        cnn_config = config['cnn']
        lstm_config = config['lstm']
        ac_config = config['actor_critic']

        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.hidden_size = lstm_config['hidden_size']
        self.num_lstm_layers = lstm_config['num_layers']

        # Determine input channels
        if len(obs_shape) == 2:
            H, W = obs_shape
            C = 1
        else:
            H, W, C = obs_shape

        self.input_channels = C
        self.spatial_shape = (H, W)
        self.max_channel_count = ac_config['max_channel_count']

        # CNN for spatial features
        self.cnn = CNNFeatureExtractor(
            input_shape=(C, H, W),
            cnn_config=cnn_config
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=cnn_config['output_size'],
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True
        )

        # Actor head (policy) - separate hidden layer to decouple from critic
        actor_hidden = ac_config['actor_hidden_size']
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_size, actor_hidden),
            nn.ReLU(),
            nn.Linear(actor_hidden, num_actions)
        )

        # Critic head (value) - separate hidden layer to decouple from actor
        critic_hidden = ac_config['critic_hidden_size']
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, 1)
        )

        # Initialize actor and critic heads
        self._init_heads(ac_config)

    def _init_heads(self, ac_config: dict):
        """Initialize actor/critic heads."""
        # Actor: hidden layer with ReLU gain, output layer with small gain
        nn.init.orthogonal_(self.actor[0].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.actor[0].bias)
        nn.init.orthogonal_(self.actor[2].weight, gain=ac_config['actor_output_gain'])
        nn.init.zeros_(self.actor[2].bias)

        # Critic: hidden layer with ReLU gain, output layer with gain=1
        nn.init.orthogonal_(self.critic[0].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.critic[0].bias)
        nn.init.orthogonal_(self.critic[2].weight, gain=ac_config['critic_output_gain'])
        nn.init.zeros_(self.critic[2].bias)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: LSTMState
    ) -> tuple[torch.Tensor, torch.Tensor, LSTMState]:
        """
        Forward pass through the network.

        Args:
            obs: Observations of shape (batch, seq_len, C, H, W) or (batch, seq_len, H, W)
            hidden_state: LSTMState with h and c tensors

        Returns:
            policy_logits: (batch, seq_len, num_actions)
            values: (batch, seq_len, 1)
            new_hidden: Updated LSTMState
        """
        if obs.dim() == 4:
            # Grayscale: (batch, seq_len, H, W) -> add channel dim
            batch_size, seq_len, H, W = obs.shape
            obs_flat = obs.reshape(batch_size * seq_len, 1, H, W)
        elif obs.dim() == 5 and obs.shape[2] <= self.max_channel_count:
            # Channel-first RGB: (batch, seq_len, C, H, W) - from train.py
            batch_size, seq_len, C, H, W = obs.shape
            obs_flat = obs.reshape(batch_size * seq_len, C, H, W)
        else:
            # Channel-last RGB: (batch, seq_len, H, W, C) - from rollout buffer
            batch_size, seq_len, H, W, C = obs.shape
            obs_flat = obs.permute(0, 1, 4, 2, 3).reshape(batch_size * seq_len, C, H, W)

        # Extract features: (batch * seq_len, hidden_size)
        features = self.cnn(obs_flat)

        # Reshape for LSTM: (batch, seq_len, hidden_size)
        features = features.reshape(batch_size, seq_len, -1)

        # LSTM forward
        lstm_out, (h_new, c_new) = self.lstm(
            features,
            hidden_state.as_tuple()
        )

        # Actor and critic heads
        policy_logits = self.actor(lstm_out)
        values = self.critic(lstm_out)

        new_hidden = LSTMState(h=h_new, c=c_new)

        return policy_logits, values, new_hidden

    def get_initial_hidden(
        self,
        batch_size: int,
        device: torch.device = None
    ) -> LSTMState:
        """Return zero-initialized hidden state."""
        if device is None:
            device = next(self.parameters()).device

        return LSTMState.zeros(
            batch_size=batch_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            device=device
        )

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden_state: LSTMState,
        action: torch.Tensor = None,
        deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, LSTMState]:
        """
        Get action, log probability, entropy, value, and new hidden state.

        Args:
            obs: Observations of shape (batch, seq_len, H, W)
            hidden_state: LSTMState
            action: Optional pre-selected actions (batch, seq_len)

        Returns:
            action: Selected actions (batch, seq_len)
            log_prob: Log probabilities (batch, seq_len)
            entropy: Policy entropy (batch, seq_len)
            value: State values (batch, seq_len)
            new_hidden: Updated LSTMState
        """
        policy_logits, values, new_hidden = self.forward(obs, hidden_state)

        # Create categorical distribution
        probs = torch.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, values.squeeze(-1), new_hidden

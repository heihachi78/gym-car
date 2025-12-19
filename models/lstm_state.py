from dataclasses import dataclass
import torch


@dataclass
class LSTMState:
    """
    Container for LSTM hidden state (h, c).

    Attributes:
        h: Hidden state tensor of shape (num_layers, batch, hidden_size)
        c: Cell state tensor of shape (num_layers, batch, hidden_size)
    """
    h: torch.Tensor
    c: torch.Tensor

    @classmethod
    def zeros(cls, batch_size: int, hidden_size: int, num_layers: int = 1,
              device: torch.device = None) -> "LSTMState":
        """Create zero-initialized hidden state."""
        if device is None:
            device = torch.device("cpu")

        return cls(
            h=torch.zeros(num_layers, batch_size, hidden_size, device=device),
            c=torch.zeros(num_layers, batch_size, hidden_size, device=device)
        )

    def detach(self) -> "LSTMState":
        """Detach hidden state from computation graph."""
        return LSTMState(
            h=self.h.detach(),
            c=self.c.detach()
        )

    def to(self, device: torch.device) -> "LSTMState":
        """Move hidden state to specified device."""
        return LSTMState(
            h=self.h.to(device),
            c=self.c.to(device)
        )

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return as tuple (h, c) for nn.LSTM input."""
        return (self.h, self.c)

    @classmethod
    def from_tuple(cls, hc: tuple[torch.Tensor, torch.Tensor]) -> "LSTMState":
        """Create LSTMState from tuple (h, c)."""
        return cls(h=hc[0], c=hc[1])

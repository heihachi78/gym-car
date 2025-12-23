import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    CNN for extracting spatial features from observations.

    Input: (batch, C, H, W) - C channels (1 for grayscale, 3 for RGB)
    Output: (batch, output_size) - flattened feature vector
    """

    def __init__(self, input_shape: tuple[int, int, int], output_size: int = 256):
        """
        Args:
            input_shape: (channels, height, width) e.g., (1, 80, 96)
            output_size: Size of output feature vector
        """
        super().__init__()

        channels, H, W = input_shape

        self.conv = nn.Sequential(
            # Conv1: input -> 64 channels
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Output: (64, 40, 48)

            # Conv2: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Output: (128, 20, 24)

            # Conv3: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Output: (256, 10, 12)

            # Conv4: 256 -> 512 channels
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Output: (512, 10, 12)

            nn.Flatten(),
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, channels, H, W)
            flat_size = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, output_size),
            nn.ReLU()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, C, H, W) with pixel values 0-255

        Returns:
            Feature tensor of shape (batch, output_size)
        """
        x = x / 255.0  # Normalize pixels to [0, 1]
        x = self.conv(x)
        x = self.fc(x)
        return x

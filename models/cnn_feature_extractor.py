import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    CNN for extracting spatial features from observations.

    Input: (batch, C, H, W) - C channels (1 for grayscale, 3 for RGB)
    Output: (batch, output_size) - flattened feature vector
    """

    def __init__(self, input_shape: tuple[int, int, int], cnn_config: dict):
        """
        Args:
            input_shape: (channels, height, width) e.g., (3, 96, 96)
            cnn_config: CNN configuration dict with layer parameters
        """
        super().__init__()

        channels, H, W = input_shape
        self.pixel_max_value = cnn_config['pixel_max_value']

        self.conv = nn.Sequential(
            # Conv1
            nn.Conv2d(
                channels,
                cnn_config['conv1_channels'],
                kernel_size=cnn_config['conv1_kernel'],
                stride=cnn_config['conv1_stride'],
                padding=cnn_config['conv1_padding']
            ),
            nn.BatchNorm2d(cnn_config['conv1_channels']),
            nn.ReLU(),

            # Conv2
            nn.Conv2d(
                cnn_config['conv1_channels'],
                cnn_config['conv2_channels'],
                kernel_size=cnn_config['conv2_kernel'],
                stride=cnn_config['conv2_stride'],
                padding=cnn_config['conv2_padding']
            ),
            nn.BatchNorm2d(cnn_config['conv2_channels']),
            nn.ReLU(),

            # Conv3
            nn.Conv2d(
                cnn_config['conv2_channels'],
                cnn_config['conv3_channels'],
                kernel_size=cnn_config['conv3_kernel'],
                stride=cnn_config['conv3_stride'],
                padding=cnn_config['conv3_padding']
            ),
            nn.BatchNorm2d(cnn_config['conv3_channels']),
            nn.ReLU(),

            # Conv4
            nn.Conv2d(
                cnn_config['conv3_channels'],
                cnn_config['conv4_channels'],
                kernel_size=cnn_config['conv4_kernel'],
                stride=cnn_config['conv4_stride'],
                padding=cnn_config['conv4_padding']
            ),
            nn.BatchNorm2d(cnn_config['conv4_channels']),
            nn.ReLU(),

            nn.Flatten(),
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, channels, H, W)
            flat_size = self.conv(dummy).shape[1]

        output_size = cnn_config['output_size']
        self.fc = nn.Sequential(
            nn.Linear(flat_size, output_size),
            nn.ReLU()
        )

        # Initialize weights
        self._init_weights(cnn_config['init_gain'])

    def _init_weights(self, init_gain: str):
        """Initialize weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain(init_gain))
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
        x = x / self.pixel_max_value  # Normalize pixels to [0, 1]
        x = self.conv(x)
        x = self.fc(x)
        return x

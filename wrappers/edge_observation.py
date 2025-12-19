from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import numpy as np
import cv2


class EdgeObservation(ObservationWrapper):
    """
    A Gymnasium observation wrapper that applies edge detection to image observations.

    Uses the Canny edge detection algorithm to extract edges from the image.
    Useful for highlighting structural features like road boundaries.

    Note: This wrapper converts the output to a binary edge image (0 or 255).
    The output is always 2D grayscale (H, W) regardless of input format.

    This wrapper only performs edge detection. For other operations, use:
    - GrayscaleObservation (gymnasium.wrappers) for grayscale conversion
    - CropObservation (crop_observation.py) for cropping
    - ResizeObservation (gymnasium.wrappers) for resizing
    - SharpenObservation (sharpen_observation.py) for sharpening
    - RenderObservation (render_observation.py) for visualization

    Args:
        env: The Gymnasium environment to wrap.
        low_threshold: Lower threshold for Canny edge detection (default: 50).
                       Edges with gradient below this are discarded.
        high_threshold: Upper threshold for Canny edge detection (default: 150).
                        Edges with gradient above this are kept as strong edges.

    Example:
        >>> env = gym.make("CarRacing-v3", render_mode="rgb_array")
        >>> env = GrayscaleObservation(env, keep_dim=True)  # (96, 96, 1)
        >>> env = CropObservation(env, height=80, width=96)  # (80, 96, 1)
        >>> env = ResizeObservation(env, (40, 48))  # (40, 48)
        >>> env = EdgeObservation(env, low_threshold=50, high_threshold=150)  # (40, 48)
        >>> env = RenderObservation(env, scale=10.0)  # visualization

    Attributes:
        low_threshold: Lower Canny threshold.
        high_threshold: Upper Canny threshold.
        observation_space: Updated to 2D grayscale (H, W).
    """

    def __init__(self, env, low_threshold=50, high_threshold=150):
        super().__init__(env)

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Get input dimensions
        input_shape = env.observation_space.shape
        if len(input_shape) == 2:
            height, width = input_shape
        elif len(input_shape) == 3:
            height, width, _ = input_shape
        else:
            raise ValueError(f"Unexpected observation space shape: {input_shape}")

        # Output is always 2D grayscale (edge map)
        self.observation_space = Box(
            low=0, high=255, shape=(height, width), dtype=np.uint8
        )

    def observation(self, observation):
        """
        Apply Canny edge detection to the observation.

        Args:
            observation: Image observation from the wrapped environment.

        Returns:
            Binary edge map with shape (H, W) and values 0 or 255.
        """
        # Convert to 2D grayscale if needed
        if observation.ndim == 3:
            if observation.shape[-1] == 1:
                # (H, W, 1) -> (H, W)
                img = observation[:, :, 0]
            else:
                # RGB (H, W, 3) -> grayscale (H, W)
                img = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        else:
            # Already 2D
            img = observation

        # Apply Canny edge detection
        edges = cv2.Canny(img, self.low_threshold, self.high_threshold)

        return edges

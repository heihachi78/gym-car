from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import numpy as np


class CropObservation(ObservationWrapper):
    """
    A Gymnasium observation wrapper that crops environment observations.

    Extracts a region from the top-left corner of the observation.
    Useful for removing UI elements like the dashboard in CarRacing.

    This wrapper only performs cropping. For other operations, use:
    - GrayscaleObservation (gymnasium.wrappers) for grayscale conversion
    - ResizeObservation (gymnasium.wrappers) for resizing
    - RenderObservation (render_observation.py) for visualization

    Supports:
    - RGB images: (H, W, 3)
    - Grayscale with channel dimension: (H, W, 1)
    - Grayscale without channel dimension: (H, W)

    Args:
        env: The Gymnasium environment to wrap.
        height: Height of the cropped region in pixels (default: 96).
        width: Width of the cropped region in pixels (default: 96).

    Example:
        >>> # Basic cropping
        >>> env = gym.make("CarRacing-v3", render_mode="rgb_array")
        >>> env = CropObservation(env, height=80, width=96)
        >>> obs, info = env.reset()  # obs.shape = (80, 96, 3)

        >>> # Full pipeline (recommended order)
        >>> env = gym.make("CarRacing-v3", render_mode="rgb_array")
        >>> env = GrayscaleObservation(env, keep_dim=True)  # (96, 96, 1)
        >>> env = CropObservation(env, height=80, width=96)  # (80, 96, 1)
        >>> env = ResizeObservation(env, (40, 48))  # (40, 48)
        >>> env = RenderObservation(env, scale=10.0)  # visualization

    Attributes:
        height: The crop height in pixels.
        width: The crop width in pixels.
        observation_space: Updated Box space reflecting the cropped observation shape.
    """

    def __init__(self, env, height=96, width=96):
        super().__init__(env)

        self.height = height
        self.width = width

        # Determine output shape based on input observation space
        input_shape = env.observation_space.shape
        if len(input_shape) == 2:
            output_shape = (height, width)
        elif len(input_shape) == 3:
            output_shape = (height, width, input_shape[2])
        else:
            raise ValueError(f"Unexpected observation space shape: {input_shape}")

        self.observation_space = Box(
            low=0, high=255, shape=output_shape, dtype=np.uint8
        )

    def observation(self, observation):
        """
        Crop the observation to the specified region from top-left corner.

        Args:
            observation: Raw observation from the wrapped environment.

        Returns:
            Cropped observation array.
        """
        if observation.ndim == 2:
            return observation[:self.height, :self.width]
        else:
            return observation[:self.height, :self.width, :]

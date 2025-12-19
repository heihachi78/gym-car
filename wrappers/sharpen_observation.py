from gymnasium import ObservationWrapper
import numpy as np
import cv2


class SharpenObservation(ObservationWrapper):
    """
    A Gymnasium observation wrapper that applies sharpening to image observations.

    Uses a convolution kernel to enhance edges and details in the image.
    Useful for making features more distinct for neural network processing.

    This wrapper only performs sharpening. For other operations, use:
    - GrayscaleObservation (gymnasium.wrappers) for grayscale conversion
    - CropObservation (crop_observation.py) for cropping
    - ResizeObservation (gymnasium.wrappers) for resizing
    - RenderObservation (render_observation.py) for visualization

    Supports:
    - RGB images: (H, W, 3)
    - Grayscale with channel dimension: (H, W, 1)
    - Grayscale without channel dimension: (H, W)

    Args:
        env: The Gymnasium environment to wrap.
        strength: Sharpening strength (default: 1.0).
                  Higher values increase the sharpening effect.
                  Values around 0.5-2.0 are typical.

    Example:
        >>> env = gym.make("CarRacing-v3", render_mode="rgb_array")
        >>> env = GrayscaleObservation(env, keep_dim=True)  # (96, 96, 1)
        >>> env = CropObservation(env, height=80, width=96)  # (80, 96, 1)
        >>> env = ResizeObservation(env, (40, 48))  # (40, 48)
        >>> env = SharpenObservation(env, strength=1.0)  # sharpen
        >>> env = RenderObservation(env, scale=10.0)  # visualization

    Attributes:
        strength: The sharpening strength multiplier.
        kernel: The sharpening convolution kernel.
        observation_space: Same as wrapped environment (unchanged).
    """

    def __init__(self, env, strength=1.0):
        super().__init__(env)

        self.strength = strength

        # Standard sharpening kernel, scaled by strength
        # Center value increases with strength, surrounding values become more negative
        self.kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

        # Adjust kernel based on strength
        # At strength=1.0, use standard kernel
        # At strength>1.0, increase sharpening effect
        # At strength<1.0, reduce sharpening effect
        if strength != 1.0:
            # Interpolate between identity kernel and sharpening kernel
            identity = np.array([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ], dtype=np.float32)
            self.kernel = identity + strength * (self.kernel - identity)

    def observation(self, observation):
        """
        Apply sharpening filter to the observation.

        Args:
            observation: Image observation from the wrapped environment.

        Returns:
            Sharpened observation array with same shape and dtype as input.
        """
        # Handle 2D grayscale (H, W)
        if observation.ndim == 2:
            sharpened = cv2.filter2D(observation, -1, self.kernel)
            return np.clip(sharpened, 0, 255).astype(np.uint8)

        # Handle 3D with single channel (H, W, 1)
        if observation.shape[-1] == 1:
            img = observation[:, :, 0]
            sharpened = cv2.filter2D(img, -1, self.kernel)
            return np.clip(sharpened, 0, 255).astype(np.uint8)[:, :, np.newaxis]

        # Handle RGB (H, W, 3)
        sharpened = cv2.filter2D(observation, -1, self.kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

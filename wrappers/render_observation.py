from gymnasium import Wrapper
import numpy as np
import pygame


class RenderObservation(Wrapper):
    """
    A Gymnasium wrapper that renders observations in a pygame window.

    This wrapper does NOT modify observations - it only provides visualization.
    Apply this as the outermost wrapper to render the final processed observation.

    Supports:
    - RGB images: (H, W, 3)
    - Grayscale with channel dimension: (H, W, 1)
    - Grayscale without channel dimension: (H, W)
    - Stacked frames: (stack_size, H, W) - rendered side-by-side

    Note: Gymnasium's ResizeObservation may drop the channel dimension for
    grayscale images, converting (H, W, 1) to (H, W). This wrapper handles
    both cases.

    Args:
        env: The Gymnasium environment to wrap.
        scale: Scale factor for the display window (default: 1.0).
               Enlarges the window for better visibility of small observations.

    Example:
        >>> env = gym.make("CarRacing-v3", render_mode="rgb_array")
        >>> env = GrayscaleObservation(env, keep_dim=True)  # (96, 96, 1)
        >>> env = CropObservation(env, height=80, width=96)  # (80, 96, 1)
        >>> env = ResizeObservation(env, (40, 48))  # (40, 48) - channel dropped
        >>> env = FrameStackObservation(env, stack_size=4)  # (4, 40, 48)
        >>> env = RenderObservation(env, scale=10.0)  # Apply last
        >>> obs, info = env.reset()
        >>> env.render()  # Shows 4 frames side-by-side

    Attributes:
        scale: The display window scale factor.
        observation_space: Same as wrapped environment (unchanged).
    """

    def __init__(self, env, scale=1.0):
        super().__init__(env)

        self.scale = scale
        self.screen = None
        self._last_obs = None
        self._is_stacked = False
        self._stack_size = 1

        # Get observation dimensions for rendering
        # Note: ResizeObservation may drop channel dim for grayscale (H, W, 1) -> (H, W)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 2:
            # 2D grayscale (H, W)
            self._obs_height, self._obs_width = obs_shape
        elif len(obs_shape) == 3:
            # Could be (H, W, C) or stacked frames (stack_size, H, W)
            # Heuristic: if first dim is small (<=16) and last dim is larger, it's stacked
            if obs_shape[0] <= 16 and obs_shape[1] > obs_shape[0] and obs_shape[2] > obs_shape[0]:
                # Stacked frames: (stack_size, H, W)
                self._is_stacked = True
                self._stack_size = obs_shape[0]
                self._obs_height, self._obs_width = obs_shape[1], obs_shape[2]
            else:
                # Regular (H, W, C)
                self._obs_height, self._obs_width = obs_shape[0], obs_shape[1]
        else:
            raise ValueError(f"Unexpected observation space shape: {obs_shape}")

    def step(self, action):
        """Execute action and cache observation for rendering."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and cache observation for rendering."""
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def render(self):
        """
        Render the current observation in a pygame window.

        Grayscale observations are automatically converted to RGB for display.
        Stacked frames are rendered side-by-side horizontally.

        Returns:
            The current observation array, or None if window was closed.
        """
        if self._last_obs is None:
            return None

        obs = self._last_obs

        # Calculate display window dimensions
        if self._is_stacked:
            # Stacked frames: render side-by-side
            total_width = self._obs_width * self._stack_size
            render_width = int(total_width * self.scale)
            render_height = int(self._obs_height * self.scale)
        else:
            render_width = int(self._obs_width * self.scale)
            render_height = int(self._obs_height * self.scale)

        # Initialize pygame window on first render call
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((render_width, render_height))
            pygame.display.set_caption("Observation")

        if self._is_stacked:
            # Stacked frames: (stack_size, H, W) -> concatenate horizontally
            frames = []
            for i in range(self._stack_size):
                frame = obs[i]  # (H, W)
                # Convert to RGB
                frame_3d = np.stack([frame] * 3, axis=-1)
                frames.append(frame_3d)
            # Concatenate horizontally: (H, W*stack_size, 3)
            obs_3d = np.concatenate(frames, axis=1)
        elif obs.ndim == 2:
            # 2D grayscale (H, W) -> RGB (H, W, 3)
            obs_3d = np.stack([obs] * 3, axis=-1)
        elif obs.shape[-1] == 1:
            # Grayscale with channel (H, W, 1) -> RGB (H, W, 3)
            obs_3d = np.concatenate([obs] * 3, axis=-1)
        else:
            # Already RGB (H, W, 3)
            obs_3d = obs

        # Create pygame surface
        # Pygame expects (W, H, 3) format, so transpose from (H, W, 3)
        surface = pygame.surfarray.make_surface(np.transpose(obs_3d, (1, 0, 2)))

        # Scale surface for display
        if self.scale != 1:
            surface = pygame.transform.scale(surface, (render_width, render_height))

        # Draw to screen
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Process pygame events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

        return self._last_obs

    def close(self):
        """Clean up pygame resources and close the wrapped environment."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        super().close()

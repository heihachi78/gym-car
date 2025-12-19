import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, NormalizeObservation
from gymnasium.vector import AsyncVectorEnv

from wrappers import CropObservation, SharpenObservation, EdgeObservation, RenderObservation


def get_normalize_wrapper(env: gym.Env) -> NormalizeObservation | None:
    """
    Find NormalizeObservation wrapper in the environment wrapper chain.

    Args:
        env: Gymnasium environment (possibly wrapped)

    Returns:
        NormalizeObservation wrapper if found, None otherwise
    """
    current = env
    while current is not None:
        if isinstance(current, NormalizeObservation):
            return current
        current = getattr(current, 'env', None)
    return None


def make_env(
    render_mode: str = "rgb_array",
    render_observation: bool = False,
    render_scale: float = 10.0,
    lap_complete_percent: float = 1.0,
    domain_randomize: bool = False,
    normalize: bool = True
) -> gym.Env:
    """
    Create CarRacing-v3 environment with observation pipeline optimized for LSTM agent.

    Pipeline:
        Raw (96x96x3) -> Crop (80x96x3) -> Normalize

    No FrameStack - LSTM handles temporal information.

    Args:
        render_mode: "rgb_array" for headless, "human" for display
        render_observation: Whether to add RenderObservation wrapper for visualization
        render_scale: Scale factor for RenderObservation
        lap_complete_percent: Fraction of track to complete for episode end
        domain_randomize: Whether to randomize track appearance
        normalize: Whether to apply NormalizeObservation (disable for inference)

    Returns:
        Configured Gymnasium environment
    """
    env = gym.make(
        "CarRacing-v3",
        render_mode=render_mode,
        lap_complete_percent=lap_complete_percent,
        domain_randomize=domain_randomize,
        continuous=False  # Discrete action space
    )

    # Spatial preprocessing (keep RGB for color information)
    env = CropObservation(env, height=80, width=96)
    #env = SharpenObservation(env, strength=1)
    #env = EdgeObservation(env, low_threshold=50, high_threshold=150)
    #env = ResizeObservation(env, (40, 48))

    # Normalize observation using running mean/std
    if normalize:
        env = NormalizeObservation(env)

    # Optional visualization
    if render_observation:
        env = RenderObservation(env, scale=render_scale)

    return env


def get_obs_shape() -> tuple[int, int, int]:
    """Return the observation shape after preprocessing (H, W, C)."""
    env = make_env(render_mode="rgb_array", normalize=False)
    shape = env.observation_space.shape
    env.close()
    return shape


def get_num_actions() -> int:
    """Return the number of discrete actions."""
    env = make_env(render_mode="rgb_array", normalize=False)
    n_actions = env.action_space.n
    env.close()
    return n_actions


def make_vec_env(
    num_envs: int = 8,
    render_mode: str = "rgb_array",
    normalize: bool = True,
    **kwargs
) -> AsyncVectorEnv:
    """
    Create vectorized CarRacing-v3 environments for parallel data collection.

    Args:
        num_envs: Number of parallel environments
        render_mode: Rendering mode (use "rgb_array" for training)
        normalize: Whether to apply observation normalization
        **kwargs: Additional arguments passed to make_env()

    Returns:
        AsyncVectorEnv with num_envs parallel environments (subprocess-based)
    """
    def _make_env():
        return make_env(render_mode=render_mode, normalize=normalize, **kwargs)

    return AsyncVectorEnv([_make_env for _ in range(num_envs)])

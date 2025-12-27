import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv


def make_env(
    render_mode: str = "rgb_array",
    lap_complete_percent: float = 1.0,
    domain_randomize: bool = False,
    continuous: bool = False,
) -> gym.Env:
    """
    Create CarRacing-v3 environment.

    No preprocessing - raw RGB (96x96x3) observations.
    Pixel normalization (/255.0) is done in the CNN forward pass.
    No FrameStack - LSTM handles temporal information.

    Args:
        render_mode: "rgb_array" for headless, "human" for display
        lap_complete_percent: Fraction of track to complete for episode end
        domain_randomize: Whether to randomize track appearance
        continuous: Whether to use continuous action space

    Returns:
        Configured Gymnasium environment
    """
    env = gym.make(
        "CarRacing-v3",
        render_mode=render_mode,
        lap_complete_percent=lap_complete_percent,
        domain_randomize=domain_randomize,
        continuous=continuous,
    )

    return env


def get_obs_shape() -> tuple[int, int, int]:
    """Return the observation shape (H, W, C)."""
    env = make_env(render_mode="rgb_array")
    shape = env.observation_space.shape
    env.close()
    return shape


def get_num_actions() -> int:
    """Return the number of discrete actions."""
    env = make_env(render_mode="rgb_array")
    n_actions = env.action_space.n
    env.close()
    return n_actions


def make_vec_env(
    num_envs: int = 8,
    render_mode: str = "rgb_array",
    **kwargs
) -> AsyncVectorEnv:
    """
    Create vectorized CarRacing-v3 environments for parallel data collection.

    Args:
        num_envs: Number of parallel environments
        render_mode: Rendering mode (use "rgb_array" for training)
        **kwargs: Additional arguments passed to make_env()

    Returns:
        AsyncVectorEnv with num_envs parallel environments (subprocess-based)
    """
    def _make_env():
        return make_env(render_mode=render_mode, **kwargs)

    return AsyncVectorEnv([_make_env for _ in range(num_envs)])

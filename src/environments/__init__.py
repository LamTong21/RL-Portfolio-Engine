"""Environment definitions and Gymnasium registry bindings."""

from gymnasium.envs.registration import register
from src.environments.portfolio_env import PortfolioEnv

register(
    id="PortfolioEnv-v0",
    entry_point="src.environments.portfolio_env:PortfolioEnv",
    max_episode_steps=1000,
)

__all__ = ["PortfolioEnv"]
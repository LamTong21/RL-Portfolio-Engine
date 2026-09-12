"""Unit tests verifying invariants and Gym interfaces for PortfolioEnv."""

import numpy as np
import pytest

from src.environments.portfolio_env import PortfolioEnv


@pytest.fixture
def synthetic_market_data() -> np.ndarray:
    """Generates synthetic log-returns for 3 assets across 150 timesteps."""
    np.random.seed(1337)
    return np.random.normal(loc=0.0002, scale=0.012, size=(150, 3)).astype(np.float32)


class TestPortfolioEnvironment:
    """Validates financial dynamics and execution integrity."""

    def test_observation_space_dimension(self, synthetic_market_data: np.ndarray) -> None:
        lookback = 25
        num_risky = synthetic_market_data.shape[1]
        num_total_assets = num_risky + 1  # Risky assets + cash

        env = PortfolioEnv(returns_data=synthetic_market_data, lookback_window=lookback)
        obs, _ = env.reset()

        expected_dim = (lookback * num_risky) + num_total_assets
        assert obs.shape == (expected_dim,)
        assert env.observation_space.shape == (expected_dim,)

    def test_action_weights_sum_to_one(self, synthetic_market_data: np.ndarray) -> None:
        env = PortfolioEnv(returns_data=synthetic_market_data, lookback_window=10)
        env.reset()

        # Execute arbitrary continuous actions (even with large unnormalized scales)
        arbitrary_actions = [
            np.array([10.0, -5.0, 2.0, 0.0]),
            np.array([-100.0, -100.0, -100.0, -100.0]),
            np.zeros(4),
        ]

        for act in arbitrary_actions:
            _, _, _, _, _ = env.step(act)
            # Weights must strictly adhere to the unit simplex
            assert np.isclose(np.sum(env.weights), 1.0, atol=1e-5)
            assert np.all(env.weights >= 0.0)

    def test_turnover_transaction_cost_deduction(self, synthetic_market_data: np.ndarray) -> None:
        cost_rate = 0.01  # 1% transaction fee to observe distinct friction
        initial_cash = 100_000.0

        # Market with 0 returns to isolate trading costs from price changes
        zero_returns = np.zeros((50, 2), dtype=np.float32)
        env = PortfolioEnv(
            returns_data=zero_returns,
            lookback_window=5,
            transaction_cost_pct=cost_rate,
            initial_balance=initial_cash,
        )
        env.reset()

        # Step: Shift from 100% Cash to 100% in Risky Asset 0
        # Target weight vector: [1.0, 0.0, 0.0] -> Turnover is |1-0| + |0-0| + |0-1| = 2.0
        rebalance_action = np.array([100.0, -100.0, -100.0])
        _, _, _, _, info = env.step(rebalance_action)

        expected_turnover = 2.0
        expected_cost = initial_cash * expected_turnover * cost_rate
        expected_nav = initial_cash - expected_cost

        assert np.isclose(info["turnover"], expected_turnover, atol=1e-3)
        assert np.isclose(info["portfolio_value"], expected_nav, atol=1e-2)

    def test_episode_termination_and_truncation(self) -> None:
        # Mini dataset to check end-of-data termination
        short_data = np.zeros((20, 2), dtype=np.float32)
        env = PortfolioEnv(returns_data=short_data, lookback_window=10)
        env.reset()

        terminated = False
        step_count = 0
        while not terminated:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            step_count += 1
            if step_count > 30:
                pytest.fail("Environment failed to terminate at dataset boundary.")

        assert terminated is True

    def test_drawdown_truncation_guard(self) -> None:
        # Crash market with -99% returns per period
        catastrophic_crash = np.full((50, 2), fill_value=-0.99, dtype=np.float32)
        env = PortfolioEnv(returns_data=catastrophic_crash, lookback_window=5)
        env.reset()

        # Allocate everything to risky assets
        all_in_risky = np.array([50.0, 50.0, -50.0])
        _, _, _, truncated, info = env.step(all_in_risky)

        # Truncation must trigger when NAV falls below 10% of initial balance
        assert truncated is True
        assert info["portfolio_value"] < 10_000.0
"""Custom Gymnasium Environment for Continuous Dynamic Portfolio Rebalancing.

Models a discrete-time financial market with N risky assets and 1 cash asset,
incorporating transaction frictions and continuous allocation updates.
"""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PortfolioEnv(gym.Env):
    """Dynamic Asset Allocation Environment adhering to Farama Gymnasium API."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        returns_data: np.ndarray,
        lookback_window: int = 30,
        transaction_cost_pct: float = 0.001,  # 10 bps
        risk_free_rate: float = 0.0,
        risk_aversion: float = 0.5,
        initial_balance: float = 100_000.0,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Args:
            returns_data: 2D array of shape (T, N) holding asset returns over time.
            lookback_window: Number of historical time steps in observation.
            transaction_cost_pct: Proportional cost applied to turnover rate.
            risk_free_rate: Per-step return on cash reserves.
            risk_aversion: Lambda penalty scaling realized portfolio variance.
            initial_balance: Starting portfolio net asset value (NAV).
            render_mode: None or 'human'.
        """
        super().__init__()

        self.returns_data = returns_data.astype(np.float32)
        self.num_steps_total, self.num_risky_assets = returns_data.shape
        self.num_assets = self.num_risky_assets + 1  # Assets + 1 Cash component

        self.lookback_window = lookback_window
        self.cost_pct = transaction_cost_pct
        self.rf_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.initial_balance = initial_balance
        self.render_mode = render_mode

        if self.num_steps_total <= self.lookback_window + 1:
            raise ValueError("Dataset size must exceed lookback_window length.")

        # Action: Target weights for (Risky Assets + Cash). Softmax applied internally.
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.num_assets,),
            dtype=np.float32,
        )

        # Observation: [Lookback returns matrix (lookback * N) + current weights (N+1)]
        obs_dim = (self.lookback_window * self.num_risky_assets) + self.num_assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Internal episode states
        self.current_step: int = 0
        self.weights: np.ndarray = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value: float = initial_balance
        self.history_portfolio_values: list[float] = []

    def _get_observation(self) -> np.ndarray:
        """Flattens recent market returns window and concatenates existing weights."""
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        history_slice = self.returns_data[start_idx:end_idx].flatten()
        return np.concatenate([history_slice, self.weights]).astype(np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Projects unconstrained action vectors onto the standard simplex."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the simulation to the initial condition."""
        super().reset(seed=seed)

        self.current_step = self.lookback_window
        # Initial portfolio allocation: 100% Cash
        self.weights = np.zeros(self.num_assets, dtype=np.float32)
        self.weights[-1] = 1.0

        self.portfolio_value = self.initial_balance
        self.history_portfolio_values = [self.portfolio_value]

        observation = self._get_observation()
        info: Dict[str, Any] = {"portfolio_value": self.portfolio_value}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies rebalancing action, steps the market forward, and calculates reward."""
        target_weights = self._softmax(action)

        # 1. Turnover & Transaction Cost
        turnover = np.sum(np.abs(target_weights - self.weights))
        transaction_cost = self.portfolio_value * turnover * self.cost_pct

        # 2. Asset Return Realization for Step t -> t+1
        market_returns = self.returns_data[self.current_step]
        period_returns = np.append(market_returns, self.rf_rate)

        # 3. Portfolio Growth Calculation
        net_wealth_pre_growth = self.portfolio_value - transaction_cost
        gross_return = float(np.dot(target_weights, period_returns))
        self.portfolio_value = float(net_wealth_pre_growth * (1.0 + gross_return))
        self.history_portfolio_values.append(self.portfolio_value)

        # Drifted weights after asset price fluctuations
        drifted_values = target_weights * (1.0 + period_returns)
        self.weights = drifted_values / np.sum(drifted_values)

        # 4. Reward Computation: Net return penalized by quadratic transaction & variance
        net_return = (self.portfolio_value - net_wealth_pre_growth) / net_wealth_pre_growth
        reward = float(np.log(max(1.0 + net_return, 1e-8)) - (self.cost_pct * turnover))

        # 5. Advancement and Termination Checks
        self.current_step += 1
        terminated = bool(self.current_step >= self.num_steps_total - 1)
        truncated = bool(self.portfolio_value <= (0.1 * self.initial_balance))  # 90% Drawdown guard

        next_obs = self._get_observation() if not terminated else np.zeros_like(self.observation_space.sample())
        info = {
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "net_return": net_return,
            "step": self.current_step,
        }

        return next_obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Simple text render of the portfolio status."""
        if self.render_mode == "human":
            print(
                f"Step: {self.current_step:04d} | "
                f"NAV: ${self.portfolio_value:,.2f} | "
                f"Cash: {self.weights[-1] * 100:.1f}% | "
                f"Weights: {np.round(self.weights[:-1], 3)}"
            )


if __name__ == "__main__":
    # Smoke test initialization with synthetic Gaussian market returns
    np.random.seed(42)
    synthetic_returns = np.random.normal(loc=0.0005, scale=0.015, size=(500, 4))

    env = PortfolioEnv(returns_data=synthetic_returns, lookback_window=20)
    obs, _ = env.reset()
    print(f"Environment instantiated. Observation shape: {obs.shape}")

    # Random step test
    dummy_action = env.action_space.sample()
    n_obs, rew, term, trunc, step_info = env.step(dummy_action)
    print(f"Sample step successful. Reward: {rew:.4f}, Next NAV: ${step_info['portfolio_value']:.2f}")
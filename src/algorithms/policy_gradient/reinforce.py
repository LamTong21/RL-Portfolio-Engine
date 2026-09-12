"""Monte Carlo Policy Gradient (REINFORCE) in PyTorch.

Reference:
    Williams, R. J. (1992). Simple statistical gradient-following algorithms for
    connectionist reinforcement learning. Machine learning, 8(3), 229-256.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.common.utils import compute_discounted_returns


class PolicyNetwork(nn.Module):
    """Parameterizes policy distribution \pi(a | s; \theta)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> Categorical:
        probs = self.net(state)
        return Categorical(probs)


class REINFORCEAgent:
    """REINFORCE Agent collecting episodic trajectories before batch parameter updates."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        device: Optional[torch.device] = None,
    ) -> None:
        self.gamma: float = gamma
        self.device: torch.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy: PolicyNetwork = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer: optim.Optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Episode trajectory buffers
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        """Samples an action from the categorical probability distribution."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        dist = self.policy(state_tensor)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        return int(action.item())

    def record_reward(self, reward: float) -> None:
        """Appends step reward to episode trace."""
        self.rewards.append(float(reward))

    def update(self) -> float:
        """Updates policy parameters using cumulative discounted trajectory returns."""
        returns = compute_discounted_returns(self.rewards, gamma=self.gamma, standardize=True)
        returns = returns.to(self.device)

        policy_loss: List[torch.Tensor] = []
        for log_prob, g_t in zip(self.log_probs, returns):
            # Objective: Maximize E[log \pi(a|s) * G_t] => Minimize -log \pi(a|s) * G_t
            policy_loss.append(-log_prob * g_t)

        self.optimizer.zero_grad()
        total_loss = torch.cat(policy_loss).sum()
        total_loss.backward()
        self.optimizer.step()

        # Clear trajectory buffers
        self.log_probs.clear()
        self.rewards.clear()

        return float(total_loss.item())


def train_reinforce(
    env_name: str = "CartPole-v1",
    num_episodes: int = 500,
    seed: int = 42,
) -> Tuple[REINFORCEAgent, List[float]]:
    """Training pipeline for REINFORCE."""
    import gymnasium as gym
    from src.common.utils import set_seed

    set_seed(seed)
    env = gym.make(env_name)
    agent = REINFORCEAgent(
        state_dim=env.observation_space.shape[0],  # type: ignore[attr-defined]
        action_dim=env.action_space.n,  # type: ignore[attr-defined]
    )

    episode_rewards: List[float] = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            state, reward, term, trunc, _ = env.step(action)
            agent.record_reward(reward)
            ep_reward += float(reward)
            done = term or trunc

        agent.update()
        episode_rewards.append(ep_reward)

    env.close()
    return agent, episode_rewards


if __name__ == "__main__":
    _, rewards = train_reinforce(num_episodes=300)
    print(f"REINFORCE Training Complete. Last 20 avg reward: {np.mean(rewards[-20:]):.2f}")
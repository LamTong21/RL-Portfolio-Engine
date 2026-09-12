"""Proximal Policy Optimization (PPO) with Clipped Objective & GAE for Continuous Actions.

Reference:
    Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
    arXiv preprint arXiv:1707.06347.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Actor-Critic network supporting continuous multivariate Gaussian policies."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        # Critic network: V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Actor network: \mu(s)
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Learnable log standard deviation vector
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward_critic(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)

    def get_action_dist(self, state: torch.Tensor) -> Normal:
        mean = self.actor_mean(state)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        return Normal(mean, std)


class PPOAgent:
    """PPO-Clip Agent utilizing Generalized Advantage Estimation (GAE)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        k_epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.gamma: float = gamma
        self.gae_lambda: float = gae_lambda
        self.clip_ratio: float = clip_ratio
        self.k_epochs: int = k_epochs
        self.batch_size: int = batch_size
        self.entropy_coef: float = entropy_coef
        self.value_coef: float = value_coef

        self.device: torch.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.ac: ActorCritic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer: optim.Optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Selects action and records corresponding log-probability and state value.

        Returns:
            action, log_prob, value
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.ac.get_action_dist(state_t)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.ac.forward_critic(state_t)

        return (
            action.cpu().numpy().squeeze(0),
            float(log_prob.item()),
            float(value.item()),
        )

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Generalized Advantage Estimators (GAE) and Target Returns."""
        advantages = []
        gae = 0.0
        extended_values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * extended_values[t + 1] * (1.0 - dones[t]) - extended_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)

        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = adv_tensor + torch.tensor(values, dtype=torch.float32, device=self.device)
        # Normalize advantages for training stability
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        return adv_tensor, returns_tensor

    def update(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        old_log_probs: List[float],
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Runs mini-batch surrogate objective optimization over K epochs."""
        states_t = torch.from_numpy(np.array(states)).float().to(self.device)
        actions_t = torch.from_numpy(np.array(actions)).float().to(self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)

        dataset_size = len(states)
        loss_tracker = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        for _ in range(self.k_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                # Forward passes
                dist = self.ac.get_action_dist(b_states)
                new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                values = self.ac.forward_critic(b_states).squeeze(-1)

                # Ratio r_t(\theta)
                ratios = torch.exp(new_log_probs - b_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value function squared loss
                critic_loss = nn.functional.mse_loss(values, b_returns)

                # Total loss
                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                self.optimizer.step()

                loss_tracker["actor_loss"] += float(actor_loss.item())
                loss_tracker["critic_loss"] += float(critic_loss.item())
                loss_tracker["entropy"] += float(entropy.item())

        return loss_tracker
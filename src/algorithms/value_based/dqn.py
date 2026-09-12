"""Deep Q-Network (DQN) with Target Network and Experience Replay in PyTorch.

Reference:
    Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
    Nature, 518(7540), 529-533.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.common.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """Multi-Layer Perceptron parameterizing action-value function Q(s, a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DQNAgent:
    """DQN Agent handling interaction, replay buffer sampling, and gradient descent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 200,
        tau: float = 1.0,  # tau=1.0: hard update; tau<1.0: Polyak soft update
        device: Optional[torch.device] = None,
    ) -> None:
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.gamma: float = gamma
        self.batch_size: int = batch_size
        self.target_update_freq: int = target_update_freq
        self.tau: float = tau

        self.epsilon: float = epsilon_start
        self.epsilon_end: float = epsilon_end
        self.epsilon_decay: float = epsilon_decay

        self.device: torch.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Q-Network and Target Network
        self.q_net: QNetwork = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net: QNetwork = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer: optim.Optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.criterion: nn.Module = nn.SmoothL1Loss()  # Huber loss

        self.replay_buffer: ReplayBuffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            action_dim=1,
            is_action_discrete=True,
        )
        self.total_steps: int = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Selects action via epsilon-greedy strategy."""
        if not evaluate and np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.action_dim))

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self) -> Optional[float]:
        """Samples a transition batch and executes one step of Adam optimization."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size, device=self.device)
        states = batch["states"]
        actions = batch["actions"].unsqueeze(1)
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Current Q-values: Q(s, a; \theta)
        current_q = self.q_net(states).gather(1, actions)

        # Target Q-values: r + \gamma * max_a' Q(s', a'; \theta^-) * (1 - done)
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q_max

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents gradient explosion in deep RL
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_steps += 1
        self._update_target_network()
        return float(loss.item())

    def _update_target_network(self) -> None:
        """Updates target network weights via hard or soft Polyak averaging."""
        if self.tau < 1.0:
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        elif self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self) -> None:
        """Decays epsilon after each full episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_dqn(
    env_name: str = "CartPole-v1",
    num_episodes: int = 400,
    seed: int = 42,
) -> Tuple[DQNAgent, List[float]]:
    """Quickstart training routine on Gymnasium environments."""
    import gymnasium as gym
    from src.common.utils import set_seed

    set_seed(seed)
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]  # type: ignore[attr-defined]
    action_dim = env.action_space.n  # type: ignore[attr-defined]

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    episode_rewards: List[float] = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            ep_reward += float(reward)

        agent.decay_epsilon()
        episode_rewards.append(ep_reward)

    env.close()
    return agent, episode_rewards


if __name__ == "__main__":
    trained_agent, rewards = train_dqn(num_episodes=250)
    print(f"DQN Training Complete. Last 20 avg reward: {np.mean(rewards[-20:]):.2f}")
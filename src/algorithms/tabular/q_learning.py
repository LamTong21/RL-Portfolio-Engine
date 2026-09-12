"""Off-Policy Temporal Difference Control: Q-Learning.

Reference:
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class QLearningAgent:
    """Tabular Q-Learning Agent implemented in NumPy."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ) -> None:
        self.num_states: int = num_states
        self.num_actions: int = num_actions
        self.lr: float = learning_rate
        self.gamma: float = gamma
        self.epsilon: float = epsilon_start
        self.epsilon_end: float = epsilon_end
        self.epsilon_decay: float = epsilon_decay

        self.rng: np.random.Generator = np.random.default_rng(seed)
        # Initialize Q-table with zeros
        self.q_table: np.ndarray = np.zeros((num_states, num_actions), dtype=np.float64)

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """Selects an action using an epsilon-greedy policy.

        Args:
            state: Discrete state index.
            evaluate: If True, executes pure greedy exploitation.
        """
        if not evaluate and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))

        # Tie-breaking randomly among max values
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """Performs off-policy Bellman Optimality update.

        Returns:
            td_error: Scalar temporal difference error for diagnostics.
        """
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        """Decays exploration parameter epsilon monotonically."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Saves Q-table to disk."""
        np.save(filepath, self.q_table)

    def load(self, filepath: str) -> None:
        """Loads Q-table from disk."""
        self.q_table = np.load(filepath)


def train_q_learning(
    env_name: str = "CliffWalking-v0",
    num_episodes: int = 500,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    seed: int = 42,
) -> Tuple[QLearningAgent, List[float]]:
    """Convenience pipeline for training Q-Learning on discrete environments."""
    import gymnasium as gym

    env = gym.make(env_name)
    agent = QLearningAgent(
        num_states=env.observation_space.n,  # type: ignore[attr-defined]
        num_actions=env.action_space.n,  # type: ignore[attr-defined]
        learning_rate=learning_rate,
        gamma=gamma,
        seed=seed,
    )

    episode_rewards: List[float] = []

    for _ in range(num_episodes):
        state, _ = env.reset(seed=seed)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, float(reward), next_state, done)
            state = next_state
            total_reward += float(reward)

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

    env.close()
    return agent, episode_rewards


if __name__ == "__main__":
    trained_agent, rewards = train_q_learning(num_episodes=300)
    print(f"Q-Learning training complete. Mean reward (last 50 eps): {np.mean(rewards[-50:]):.2f}")
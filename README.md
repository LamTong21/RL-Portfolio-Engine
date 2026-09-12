# Reinforcement Learning: Foundations, Implementations & Financial Applications

```
![Python 3.10+](https://www.python.org/downloads/)
![PyTorch](https://pytorch.org/)
![Gymnasium](https://gymnasium.farama.org/)
![License: MIT](https://opensource.org/licenses/MIT) 
```

A research-oriented repository demonstrating core Reinforcement Learning (RL) concepts, ranging from classical tabular methods implemented from scratch in NumPy to modern Deep RL architectures (DQN, PPO) implemented in PyTorch. The repository culminates in a custom financial environment modeling continuous dynamic portfolio allocation under real-world market frictions.

---

## 📌 Key Highlights

- **From-Scratch Implementations:** Tabular Q-Learning, SARSA, and DQN written without high-level wrappers (e.g., Stable-Baselines3) to demonstrate mathematical and algorithmic understanding.
- **Mathematical Rigor:** Clear documentation of Bellman Equations, convergence properties, policy gradient derivations, and advantage estimation.
- **Custom Financial Environment:** An OpenAI/Farama `Gymnasium`-compatible environment simulating continuous portfolio rebalancing with proportional transaction costs and slippage.
- **Reproducibility & Engineering Standards:** Fully typed codebases (`mypy`), modular architecture, hyperparameter management via YAML, and CI testing.

---

## 📐 Mathematical Overview

### 1. Markov Decision Process (MDP) & Bellman Optimality

Every agent interacts within a formal MDP framework defined by $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

$$V^*(s) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' | s, a) V^*(s') \right]$$

$$Q^*(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' | s, a) \max_{a' \in \mathcal{A}} Q^*(s', a')$$

### 2. Policy Gradient Theorem

For continuous action spaces, parameterized policy updates follow:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \hat{A}_t \right]$$

*(Full LaTeX proofs and mathematical breakdowns are available in `docs/math_derivations.md`).*

---

## 🗂 Project Architecture & Implementations

| Algorithm | Type | State / Action Space | Framework | Reference Target |
| :--- | :--- | :--- | :--- | :--- |
| **Q-Learning** | Model-Free TD Control | Discrete / Discrete | NumPy | `FrozenLake-v1`, `CliffWalking-v0` |
| **SARSA** | On-Policy TD Control | Discrete / Discrete | NumPy | `CliffWalking-v0` |
| **Deep Q-Network (DQN)** | Value-Based Deep RL | Continuous / Discrete | PyTorch | `CartPole-v1`, `LunarLander-v2` |
| **PPO (Clipped)** | Policy Gradient | Continuous / Continuous | PyTorch | `Custom PortfolioEnv-v0` |

---

## 📈 Featured Case Study: Continuous Dynamic Portfolio Allocation

An implementation of a custom `gymnasium.Env` designed to solve the dynamic multi-asset rebalancing problem:

- **State Space $\mathcal{S}_t$:** Rolling normalized log-returns, momentum indicators, realized volatility, and current portfolio weight vector $w_{t-1}$.
- **Action Space $\mathcal{A}_t$:** Target portfolio allocation vector $w_t \in \Delta^n$ subject to $\sum_{i=1}^n w_{i,t} = 1$ and $w_{i,t} \ge 0$ (Simplex projection via Softmax or Dirichlet parameterization).
- **Reward Formulation:** Differential Sharpe ratio or net return penalized by quadratic transaction costs:
    
    $$R_t = \sum_{i=1}^n w_{i,t} r_{i,t} - c \sum_{i=1}^n |w_{i,t} - w_{i,t}^+| - \lambda \sigma_p^2$$
    
    where $c$ is the transaction fee and $\lambda$ is the risk-aversion parameter.
    

---

## 🚀 Quickstart

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

```
git clone https://github.com/your-username/reinforcement-learning-foundations.git
cd reinforcement-learning-foundations
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running an Agent

Train a Deep Q-Network on `CartPole-v1`:

```
python -m src.algorithms.value_based.dqn --config configs/dqn_cartpole.yaml
```

Run backtesting on the Financial Portfolio environment:

```
python -m src.algorithms.policy_gradient.ppo --config configs/ppo_portfolio.yaml --eval
```

## 🧪 Testing & Verification

Unit tests cover Bellman error convergence, replay buffer memory constraints, and environment step invariances:

```
pytest tests/ -v
```

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
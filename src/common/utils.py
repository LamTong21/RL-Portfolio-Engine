"""General numerical utilities, seeding, and plotting helpers."""

import os
import random
from typing import List, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Sets random seeds across standard Python, NumPy, and PyTorch backends.
    
    Args:
        seed: Target integer seed.
        deterministic: If True, forces cuDNN backend into deterministic mode.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon acceleration backend
        torch.mps.manual_seed(seed)


def moving_average(values: Sequence[float], window_size: int = 50) -> np.ndarray:
    """Computes simple moving average using cumulative sums."""
    if len(values) < window_size:
        return np.array(values, dtype=np.float32)
    cumsum = np.cumsum(np.insert(values, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


def compute_discounted_returns(
    rewards: Sequence[float], gamma: float = 0.99, standardize: bool = True
) -> torch.Tensor:
    """Computes discounted cumulative returns: G_t = sum_{k=0}^{T-t-1} gamma^k * R_{t+k+1}.
    
    Args:
        rewards: Sequence of scalar immediate rewards.
        gamma: Discount factor in range (0, 1].
        standardize: Whether to normalize returns to mean 0 and std 1.
    """
    discounted_returns: List[float] = []
    running_add = 0.0
    for r in reversed(rewards):
        running_add = r + gamma * running_add
        discounted_returns.insert(0, running_add)

    returns = torch.tensor(discounted_returns, dtype=torch.float32)
    if standardize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def plot_learning_curve(
    rewards: Sequence[float],
    losses: Optional[Sequence[float]] = None,
    window_size: int = 50,
    save_path: Optional[str] = "docs/images/learning_curve.png",
    title: str = "Training Convergence Diagnostics",
) -> None:
    """Generates dual-axis or subpanel diagnostic learning curves."""
    num_plots = 2 if losses is not None and len(losses) > 0 else 1
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=False)
    
    if num_plots == 1:
        axes = [axes]

    # Episode Rewards plot
    ax_rew = axes[0]
    ax_rew.plot(rewards, alpha=0.3, label="Raw Episode Reward", color="dodgerblue")
    if len(rewards) >= window_size:
        ma_rew = moving_average(rewards, window_size)
        ax_rew.plot(
            range(window_size - 1, len(rewards)),
            ma_rew,
            label=f"SMA ({window_size} eps)",
            color="royalblue",
            linewidth=2.0,
        )
    ax_rew.set_title(title, fontsize=12, fontweight="bold")
    ax_rew.set_ylabel("Reward / Return", fontsize=10)
    ax_rew.set_xlabel("Episode", fontsize=10)
    ax_rew.grid(True, linestyle="--", alpha=0.6)
    ax_rew.legend(loc="upper left")

    # Training Loss plot
    if num_plots == 2 and losses is not None:
        ax_loss = axes[1]
        ax_loss.plot(losses, alpha=0.4, label="Loss Step", color="crimson")
        if len(losses) >= window_size:
            ma_loss = moving_average(losses, window_size)
            ax_loss.plot(
                range(window_size - 1, len(losses)),
                ma_loss,
                label=f"SMA Loss ({window_size} steps)",
                color="darkred",
                linewidth=1.8,
            )
        ax_loss.set_ylabel("Loss Magnitude", fontsize=10)
        ax_loss.set_xlabel("Optimization Step", fontsize=10)
        ax_loss.set_yscale("log")
        ax_loss.grid(True, linestyle="--", alpha=0.6)
        ax_loss.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
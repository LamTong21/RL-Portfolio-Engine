"""Experience Replay mechanisms for Value-Based and Off-Policy RL algorithms."""

from typing import Any, Dict, NamedTuple, Tuple
import numpy as np
import torch


class Transition(NamedTuple):
    state: np.ndarray
    action: int | float | np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Standard Uniform Experience Replay Buffer implemented via pre-allocated NumPy arrays."""

    def __init__(
        self,
        capacity: int,
        state_dim: Tuple[int, ...] | int,
        action_dim: int = 1,
        is_action_discrete: bool = True,
    ) -> None:
        self.capacity: int = int(capacity)
        self.state_dim: Tuple[int, ...] = (
            (state_dim,) if isinstance(state_dim, int) else tuple(state_dim)
        )
        self.action_dim: int = action_dim
        self.is_action_discrete: bool = is_action_discrete

        self.ptr: int = 0
        self.size: int = 0

        # Pre-allocate contiguous memory blocks
        self.states: np.ndarray = np.zeros((self.capacity, *self.state_dim), dtype=np.float32)
        self.next_states: np.ndarray = np.zeros((self.capacity, *self.state_dim), dtype=np.float32)
        
        action_dtype = np.int64 if is_action_discrete else np.float32
        action_shape = (self.capacity,) if is_action_discrete else (self.capacity, self.action_dim)
        self.actions: np.ndarray = np.zeros(action_shape, dtype=action_dtype)
        
        self.rewards: np.ndarray = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones: np.ndarray = np.zeros((self.capacity, 1), dtype=np.bool_)

    def push(
        self,
        state: np.ndarray,
        action: int | float | np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Stores an environment transition tuple."""
        self.states[self.ptr] = np.asarray(state, dtype=np.float32)
        self.next_states[self.ptr] = np.asarray(next_state, dtype=np.float32)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = bool(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> Dict[str, torch.Tensor]:
        """Samples a uniform random batch of transitions converted to PyTorch Tensors."""
        if self.size < batch_size:
            raise ValueError(f"Insufficient samples. Current size: {self.size}, Required: {batch_size}")

        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": torch.from_numpy(self.states[idxs]).to(device),
            "actions": torch.from_numpy(self.actions[idxs]).to(device),
            "rewards": torch.from_numpy(self.rewards[idxs]).to(device),
            "next_states": torch.from_numpy(self.next_states[idxs]).to(device),
            "dones": torch.from_numpy(self.dones[idxs].astype(np.float32)).to(device),
        }

    def __len__(self) -> int:
        return self.size


class SumTree:
    """Binary Sum Tree data structure for O(log N) priority updates and sampling.
    
    Leaves store priorities p_i > 0, while internal nodes store intermediate sums.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.tree: np.ndarray = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_ptr: int = 0
        self.size: int = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Recursively update parent nodes."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, tree_idx: int, priority: float) -> None:
        """Update leaf node priority and propagate changes upwards."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority: float) -> int:
        """Add new priority at leaf node pointing to internal data index."""
        tree_idx = self.data_ptr + self.capacity - 1
        self.update(tree_idx, priority)
        
        assigned_data_idx = self.data_ptr
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return assigned_data_idx

    def get_leaf(self, value: float) -> Tuple[int, float, int]:
        """Traverse tree to find leaf index corresponding to prefix sum value.
        
        Returns:
            tree_idx: Node index in the binary tree array.
            priority: Assigned priority value.
            data_idx: Corresponding index in data buffers.
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            if value <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                value -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, float(self.tree[leaf_idx]), data_idx

    @property
    def total_priority(self) -> float:
        """Returns the sum of all priorities (root node)."""
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """Returns maximum priority stored in active leaves."""
        leaves = self.tree[self.capacity - 1 : self.capacity - 1 + self.size]
        return float(np.max(leaves)) if self.size > 0 else 1.0


SegmentTree = SumTree


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay (PER) using Proportional Prioritization.
    
    Reference:
        Schaul et al. (2015): https://arxiv.org/abs/1511.05952
    """

    def __init__(
        self,
        capacity: int,
        state_dim: Tuple[int, ...] | int,
        action_dim: int = 1,
        is_action_discrete: bool = True,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(capacity, state_dim, action_dim, is_action_discrete)
        self.tree = SumTree(self.capacity)
        self.alpha: float = alpha
        self.beta: float = beta_start
        self.beta_start: float = beta_start
        self.beta_frames: int = beta_frames
        self.eps: float = eps
        self.frame: int = 1

    def push(
        self,
        state: np.ndarray,
        action: int | float | np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Pushes transition and assigns max existing priority for optimistic exploration."""
        max_p = self.tree.max_priority
        if max_p <= 0:
            max_p = 1.0

        idx = self.tree.data_ptr
        super().push(state, action, reward, next_state, done)
        self.tree.add(max_p ** self.alpha)

    def sample(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:
        """Samples a priority-weighted batch with Importance Sampling (IS) weights."""
        if self.tree.size < batch_size:
            raise ValueError(f"Buffer underflow: current size {self.tree.size} < batch {batch_size}")

        # Linearly anneal beta to 1.0
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        b_idxs = np.empty((batch_size,), dtype=np.int32)
        b_tree_idxs = np.empty((batch_size,), dtype=np.int32)
        is_weights = np.empty((batch_size, 1), dtype=np.float32)

        p_total = self.tree.total_priority
        segment = p_total / batch_size

        # Minimum probability across buffer for normalisation
        leaves = self.tree.tree[self.capacity - 1 : self.capacity - 1 + self.tree.size]
        min_prob = np.min(leaves[leaves > 0]) / p_total
        max_weight = (self.tree.size * min_prob) ** (-self.beta)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            tree_idx, priority, data_idx = self.tree.get_leaf(s)

            # Defensive clamp for floating point boundary cases
            data_idx = min(data_idx, self.tree.size - 1)
            b_idxs[i] = data_idx
            b_tree_idxs[i] = tree_idx

            prob = priority / p_total
            is_weights[i] = ((self.tree.size * prob) ** (-self.beta)) / max_weight

        return {
            "states": torch.from_numpy(self.states[b_idxs]).to(device),
            "actions": torch.from_numpy(self.actions[b_idxs]).to(device),
            "rewards": torch.from_numpy(self.rewards[b_idxs]).to(device),
            "next_states": torch.from_numpy(self.next_states[b_idxs]).to(device),
            "dones": torch.from_numpy(self.dones[b_idxs].astype(np.float32)).to(device),
            "tree_indices": b_tree_idxs,
            "weights": torch.from_numpy(is_weights).to(device),
        }

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update transition priorities via |TD-error| + eps."""
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(tree_idx, priority)
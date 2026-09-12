"""Unit tests for Experience Replay Buffers and SumTree data structures."""

import numpy as np
import pytest
import torch

from src.common.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer, SumTree


class TestReplayBuffer:
    """Validates standard circular ReplayBuffer properties."""

    def test_buffer_push_and_capacity(self) -> None:
        capacity = 5
        state_dim = 4
        buffer = ReplayBuffer(capacity=capacity, state_dim=state_dim, is_action_discrete=True)

        for i in range(7):
            s = np.full((state_dim,), fill_value=i, dtype=np.float32)
            buffer.push(state=s, action=i % 2, reward=float(i), next_state=s, done=False)

        assert len(buffer) == capacity
        # Buffer has wrapped around; first two slots should contain items 5 and 6
        assert buffer.states[0, 0] == 5.0
        assert buffer.states[1, 0] == 6.0

    def test_sample_shapes_and_types(self) -> None:
        buffer = ReplayBuffer(capacity=50, state_dim=8, is_action_discrete=True)
        for _ in range(30):
            buffer.push(
                state=np.random.randn(8).astype(np.float32),
                action=1,
                reward=1.0,
                next_state=np.random.randn(8).astype(np.float32),
                done=False,
            )

        batch_size = 16
        batch = buffer.sample(batch_size=batch_size)

        assert isinstance(batch["states"], torch.Tensor)
        assert batch["states"].shape == (batch_size, 8)
        assert batch["actions"].shape == (batch_size,)
        assert batch["rewards"].shape == (batch_size, 1)
        assert batch["dones"].shape == (batch_size, 1)

    def test_sample_insufficient_data_raises(self) -> None:
        buffer = ReplayBuffer(capacity=10, state_dim=2)
        buffer.push(np.zeros(2), 0, 0.0, np.zeros(2), False)
        with pytest.raises(ValueError, match="Insufficient samples"):
            buffer.sample(batch_size=5)


class TestSumTree:
    """Validates binary SumTree operations and mathematical invariants."""

    def test_sumtree_root_total(self) -> None:
        capacity = 4
        tree = SumTree(capacity=capacity)
        priorities = [1.5, 2.5, 3.0, 1.0]

        for p in priorities:
            tree.add(p)

        assert np.isclose(tree.total_priority, sum(priorities))

    def test_sumtree_update(self) -> None:
        tree = SumTree(capacity=4)
        for p in [1.0, 2.0, 3.0, 4.0]:
            tree.add(p)

        # Leaf index for first data element is (capacity - 1) = 3
        first_leaf_idx = 3
        tree.update(first_leaf_idx, 5.0)  # Changed from 1.0 -> 5.0 (+4.0)
        assert np.isclose(tree.total_priority, 14.0)

    def test_sumtree_get_leaf_bounds(self) -> None:
        tree = SumTree(capacity=4)
        for p in [1.0, 2.0, 3.0, 4.0]:
            tree.add(p)

        # Query values within range (0, 10.0]
        _, prio_1, data_idx_1 = tree.get_leaf(0.5)
        assert data_idx_1 == 0
        assert np.isclose(prio_1, 1.0)

        _, prio_end, data_idx_end = tree.get_leaf(9.5)
        assert data_idx_end == 3
        assert np.isclose(prio_end, 4.0)


class TestPrioritizedReplayBuffer:
    """Validates Proportional Prioritized Experience Replay behavior."""

    def test_per_sampling_and_is_weights(self) -> None:
        buffer = PrioritizedReplayBuffer(
            capacity=20,
            state_dim=3,
            alpha=0.6,
            beta_start=0.4,
        )

        for i in range(15):
            buffer.push(
                state=np.ones(3, dtype=np.float32) * i,
                action=0,
                reward=float(i),
                next_state=np.ones(3, dtype=np.float32) * (i + 1),
                done=False,
            )

        batch = buffer.sample(batch_size=8)
        assert "weights" in batch
        assert "tree_indices" in batch
        assert batch["weights"].shape == (8, 1)
        # IS weights normalized by max_weight must be bounded by 1.0
        assert torch.all(batch["weights"] <= 1.0 + 1e-6)

    def test_priority_update_changes_distribution(self) -> None:
        buffer = PrioritizedReplayBuffer(capacity=10, state_dim=2)
        for _ in range(5):
            buffer.push(np.zeros(2), 0, 1.0, np.zeros(2), False)

        batch = buffer.sample(batch_size=3)
        tree_idxs = batch["tree_indices"]

        # Assign high TD error to sampled transitions
        large_td_errors = np.array([100.0, 100.0, 100.0])
        initial_total = buffer.tree.total_priority
        buffer.update_priorities(tree_idxs, large_td_errors)

        assert buffer.tree.total_priority > initial_total
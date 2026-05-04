"""Tests for models/ppo_agent.py"""

import numpy as np
import torch

from models.ppo_agent import ActorCritic, PPOTrainer, RackEnv
from utils.retail_physics import NUM_SHELVES


def _make_products(n=10, seed=42):
    """Create a small products array: (n, 4) with [price, margin, sales, width]."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(1, 10, n),    # price
        rng.uniform(10, 50, n),   # margin
        rng.integers(10, 200, n), # sales
        rng.uniform(5, 30, n),    # width
    ]).astype(np.float32)


class TestRackEnv:
    def test_reset_returns_correct_shape(self):
        products = _make_products(8)
        env = RackEnv(products, max_steps=10)
        state = env.reset()
        assert state.shape == (8, 5)

    def test_shelf_levels_in_range(self):
        env = RackEnv(_make_products(10), max_steps=5)
        env.reset()
        assert all(1 <= s <= NUM_SHELVES for s in env.shelf_levels)

    def test_step_returns_correct_tuple(self):
        env = RackEnv(_make_products(5), max_steps=10)
        env.reset()
        state, reward, done, info = env.step((0, 1))
        assert state.shape == (5, 5)
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(done, bool)
        assert "profit" in info

    def test_step_swaps_shelves(self):
        env = RackEnv(_make_products(5), max_steps=10)
        env.reset()
        shelf_a = env.shelf_levels[0]
        shelf_b = env.shelf_levels[1]
        env.step((0, 1))
        assert env.shelf_levels[0] == shelf_b
        assert env.shelf_levels[1] == shelf_a

    def test_done_after_max_steps(self):
        env = RackEnv(_make_products(5), max_steps=3)
        env.reset()
        for i in range(3):
            _, _, done, _ = env.step((0, 1))
        assert done

    def test_not_done_before_max_steps(self):
        env = RackEnv(_make_products(5), max_steps=10)
        env.reset()
        _, _, done, _ = env.step((0, 1))
        assert not done

    def test_state_dim(self):
        env = RackEnv(_make_products(5))
        assert env.state_dim == 5

    def test_n_products(self):
        env = RackEnv(_make_products(7))
        assert env.n_products == 7

    def test_total_profit_positive(self):
        env = RackEnv(_make_products(5))
        env.reset()
        assert env._total_profit() > 0


class TestActorCritic:
    def test_forward_shapes(self):
        model = ActorCritic(state_dim=5, n_products=8, hidden=32)
        x = torch.randn(1, 5 * 8)
        logits, value = model(x)
        assert logits.shape == (1, 8)
        assert value.shape == (1, 1)

    def test_batch_forward(self):
        model = ActorCritic(state_dim=5, n_products=6, hidden=32)
        x = torch.randn(4, 5 * 6)
        logits, value = model(x)
        assert logits.shape == (4, 6)
        assert value.shape == (4, 1)


class TestPPOTrainer:
    def test_select_action(self):
        env = RackEnv(_make_products(5), max_steps=10)
        trainer = PPOTrainer(env, lr=1e-3, hidden=32)
        state = env.reset()
        action, log_prob, value = trainer.select_action(state)
        assert len(action) == 2
        assert 0 <= action[0] < 5
        assert 0 <= action[1] < 5
        assert action[0] != action[1]
        assert log_prob.shape == ()
        assert value.shape == ()

    def test_train_smoke(self):
        """Train for a few episodes — just verify it doesn't crash."""
        env = RackEnv(_make_products(5), max_steps=5)
        trainer = PPOTrainer(env, lr=1e-3, hidden=32, k_epochs=2)
        rewards = trainer.train(n_episodes=3)
        assert len(rewards) == 3
        assert all(isinstance(r, (int, float, np.floating)) for r in rewards)

    def test_get_optimized_shelves(self):
        env = RackEnv(_make_products(6), max_steps=5)
        trainer = PPOTrainer(env, lr=1e-3, hidden=32)
        trainer.train(n_episodes=2)
        shelves = trainer.get_optimized_shelves()
        assert len(shelves) == 6
        assert all(1 <= s <= NUM_SHELVES for s in shelves)

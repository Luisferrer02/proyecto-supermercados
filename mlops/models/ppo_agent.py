"""
PPO Reinforcement Learning Agent for Shelf Optimization
========================================================
- **Environment**: Rack state = current product positions on 7 shelves.
- **Action**: Swap two products within the same rack.
- **Reward**: Delta in total rack profit after the swap.
- **Agent**: Actor-Critic MLP with PPO clipped objective.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.retail_physics import (
    compute_product_profit,
    get_shelf_multiplier,
    SHELF_WIDTH_CM,
    NUM_SHELVES,
)


# ---------------------------------------------------------------------------
# Custom Gym-like Environment
# ---------------------------------------------------------------------------

class RackEnv:
    """
    A simplified RL environment for shelf rearrangement.

    State  : Array of [price, margin, sales, width, shelf_level] per product.
    Action : Tuple (idx_a, idx_b) — swap shelf assignments of two products.
    Reward : Δ total rack profit.
    """

    def __init__(self, products: np.ndarray, max_steps: int = 50):
        """
        Args:
            products: (N, 4) array with [price, margin, sales, width] per product.
            max_steps: maximum swaps per episode.
        """
        self.products = products.copy()       # immutable product properties
        self.n = len(products)
        self.max_steps = max_steps
        self.step_count = 0
        self.shelf_levels = None
        self.reset()

    # -- env interface -------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Random initial shelf assignment."""
        self.shelf_levels = np.random.randint(1, NUM_SHELVES + 1, size=self.n)
        self.step_count = 0
        return self._get_state()

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute swap and return (next_state, reward, done, info)."""
        idx_a, idx_b = action
        old_profit = self._total_profit()

        # Swap shelf levels
        self.shelf_levels[idx_a], self.shelf_levels[idx_b] = (
            self.shelf_levels[idx_b],
            self.shelf_levels[idx_a],
        )

        new_profit = self._total_profit()
        reward = new_profit - old_profit

        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self._get_state(), reward, done, {"profit": new_profit}

    # -- helpers -------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """State = product features + current shelf level, shape (N, 5)."""
        shelves = self.shelf_levels.reshape(-1, 1).astype(np.float32)
        return np.concatenate([self.products, shelves], axis=1)

    def _total_profit(self) -> float:
        total = 0.0
        for i in range(self.n):
            total += compute_product_profit(
                self.products[i, 0],  # price
                self.products[i, 1],  # margin
                self.products[i, 2],  # sales
                int(self.shelf_levels[i]),
            )
        return total

    @property
    def state_dim(self) -> int:
        return 5  # price, margin, sales, width, shelf

    @property
    def n_products(self) -> int:
        return self.n


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic for discrete swap actions.
    We encode the full rack state, then produce:
      - Actor : logits for each possible swap (simplified to per-product scores).
      - Critic: V(s) scalar value estimate.
    """

    def __init__(self, state_dim: int, n_products: int, hidden: int = 128):
        super().__init__()
        input_dim = state_dim * n_products  # flattened rack state
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Actor head: score per product (we'll pick top-2 to form a swap)
        self.actor = nn.Linear(hidden, n_products)
        # Critic head
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor):
        """
        Args:
            state: (batch, n_products * state_dim)  — flattened
        Returns:
            logits: (batch, n_products)
            value : (batch, 1)
        """
        h = self.shared(state)
        return self.actor(h), self.critic(h)


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Proximal Policy Optimization with clipped objective."""

    def __init__(self, env: RackEnv, lr: float = 3e-4,
                 gamma: float = 0.99, eps_clip: float = 0.2,
                 k_epochs: int = 4, hidden: int = 128):
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(env.state_dim, env.n_products, hidden)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    # -- action selection ----------------------------------------------------

    def select_action(self, state: np.ndarray) -> Tuple[Tuple[int, int], torch.Tensor, torch.Tensor]:
        """Pick a swap action using the actor's current policy."""
        flat = torch.FloatTensor(state.flatten()).unsqueeze(0)
        logits, value = self.policy(flat)
        logits = torch.clamp(logits, -20, 20)  # prevent overflow
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        probs = probs + 1e-8  # prevent zero probabilities
        probs = probs / probs.sum()

        # Sample two distinct product indices
        indices = torch.multinomial(probs, 2, replacement=False)
        idx_a, idx_b = indices[0].item(), indices[1].item()
        log_prob = torch.log(probs[idx_a]) + torch.log(probs[idx_b])

        return (idx_a, idx_b), log_prob, value.squeeze()

    # -- training ------------------------------------------------------------

    def train(self, n_episodes: int = 200) -> List[float]:
        """Run PPO training loop. Returns list of episode rewards."""
        episode_rewards = []

        for ep in range(n_episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0.0

            states, actions, log_probs, rewards, values, dones = (
                [], [], [], [], [], [],
            )

            while not done:
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(torch.FloatTensor(state.flatten()))
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(done)

                state = next_state
                ep_reward += reward

            episode_rewards.append(ep_reward)

            # -- compute returns & advantages --------------------------------
            returns = []
            discounted = 0.0
            for r, d in zip(reversed(rewards), reversed(dones)):
                if d:
                    discounted = 0.0
                discounted = r + self.gamma * discounted
                returns.insert(0, discounted)

            returns_t = torch.FloatTensor(returns)
            values_t = torch.stack(values).detach()
            advantages = returns_t - values_t

            if advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            old_log_probs = torch.stack(log_probs).detach()
            states_t = torch.stack(states)

            # -- PPO update --------------------------------------------------
            for _ in range(self.k_epochs):
                new_log_probs = []
                new_values = []
                for i, (s, a) in enumerate(zip(states_t, actions)):
                    flat = s.unsqueeze(0)
                    logits, v = self.policy(flat)
                    logits = torch.clamp(logits, -20, 20)
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    lp = torch.log(probs[a[0]] + 1e-8) + torch.log(probs[a[1]] + 1e-8)
                    new_log_probs.append(lp)
                    new_values.append(v.squeeze())

                new_log_probs_t = torch.stack(new_log_probs)
                new_values_t = torch.stack(new_values)

                ratio = torch.exp(new_log_probs_t - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(new_values_t, returns_t)
                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return episode_rewards

    def get_optimized_shelves(self) -> np.ndarray:
        """Run one greedy episode using trained policy. Returns shelf_levels."""
        state = self.env.reset()
        done = False
        while not done:
            flat = torch.FloatTensor(state.flatten()).unsqueeze(0)
            with torch.no_grad():
                logits, _ = self.policy(flat)
            logits = torch.clamp(logits, -20, 20)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            probs = probs + 1e-8
            probs = probs / probs.sum()
            indices = torch.multinomial(probs, 2, replacement=False)
            action = (indices[0].item(), indices[1].item())
            state, _, done, _ = self.env.step(action)
        return self.env.shelf_levels.copy()

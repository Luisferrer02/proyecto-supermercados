"""
environment.py
─────────────────────────────────────────────────────────────────────────────
Custom Gymnasium Environment: ShelfSpaceEnv

Overview
────────
Each episode simulates `episode_length` days of retail operations across a
store shelf with `n_positions` slots and `n_products` SKUs. At each step,
the PPO agent assigns ONE product to ONE shelf position (the action). The
environment then simulates one full day of sales, updating inventory and
returning a reward signal.

The XGBoost forecaster is embedded inside the environment's step() function,
called to compute the lift multiplier for every occupied position, which then
scales baseline demand to produce realized sales.

State Space  (observation_space)
─────────────────────────────────
The observation vector has shape (n_products × 7 + n_positions × 3,):

  Per-product features (n_products × 7):
    [0] inventory_level         — current on-hand units (normalized 0-1)
    [1] avg_daily_sales_7d      — 7-day rolling average (normalized)
    [2] avg_daily_sales_30d     — 30-day rolling average (normalized)
    [3] base_price              — unit retail price (normalized)
    [4] margin_pct              — gross margin fraction
    [5] days_since_last_restock — shelf freshness (normalized)
    [6] current_lift_estimate   — XGBoost lift at current position (or 1.0)

  Per-position features (n_positions × 3):
    [0] is_occupied             — binary
    [1] tier_level              — shelf tier (0-3, normalized)
    [2] is_premium              — binary mask (eye-level / end-cap)

Action Space
────────────
Discrete: n_products × n_positions
Action k encodes: product_id = k // n_positions, position_id = k % n_positions

The agent chooses which product to place in which position. Placing a product
that is already there is a valid no-op (zero cost). Placing a product in an
occupied slot swaps the occupants.

Reward Function (per step = per day)
─────────────────────────────────────
  R_t = Σᵢ [ Mᵢ × Sᵢₜ  -  C_hold × Iᵢₜ  -  Φ(OOSᵢₜ) ]

  Mᵢ        : gross margin per unit for product i
  Sᵢₜ       : units sold on day t (min of demand and inventory)
  C_hold    : per-unit per-day holding cost (0.02 by default)
  Iᵢₜ       : inventory level after sales
  Φ(OOSᵢₜ) : stockout penalty (stepped, superlinear)
              = 0          if no stockout
              = 2.0 × Mᵢ   if partially stocked out (< 50% demand met)
              = 5.0 × Mᵢ   if fully stocked out
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from forecaster import DemandForecaster, N_FEATURES

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Reward / Environment Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

HOLDING_COST_PER_UNIT_DAY = 0.02      # C_hold
STOCKOUT_PENALTY_PARTIAL = 2.0        # Φ for partial stockout (×margin)
STOCKOUT_PENALTY_FULL = 5.0           # Φ for full stockout (×margin)
STOCKOUT_THRESHOLD = 0.50             # fraction of demand met below which = partial

RESTOCK_PROB_PER_DAY = 0.15           # probability of automatic restock event
RESTOCK_AMOUNT_RANGE = (20, 80)       # units added on restock

DEMAND_NOISE_STD = 0.12               # log-normal noise σ on daily demand


# ─────────────────────────────────────────────────────────────────────────────
# Product Catalogue
# ─────────────────────────────────────────────────────────────────────────────

def build_default_product_catalog(n_products: int, seed: int = 1) -> dict:
    """
    Build a dictionary of product parameters.

    Keys
    ────
    base_demand       : float  — baseline units/day demand at any position
    base_price        : float  — retail price USD
    margin_pct        : float  — fraction (0-1)
    category          : int    — category ID
    max_inventory     : int    — shelf capacity for this product
    """
    rng = np.random.default_rng(seed)
    catalog = {}
    for pid in range(n_products):
        price = rng.uniform(0.99, 29.99)
        margin = rng.uniform(0.10, 0.65)
        catalog[pid] = {
            "base_demand": rng.uniform(2.0, 20.0),  # units/day
            "base_price": round(float(price), 2),
            "margin_pct": round(float(margin), 3),
            "margin_abs": round(float(price * margin), 4),
            "category": int(rng.integers(0, 10)),
            "max_inventory": int(rng.integers(30, 120)),
        }
    return catalog


def build_default_position_catalog(n_positions: int, seed: int = 2) -> dict:
    """
    Build a dictionary of shelf-position attributes.

    Keys
    ────
    tier          : int   — 0=bottom … 3=eye-level
    facing_count  : int   — number of facings (slots)
    is_end_cap    : bool
    is_checkout   : bool
    is_premium    : bool  — True for eye-level or end-cap
    """
    rng = np.random.default_rng(seed)
    positions = {}
    for pos in range(n_positions):
        tier = int(rng.integers(0, 4))
        is_end_cap = bool(rng.binomial(1, 0.15))
        is_checkout = bool(rng.binomial(1, 0.10))
        positions[pos] = {
            "tier": tier,
            "facing_count": int(rng.integers(1, 5)),
            "is_end_cap": is_end_cap,
            "is_checkout": is_checkout,
            "is_premium": (tier == 3 or is_end_cap),
        }
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class ShelfSpaceEnv(gym.Env):
    """
    Custom Gymnasium environment for shelf-space product placement optimization.

    Parameters
    ----------
    n_products      : number of SKUs to place
    n_positions     : number of shelf positions
    episode_length  : number of days per episode
    forecaster      : a trained DemandForecaster instance (or path to .ubj)
    product_catalog : dict from build_default_product_catalog (or None = auto)
    position_catalog: dict from build_default_position_catalog (or None = auto)
    seed            : random seed for reproducibility
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        n_products: int = 10,
        n_positions: int = 15,
        episode_length: int = 30,
        forecaster: Optional[DemandForecaster | str | Path] = None,
        product_catalog: Optional[dict] = None,
        position_catalog: Optional[dict] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.n_products = n_products
        self.n_positions = n_positions
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)

        # --- Catalogs ---
        self.products = product_catalog or build_default_product_catalog(n_products, seed)
        self.positions = position_catalog or build_default_position_catalog(n_positions, seed)

        # --- Forecaster ---
        if forecaster is None:
            logger.info("No forecaster provided — training a new one on synthetic data.")
            self.forecaster = DemandForecaster()
            self.forecaster.train(verbose=False)
        elif isinstance(forecaster, (str, Path)):
            self.forecaster = DemandForecaster.load(forecaster)
        else:
            self.forecaster = forecaster

        # --- Spaces ---
        obs_dim = n_products * 7 + n_positions * 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        # Action: (product_id, position_id) encoded as flat integer
        self.action_space = spaces.Discrete(n_products * n_positions)

        # --- State buffers (allocated in reset) ---
        self.inventory: np.ndarray = np.zeros(n_products, dtype=np.float32)
        self.sales_history: np.ndarray = np.zeros((n_products, 30), dtype=np.float32)
        self.days_since_restock: np.ndarray = np.zeros(n_products, dtype=np.float32)
        self.placement: dict[int, int] = {}   # position_id → product_id
        self.current_lifts: np.ndarray = np.ones(n_products, dtype=np.float32)

        self._day = 0
        self._episode_reward = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Core Gymnasium API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Initialize inventory to 50-90% of max capacity
        self.inventory = np.array([
            self.rng.integers(
                int(0.5 * self.products[pid]["max_inventory"]),
                self.products[pid]["max_inventory"] + 1,
            )
            for pid in range(self.n_products)
        ], dtype=np.float32)

        self.sales_history = np.zeros((self.n_products, 30), dtype=np.float32)
        self.days_since_restock = self.rng.integers(0, 5, size=self.n_products).astype(np.float32)

        # Random initial placement (each position gets a random product)
        shuffled = self.rng.permutation(self.n_products)
        self.placement = {}
        for pos_id in range(min(self.n_positions, self.n_products)):
            self.placement[pos_id] = int(shuffled[pos_id])

        self._day = 0
        self._episode_reward = 0.0
        self._refresh_lifts()

        obs = self._get_observation()
        info = {"day": 0, "placement": dict(self.placement)}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one day of retail simulation.

        Step breakdown
        ──────────────
        1. Decode action → (product_id, position_id)
        2. Apply placement change (swap if needed)
        3. Refresh XGBoost lift estimates for all positions
        4. Simulate demand and sales for each placed product
        5. Compute reward R_t
        6. Update inventory and sales history
        7. Apply stochastic restocking
        8. Build and return next observation
        """
        # 1. Decode action
        product_id = int(action) // self.n_positions
        position_id = int(action) % self.n_positions

        # 2. Placement update with swap logic
        self._apply_placement(product_id, position_id)

        # 3. Refresh lifts (XGBoost call)
        self._refresh_lifts()

        # 4-6. Simulate one day
        day_of_week = int(self._day % 7)
        week_of_year = int((self._day // 7) % 52) + 1
        total_reward = 0.0
        info_products = {}

        for pos_id, pid in self.placement.items():
            pos = self.positions[pos_id]
            prod = self.products[pid]

            lift = float(self.current_lifts[pid])

            # Demand realization
            demand_mean = prod["base_demand"] * lift
            noise = float(self.rng.lognormal(0, DEMAND_NOISE_STD))
            demand = demand_mean * noise

            # Sales are capped by inventory
            units_sold = min(demand, float(self.inventory[pid]))
            units_sold = max(0.0, units_sold)

            # Inventory after sales
            new_inventory = float(self.inventory[pid]) - units_sold

            # Reward components
            margin = prod["margin_abs"]
            profit_term = margin * units_sold
            holding_term = HOLDING_COST_PER_UNIT_DAY * new_inventory
            oos_penalty = self._oos_penalty(demand, units_sold, margin)

            step_reward = profit_term - holding_term - oos_penalty
            total_reward += step_reward

            self.inventory[pid] = new_inventory
            self.days_since_restock[pid] += 1

            # Update sales history (rolling)
            self.sales_history[pid] = np.roll(self.sales_history[pid], -1)
            self.sales_history[pid, -1] = units_sold

            info_products[pid] = {
                "demand": round(demand, 2),
                "units_sold": round(units_sold, 2),
                "inventory": round(new_inventory, 2),
                "lift": round(lift, 3),
                "reward_components": {
                    "profit": round(profit_term, 4),
                    "holding": round(holding_term, 4),
                    "oos_penalty": round(oos_penalty, 4),
                },
            }

        # 7. Stochastic restock
        self._apply_restock()

        self._day += 1
        self._episode_reward += total_reward

        terminated = self._day >= self.episode_length
        truncated = False

        obs = self._get_observation()
        info = {
            "day": self._day,
            "episode_reward_so_far": self._episode_reward,
            "products": info_products,
            "placement": dict(self.placement),
        }

        return obs, float(total_reward), terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        lines = [
            f"\n{'─'*55}",
            f" Day {self._day:3d}/{self.episode_length}  |  Cumulative Reward: {self._episode_reward:9.2f}",
            f"{'─'*55}",
            f" {'Pos':>4}  {'Tier':>4}  {'Product':>7}  {'Inventory':>9}  {'Lift':>6}",
            f" {'───':>4}  {'────':>4}  {'───────':>7}  {'─────────':>9}  {'────':>6}",
        ]
        for pos_id in sorted(self.placement.keys()):
            pid = self.placement[pos_id]
            tier = self.positions[pos_id]["tier"]
            inv = int(self.inventory[pid])
            lift = self.current_lifts[pid]
            premium_flag = "★" if self.positions[pos_id]["is_premium"] else " "
            lines.append(f" {premium_flag}{pos_id:3d}  {'▁▃▅█'[tier]:>4}  P{pid:06d}  {inv:9d}  {lift:6.3f}")

        out = "\n".join(lines)
        if mode == "human":
            print(out)
        return out

    def close(self) -> None:
        pass

    # ──────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _apply_placement(self, product_id: int, position_id: int) -> None:
        """
        Place product_id at position_id, swapping if position already occupied.
        If product already at that position → no-op.
        """
        # Find where product_id currently lives (if anywhere)
        current_pos = None
        for pos, pid in self.placement.items():
            if pid == product_id:
                current_pos = pos
                break

        # Find if target position is occupied
        occupant = self.placement.get(position_id, None)

        if occupant == product_id:
            return  # already there, no-op

        if current_pos is not None and occupant is not None:
            # Swap
            self.placement[current_pos] = occupant
            self.placement[position_id] = product_id
        elif current_pos is not None:
            # Move product to empty position
            del self.placement[current_pos]
            self.placement[position_id] = product_id
        else:
            # Product wasn't placed before; just slot it in (swapping if needed)
            self.placement[position_id] = product_id

    def _refresh_lifts(self) -> None:
        """
        Query XGBoost forecaster for all currently-placed products and
        update self.current_lifts. Unplaced products retain lift=1.0.
        """
        self.current_lifts = np.ones(self.n_products, dtype=np.float32)

        if not self.placement:
            return

        day_of_week = int(self._day % 7)
        week_of_year = int((self._day // 7) % 52) + 1

        feature_rows = []
        pids_to_update = []

        for pos_id, pid in self.placement.items():
            pos = self.positions[pos_id]
            prod = self.products[pid]
            row = [
                pos["tier"],                      # shelf_tier
                pos["facing_count"],               # facing_count
                int(pos["is_end_cap"]),            # is_end_cap
                int(pos["is_checkout"]),           # is_checkout
                0,                                 # adjacency_promo (simplified)
                prod["category"],                  # product_category
                prod["base_price"],                # base_price
                prod["margin_pct"],                # margin_pct
                int(self.days_since_restock[pid]), # days_since_restock
                day_of_week,                       # day_of_week
                week_of_year,                      # week_of_year
            ]
            feature_rows.append(row)
            pids_to_update.append(pid)

        X = np.array(feature_rows, dtype=np.float32)
        lifts = self.forecaster.predict(X)

        for pid, lift in zip(pids_to_update, lifts):
            self.current_lifts[pid] = float(lift)

    def _oos_penalty(
        self,
        demand: float,
        units_sold: float,
        margin: float,
    ) -> float:
        """
        Superlinear out-of-stock penalty.
        Φ = 0                      if fully met
          = PARTIAL_MULT × margin  if 50-99% met
          = FULL_MULT × margin     if < 50% met (or total stockout)
        """
        if demand < 1e-6:
            return 0.0
        fill_rate = units_sold / demand
        if fill_rate >= 1.0 - 1e-6:
            return 0.0
        elif fill_rate >= STOCKOUT_THRESHOLD:
            return STOCKOUT_PENALTY_PARTIAL * margin
        else:
            return STOCKOUT_PENALTY_FULL * margin

    def _apply_restock(self) -> None:
        """
        Randomly restock products with probability RESTOCK_PROB_PER_DAY.
        Caps inventory at max_inventory.
        """
        for pid in range(self.n_products):
            if self.rng.random() < RESTOCK_PROB_PER_DAY:
                amount = int(self.rng.integers(*RESTOCK_AMOUNT_RANGE))
                max_inv = self.products[pid]["max_inventory"]
                self.inventory[pid] = min(self.inventory[pid] + amount, max_inv)
                self.days_since_restock[pid] = 0

    def _get_observation(self) -> np.ndarray:
        """
        Build the flat observation vector.

        Layout
        ──────
        [product_0_features (7)] ... [product_N_features (7)]
        [position_0_features (3)] ... [position_M_features (3)]
        """
        parts = []

        # Per-product block
        for pid in range(self.n_products):
            prod = self.products[pid]
            max_inv = prod["max_inventory"]

            inv_norm = float(self.inventory[pid]) / max(max_inv, 1)
            avg_7d = float(np.mean(self.sales_history[pid, -7:])) / max(prod["base_demand"] * 3, 1)
            avg_30d = float(np.mean(self.sales_history[pid, :])) / max(prod["base_demand"] * 3, 1)
            price_norm = prod["base_price"] / 50.0        # normalize by $50 max
            margin = prod["margin_pct"]
            restock_norm = min(float(self.days_since_restock[pid]) / 14.0, 1.0)
            lift_norm = float(self.current_lifts[pid]) / 4.0  # max lift ~4x

            parts.extend([
                np.clip(inv_norm, 0.0, 1.0),
                np.clip(avg_7d, 0.0, 1.0),
                np.clip(avg_30d, 0.0, 1.0),
                np.clip(price_norm, 0.0, 1.0),
                np.clip(margin, 0.0, 1.0),
                restock_norm,
                np.clip(lift_norm, 0.0, 1.0),
            ])

        # Per-position block
        for pos_id in range(self.n_positions):
            pos = self.positions[pos_id]
            is_occupied = float(pos_id in self.placement)
            tier_norm = pos["tier"] / 3.0
            is_premium = float(pos["is_premium"])
            parts.extend([is_occupied, tier_norm, is_premium])

        return np.array(parts, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        return self.n_products * 7 + self.n_positions * 3

    def decode_action(self, action: int) -> tuple[int, int]:
        """Return (product_id, position_id) for a flat action integer."""
        return action // self.n_positions, action % self.n_positions

    def get_placement_summary(self) -> list[dict]:
        rows = []
        for pos_id, pid in sorted(self.placement.items()):
            rows.append({
                "position": pos_id,
                "product": pid,
                "tier": self.positions[pos_id]["tier"],
                "is_premium": self.positions[pos_id]["is_premium"],
                "lift": round(float(self.current_lifts[pid]), 3),
                "inventory": round(float(self.inventory[pid]), 1),
            })
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    env = ShelfSpaceEnv(n_products=8, n_positions=10, episode_length=5, seed=0)
    obs, info = env.reset(seed=0)

    print(f"Observation shape : {obs.shape}")
    print(f"Action space size : {env.action_space.n}")
    print(f"Initial placement : {info['placement']}")

    total_r = 0.0
    for step_i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        print(f"  Step {step_i+1}: action={action:4d}  reward={reward:8.3f}  done={terminated}")
        if terminated:
            break

    env.render()
    print(f"\nTotal episode reward: {total_r:.3f}")
    env.close()

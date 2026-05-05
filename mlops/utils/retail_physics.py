"""
Retail Physics Engine
=====================
Implements shelf multipliers, category constraints, width validation,
profit calculations with complex dynamics, and synthetic training data.

Complex dynamics that ML models can learn but Greedy cannot:
  1. Crowding penalty  — too many products on one shelf reduces each one's sales
  2. Spread bonus      — distributing products across shelves improves visibility
  3. Price-tier affinity — products near similar price ranges sell better
  4. Diminishing returns — eye-level benefit decreases as the shelf fills up
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHELF_WIDTH_CM = 300.0
NUM_SHELVES = 7

SHELF_MULTIPLIERS: Dict[int, float] = {
    1: 0.60,   # Floor
    2: 0.80,   # Low
    3: 0.95,   # Below eye
    4: 1.15,   # Eye level (peak)
    5: 1.00,   # Above eye
    6: 0.75,   # High
    7: 0.50,   # Top
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def get_shelf_multiplier(shelf_level: int) -> float:
    """Return the sales multiplier for a given shelf level (1-7)."""
    return SHELF_MULTIPLIERS.get(shelf_level, 1.0)


def compute_product_profit(price: float,
                           margin_pct: float,
                           monthly_sales: float,
                           shelf_level: int) -> float:
    """
    Basic product profit (used for simple calculations).
    profit = price * (margin / 100) * sales * shelf_multiplier
    """
    multiplier = get_shelf_multiplier(shelf_level)
    return price * (margin_pct / 100.0) * monthly_sales * multiplier


def compute_rack_profit_advanced(df_rack: pd.DataFrame) -> float:
    """
    Advanced profit calculation with complex dynamics (vectorized).
    Includes: shelf multiplier + crowding penalty + spread bonus +
    diminishing returns on eye-level shelves.
    """
    n = len(df_rack)
    if n == 0:
        return 0.0

    shelf = df_rack["shelf_level"].astype(int)
    price = df_rack["price_numeric"]
    margin = df_rack["profit_margin_percentage"] / 100.0
    sales = df_rack["estimated_monthly_sales"]

    # Shelf multiplier (vectorized lookup)
    base_mult = shelf.map(SHELF_MULTIPLIERS).fillna(1.0)

    # Shelf-level aggregates
    shelf_counts = shelf.value_counts()
    shelf_widths = df_rack.groupby("shelf_level")["product_width_cm"].sum()
    n_shelves_used = len(shelf_counts)

    # Spread bonus
    spread_bonus = 1.0 + 0.15 * (n_shelves_used / NUM_SHELVES)

    # Crowding penalty per product (based on its shelf)
    count_on_shelf = shelf.map(shelf_counts)
    crowding_factor = np.where(
        count_on_shelf > 5,
        np.maximum(0.5, 1.0 - 0.05 * (count_on_shelf - 5)),
        1.0,
    )

    # Diminishing returns on eye-level shelves (3, 4, 5)
    fill_ratio = shelf.map(shelf_widths).fillna(0) / SHELF_WIDTH_CM
    is_eye = shelf.isin([3, 4, 5])
    dim_factor = np.where(
        is_eye & (fill_ratio > 0.6),
        np.maximum(0.7, 1.0 - 0.5 * (fill_ratio - 0.6)),
        1.0,
    )

    # Price-tier positioning bonus
    tier_bonus = np.ones(n)
    tier_bonus[(price > 5.0) & is_eye] = 1.1
    tier_bonus[(price < 2.0) & shelf.isin([1, 2])] = 1.15
    tier_bonus[(price > 5.0) & shelf.isin([1, 2])] = 0.85

    effective_mult = base_mult * crowding_factor * dim_factor * tier_bonus * spread_bonus
    profit = price * margin * sales * effective_mult
    return float(profit.sum())


def compute_rack_profit(df_rack: pd.DataFrame) -> float:
    """Total profit using advanced dynamics."""
    return compute_rack_profit_advanced(df_rack)


# ---------------------------------------------------------------------------
# Constraint checks
# ---------------------------------------------------------------------------

def check_shelf_width(df: pd.DataFrame,
                      rack_id: int,
                      shelf_level: int) -> float:
    """Return the total width used on a specific shelf of a rack."""
    mask = (df["rack_id"] == rack_id) & (df["shelf_level"] == shelf_level)
    return df.loc[mask, "product_width_cm"].sum()


def validate_all_shelves(df: pd.DataFrame) -> List[Tuple[int, int, float]]:
    """Return list of (rack_id, shelf_level, total_cm) for overflowing shelves."""
    violations = []
    for rack_id in df["rack_id"].unique():
        for shelf in range(1, NUM_SHELVES + 1):
            total = check_shelf_width(df, rack_id, shelf)
            if total > SHELF_WIDTH_CM:
                violations.append((rack_id, shelf, total))
    return violations


def enforce_shelf_constraint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Redistribute products from overflowing shelves to shelves with
    remaining capacity *within the same rack*.
    """
    df = df.copy()
    for rack_id in df["rack_id"].unique():
        rack_mask = df["rack_id"] == rack_id
        for shelf in range(1, NUM_SHELVES + 1):
            shelf_mask = rack_mask & (df["shelf_level"] == shelf)
            total = df.loc[shelf_mask, "product_width_cm"].sum()
            if total <= SHELF_WIDTH_CM:
                continue
            overflow_idx = df.loc[shelf_mask].sort_values(
                "product_width_cm", ascending=False
            ).index.tolist()
            for idx in overflow_idx:
                if total <= SHELF_WIDTH_CM:
                    break
                w = df.loc[idx, "product_width_cm"]
                for alt_shelf in range(1, NUM_SHELVES + 1):
                    if alt_shelf == shelf:
                        continue
                    alt_total = check_shelf_width(df, rack_id, alt_shelf)
                    if alt_total + w <= SHELF_WIDTH_CM:
                        df.loc[idx, "shelf_level"] = alt_shelf
                        total -= w
                        break
    return df


# ---------------------------------------------------------------------------
# Synthetic training data generation (advanced)
# ---------------------------------------------------------------------------

def generate_synthetic_training_data(df: pd.DataFrame,
                                     n_samples: int = 5000,
                                     seed: int = 42) -> pd.DataFrame:
    """
    Create training samples by simulating shelf reassignments with
    full rack context (so the model can learn crowding/spread effects).
    """
    rng = np.random.RandomState(seed)
    records = []
    rack_ids = df["rack_id"].unique()

    for _ in range(n_samples):
        # Pick a random rack and product
        rack_id = rng.choice(rack_ids)
        rack_df = df[df["rack_id"] == rack_id]
        if len(rack_df) == 0:
            continue
        local_idx = rng.randint(0, len(rack_df))
        row = rack_df.iloc[local_idx]
        original_shelf = int(row["shelf_level"])
        new_shelf = rng.randint(1, NUM_SHELVES + 1)

        # Compute profit with rack context
        base_profit = compute_rack_profit_advanced(rack_df)

        # Simulate the move
        rack_modified = rack_df.copy()
        rack_modified.iloc[local_idx, rack_modified.columns.get_loc("shelf_level")] = new_shelf
        new_profit = compute_rack_profit_advanced(rack_modified)

        profit_lift = new_profit - base_profit

        # Rack-level context features
        shelf_counts = rack_df.groupby("shelf_level").size()
        n_on_original = shelf_counts.get(original_shelf, 0)
        n_on_new = shelf_counts.get(new_shelf, 0)
        n_shelves_used = len(shelf_counts)

        records.append({
            "price_numeric": row["price_numeric"],
            "profit_margin_percentage": row["profit_margin_percentage"],
            "estimated_monthly_sales": row["estimated_monthly_sales"],
            "product_width_cm": row["product_width_cm"],
            "original_shelf": original_shelf,
            "new_shelf": new_shelf,
            "rack_id": row["rack_id"],
            "n_products_on_original_shelf": n_on_original,
            "n_products_on_new_shelf": n_on_new,
            "n_shelves_used": n_shelves_used,
            "rack_product_count": len(rack_df),
            "profit_lift": profit_lift,
        })

    return pd.DataFrame(records)


def generate_absolute_profit_data(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate training data where each sample is (product_features, shelf) → absolute_profit.

    For every product in the dataset, compute its profit on each of the 7 shelves.
    This gives 7 samples per product — the model learns to predict how much profit
    a product generates on any given shelf.
    """
    records = []

    for rack_id in df["rack_id"].unique():
        rack_df = df[df["rack_id"] == rack_id]
        if len(rack_df) == 0:
            continue

        for _, row in rack_df.iterrows():
            base_profit = (
                row["price_numeric"]
                * (row["profit_margin_percentage"] / 100.0)
                * row["estimated_monthly_sales"]
            )
            for shelf in range(1, NUM_SHELVES + 1):
                mult = SHELF_MULTIPLIERS.get(shelf, 1.0)
                profit = base_profit * mult
                records.append({
                    "price_numeric": row["price_numeric"],
                    "profit_margin_percentage": row["profit_margin_percentage"],
                    "estimated_monthly_sales": row["estimated_monthly_sales"],
                    "product_width_cm": row["product_width_cm"],
                    "shelf": shelf,
                    "profit": profit,
                })

    return pd.DataFrame(records)


def optimize_rack_greedy(df_rack: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy optimizer: for each product in the rack, try every shelf
    and assign it to the one that maximises its individual profit
    (using basic shelf multiplier only — no awareness of crowding/spread).
    """
    df_opt = df_rack.copy()
    df_opt["_base_profit"] = df_opt.apply(
        lambda r: r["price_numeric"]
        * (r["profit_margin_percentage"] / 100.0)
        * r["estimated_monthly_sales"],
        axis=1,
    )
    df_opt = df_opt.sort_values("_base_profit", ascending=False)

    # Start with zero capacity (we reassign all products)
    capacity = dict.fromkeys(range(1, NUM_SHELVES + 1), SHELF_WIDTH_CM)

    for idx in df_opt.index:
        w = df_opt.loc[idx, "product_width_cm"]
        best_shelf = int(df_opt.loc[idx, "shelf_level"])
        best_mult = -1.0

        for s in range(1, NUM_SHELVES + 1):
            m = get_shelf_multiplier(s)
            if m > best_mult and capacity[s] >= w:
                best_shelf = s
                best_mult = m

        df_opt.loc[idx, "shelf_level"] = best_shelf
        capacity[best_shelf] -= w

    return df_opt.drop(columns=["_base_profit"])




def optimize_rack_advanced(df_rack: pd.DataFrame) -> pd.DataFrame:
    """Greedy optimizer using the advanced profit formula.

    For each product (highest base profit first), try every shelf and pick
    the one that maximises the total rack profit including crowding, spread,
    and diminishing returns.
    """
    df_opt = df_rack.copy()
    df_opt["_base_profit"] = (
        df_opt["price_numeric"]
        * (df_opt["profit_margin_percentage"] / 100.0)
        * df_opt["estimated_monthly_sales"]
    )
    df_opt = df_opt.sort_values("_base_profit", ascending=False)

    capacity = dict.fromkeys(range(1, NUM_SHELVES + 1), SHELF_WIDTH_CM)

    for idx in df_opt.index:
        w = df_opt.loc[idx, "product_width_cm"]
        original_shelf = int(df_opt.loc[idx, "shelf_level"])
        best_shelf = original_shelf
        best_profit = -float("inf")

        for s in range(1, NUM_SHELVES + 1):
            if capacity[s] < w:
                continue
            df_opt.loc[idx, "shelf_level"] = s
            rack_profit = compute_rack_profit_advanced(df_opt)
            if rack_profit > best_profit:
                best_profit = rack_profit
                best_shelf = s

        df_opt.loc[idx, "shelf_level"] = best_shelf
        capacity[best_shelf] -= w

    return df_opt.drop(columns=["_base_profit"])

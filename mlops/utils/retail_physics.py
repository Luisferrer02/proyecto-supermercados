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

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHELF_WIDTH_CM = 300.0
NUM_SHELVES = 7

SHELF_MULTIPLIERS: Dict[int, float] = {
    1: 0.7,   # Bottom
    2: 0.7,   # Bottom
    3: 1.2,   # Eye level
    4: 1.2,   # Eye level
    5: 1.2,   # Eye level
    6: 0.8,   # Top
    7: 0.8,   # Top
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
    profit = price × (margin / 100) × sales × shelf_multiplier
    """
    multiplier = get_shelf_multiplier(shelf_level)
    return price * (margin_pct / 100.0) * monthly_sales * multiplier


def compute_rack_profit_advanced(df_rack: pd.DataFrame) -> float:
    """
    Advanced profit calculation with complex dynamics.
    Includes: shelf multiplier + crowding penalty + spread bonus +
    diminishing returns on eye-level shelves.
    """
    total_profit = 0.0
    n = len(df_rack)
    if n == 0:
        return 0.0

    # Count products per shelf
    shelf_counts = df_rack.groupby("shelf_level").size().to_dict()
    shelf_widths = df_rack.groupby("shelf_level")["product_width_cm"].sum().to_dict()

    # How many unique shelves are used
    n_shelves_used = len(shelf_counts)
    # Spread bonus: using more shelves = better visibility (up to +15%)
    spread_bonus = 1.0 + 0.15 * (n_shelves_used / NUM_SHELVES)

    for _, row in df_rack.iterrows():
        shelf = int(row["shelf_level"])
        base_mult = get_shelf_multiplier(shelf)

        # 1. Crowding penalty: more products on a shelf = less attention each
        #    Penalty kicks in after 5 products on a shelf
        count_on_shelf = shelf_counts.get(shelf, 1)
        crowding_factor = 1.0
        if count_on_shelf > 5:
            crowding_factor = max(0.5, 1.0 - 0.05 * (count_on_shelf - 5))

        # 2. Diminishing returns on eye-level shelves
        #    As eye-level shelves fill up (by width), the multiplier decreases
        dim_factor = 1.0
        if shelf in [3, 4, 5]:
            fill_ratio = shelf_widths.get(shelf, 0) / SHELF_WIDTH_CM
            # Starts penalizing after 60% full
            if fill_ratio > 0.6:
                dim_factor = max(0.7, 1.0 - 0.5 * (fill_ratio - 0.6))

        # 3. Price-tier positioning bonus
        #    Expensive products (>5€) get a bonus on eye-level (shoppers see premium first)
        #    Cheap products (<2€) get a bonus on bottom shelves (bulk buys)
        tier_bonus = 1.0
        price = row["price_numeric"]
        if price > 5.0 and shelf in [3, 4, 5]:
            tier_bonus = 1.1  # +10% for premium at eye level
        elif price < 2.0 and shelf in [1, 2]:
            tier_bonus = 1.15  # +15% for budget items at bottom (bulk)
        elif price > 5.0 and shelf in [1, 2]:
            tier_bonus = 0.85  # penalty: premium hidden at bottom

        effective_mult = base_mult * crowding_factor * dim_factor * tier_bonus * spread_bonus

        profit = (
            price
            * (row["profit_margin_percentage"] / 100.0)
            * row["estimated_monthly_sales"]
            * effective_mult
        )
        total_profit += profit

    return total_profit


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
                w = df.at[idx, "product_width_cm"]
                for alt_shelf in range(1, NUM_SHELVES + 1):
                    if alt_shelf == shelf:
                        continue
                    alt_total = check_shelf_width(df, rack_id, alt_shelf)
                    if alt_total + w <= SHELF_WIDTH_CM:
                        df.at[idx, "shelf_level"] = alt_shelf
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

    capacity = {s: SHELF_WIDTH_CM for s in range(1, NUM_SHELVES + 1)}

    for idx in df_opt.index:
        w = df_opt.at[idx, "product_width_cm"]
        best_shelf = df_opt.at[idx, "shelf_level"]
        best_mult = get_shelf_multiplier(best_shelf)

        for s in range(1, NUM_SHELVES + 1):
            m = get_shelf_multiplier(s)
            if m > best_mult and capacity[s] >= w:
                best_shelf = s
                best_mult = m

        df_opt.at[idx, "shelf_level"] = best_shelf
        capacity[best_shelf] -= w

    df_opt.drop(columns=["_base_profit"], inplace=True)
    return df_opt

"""Tests for utils/retail_physics.py"""

import numpy as np
import pandas as pd

from utils.retail_physics import (
    NUM_SHELVES,
    SHELF_WIDTH_CM,
    compute_rack_profit,
    enforce_shelf_constraint,
    generate_synthetic_training_data,
    get_shelf_multiplier,
    optimize_rack_greedy,
    validate_all_shelves,
)


def _make_rack(n=10, rack_id=0, seed=42):
    """Helper: create a small rack DataFrame for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "rack_id": rack_id,
        "Category": "TestCat",
        "name": [f"product_{i}" for i in range(n)],
        "price_numeric": rng.uniform(1, 10, n).round(2),
        "profit_margin_percentage": rng.uniform(10, 50, n).round(1),
        "estimated_monthly_sales": rng.integers(10, 200, n),
        "product_width_cm": rng.uniform(5, 30, n).round(1),
        "shelf_level": rng.integers(1, NUM_SHELVES + 1, n),
    })


class TestShelfMultiplier:
    def test_eye_level_highest(self):
        assert get_shelf_multiplier(4) > get_shelf_multiplier(1)
        assert get_shelf_multiplier(4) > get_shelf_multiplier(7)

    def test_all_shelves_positive(self):
        for s in range(1, NUM_SHELVES + 1):
            assert get_shelf_multiplier(s) > 0

    def test_unknown_shelf_returns_1(self):
        assert get_shelf_multiplier(99) == 1.0


class TestComputeRackProfit:
    def test_positive_profit(self):
        rack = _make_rack()
        profit = compute_rack_profit(rack)
        assert profit > 0

    def test_empty_rack_zero(self):
        empty = pd.DataFrame(columns=_make_rack().columns)
        assert compute_rack_profit(empty) == 0.0

    def test_eye_level_more_profitable(self):
        rack = _make_rack(n=1)
        rack_eye = rack.copy()
        rack_eye["shelf_level"] = 4
        rack_floor = rack.copy()
        rack_floor["shelf_level"] = 1
        assert compute_rack_profit(rack_eye) > compute_rack_profit(rack_floor)


class TestEnforceShelfConstraint:
    def test_no_violations_unchanged(self):
        rack = _make_rack(n=3)
        rack["product_width_cm"] = 10.0  # 3 * 10 = 30 << 300
        result = enforce_shelf_constraint(rack)
        assert len(result) == len(rack)

    def test_fixes_overflow(self):
        rack = _make_rack(n=5)
        rack["shelf_level"] = 1  # all on shelf 1
        rack["product_width_cm"] = 100.0  # 5 * 100 = 500 > 300
        result = enforce_shelf_constraint(rack)
        violations = validate_all_shelves(result)
        assert len(violations) == 0


class TestValidateAllShelves:
    def test_no_violations(self):
        rack = _make_rack(n=3)
        rack["product_width_cm"] = 10.0
        assert validate_all_shelves(rack) == []

    def test_detects_overflow(self):
        rack = _make_rack(n=5)
        rack["shelf_level"] = 1
        rack["product_width_cm"] = 100.0
        violations = validate_all_shelves(rack)
        assert len(violations) > 0
        assert violations[0][2] > SHELF_WIDTH_CM


class TestGenerateSyntheticTrainingData:
    def test_returns_expected_columns(self):
        rack = _make_rack()
        data = generate_synthetic_training_data(rack, n_samples=50, seed=42)
        assert "profit_lift" in data.columns
        assert "original_shelf" in data.columns
        assert "new_shelf" in data.columns
        assert len(data) > 0

    def test_deterministic(self):
        rack = _make_rack()
        d1 = generate_synthetic_training_data(rack, n_samples=20, seed=1)
        d2 = generate_synthetic_training_data(rack, n_samples=20, seed=1)
        pd.testing.assert_frame_equal(d1, d2)


class TestOptimizeRackGreedy:
    def test_respects_shelf_width(self):
        rack = _make_rack(n=15)
        result = optimize_rack_greedy(rack)
        for shelf in range(1, NUM_SHELVES + 1):
            total_w = result.loc[result["shelf_level"] == shelf, "product_width_cm"].sum()
            assert total_w <= SHELF_WIDTH_CM + 0.01  # float tolerance

    def test_produces_valid_layout(self):
        rack = _make_rack()
        optimized = optimize_rack_greedy(rack)
        # Should produce a valid layout with positive profit
        assert compute_rack_profit(optimized) > 0
        assert len(optimized) == len(rack)

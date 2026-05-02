"""Tests for utils/explainability.py"""

import pandas as pd

from utils.explainability import explain_all, explain_rack


def _make_rack(shelves_orig, shelves_opt):
    """Helper: create original and optimized rack DataFrames."""
    n = len(shelves_orig)
    base = {
        "rack_id": 0,
        "Category": "Test",
        "name": [f"product_{i}" for i in range(n)],
        "price_numeric": [5.0] * n,
        "profit_margin_percentage": [30.0] * n,
        "estimated_monthly_sales": [100] * n,
        "product_width_cm": [10.0] * n,
    }
    orig = pd.DataFrame({**base, "shelf_level": shelves_orig})
    opt = pd.DataFrame({**base, "shelf_level": shelves_opt})
    return orig, opt


class TestExplainRack:
    def test_no_moves_empty(self):
        orig, opt = _make_rack([1, 2, 3], [1, 2, 3])
        assert explain_rack(orig, opt) == []

    def test_moved_product_gets_explanation(self):
        orig, opt = _make_rack([1, 2, 3], [4, 2, 3])  # product_0 moved 1→4
        explanations = explain_rack(orig, opt)
        assert len(explanations) == 1
        assert explanations[0]["product"] == "product_0"
        assert explanations[0]["old_shelf"] == 1
        assert explanations[0]["new_shelf"] == 4
        assert len(explanations[0]["reasons"]) > 0

    def test_eye_level_promotion_reason(self):
        orig, opt = _make_rack([1], [4])  # moved to eye level
        explanations = explain_rack(orig, opt)
        codes = [r["code"] for r in explanations[0]["reasons"]]
        assert "eye_level_promotion" in codes

    def test_max_three_reasons(self):
        orig, opt = _make_rack([1, 1, 1, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1, 1])
        explanations = explain_rack(orig, opt)
        assert len(explanations[0]["reasons"]) <= 3


class TestExplainAll:
    def test_groups_by_rack(self):
        orig = pd.DataFrame({
            "rack_id": [0, 0, 1, 1],
            "name": ["a", "b", "c", "d"],
            "shelf_level": [1, 2, 1, 2],
            "price_numeric": [5.0] * 4,
            "profit_margin_percentage": [30.0] * 4,
            "estimated_monthly_sales": [100] * 4,
            "product_width_cm": [10.0] * 4,
            "Category": ["X"] * 4,
        })
        opt = orig.copy()
        opt.loc[0, "shelf_level"] = 4  # move product in rack 0
        opt.loc[2, "shelf_level"] = 5  # move product in rack 1

        result = explain_all(orig, opt)
        assert "0" in result
        assert "1" in result

    def test_empty_when_no_moves(self):
        orig = pd.DataFrame({
            "rack_id": [0], "name": ["a"], "shelf_level": [3],
            "price_numeric": [5.0], "profit_margin_percentage": [30.0],
            "estimated_monthly_sales": [100], "product_width_cm": [10.0],
            "Category": ["X"],
        })
        assert explain_all(orig, orig.copy()) == {}

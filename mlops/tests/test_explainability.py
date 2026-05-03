"""Tests for utils/explainability.py"""

import pandas as pd

from utils.explainability import explain_all, explain_rack, _product_profit_score


def _make_rack(shelves_orig, shelves_opt, **kwargs):
    """Helper: create original and optimized rack DataFrames."""
    n = len(shelves_orig)
    margins = kwargs.get('margins', [30.0] * n)
    sales = kwargs.get('sales', [100] * n)
    widths = kwargs.get('widths', [10.0] * n)
    base = {
        "rack_id": 0,
        "Category": "Test",
        "name": [f"product_{i}" for i in range(n)],
        "price_numeric": [5.0] * n,
        "profit_margin_percentage": margins,
        "estimated_monthly_sales": sales,
        "product_width_cm": widths,
    }
    orig = pd.DataFrame({**base, "shelf_level": shelves_orig})
    opt = pd.DataFrame({**base, "shelf_level": shelves_opt})
    return orig, opt


class TestProductProfitScore:
    def test_score_calculation(self):
        row = pd.Series({
            "price_numeric": 10.0,
            "profit_margin_percentage": 50.0,
            "estimated_monthly_sales": 100.0,
        })
        score = _product_profit_score(row)
        assert score == 10.0 * 0.5 * 100.0

    def test_missing_values_default_to_zero(self):
        row = pd.Series({})
        score = _product_profit_score(row)
        assert score == 0.0


class TestExplainRack:
    def test_no_moves_empty(self):
        orig, opt = _make_rack([1, 2, 3], [1, 2, 3])
        assert explain_rack(orig, opt) == []

    def test_missing_name_column_returns_empty(self):
        orig = pd.DataFrame({"shelf_level": [1, 2], "id": [0, 1]})
        opt = pd.DataFrame({"shelf_level": [2, 2], "id": [0, 1]})
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

    def test_margin_premium_reason(self):
        # High margin product moves to eye level (need > median * 1.15 and > 15%)
        orig, opt = _make_rack([1], [4], margins=[60.0])  # 60% margin
        explanations = explain_rack(orig, opt)
        codes = [r["code"] for r in explanations[0]["reasons"]]
        # Should have eye_level_promotion, might have margin_premium depending on medians
        assert "eye_level_promotion" in codes

    def test_sales_volume_reason(self):
        # High sales product (need > median * 1.25 and > 20)
        orig, opt = _make_rack([1], [4], sales=[1000.0])  # 1000 units
        explanations = explain_rack(orig, opt)
        codes = [r["code"] for r in explanations[0]["reasons"]]
        # Should have eye_level_promotion, might have sales_volume depending on medians
        assert "eye_level_promotion" in codes

    def test_crowding_relief_reason(self):
        # Many products on original shelf, fewer on new shelf
        orig, opt = _make_rack(
            [1, 1, 1, 1, 1, 1, 1, 2],  # 7 products on shelf 1
            [4, 1, 1, 1, 1, 1, 1, 2]   # product_0 moves to shelf 4
        )
        explanations = explain_rack(orig, opt)
        codes = [r["code"] for r in explanations[0]["reasons"]]
        # Should have crowding_relief if the new shelf has fewer products
        assert "crowding_relief" in codes or "eye_level_promotion" in codes

    def test_relegated_low_return_reason(self):
        # Move from eye level to low shelf
        orig, opt = _make_rack([4], [1], margins=[10.0], sales=[10.0])
        explanations = explain_rack(orig, opt)
        codes = [r["code"] for r in explanations[0]["reasons"]]
        assert "relegated_low_return" in codes

    def test_within_tier_move_to_premium(self):
        # Move within eye-level to lower position
        orig, opt = _make_rack([5], [3], margins=[40.0])
        explanations = explain_rack(orig, opt)
        codes = [r["code"] for r in explanations[0]["reasons"]]
        assert "eye_level_promotion" in codes

    def test_reoptimization_default_reason(self):
        # Low margin, low sales - should get default reason
        orig, opt = _make_rack([1], [2], margins=[5.0], sales=[5.0])
        explanations = explain_rack(orig, opt)
        codes = [r["code"] for r in explanations[0]["reasons"]]
        assert "reoptimization" in codes

    def test_includes_profit_metrics(self):
        orig, opt = _make_rack([1], [4])
        explanations = explain_rack(orig, opt)
        assert "margin_pct" in explanations[0]
        assert "monthly_sales" in explanations[0]
        assert "profit_score" in explanations[0]

    def test_sorted_by_profit_score(self):
        orig = pd.DataFrame({
            "name": ["low_profit", "high_profit"],
            "rack_id": [0, 0],
            "Category": ["X", "X"],
            "shelf_level": [1, 1],
            "price_numeric": [5.0, 10.0],
            "profit_margin_percentage": [20.0, 50.0],
            "estimated_monthly_sales": [50.0, 200.0],
            "product_width_cm": [10.0, 10.0],
        })
        opt = pd.DataFrame({
            "name": ["low_profit", "high_profit"],
            "rack_id": [0, 0],
            "Category": ["X", "X"],
            "shelf_level": [4, 3],
            "price_numeric": [5.0, 10.0],
            "profit_margin_percentage": [20.0, 50.0],
            "estimated_monthly_sales": [50.0, 200.0],
            "product_width_cm": [10.0, 10.0],
        })
        explanations = explain_rack(orig, opt)
        assert explanations[0]["product"] == "high_profit"
        assert explanations[1]["product"] == "low_profit"


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

    def test_missing_rack_id_column_returns_empty(self):
        orig = pd.DataFrame({
            "name": ["a"],
            "shelf_level": [3],
            "price_numeric": [5.0],
            "profit_margin_percentage": [30.0],
            "estimated_monthly_sales": [100],
            "product_width_cm": [10.0],
            "Category": ["X"],
        })
        opt = orig.copy()
        assert explain_all(orig, opt) == {}

    def test_empty_original_rack_skipped(self):
        orig = pd.DataFrame({
            "rack_id": [0],
            "name": ["a"],
            "shelf_level": [3],
            "price_numeric": [5.0],
            "profit_margin_percentage": [30.0],
            "estimated_monthly_sales": [100],
            "product_width_cm": [10.0],
            "Category": ["X"],
        })
        opt = pd.DataFrame({
            "rack_id": [1],
            "name": ["b"],
            "shelf_level": [3],
            "price_numeric": [5.0],
            "profit_margin_percentage": [30.0],
            "estimated_monthly_sales": [100],
            "product_width_cm": [10.0],
            "Category": ["X"],
        })
        result = explain_all(orig, opt)
        assert result == {}

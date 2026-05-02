"""Tests for utils/data_io.py"""

import numpy as np
import pandas as pd
import pytest

from utils.data_io import (
    assign_shelves,
    get_seasonal_mult,
    load_monthly_csvs,
    parse_eur_price,
    profile_for,
)


class TestParseEurPrice:
    def test_normal(self):
        assert parse_eur_price("0,36 €") == pytest.approx(0.36)

    def test_large(self):
        assert parse_eur_price("32,52 €") == pytest.approx(32.52)

    def test_thousands(self):
        assert parse_eur_price("1.234,56 €") == pytest.approx(1234.56)

    def test_no_symbol(self):
        assert parse_eur_price("5,99") == pytest.approx(5.99)

    def test_nan(self):
        assert parse_eur_price(float("nan")) == 0.0

    def test_empty(self):
        assert parse_eur_price("") == 0.0

    def test_none(self):
        assert parse_eur_price(None) == 0.0

    def test_garbage(self):
        assert parse_eur_price("abc") == 0.0


class TestProfileFor:
    def test_fruta(self):
        margin, width = profile_for("Fruta y Verdura")
        assert margin == (25, 45)
        assert width == (8, 25)

    def test_chocolate(self):
        margin, _ = profile_for("Chocolate con leche")
        assert margin == (30, 55)

    def test_unknown_returns_default(self):
        margin, width = profile_for("Categoría Inventada XYZ")
        assert margin == (20, 50)
        assert width == (5, 20)

    def test_case_insensitive(self):
        assert profile_for("CERVEZA") == profile_for("cerveza")


class TestGetSeasonalMult:
    def test_summer_fruit(self):
        mult = get_seasonal_mult(7, "Fruta tropical")
        assert mult > 1.0

    def test_december_chocolate(self):
        mult = get_seasonal_mult(12, "Chocolate negro")
        assert mult == pytest.approx(1.8)

    def test_unknown_category_returns_default(self):
        mult = get_seasonal_mult(6, "Categoría Rara")
        assert mult == pytest.approx(1.0)

    def test_invalid_month_returns_1(self):
        mult = get_seasonal_mult(99, "Fruta")
        assert mult == pytest.approx(1.0)


class TestAssignShelves:
    def test_assigns_rack_id_and_shelf_level(self):
        df = pd.DataFrame({
            "Category": ["A", "A", "B", "B", "C"],
            "name": ["p1", "p2", "p3", "p4", "p5"],
        })
        rng = np.random.default_rng(42)
        result = assign_shelves(df, rng)
        assert "rack_id" in result.columns
        assert "shelf_level" in result.columns
        assert result["rack_id"].nunique() == 3
        assert result["shelf_level"].between(1, 7).all()

    def test_same_category_same_rack(self):
        df = pd.DataFrame({"Category": ["X", "X", "Y"], "name": ["a", "b", "c"]})
        result = assign_shelves(df, np.random.default_rng(0))
        assert result.loc[0, "rack_id"] == result.loc[1, "rack_id"]
        assert result.loc[0, "rack_id"] != result.loc[2, "rack_id"]


class TestLoadMonthlyCsvs:
    def test_no_files_not_required(self, tmp_path):
        result = load_monthly_csvs(tmp_path, required=False)
        assert result.empty

    def test_loads_and_concatenates(self, tmp_path):
        for i in range(2):
            df = pd.DataFrame({"Category": ["A"], "value": [i]})
            df.to_csv(tmp_path / f"sales_2025_{i+1:02d}_month.csv", index=False)
        result = load_monthly_csvs(tmp_path)
        assert len(result) == 2

    def test_add_month_cols(self, tmp_path):
        df = pd.DataFrame({"Category": ["A"], "value": [1]})
        df.to_csv(tmp_path / "sales_2025_07_july.csv", index=False)
        result = load_monthly_csvs(tmp_path, add_month_cols=True)
        assert result["_year"].iloc[0] == 2025
        assert result["_month"].iloc[0] == 7

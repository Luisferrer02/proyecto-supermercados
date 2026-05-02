"""Tests for utils/csv_schema.py"""

import pandas as pd

from utils.csv_schema import all_ok, validate_csv, validate_directory


def _write_valid_csv(path):
    """Write a minimal valid monthly sales CSV."""
    df = pd.DataFrame({
        "Category": ["Fruta", "Verdura"],
        "name": ["Manzana", "Lechuga"],
        "price_numeric": [1.50, 0.99],
        "profit_margin_percentage": [30.0, 25.0],
        "estimated_monthly_sales": [200, 150],
        "product_width_cm": [10.0, 15.0],
        "rack_id": [0, 1],
        "shelf_level": [3, 5],
    })
    df.to_csv(path, index=False)


class TestValidateCsv:
    def test_valid_file(self, tmp_path):
        csv = tmp_path / "sales_2025_01_january.csv"
        _write_valid_csv(csv)
        result = validate_csv(csv)
        assert result.ok
        assert result.row_count == 2
        assert result.errors == []

    def test_missing_file(self, tmp_path):
        result = validate_csv(tmp_path / "nonexistent.csv")
        assert not result.ok
        assert "does not exist" in result.errors[0]

    def test_missing_columns(self, tmp_path):
        csv = tmp_path / "sales_2025_01_january.csv"
        pd.DataFrame({"Category": ["A"], "name": ["B"]}).to_csv(csv, index=False)
        result = validate_csv(csv)
        assert not result.ok
        assert any("Missing required" in e for e in result.errors)

    def test_empty_csv(self, tmp_path):
        csv = tmp_path / "sales_2025_01_january.csv"
        pd.DataFrame(columns=["Category"]).to_csv(csv, index=False)
        result = validate_csv(csv)
        assert not result.ok

    def test_out_of_range_warns(self, tmp_path):
        csv = tmp_path / "sales_2025_01_january.csv"
        df = pd.DataFrame({
            "Category": ["A"],
            "name": ["B"],
            "price_numeric": [9999.0],  # out of range
            "profit_margin_percentage": [30.0],
            "estimated_monthly_sales": [100],
            "product_width_cm": [10.0],
            "rack_id": [0],
            "shelf_level": [3],
        })
        df.to_csv(csv, index=False)
        result = validate_csv(csv)
        assert result.ok  # warnings don't fail
        assert len(result.warnings) > 0


class TestValidateDirectory:
    def test_valid_directory(self, tmp_path):
        _write_valid_csv(tmp_path / "sales_2025_01_january.csv")
        _write_valid_csv(tmp_path / "sales_2025_02_february.csv")
        results = validate_directory(tmp_path)
        assert all_ok(results)
        assert len(results) == 2

    def test_empty_directory(self, tmp_path):
        results = validate_directory(tmp_path)
        assert not all_ok(results)

    def test_nonexistent_directory(self):
        results = validate_directory("/nonexistent/path")
        assert not all_ok(results)

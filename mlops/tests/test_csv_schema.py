"""Tests for utils/csv_schema.py"""

import pandas as pd

from utils.csv_schema import all_ok, format_report, validate_csv, validate_directory


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

    def test_non_numeric_values_error(self, tmp_path):
        csv = tmp_path / "sales_2025_01_january.csv"
        df = pd.DataFrame({
            "Category": ["A"],
            "name": ["B"],
            "price_numeric": ["not_a_number"],
            "profit_margin_percentage": [30.0],
            "estimated_monthly_sales": [100],
            "product_width_cm": [10.0],
            "rack_id": [0],
            "shelf_level": [3],
        })
        df.to_csv(csv, index=False)
        result = validate_csv(csv)
        assert not result.ok
        assert any("non-numeric" in e for e in result.errors)

    def test_null_values_in_required_column_warns(self, tmp_path):
        csv = tmp_path / "sales_2025_01_january.csv"
        df = pd.DataFrame({
            "Category": ["A", None],
            "name": ["B", "C"],
            "price_numeric": [1.50, 2.00],
            "profit_margin_percentage": [30.0, 25.0],
            "estimated_monthly_sales": [100, 200],
            "product_width_cm": [10.0, 15.0],
            "rack_id": [0, 1],
            "shelf_level": [3, 5],
        })
        df.to_csv(csv, index=False)
        result = validate_csv(csv)
        assert result.ok  # warnings don't fail
        assert any("null" in w.lower() for w in result.warnings)

    def test_wrong_filename_pattern_warns(self, tmp_path):
        csv = tmp_path / "wrong_file_name.csv"
        _write_valid_csv(csv)
        result = validate_csv(csv)
        assert result.ok  # warnings don't fail
        assert any("sales_" in w and ".csv" in w for w in result.warnings)

    def test_non_numeric_parse_error(self, tmp_path):
        csv = tmp_path / "sales_2025_01_january.csv"
        df = pd.DataFrame({
            "Category": ["A"],
            "name": ["B"],
            "price_numeric": ["abc"],
            "profit_margin_percentage": ["xyz"],  # non-numeric
            "estimated_monthly_sales": [100],
            "product_width_cm": [10.0],
            "rack_id": [0],
            "shelf_level": [3],
        })
        df.to_csv(csv, index=False)
        result = validate_csv(csv)
        assert not result.ok
        assert any("non-numeric" in e for e in result.errors)


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


class TestFormatReport:
    def test_format_single_ok_result(self, tmp_path):
        csv = tmp_path / "sales_2025_01_january.csv"
        _write_valid_csv(csv)
        results = [validate_csv(csv)]
        report = format_report(results)
        assert "OK" in report
        assert "1" in report  # total files
        assert str(csv) in report

    def test_format_multiple_mixed_results(self, tmp_path):
        csv1 = tmp_path / "sales_2025_01_january.csv"
        csv2 = tmp_path / "invalid.csv"
        _write_valid_csv(csv1)
        csv2.write_text("invalid")
        results = [validate_csv(csv1), validate_csv(csv2)]
        report = format_report(results)
        assert "OK: 1" in report
        assert "Failed: 1" in report


class TestAllOk:
    def test_all_ok_with_empty_list(self):
        assert not all_ok([])

    def test_all_ok_with_all_passing(self, tmp_path):
        csv = tmp_path / "sales_2025_01_january.csv"
        _write_valid_csv(csv)
        results = [validate_csv(csv)]
        assert all_ok(results)

    def test_all_ok_with_one_failing(self, tmp_path):
        csv = tmp_path / "sales_2025_01_january.csv"
        _write_valid_csv(csv)
        valid_result = validate_csv(csv)
        invalid_result = validate_csv(tmp_path / "nonexistent.csv")
        results = [valid_result, invalid_result]
        assert not all_ok(results)

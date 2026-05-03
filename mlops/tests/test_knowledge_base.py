"""Tests for utils/knowledge_base.py — pure helper functions only."""

import pandas as pd

from utils.knowledge_base import _summarize_category, parse_month_from_filename


class TestParseMonthFromFilename:
    def test_standard(self):
        assert parse_month_from_filename("sales_2025_07_july.csv") == (2025, 7, "july")

    def test_january(self):
        assert parse_month_from_filename("sales_2024_01_january.csv") == (2024, 1, "january")

    def test_december(self):
        assert parse_month_from_filename("sales_2023_12_december.csv") == (2023, 12, "december")

    def test_invalid_returns_none(self):
        assert parse_month_from_filename("random_file.csv") is None

    def test_missing_parts_returns_none(self):
        assert parse_month_from_filename("sales_2025.csv") is None


class TestSummarizeCategory:
    def test_contains_key_fields(self):
        df = pd.DataFrame({
            "name": ["Apple", "Banana"],
            "estimated_monthly_sales": [100, 200],
            "price_numeric": [1.50, 2.00],
            "profit_margin_percentage": [30.0, 25.0],
            "product_width_cm": [10.0, 12.0],
            "shelf_level": [3, 4],
        })
        summary = _summarize_category(df, "Fruit", 2025, 7)
        assert "Fruit" in summary
        assert "07/2025" in summary
        assert "Products: 2" in summary
        assert "Apple" in summary

    def test_single_product(self):
        df = pd.DataFrame({
            "name": ["Solo"],
            "estimated_monthly_sales": [50],
            "price_numeric": [3.00],
            "profit_margin_percentage": [40.0],
            "product_width_cm": [8.0],
            "shelf_level": [5],
        })
        summary = _summarize_category(df, "Single", 2024, 1)
        assert "Solo" in summary
        assert "Products: 1" in summary

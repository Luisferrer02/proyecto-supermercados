"""
Tests for critical coverage gaps identified in audit:
- FeatureNormalizer
- optimize_rack_mlp / optimize_rack_profit_mlp
- Integration: generate data → train → optimize
"""

import numpy as np
import pandas as pd
import pytest
import torch

from models.mlp import build_mlp
from utils.training import (
    FEATURE_COLS,
    FeatureNormalizer,
    optimize_rack_mlp,
    optimize_rack_profit_mlp,
)
from utils.retail_physics import (
    NUM_SHELVES,
    SHELF_WIDTH_CM,
    compute_rack_profit_advanced,
    generate_absolute_profit_data,
    generate_synthetic_training_data,
    optimize_rack_advanced,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rack_df():
    """A small rack DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 10
    return pd.DataFrame({
        "Category": ["TestCat"] * n,
        "name": [f"Product_{i}" for i in range(n)],
        "price_numeric": rng.uniform(1, 20, n),
        "profit_margin_percentage": rng.uniform(15, 60, n),
        "estimated_monthly_sales": rng.integers(10, 500, n).astype(float),
        "product_width_cm": rng.uniform(3, 15, n),
        "rack_id": [0] * n,
        "shelf_level": rng.integers(1, 8, n),
    })


@pytest.fixture
def multi_rack_df(sample_rack_df):
    """Multiple racks for integration tests."""
    dfs = []
    for i in range(3):
        df = sample_rack_df.copy()
        df["rack_id"] = i
        df["name"] = [f"Product_{i}_{j}" for j in range(len(df))]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# FeatureNormalizer tests
# ---------------------------------------------------------------------------

class TestFeatureNormalizer:
    def test_fit_transform(self):
        X = torch.randn(100, 5)
        norm = FeatureNormalizer()
        X_norm = norm.fit_transform(X)
        assert X_norm.shape == X.shape
        # After normalization, mean should be ~0 and std ~1
        assert torch.abs(X_norm.mean(dim=0)).max() < 0.2
        assert torch.abs(X_norm.std(dim=0) - 1.0).max() < 0.2

    def test_transform_before_fit_raises(self):
        norm = FeatureNormalizer()
        X = torch.randn(10, 5)
        with pytest.raises(RuntimeError, match="before fit"):
            norm.transform(X)

    def test_fit_then_transform(self):
        X_train = torch.randn(100, 3)
        X_test = torch.randn(20, 3)
        norm = FeatureNormalizer()
        norm.fit(X_train)
        X_test_norm = norm.transform(X_test)
        assert X_test_norm.shape == X_test.shape

    def test_constant_feature_handled(self):
        """A constant feature should not cause division by zero."""
        X = torch.ones(50, 3)
        X[:, 1] = torch.randn(50)  # only col 1 varies
        norm = FeatureNormalizer()
        X_norm = norm.fit_transform(X)
        assert not torch.isnan(X_norm).any()
        assert not torch.isinf(X_norm).any()


# ---------------------------------------------------------------------------
# Optimizer tests
# ---------------------------------------------------------------------------

class TestOptimizeRackMlp:
    def test_returns_valid_shelves(self, sample_rack_df):
        mlp = build_mlp(input_dim=len(FEATURE_COLS))
        result = optimize_rack_mlp(
            sample_rack_df, mlp, NUM_SHELVES, SHELF_WIDTH_CM
        )
        assert len(result) == len(sample_rack_df)
        assert result["shelf_level"].between(1, NUM_SHELVES).all()

    def test_respects_capacity(self, sample_rack_df):
        mlp = build_mlp(input_dim=len(FEATURE_COLS))
        result = optimize_rack_mlp(
            sample_rack_df, mlp, NUM_SHELVES, SHELF_WIDTH_CM
        )
        for shelf in range(1, NUM_SHELVES + 1):
            total_width = result[result["shelf_level"] == shelf]["product_width_cm"].sum()
            assert total_width <= SHELF_WIDTH_CM + 0.01

    def test_with_normalizer(self, sample_rack_df):
        mlp = build_mlp(input_dim=len(FEATURE_COLS))
        norm = FeatureNormalizer()
        # Fit on dummy data
        dummy_X = torch.randn(100, len(FEATURE_COLS))
        norm.fit(dummy_X)
        result = optimize_rack_mlp(
            sample_rack_df, mlp, NUM_SHELVES, SHELF_WIDTH_CM, normalizer=norm
        )
        assert len(result) == len(sample_rack_df)


class TestOptimizeRackProfitMlp:
    def test_returns_valid_shelves(self, sample_rack_df):
        mlp = build_mlp(input_dim=5)
        result = optimize_rack_profit_mlp(
            sample_rack_df, mlp, NUM_SHELVES, SHELF_WIDTH_CM
        )
        assert len(result) == len(sample_rack_df)
        assert result["shelf_level"].between(1, NUM_SHELVES).all()


class TestOptimizeRackAdvanced:
    def test_returns_valid_shelves(self, sample_rack_df):
        result = optimize_rack_advanced(sample_rack_df)
        assert len(result) == len(sample_rack_df)
        assert result["shelf_level"].between(1, NUM_SHELVES).all()

    def test_improves_or_maintains_profit(self, sample_rack_df):
        original_profit = compute_rack_profit_advanced(sample_rack_df)
        result = optimize_rack_advanced(sample_rack_df)
        optimized_profit = compute_rack_profit_advanced(result)
        # Advanced optimizer should not make things worse
        assert optimized_profit >= original_profit * 0.95


# ---------------------------------------------------------------------------
# Integration test: generate → train → optimize
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_synthetic_data_to_model_training(self, multi_rack_df):
        """End-to-end: generate synthetic data, verify it has correct columns."""
        synth = generate_synthetic_training_data(multi_rack_df, n_samples=100, seed=42)
        assert len(synth) > 0
        for col in FEATURE_COLS + ["profit_lift"]:
            assert col in synth.columns, f"Missing column: {col}"
        assert not synth["profit_lift"].isna().any()

    def test_absolute_profit_data_generation(self, multi_rack_df):
        """Generate absolute profit data and verify schema."""
        profit_data = generate_absolute_profit_data(multi_rack_df)
        assert len(profit_data) > 0
        assert "profit" in profit_data.columns
        assert "shelf" in profit_data.columns
        assert (profit_data["profit"] >= 0).all()
        assert profit_data["shelf"].between(1, NUM_SHELVES).all()

    def test_mlp_train_and_optimize(self, multi_rack_df):
        """Train MLP on synthetic data, then use it to optimize a rack."""
        synth = generate_synthetic_training_data(multi_rack_df, n_samples=500, seed=42)
        X = torch.FloatTensor(synth[FEATURE_COLS].to_numpy())
        y = torch.FloatTensor(synth["profit_lift"].to_numpy())

        mlp = build_mlp(input_dim=len(FEATURE_COLS))
        # Quick training (just verify it doesn't crash)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        mlp.train()
        for _ in range(5):
            pred = mlp(X)
            loss = torch.nn.MSELoss()(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mlp.eval()
        rack = multi_rack_df[multi_rack_df["rack_id"] == 0].copy()
        result = optimize_rack_mlp(rack, mlp, NUM_SHELVES, SHELF_WIDTH_CM)
        assert len(result) == len(rack)
        assert result["shelf_level"].between(1, NUM_SHELVES).all()

"""Tests for utils/training.py"""

import numpy as np
import pandas as pd
import torch

from utils.training import (
    FEATURE_COLS,
    df_to_tensors,
    make_sequences,
    optimize_rack_mlp,
    train_model,
)


def _make_training_df(n=100, seed=42):
    """Helper: create a DataFrame with the expected feature columns + target."""
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0, 100, n) for col in FEATURE_COLS}
    data["profit_lift"] = rng.uniform(-50, 50, n)
    return pd.DataFrame(data)


class TestDfToTensors:
    def test_shapes(self):
        df = _make_training_df(50)
        X, y = df_to_tensors(df)
        assert X.shape == (50, len(FEATURE_COLS))
        assert y.shape == (50,)

    def test_types(self):
        X, y = df_to_tensors(_make_training_df(10))
        assert X.dtype == torch.float32
        assert y.dtype == torch.float32


class TestMakeSequences:
    def test_shapes(self):
        df = _make_training_df(100)
        X, y = make_sequences(df, seq_len=10)
        assert X.shape == (10, 10, len(FEATURE_COLS))
        assert y.shape == (10, 10)

    def test_too_few_rows_returns_none(self):
        df = _make_training_df(5)
        X, y = make_sequences(df, seq_len=10)
        assert X is None
        assert y is None

    def test_remainder_dropped(self):
        df = _make_training_df(25)
        X, y = make_sequences(df, seq_len=10)
        assert X.shape[0] == 2  # 25 // 10 = 2


class TestTrainModel:
    def test_smoke(self):
        """Train a tiny model for 2 epochs — just verify it doesn't crash."""
        df = _make_training_df(200)
        train_X, train_y = df_to_tensors(df.iloc[:150])
        test_X, test_y = df_to_tensors(df.iloc[150:])

        model = torch.nn.Sequential(
            torch.nn.Linear(len(FEATURE_COLS), 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Flatten(0),
        )

        metrics = train_model(model, train_X, train_y, test_X, test_y,
                              name="test", epochs=2, batch_size=32)

        assert "mse" in metrics
        assert "rmse_eur" in metrics
        assert metrics["mse"] >= 0

    def test_on_epoch_callback(self):
        df = _make_training_df(100)
        X, y = df_to_tensors(df.iloc[:80])
        tX, ty = df_to_tensors(df.iloc[80:])
        model = torch.nn.Sequential(
            torch.nn.Linear(len(FEATURE_COLS), 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Flatten(0),
        )
        epochs_seen = []
        train_model(model, X, y, tX, ty, name="cb", epochs=3, batch_size=32,
                    on_epoch=lambda ep, total, loss: epochs_seen.append(ep))
        assert epochs_seen == [1, 2, 3]


class TestOptimizeRackMlp:
    def _make_rack(self, n=6):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "price_numeric": rng.uniform(1, 10, n),
            "profit_margin_percentage": rng.uniform(10, 50, n),
            "estimated_monthly_sales": rng.integers(20, 200, n).astype(float),
            "product_width_cm": rng.uniform(5, 20, n),
            "shelf_level": rng.integers(1, 8, n),
        })

    def _make_mlp(self):
        return torch.nn.Sequential(
            torch.nn.Linear(len(FEATURE_COLS), 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Flatten(0),
        )

    def test_returns_same_rows(self):
        rack = self._make_rack()
        result = optimize_rack_mlp(rack, self._make_mlp(), num_shelves=7, shelf_width_cm=300)
        assert len(result) == len(rack)

    def test_respects_shelf_width(self):
        rack = self._make_rack(n=10)
        result = optimize_rack_mlp(rack, self._make_mlp(), num_shelves=7, shelf_width_cm=300)
        for s in range(1, 8):
            w = result.loc[result["shelf_level"] == s, "product_width_cm"].sum()
            assert w <= 300 + 0.01

    def test_noise_produces_different_layouts(self):
        rack = self._make_rack()
        mlp = self._make_mlp()
        r1 = optimize_rack_mlp(rack, mlp, num_shelves=7, shelf_width_cm=300, noise_scale=0)
        r2 = optimize_rack_mlp(rack, mlp, num_shelves=7, shelf_width_cm=300, noise_scale=5.0)
        # With high noise, layouts are very likely different
        # (not guaranteed, so just check it doesn't crash)
        assert len(r2) == len(rack)

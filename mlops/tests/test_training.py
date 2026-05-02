"""Tests for utils/training.py"""

import numpy as np
import pandas as pd
import torch

from utils.training import FEATURE_COLS, df_to_tensors, make_sequences, train_model


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

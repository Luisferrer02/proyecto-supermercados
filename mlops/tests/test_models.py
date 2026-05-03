"""Tests for models/ — smoke tests for forward pass dimensions."""

import torch

from models.lstm_model import build_lstm
from models.mlp import build_mlp
from models.transformer_model import build_transformer
from utils.training import FEATURE_COLS

INPUT_DIM = len(FEATURE_COLS)


class TestMLP:
    def test_forward_shape(self):
        model = build_mlp(input_dim=INPUT_DIM)
        x = torch.randn(8, INPUT_DIM)
        out = model(x)
        assert out.shape == (8,)

    def test_single_sample(self):
        model = build_mlp(input_dim=INPUT_DIM)
        model.eval()  # BatchNorm requires eval mode for batch_size=1
        x = torch.randn(1, INPUT_DIM)
        out = model(x)
        assert out.shape == (1,)


class TestLSTM:
    def test_forward_shape(self):
        model = build_lstm(input_dim=INPUT_DIM)
        x = torch.randn(4, 10, INPUT_DIM)  # batch=4, seq=10
        out = model(x)
        assert out.shape == (4, 10)

    def test_variable_seq_len(self):
        model = build_lstm(input_dim=INPUT_DIM)
        x = torch.randn(2, 5, INPUT_DIM)
        out = model(x)
        assert out.shape == (2, 5)


class TestTransformer:
    def test_forward_shape(self):
        model = build_transformer(input_dim=INPUT_DIM)
        x = torch.randn(4, 10, INPUT_DIM)
        out = model(x)
        assert out.shape == (4, 10)

    def test_variable_seq_len(self):
        model = build_transformer(input_dim=INPUT_DIM)
        x = torch.randn(2, 20, INPUT_DIM)
        out = model(x)
        assert out.shape == (2, 20)

    def test_single_sequence(self):
        """Test transformer with single sequence."""
        model = build_transformer(input_dim=INPUT_DIM)
        model.eval()  # BatchNorm requires eval for small batch
        x = torch.randn(1, 1, INPUT_DIM)
        out = model(x)
        assert out.shape == (1, 1)

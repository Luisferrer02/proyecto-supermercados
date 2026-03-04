"""
LSTM Model for Profit Lift Prediction
======================================
Treats products on a shelf as a left-to-right sequence to capture
"neighbouring product" effects.

Input  : Sequence of product feature vectors (batch, seq_len, feat_dim)
Output : Per-product profit lift predictions  (batch, seq_len)
"""

import torch
import torch.nn as nn


class ProfitLiftLSTM(nn.Module):
    """Two-layer LSTM for sequential shelf modelling."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            predictions: (batch, seq_len)
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        out = self.fc(lstm_out)             # (batch, seq_len, 1)
        return out.squeeze(-1)


def build_lstm(input_dim: int = 6) -> ProfitLiftLSTM:
    """Factory helper."""
    return ProfitLiftLSTM(input_dim=input_dim)

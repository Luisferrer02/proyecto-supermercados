"""
Transformer Model for Profit Lift Prediction
=============================================
Uses Self-Attention to find dependencies between products on
different shelves of the same rack.

Input  : Set of product feature vectors across all shelves (batch, n_products, feat_dim)
Output : Per-product profit lift predictions (batch, n_products)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Simple sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ProfitLiftTransformer(nn.Module):
    """Transformer encoder with Pre-LN and feature standardization."""

    def __init__(self, input_dim: int = 10, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()

        # Feature standardization (learned per-feature scale/shift)
        self.feature_norm = nn.BatchNorm1d(input_dim)

        # Project input features to d_model dimensions
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc = PositionalEncoding(d_model)

        # Pre-LN Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output head
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_products, input_dim)
        Returns:
            predictions: (batch, n_products)
        """
        batch, seq_len, feat = x.shape

        # Standardize features across the batch
        x_flat = x.reshape(-1, feat)             # (batch*seq, feat)
        x_flat = self.feature_norm(x_flat)       # BatchNorm per feature
        x = x_flat.reshape(batch, seq_len, feat)

        x = self.input_proj(x)                   # (batch, n, d_model)
        x = self.input_norm(x)
        x = self.pos_enc(x)
        x = self.encoder(x)                      # (batch, n, d_model)
        out = self.fc_out(x)                     # (batch, n, 1)
        return out.squeeze(-1)


def build_transformer(input_dim: int = 6) -> ProfitLiftTransformer:
    """Factory helper."""
    return ProfitLiftTransformer(input_dim=input_dim)

"""
MLP Model for Profit Lift Prediction
=====================================
Input  : Flattened feature vector (price, margin, sales, width, shelf levels)
Output : Predicted profit lift (scalar)
"""

import torch
import torch.nn as nn


class ProfitLiftMLP(nn.Module):
    """Three-hidden-layer MLP for profit-lift regression."""

    def __init__(self, input_dim: int = 10, hidden_dims=(256, 128, 64), dropout: float = 0.15):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_mlp(input_dim: int = 6) -> ProfitLiftMLP:
    """Factory helper."""
    return ProfitLiftMLP(input_dim=input_dim)

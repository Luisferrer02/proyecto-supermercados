"""
utils/training.py — Shared model training helpers
==================================================
Used by both 02_train_models.py and 04_ingest.py to avoid duplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

FEATURE_COLS = [
    "price_numeric",
    "profit_margin_percentage",
    "estimated_monthly_sales",
    "product_width_cm",
    "original_shelf",
    "new_shelf",
    "n_products_on_original_shelf",
    "n_products_on_new_shelf",
    "n_shelves_used",
    "rack_product_count",
]

# Simpler features for absolute-profit MLP
PROFIT_FEATURE_COLS = [
    "price_numeric",
    "profit_margin_percentage",
    "estimated_monthly_sales",
    "product_width_cm",
    "shelf",
]

EPOCHS = 80
BATCH = 128
LR = 5e-4


class FeatureNormalizer:
    """Z-score normalizer: (x - mean) / std. Fit on train, apply to all."""

    def __init__(self):
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, X: torch.Tensor) -> "FeatureNormalizer":
        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0).clamp(min=1e-2)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureNormalizer.transform() called before fit()")
        return (X - self.mean) / self.std

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)


def df_to_tensors(data_df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert DataFrame to (features, targets) float tensors."""
    X = torch.FloatTensor(data_df[FEATURE_COLS].to_numpy().copy())
    y = torch.FloatTensor(data_df["profit_lift"].to_numpy().copy())
    return X, y


def make_sequences(
    data_df: pd.DataFrame, seq_len: int = 10
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Shape a DataFrame into (n_seq, seq_len, n_features) tensors.

    Returns (None, None) if there are fewer than seq_len rows.
    """
    X_all = data_df[FEATURE_COLS].to_numpy().copy().astype(np.float32)
    y_all = data_df["profit_lift"].to_numpy().copy().astype(np.float32)
    n_seq = len(X_all) // seq_len
    if n_seq == 0:
        return None, None
    X_seq = X_all[: n_seq * seq_len].reshape(n_seq, seq_len, -1)
    y_seq = y_all[: n_seq * seq_len].reshape(n_seq, seq_len)
    return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)


def train_model(
    model: nn.Module,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    *,
    name: str = "model",
    epochs: int = EPOCHS,
    batch_size: int = BATCH,
    lr: float = LR,
    clip_grad: float = 0.0,
    on_epoch: object = None,
    patience: int = 30,
) -> dict:
    """Train a supervised model (flat or sequence) and return test metrics.

    Args:
        model:      Any nn.Module whose forward() accepts train_X.
        train_X:    Training features tensor.
        train_y:    Training targets tensor.
        test_X:     Test features tensor.
        test_y:     Test targets tensor.
        name:       Label used in progress messages.
        epochs:     Number of training epochs.
        batch_size: Mini-batch size.
        lr:         Adam learning rate.
        clip_grad:  If > 0, clip gradient norm to this value.
        on_epoch:   Optional callable(epoch, total, avg_loss) for progress.
        patience:   Early stopping patience (epochs without improvement).

    Returns:
        dict with keys: mse, mse_eur2, rmse_eur, mae_eur
    """
    # Auto-select best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"   [{name}] Using device: {device}")

    model = model.to(device)
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    test_X = test_X.to(device)
    test_y = test_y.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)

    model.train()
    best_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    for epoch in range(epochs):
        total_loss, count = 0.0, 0
        for bx, by in loader:
            pred = model(bx)
            loss = criterion(pred, by)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   [{name}] NaN/Inf loss at epoch {epoch+1}, stopping.")
                if best_state:
                    model.load_state_dict(best_state)
                break
            optimizer.zero_grad()
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            total_loss += loss.item() * len(bx)
            count += len(bx)
        else:
            avg_loss = total_loss / max(count, 1)
            scheduler.step(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
            if on_epoch:
                on_epoch(epoch + 1, epochs, avg_loss)
            elif (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"   [{name}] Epoch {epoch+1}/{epochs}  Train MSE: {avg_loss:.4f}")
            if epochs_no_improve >= patience:
                print(f"   [{name}] Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
            continue
        break  # NaN detected

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(test_X)
        mse = criterion(pred, test_y).item()
    rmse = float(mse**0.5)
    mae = float(torch.mean(torch.abs(pred - test_y)).item())
    print(f"   [{name}] Test MSE: {mse:.2f} €²  |  RMSE: {rmse:.2f} €  |  MAE: {mae:.2f} €")

    # Move back to CPU for saving and inference
    model = model.cpu()
    return {"mse": mse, "mse_eur2": mse, "rmse_eur": rmse, "mae_eur": mae}


def optimize_rack_profit_mlp(
    rack_df: pd.DataFrame,
    mlp_model: nn.Module,
    num_shelves: int,
    shelf_width_cm: float,
    noise_scale: float = 0.0,
    normalizer: "FeatureNormalizer | None" = None,
) -> pd.DataFrame:
    """Assign each product to the shelf with highest predicted absolute profit.

    For each product, predict profit on all 7 shelves and pick the best one
    that still has capacity. Simple, robust, no normalizer needed.
    """
    result = rack_df.copy()
    result["_pp"] = (
        result["price_numeric"]
        * (result["profit_margin_percentage"] / 100.0)
        * result["estimated_monthly_sales"]
    )
    result = result.sort_values("_pp", ascending=False)

    capacity = dict.fromkeys(range(1, num_shelves + 1), shelf_width_cm)
    mlp_model.eval()
    _rng = np.random.default_rng()
    assignments: dict = {}

    for idx in result.index:
        row = result.loc[idx]
        w = row["product_width_cm"]
        best_shelf = int(row["shelf_level"])
        best_profit = -float("inf")

        for s in range(1, num_shelves + 1):
            if capacity[s] < w:
                continue
            features = torch.FloatTensor([[
                row["price_numeric"], row["profit_margin_percentage"],
                row["estimated_monthly_sales"], row["product_width_cm"],
                s,
            ]])
            if normalizer is not None:
                features = normalizer.transform(features)
            with torch.no_grad():
                pred_profit = mlp_model(features).item()
            if noise_scale > 0:
                pred_profit += float(_rng.normal(0, noise_scale))
            if pred_profit > best_profit:
                best_profit = pred_profit
                best_shelf = s

        assignments[idx] = best_shelf
        capacity[best_shelf] -= w

    result = result.drop(columns=["_pp"])
    result["shelf_level"] = pd.Series(assignments)
    return result


def optimize_rack_mlp(
    rack_df: pd.DataFrame,
    mlp_model: nn.Module,
    num_shelves: int,
    shelf_width_cm: float,
    noise_scale: float = 0.0,
    normalizer: "FeatureNormalizer | None" = None,
) -> pd.DataFrame:
    """Greedy MLP-guided shelf assignment for a single rack.

    Sorts products by profit potential (highest first), then for each product
    tries every shelf and picks the one with the highest predicted profit lift.
    ``noise_scale > 0`` adds Gaussian noise to the predictions to generate
    diverse candidate layouts for ensemble scoring.

    Returns a copy of ``rack_df`` with updated ``shelf_level``.
    """
    result = rack_df.copy()
    result["_pp"] = (
        result["price_numeric"]
        * (result["profit_margin_percentage"] / 100.0)
        * result["estimated_monthly_sales"]
    )
    result = result.sort_values("_pp", ascending=False)

    capacity = dict.fromkeys(range(1, num_shelves + 1), shelf_width_cm)
    shelf_counts = dict.fromkeys(range(1, num_shelves + 1), 0)
    mlp_model.eval()
    _rng = np.random.default_rng()

    assignments: dict = {}

    for idx in result.index:
        row = result.loc[idx]
        original_shelf = int(row["shelf_level"])
        w = row["product_width_cm"]
        best_shelf = original_shelf
        n_shelves_used = sum(1 for c in shelf_counts.values() if c > 0)

        # Batch all 7 shelves in one forward pass
        valid_shelves = [s for s in range(1, num_shelves + 1) if capacity[s] >= w]
        if not valid_shelves:
            assignments[idx] = original_shelf
            continue

        batch_features = torch.FloatTensor([
            [row["price_numeric"], row["profit_margin_percentage"],
             row["estimated_monthly_sales"], row["product_width_cm"],
             original_shelf, s,
             shelf_counts.get(original_shelf, 0), shelf_counts.get(s, 0),
             n_shelves_used, len(result)]
            for s in valid_shelves
        ])
        if normalizer is not None:
            batch_features = normalizer.transform(batch_features)
        with torch.no_grad():
            lifts = mlp_model(batch_features).numpy()
        if noise_scale > 0:
            lifts = lifts + _rng.normal(0, noise_scale, size=lifts.shape)

        best_idx = int(lifts.argmax())
        best_shelf = valid_shelves[best_idx]

        assignments[idx] = best_shelf
        capacity[best_shelf] -= w
        shelf_counts[best_shelf] += 1

    result = result.drop(columns=["_pp"])
    result["shelf_level"] = pd.Series(assignments)
    return result



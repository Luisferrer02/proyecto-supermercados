#!/usr/bin/env python3
"""
02_train_models.py — Model Training & Comparison Framework
===========================================================
Trains four architectures on synthetic profit-lift data generated from the
augmented product CSV:
  1. MLP
  2. LSTM
  3. Transformer
  4. PPO (Reinforcement Learning)

Each model uses its OWN predictions to optimize shelf layouts.

Usage:
    python 02_train_models.py                     # Full training
    python 02_train_models.py --sample-size 50    # Quick smoke test
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.retail_physics import (
    generate_synthetic_training_data,
    compute_rack_profit,
    optimize_rack_greedy,
    SHELF_WIDTH_CM,
    NUM_SHELVES,
)
from models.mlp import build_mlp
from models.lstm_model import build_lstm
from models.transformer_model import build_transformer
from models.ppo_agent import RackEnv, PPOTrainer
from utils.model_persistence import save_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MONTHLY_DIR = BASE_DIR / "data" / "monthly"
RESULTS_DIR = BASE_DIR / "results"
EPOCHS = 80
BATCH = 128
LR = 5e-4

# Extended feature set (includes rack context)
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


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_prepare(sample_size: int | None = None,
                     val_fraction: float = 0.15,
                     test_fraction: float = 0.15,
                     rack_holdout_fraction: float = 0.20,
                     split_seed: int = 42):
    """Load monthly CSVs and build a train/val/test split + rack holdout.

    The existing code used two independent draws (seed=42 for train,
    seed=99 for test) from the same population. Auditors flagged that as
    insufficiently independent. We now:

      1. Reserve `rack_holdout_fraction` of racks entirely — no sample
         from these racks ever enters train/val/test.
      2. From the remaining racks, generate a single pool of synthetic
         samples, shuffle with `split_seed`, and slice into train / val
         / test in `(1 - val - test) / val / test` proportions.
      3. Save a SHA-256 hash of the test indices and the holdout rack
         list to `results/test_split_hash.json` so the split is
         reproducible across runs.

    Returns: (df, train_df, val_df, test_df, holdout_df)
    """
    import hashlib
    import json as _json

    print(f"Loading monthly data from {MONTHLY_DIR}...")

    csv_files = sorted(MONTHLY_DIR.glob("sales_*.csv"))
    if not csv_files:
        print(f"ERROR: No sales_*.csv files found in {MONTHLY_DIR}")
        print("       Run 01_generate_monthly_sales.py first.")
        sys.exit(1)

    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))
        print(f"   Loaded {f.name} ({len(dfs[-1])} products)")
    df = pd.concat(dfs, ignore_index=True)

    if sample_size:
        df = df.head(sample_size)
    print(f"   Total: {len(df)} product-month records")

    # ---- Rack holdout -----------------------------------------------------
    rng = np.random.default_rng(split_seed)
    all_racks = np.sort(df["rack_id"].unique())
    n_holdout = max(1, int(round(len(all_racks) * rack_holdout_fraction)))
    holdout_racks = rng.choice(all_racks, size=n_holdout, replace=False)
    holdout_racks = sorted(int(r) for r in holdout_racks)
    train_pool_racks = [int(r) for r in all_racks if int(r) not in set(holdout_racks)]

    print(f"   Racks: {len(all_racks)} total, "
          f"{len(train_pool_racks)} for train/val/test, "
          f"{len(holdout_racks)} held out")

    # ---- Synthetic sample generation --------------------------------------
    pool_df_source = df[df["rack_id"].isin(train_pool_racks)].reset_index(drop=True)
    holdout_df_source = df[df["rack_id"].isin(holdout_racks)].reset_index(drop=True)

    # Generate roughly 3 swaps per product, split into train/val/test
    n_total = max(20000, len(pool_df_source) * 3) + max(3000, len(pool_df_source))
    pool_samples = generate_synthetic_training_data(
        pool_df_source, n_samples=n_total, seed=split_seed)

    # Shuffle with a fixed permutation, then slice
    perm = rng.permutation(len(pool_samples))
    pool_samples = pool_samples.iloc[perm].reset_index(drop=True)
    n = len(pool_samples)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test_df = pool_samples.iloc[:n_test].reset_index(drop=True)
    val_df = pool_samples.iloc[n_test:n_test + n_val].reset_index(drop=True)
    train_df = pool_samples.iloc[n_test + n_val:].reset_index(drop=True)

    # Rack-holdout samples (smaller N because they're only for reporting)
    holdout_n = max(3000, len(holdout_df_source))
    holdout_df = generate_synthetic_training_data(
        holdout_df_source, n_samples=holdout_n, seed=split_seed + 1)

    print(f"   Split: train={len(train_df)}, val={len(val_df)}, "
          f"test={len(test_df)}, rack_holdout={len(holdout_df)}")

    # ---- Persist the split hash for reproducibility -----------------------
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        test_idx_bytes = perm[:n_test].tobytes()
        split_hash = hashlib.sha256(test_idx_bytes).hexdigest()
        with open(RESULTS_DIR / "test_split_hash.json", "w") as f:
            _json.dump({
                "split_seed": split_seed,
                "test_indices_hash": split_hash,
                "holdout_racks": holdout_racks,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "n_holdout": len(holdout_df),
            }, f, indent=2)
        print(f"   Split hash (first 16 chars): {split_hash[:16]}…")
    except Exception as exc:
        print(f"   ⚠  Could not write split hash: {exc}")

    return df, train_df, val_df, test_df, holdout_df


def df_to_tensors(data_df: pd.DataFrame):
    """Convert DataFrame to feature & target tensors."""
    X = torch.FloatTensor(data_df[FEATURE_COLS].values.copy())
    y = torch.FloatTensor(data_df["profit_lift"].values.copy())
    return X, y


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_supervised(model, train_X, train_y, test_X, test_y, name: str,
                     epochs=EPOCHS, batch_size=BATCH, lr=LR):
    """Train a supervised model and return test MSE."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    dataset = TensorDataset(train_X, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for bx, by in loader:
            pred = model(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        avg_loss = total_loss / len(train_X)
        scheduler.step(avg_loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"   [{name}] Epoch {epoch+1}/{epochs}  Train MSE: {avg_loss:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(test_X)
        mse = criterion(pred, test_y).item()
        rmse = float(mse ** 0.5)
        mae = float(torch.mean(torch.abs(pred - test_y)).item())
    print(f"   [{name}] Test MSE: {mse:.2f} €²  |  RMSE: {rmse:.2f} €  |  MAE: {mae:.2f} €")
    metrics = {"mse_eur2": mse, "rmse_eur": rmse, "mae_eur": mae, "mse": mse}
    return metrics, model


def _make_sequences(data_df, seq_len: int = 10):
    """Shape a DataFrame of samples into (n_seq, seq_len, n_features) batches."""
    X_all = data_df[FEATURE_COLS].values.copy()
    y_all = data_df["profit_lift"].values.copy()
    n_seq = len(X_all) // seq_len
    X_seq = X_all[:n_seq * seq_len].reshape(n_seq, seq_len, -1)
    y_seq = y_all[:n_seq * seq_len].reshape(n_seq, seq_len)
    return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)


def eval_seq_mse(model, df, seq_len: int = 10) -> float:
    """Evaluate a sequence model on an arbitrary DataFrame split."""
    X, y = _make_sequences(df, seq_len)
    model.eval()
    with torch.no_grad():
        pred = model(X)
        return nn.MSELoss()(pred, y).item()


def train_sequence_model(model, train_df, test_df, name: str,
                         seq_len: int = 10, epochs=EPOCHS, lr=LR,
                         clip_grad: float = 0.0):
    """Train LSTM / Transformer on padded sequences."""
    def make_sequences(data_df, seq_len):
        return _make_sequences(data_df, seq_len)

    train_X, train_y = make_sequences(train_df, seq_len)
    test_X, test_y = make_sequences(test_df, seq_len)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    dataset = TensorDataset(train_X, train_y)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for bx, by in loader:
            pred = model(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            total_loss += loss.item() * len(bx)
            count += len(bx)
        avg_loss = total_loss / max(count, 1)
        scheduler.step(avg_loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"   [{name}] Epoch {epoch+1}/{epochs}  Train MSE: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(test_X)
        mse = criterion(pred, test_y).item()
        rmse = float(mse ** 0.5)
        mae = float(torch.mean(torch.abs(pred - test_y)).item())
    print(f"   [{name}] Test MSE: {mse:.2f} €²  |  RMSE: {rmse:.2f} €  |  MAE: {mae:.2f} €")
    metrics = {"mse_eur2": mse, "rmse_eur": rmse, "mae_eur": mae, "mse": mse}
    return metrics, model


def train_ppo(df: pd.DataFrame, n_episodes: int = 500) -> dict:
    """Train PPO agent on a representative rack and return metrics."""
    rack_counts = df.groupby("rack_id").size()
    target_rack = rack_counts.idxmax()
    rack_df = df[df["rack_id"] == target_rack].copy()

    if len(rack_df) > 40:
        rack_df = rack_df.head(40)

    products = rack_df[["price_numeric", "profit_margin_percentage",
                        "estimated_monthly_sales", "product_width_cm"]].values.astype(np.float32)

    env = RackEnv(products, max_steps=50)
    trainer = PPOTrainer(env, lr=3e-4, hidden=128)
    print(f"   [PPO] Training on rack {target_rack} ({len(rack_df)} products) for {n_episodes} episodes …")
    rewards = trainer.train(n_episodes=n_episodes)

    opt_shelves = trainer.get_optimized_shelves()

    rack_df_orig = rack_df.copy()
    rack_df_opt = rack_df.copy()
    rack_df_opt["shelf_level"] = opt_shelves

    orig_profit = compute_rack_profit(rack_df_orig)
    opt_profit = compute_rack_profit(rack_df_opt)

    print(f"   [PPO] Original profit: {orig_profit:.2f} → Optimized: {opt_profit:.2f}  "
          f"(Δ = {opt_profit - orig_profit:+.2f})")

    return {
        "episode_rewards": rewards,
        "original_profit": orig_profit,
        "optimized_profit": opt_profit,
        "optimized_shelves": opt_shelves.tolist(),
        "rack_id": int(target_rack),
        "n_products": len(rack_df),
    }


# ---------------------------------------------------------------------------
# Model-guided optimization
# ---------------------------------------------------------------------------

def optimize_with_model(model, rack_df, model_type="flat"):
    """
    Use a trained model's predictions to assign each product to its
    best shelf. For each product, predict profit-lift for all 7 shelf
    options and pick the highest.
    """
    opt_df = rack_df.copy()
    opt_df["_base_profit_potential"] = (
        opt_df["price_numeric"]
        * (opt_df["profit_margin_percentage"] / 100.0)
        * opt_df["estimated_monthly_sales"]
    )
    opt_df = opt_df.sort_values("_base_profit_potential", ascending=False)

    capacity = {s: SHELF_WIDTH_CM for s in range(1, NUM_SHELVES + 1)}
    shelf_counts = {s: 0 for s in range(1, NUM_SHELVES + 1)}
    model.eval()

    for idx in opt_df.index:
        row = opt_df.loc[idx]
        original_shelf = row["shelf_level"]
        w = row["product_width_cm"]
        best_shelf = int(original_shelf)
        best_lift = -float("inf")

        n_shelves_used = sum(1 for s in shelf_counts.values() if s > 0)

        for candidate_shelf in range(1, NUM_SHELVES + 1):
            if capacity[candidate_shelf] < w:
                continue

            features = torch.FloatTensor([[
                row["price_numeric"],
                row["profit_margin_percentage"],
                row["estimated_monthly_sales"],
                row["product_width_cm"],
                original_shelf,
                candidate_shelf,
                shelf_counts.get(int(original_shelf), 0),
                shelf_counts.get(candidate_shelf, 0),
                n_shelves_used,
                len(opt_df),
            ]])

            with torch.no_grad():
                if model_type == "flat":
                    pred_lift = model(features).item()
                else:
                    pred_lift = model(features.unsqueeze(0)).item()

            if pred_lift > best_lift:
                best_lift = pred_lift
                best_shelf = candidate_shelf

        opt_df.at[idx, "shelf_level"] = best_shelf
        capacity[best_shelf] -= w
        shelf_counts[best_shelf] += 1

    opt_df.drop(columns=["_base_profit_potential"], inplace=True)
    return opt_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train & Compare Models")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--ppo-episodes", type=int, default=500)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data with train/val/test + rack-holdout split
    df, train_df, val_df, test_df, holdout_df = load_and_prepare(args.sample_size)
    input_dim = len(FEATURE_COLS)

    # Prepare flat tensors
    train_X, train_y = df_to_tensors(train_df)
    test_X, test_y = df_to_tensors(test_df)
    val_X, val_y = df_to_tensors(val_df)
    holdout_X, holdout_y = df_to_tensors(holdout_df)

    results = {}

    # ---- 1. MLP (larger) ----
    print("\n🔵 Training MLP …")
    mlp = build_mlp(input_dim=input_dim)
    metrics_mlp, mlp = train_supervised(mlp, train_X, train_y, test_X, test_y,
                                         "MLP", epochs=args.epochs)
    mlp_version = save_model(mlp, "mlp", RESULTS_DIR,
                              metadata={"origin": "02_train_models.py",
                                        "epochs": args.epochs, **metrics_mlp})
    print(f"   💾 MLP saved (hash {mlp_version['hash']}) → {mlp_version['archive_path']}")

    # Additional evaluations on validation + rack-holdout splits
    mlp.eval()
    with torch.no_grad():
        val_mse = nn.MSELoss()(mlp(val_X), val_y).item()
        holdout_mse = nn.MSELoss()(mlp(holdout_X), holdout_y).item()
    print(f"   [MLP] Val MSE: {val_mse:.2f} €²  |  Holdout(rack) MSE: {holdout_mse:.2f} €²  "
          f"|  Holdout RMSE: {holdout_mse ** 0.5:.2f} €")
    results["MLP"] = {
        **dict(metrics_mlp),
        "val_mse_eur2": val_mse,
        "val_rmse_eur": val_mse ** 0.5,
        "holdout_rack_mse_eur2": holdout_mse,
        "holdout_rack_rmse_eur": holdout_mse ** 0.5,
        "version": mlp_version,
    }

    # ---- 2. LSTM ----
    print("\n🟢 Training LSTM …")
    lstm = build_lstm(input_dim=input_dim)
    metrics_lstm, lstm = train_sequence_model(lstm, train_df, test_df,
                                               "LSTM", epochs=args.epochs)
    lstm_version = save_model(lstm, "lstm", RESULTS_DIR,
                               metadata={"origin": "02_train_models.py",
                                         "epochs": args.epochs, **metrics_lstm})
    print(f"   💾 LSTM saved (hash {lstm_version['hash']}) → {lstm_version['archive_path']}")
    lstm_val_mse = eval_seq_mse(lstm, val_df)
    lstm_holdout_mse = eval_seq_mse(lstm, holdout_df)
    print(f"   [LSTM] Val MSE: {lstm_val_mse:.2f} €²  |  Holdout(rack) MSE: {lstm_holdout_mse:.2f} €²")
    results["LSTM"] = {
        **dict(metrics_lstm),
        "val_mse_eur2": lstm_val_mse,
        "val_rmse_eur": lstm_val_mse ** 0.5,
        "holdout_rack_mse_eur2": lstm_holdout_mse,
        "holdout_rack_rmse_eur": lstm_holdout_mse ** 0.5,
        "version": lstm_version,
    }

    # ---- 3. Transformer ----
    print("\n🟡 Training Transformer …")
    transformer = build_transformer(input_dim=input_dim)
    metrics_trans, transformer = train_sequence_model(transformer, train_df, test_df,
                                                       "Transformer", epochs=150,
                                                       lr=1e-4, clip_grad=1.0)
    trans_version = save_model(transformer, "transformer", RESULTS_DIR,
                                metadata={"origin": "02_train_models.py",
                                          "epochs": 150, **metrics_trans})
    print(f"   💾 Transformer saved (hash {trans_version['hash']}) → {trans_version['archive_path']}")
    trans_val_mse = eval_seq_mse(transformer, val_df)
    trans_holdout_mse = eval_seq_mse(transformer, holdout_df)
    print(f"   [Trans] Val MSE: {trans_val_mse:.2f} €²  |  Holdout(rack) MSE: {trans_holdout_mse:.2f} €²")
    results["Transformer"] = {
        **dict(metrics_trans),
        "val_mse_eur2": trans_val_mse,
        "val_rmse_eur": trans_val_mse ** 0.5,
        "holdout_rack_mse_eur2": trans_holdout_mse,
        "holdout_rack_rmse_eur": trans_holdout_mse ** 0.5,
        "version": trans_version,
    }

    # ---- Prediction baselines: Identity (predict 0) and Random ----
    identity_mse = float(torch.mean(test_y ** 2).item())
    random_pred = torch.randn_like(test_y) * float(test_y.std())
    random_mse = float(torch.mean((random_pred - test_y) ** 2).item())
    print(f"\n📏 Prediction baselines on test set ({test_y.numel()} samples):")
    print(f"   [Identity] Predicts 0 → MSE: {identity_mse:.2f} €²  "
          f"| RMSE: {identity_mse ** 0.5:.2f} €")
    print(f"   [Random]   Gaussian    → MSE: {random_mse:.2f} €²  "
          f"| RMSE: {random_mse ** 0.5:.2f} €")
    results["_baseline_identity_prediction"] = {
        "mse_eur2": identity_mse,
        "rmse_eur": identity_mse ** 0.5,
        "description": "Predicts profit_lift=0 for every sample (no-model baseline)",
    }
    results["_baseline_random_prediction"] = {
        "mse_eur2": random_mse,
        "rmse_eur": random_mse ** 0.5,
        "description": "Gaussian noise with same std as target (noise-floor baseline)",
    }

    # ---- 4. PPO ----
    print("\n🔴 Training PPO …")
    ppo_results = train_ppo(df, n_episodes=args.ppo_episodes)
    results["PPO"] = {
        "original_profit": ppo_results["original_profit"],
        "optimized_profit": ppo_results["optimized_profit"],
    }

    # ---- Optimize using each model's own predictions ----
    print("\n📊 Computing model-guided optimizations …")
    target_rack = ppo_results["rack_id"]
    rack_df = df[df["rack_id"] == target_rack].copy()
    if len(rack_df) > 40:
        rack_df = rack_df.head(40)

    orig_profit = compute_rack_profit(rack_df)

    supervised_models = {
        "MLP": (mlp, "flat"),
        "LSTM": (lstm, "seq"),
        "Transformer": (transformer, "seq"),
    }

    rack_layouts = {"Original": rack_df.copy()}

    for model_name, (model, mtype) in supervised_models.items():
        opt_rack = optimize_with_model(model, rack_df, model_type=mtype)
        opt_profit = compute_rack_profit(opt_rack)
        results[model_name]["original_profit"] = orig_profit
        results[model_name]["optimized_profit"] = opt_profit
        lift = opt_profit - orig_profit
        print(f"   [{model_name}] Orig: €{orig_profit:.2f} → Opt: €{opt_profit:.2f} (Δ = {lift:+.2f})")
        rack_layouts[model_name] = opt_rack

    # Greedy baseline
    greedy_rack = optimize_rack_greedy(rack_df)
    greedy_profit = compute_rack_profit(greedy_rack)
    results["Greedy"] = {
        "original_profit": orig_profit,
        "optimized_profit": greedy_profit,
    }
    rack_layouts["Greedy"] = greedy_rack
    print(f"   [Greedy] Orig: €{orig_profit:.2f} → Opt: €{greedy_profit:.2f} (Δ = {greedy_profit - orig_profit:+.2f})")

    # Save results
    results_file = RESULTS_DIR / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {results_file}")

    # Save rack layouts for visualization
    for name, layout_df in rack_layouts.items():
        layout_df.to_csv(RESULTS_DIR / f"rack_layout_{name.lower()}.csv", index=False)

    # Print comparison table (MSE in €², RMSE in €, profits in €)
    print("\n" + "="*88)
    print(f"{'Model':<15} {'MSE (€²)':>12} {'RMSE (€)':>10} "
          f"{'Orig (€)':>14} {'Opt (€)':>14} {'Lift (€)':>12}")
    print("-"*88)
    for name, r in results.items():
        if name.startswith("_"):  # baselines for prediction, not optimization
            continue
        mse_val = r.get("mse_eur2", r.get("mse"))
        mse_str = f"{mse_val:.2f}" if isinstance(mse_val, (int, float)) else "N/A"
        rmse_val = r.get("rmse_eur")
        rmse_str = f"{rmse_val:.2f}" if isinstance(rmse_val, (int, float)) else "N/A"
        orig = r.get("original_profit", 0)
        opt = r.get("optimized_profit", 0)
        lift = opt - orig
        print(f"{name:<15} {mse_str:>12} {rmse_str:>10} "
              f"{orig:>14.2f} {opt:>14.2f} {lift:>+12.2f}")
    print("="*88)
    print("Note: MSE is in €² (squared errors), RMSE is the interpretable")
    print("      typical error in €. Greedy is the heuristic baseline for")
    print("      optimization; Identity/Random are prediction baselines.")
    print("="*88)


if __name__ == "__main__":
    main()

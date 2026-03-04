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
import os
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
    compute_rack_profit_advanced,
    optimize_rack_greedy,
    get_shelf_multiplier,
    SHELF_WIDTH_CM,
    NUM_SHELVES,
)
from models.mlp import build_mlp
from models.lstm_model import build_lstm
from models.transformer_model import build_transformer
from models.ppo_agent import RackEnv, PPOTrainer


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

def load_and_prepare(sample_size: int | None = None):
    """Load monthly CSVs and generate synthetic training data."""
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
    print(f"   Total: {len(df)} product-month records ")

    # Generate synthetic training data from the combined dataset
    train_df = generate_synthetic_training_data(df, n_samples=max(20000, len(df) * 3), seed=42)
    test_df = generate_synthetic_training_data(df, n_samples=max(3000, len(df)), seed=99)
    print(f"   Training samples: {len(train_df)}, Test samples: {len(test_df)}")

    return df, train_df, test_df


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
    print(f"   [{name}] Test MSE: {mse:.4f}")
    return mse, model


def train_sequence_model(model, train_df, test_df, name: str,
                         seq_len: int = 10, epochs=EPOCHS, lr=LR,
                         clip_grad: float = 0.0):
    """Train LSTM / Transformer on padded sequences."""
    def make_sequences(data_df, seq_len):
        X_all = data_df[FEATURE_COLS].values.copy()
        y_all = data_df["profit_lift"].values.copy()
        n_seq = len(X_all) // seq_len
        X_seq = X_all[:n_seq * seq_len].reshape(n_seq, seq_len, -1)
        y_seq = y_all[:n_seq * seq_len].reshape(n_seq, seq_len)
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)

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
    print(f"   [{name}] Test MSE: {mse:.4f}")
    return mse, model


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

    # Load data
    df, train_df, test_df = load_and_prepare(args.sample_size)
    input_dim = len(FEATURE_COLS)

    # Prepare flat tensors
    train_X, train_y = df_to_tensors(train_df)
    test_X, test_y = df_to_tensors(test_df)

    results = {}

    # ---- 1. MLP (larger) ----
    print("\n🔵 Training MLP …")
    mlp = build_mlp(input_dim=input_dim)
    mse_mlp, mlp = train_supervised(mlp, train_X, train_y, test_X, test_y,
                                     "MLP", epochs=args.epochs)
    torch.save(mlp.state_dict(), RESULTS_DIR / "mlp.pth")
    results["MLP"] = {"mse": mse_mlp}

    # ---- 2. LSTM ----
    print("\n🟢 Training LSTM …")
    lstm = build_lstm(input_dim=input_dim)
    mse_lstm, lstm = train_sequence_model(lstm, train_df, test_df,
                                           "LSTM", epochs=args.epochs)
    torch.save(lstm.state_dict(), RESULTS_DIR / "lstm.pth")
    results["LSTM"] = {"mse": mse_lstm}

    # ---- 3. Transformer ----
    print("\n🟡 Training Transformer …")
    transformer = build_transformer(input_dim=input_dim)
    mse_trans, transformer = train_sequence_model(transformer, train_df, test_df,
                                                   "Transformer", epochs=150,
                                                   lr=1e-4, clip_grad=1.0)
    torch.save(transformer.state_dict(), RESULTS_DIR / "transformer.pth")
    results["Transformer"] = {"mse": mse_trans}

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

    # Print comparison table
    print("\n" + "="*70)
    print(f"{'Model':<15} {'MSE':>10} {'Orig Profit':>15} {'Opt Profit':>15} {'Lift':>10}")
    print("-"*70)
    for name, r in results.items():
        mse_str = f"{r.get('mse', '-'):.4f}" if 'mse' in r else "N/A"
        orig = r.get("original_profit", 0)
        opt = r.get("optimized_profit", 0)
        lift = opt - orig
        print(f"{name:<15} {mse_str:>10} {orig:>15.2f} {opt:>15.2f} {lift:>+10.2f}")
    print("="*70)


if __name__ == "__main__":
    main()

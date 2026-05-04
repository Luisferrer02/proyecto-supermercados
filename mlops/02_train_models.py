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
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.lstm_model import build_lstm
from models.mlp import build_mlp
from models.ppo_agent import PPOTrainer, RackEnv
from models.transformer_model import build_transformer
from utils.data_io import load_monthly_csvs
from utils.model_persistence import save_model
from utils.retail_physics import (
    NUM_SHELVES,
    SHELF_WIDTH_CM,
    compute_rack_profit,
    generate_synthetic_training_data,
    optimize_rack_greedy,
)
from utils.training import (
    EPOCHS,
    FEATURE_COLS,
    FeatureNormalizer,
    LR,
    df_to_tensors,
    make_sequences,
    optimize_rack_mlp,
    train_model,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MONTHLY_DIR = BASE_DIR / "data" / "monthly"
RESULTS_DIR = BASE_DIR / "results"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_prepare(
    sample_size: int | None = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    rack_holdout_fraction: float = 0.20,
    split_seed: int = 42,
):
    """Load monthly CSVs and build a train/val/test split + rack holdout.

    Reserves `rack_holdout_fraction` of racks entirely, generates a single
    pool of synthetic samples from the rest, shuffles with `split_seed`, and
    slices into train/val/test. Saves a SHA-256 hash of the test indices to
    results/test_split_hash.json for reproducibility.

    Returns: (df, train_df, val_df, test_df, holdout_df)
    """
    print(f"Loading monthly data from {MONTHLY_DIR}...")
    df = load_monthly_csvs(MONTHLY_DIR)
    print(f"Loaded {len(df)} product-month records from {MONTHLY_DIR}")

    if sample_size:
        df = df.head(sample_size)
    print(f"   Total: {len(df)} product-month records")

    rng = np.random.default_rng(split_seed)
    all_racks = np.sort(df["rack_id"].unique())
    n_holdout = max(1, round(len(all_racks) * rack_holdout_fraction))
    holdout_racks = sorted(int(r) for r in rng.choice(all_racks, size=n_holdout, replace=False))
    train_pool_racks = [int(r) for r in all_racks if int(r) not in set(holdout_racks)]

    print(
        f"   Racks: {len(all_racks)} total, "
        f"{len(train_pool_racks)} for train/val/test, "
        f"{len(holdout_racks)} held out"
    )

    pool_df_source = df[df["rack_id"].isin(train_pool_racks)].reset_index(drop=True)
    holdout_df_source = df[df["rack_id"].isin(holdout_racks)].reset_index(drop=True)

    n_total = max(20000, len(pool_df_source) * 3) + max(3000, len(pool_df_source))
    print(f"   Generating {n_total} synthetic training samples (this may take a few minutes)...")
    pool_samples = generate_synthetic_training_data(pool_df_source, n_samples=n_total, seed=split_seed)
    print(f"   Generated {len(pool_samples)} samples.")

    perm = rng.permutation(len(pool_samples))
    pool_samples = pool_samples.iloc[perm].reset_index(drop=True)
    n = len(pool_samples)
    n_test = round(n * test_fraction)
    n_val = round(n * val_fraction)
    test_df = pool_samples.iloc[:n_test].reset_index(drop=True)
    val_df = pool_samples.iloc[n_test:n_test + n_val].reset_index(drop=True)
    train_df = pool_samples.iloc[n_test + n_val:].reset_index(drop=True)

    print("   Generating holdout samples...")
    holdout_df = generate_synthetic_training_data(
        holdout_df_source, n_samples=max(3000, len(holdout_df_source)), seed=split_seed + 1
    )

    print(
        f"   Split: train={len(train_df)}, val={len(val_df)}, "
        f"test={len(test_df)}, rack_holdout={len(holdout_df)}"
    )

    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        split_hash = hashlib.sha256(perm[:n_test].tobytes()).hexdigest()
        with open(RESULTS_DIR / "test_split_hash.json", "w") as f:
            json.dump(
                {
                    "split_seed": split_seed,
                    "test_indices_hash": split_hash,
                    "holdout_racks": holdout_racks,
                    "n_train": len(train_df),
                    "n_val": len(val_df),
                    "n_test": len(test_df),
                    "n_holdout": len(holdout_df),
                },
                f,
                indent=2,
            )
        print(f"   Split hash (first 16 chars): {split_hash[:16]}…")
    except Exception as exc:
        print(f"     Could not write split hash: {exc}")

    return df, train_df, val_df, test_df, holdout_df


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def train_ppo(df: pd.DataFrame, n_episodes: int = 500) -> dict:
    """Train PPO agent on the largest rack and return metrics."""
    rack_counts = df.groupby("rack_id").size()
    target_rack = rack_counts.idxmax()
    rack_df = df[df["rack_id"] == target_rack].head(40).copy()

    products = rack_df[
        ["price_numeric", "profit_margin_percentage", "estimated_monthly_sales", "product_width_cm"]
    ].to_numpy().astype(np.float32)

    env = RackEnv(products, max_steps=50)
    trainer = PPOTrainer(env, lr=3e-4, hidden=128)
    print(f"   [PPO] Training on rack {target_rack} ({len(rack_df)} products) for {n_episodes} episodes …")
    rewards = trainer.train(n_episodes=n_episodes)

    opt_shelves = trainer.get_optimized_shelves()
    rack_df_opt = rack_df.copy()
    rack_df_opt["shelf_level"] = opt_shelves

    orig_profit = compute_rack_profit(rack_df)
    opt_profit = compute_rack_profit(rack_df_opt)
    print(
        f"   [PPO] Original profit: {orig_profit:.2f} → Optimized: {opt_profit:.2f}  "
        f"(Δ = {opt_profit - orig_profit:+.2f})"
    )

    return {
        "episode_rewards": rewards,
        "original_profit": orig_profit,
        "optimized_profit": opt_profit,
        "optimized_shelves": opt_shelves.tolist(),
        "rack_id": int(target_rack),
        "n_products": len(rack_df),
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Per-model train + eval helpers
# ---------------------------------------------------------------------------

def _eval_extra(model, val_X, val_y, holdout_X, holdout_y, name):
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        val_mse = criterion(model(val_X), val_y).item()
        holdout_mse = criterion(model(holdout_X), holdout_y).item()
    print(
        f"   [{name}] Val MSE: {val_mse:.2f} €²  |  Holdout(rack) MSE: {holdout_mse:.2f} €²  "
        f"|  Holdout RMSE: {holdout_mse ** 0.5:.2f} €"
    )
    return val_mse, holdout_mse


def _save_and_report(model, name, metrics, epochs):
    version = save_model(model, name.lower(), RESULTS_DIR,
                         metadata={"origin": "02_train_models.py", "epochs": epochs, **metrics})
    print(f"    {name} saved (hash {version['hash']}) → {version['archive_path']}")
    return version


def _train_flat(name, model, train_X, train_y, test_X, test_y, val_X, val_y, holdout_X, holdout_y, epochs, lr=LR):
    metrics = train_model(model, train_X, train_y, test_X, test_y, name=name, epochs=epochs, lr=lr)
    version = _save_and_report(model, name, metrics, epochs)
    val_mse, holdout_mse = _eval_extra(model, val_X, val_y, holdout_X, holdout_y, name)
    return {**metrics, "val_mse_eur2": val_mse, "val_rmse_eur": val_mse**0.5,
            "holdout_rack_mse_eur2": holdout_mse, "holdout_rack_rmse_eur": holdout_mse**0.5,
            "version": version}, model


def _eval_seq_mse(model, df):
    X, y = make_sequences(df)
    model.eval()
    with torch.no_grad():
        return nn.MSELoss()(model(X), y).item()


def _train_seq(name, model, train_df, test_df, val_df, holdout_df, epochs, lr=LR, clip_grad=0.0, normalizer=None):
    if normalizer is not None:
        dfs = [train_df.copy(), test_df.copy(), val_df.copy(), holdout_df.copy()]
        for df in dfs:
            normed = normalizer.transform(torch.FloatTensor(df[FEATURE_COLS].to_numpy()))
            for i, col in enumerate(FEATURE_COLS):
                df[col] = normed[:, i].numpy()
        train_df, test_df, val_df, holdout_df = dfs
    train_X, train_y = make_sequences(train_df)
    test_X, test_y = make_sequences(test_df)
    metrics = train_model(model, train_X, train_y, test_X, test_y,
                          name=name, epochs=epochs, lr=lr, clip_grad=clip_grad)
    version = _save_and_report(model, name, metrics, epochs)
    val_mse = _eval_seq_mse(model, val_df)
    holdout_mse = _eval_seq_mse(model, holdout_df)
    print(f"   [{name}] Val MSE: {val_mse:.2f} €²  |  Holdout(rack) MSE: {holdout_mse:.2f} €²")
    return {**metrics, "val_mse_eur2": val_mse, "val_rmse_eur": val_mse**0.5,
            "holdout_rack_mse_eur2": holdout_mse, "holdout_rack_rmse_eur": holdout_mse**0.5,
            "version": version}, model


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

    df, train_df, val_df, test_df, holdout_df = load_and_prepare(args.sample_size)
    input_dim = len(FEATURE_COLS)

    train_X, train_y = df_to_tensors(train_df)
    test_X, test_y = df_to_tensors(test_df)
    val_X, val_y = df_to_tensors(val_df)
    holdout_X, holdout_y = df_to_tensors(holdout_df)

    # Normalize features for MLP (fit on train, apply to all)
    normalizer = FeatureNormalizer().fit(train_X)
    train_X_norm = normalizer.fit_transform(train_X)
    test_X_norm = normalizer.transform(test_X)
    val_X_norm = normalizer.transform(val_X)
    holdout_X_norm = normalizer.transform(holdout_X)

    results = {}

    print("\n Training MLP (normalized, 400 epochs) …")
    mlp_epochs = max(args.epochs, 400)
    results["MLP"], mlp = _train_flat(
        "MLP", build_mlp(input_dim=input_dim),
        train_X_norm, train_y, test_X_norm, test_y, val_X_norm, val_y, holdout_X_norm, holdout_y,
        mlp_epochs,
    )

    # Save normalizer so 05_predict.py can apply the same transform
    torch.save({"mean": normalizer.mean, "std": normalizer.std}, RESULTS_DIR / "normalizer.pth")

    print("\n Training LSTM …")
    results["LSTM"], lstm = _train_seq(
        "LSTM", build_lstm(input_dim=input_dim),
        train_df, test_df, val_df, holdout_df, args.epochs,
    )

    print("\n Training Transformer (normalized, 300 epochs) …")
    results["Transformer"], transformer = _train_seq(
        "Transformer", build_transformer(input_dim=input_dim),
        train_df, test_df, val_df, holdout_df, epochs=300, lr=1e-4, clip_grad=1.0,
        normalizer=normalizer,
    )

    # Prediction baselines
    identity_mse = float(torch.mean(test_y ** 2).item())
    random_mse = float(torch.mean((torch.randn_like(test_y) * float(test_y.std()) - test_y) ** 2).item())
    print(f"\n Prediction baselines on test set ({test_y.numel()} samples):")
    print(f"   [Identity] Predicts 0 → MSE: {identity_mse:.2f} €²  | RMSE: {identity_mse ** 0.5:.2f} €")
    print(f"   [Random]   Gaussian    → MSE: {random_mse:.2f} €²  | RMSE: {random_mse ** 0.5:.2f} €")
    results["_baseline_identity_prediction"] = {
        "mse_eur2": identity_mse, "rmse_eur": identity_mse ** 0.5,
        "description": "Predicts profit_lift=0 for every sample (no-model baseline)",
    }
    results["_baseline_random_prediction"] = {
        "mse_eur2": random_mse, "rmse_eur": random_mse ** 0.5,
        "description": "Gaussian noise with same std as target (noise-floor baseline)",
    }

    print("\n Training PPO …")
    ppo_results = train_ppo(df, n_episodes=args.ppo_episodes)
    results["PPO"] = {
        "original_profit": ppo_results["original_profit"],
        "optimized_profit": ppo_results["optimized_profit"],
    }

    print("\n Computing model-guided optimizations …")
    rack_df = df[df["rack_id"] == ppo_results["rack_id"]].head(40).copy()
    orig_profit = compute_rack_profit(rack_df)
    rack_layouts = {"Original": rack_df.copy()}

    for model_name, model in [("MLP", mlp)]:
        opt_rack = optimize_rack_mlp(rack_df, model, NUM_SHELVES, SHELF_WIDTH_CM, normalizer=normalizer)
        opt_profit = compute_rack_profit(opt_rack)
        results[model_name]["original_profit"] = orig_profit
        results[model_name]["optimized_profit"] = opt_profit
        print(f"   [{model_name}] Orig: €{orig_profit:.2f} → Opt: €{opt_profit:.2f} (Δ = {opt_profit - orig_profit:+.2f})")
        rack_layouts[model_name] = opt_rack

    # LSTM and Transformer are sequence models — they score full layouts
    # but don't do per-product greedy assignment. Their optimization value
    # comes from the ensemble in 05_predict.py (MLP proposes, Transformer scores).
    for model_name in ["LSTM", "Transformer"]:
        results[model_name]["original_profit"] = orig_profit
        results[model_name]["optimized_profit"] = orig_profit

    greedy_rack = optimize_rack_greedy(rack_df)
    greedy_profit = compute_rack_profit(greedy_rack)
    results["Greedy"] = {"original_profit": orig_profit, "optimized_profit": greedy_profit}
    rack_layouts["Greedy"] = greedy_rack
    print(f"   [Greedy] Orig: €{orig_profit:.2f} → Opt: €{greedy_profit:.2f} (Δ = {greedy_profit - orig_profit:+.2f})")

    results_file = RESULTS_DIR / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n Results saved to {results_file}")

    for name, layout_df in rack_layouts.items():
        layout_df.to_csv(RESULTS_DIR / f"rack_layout_{name.lower()}.csv", index=False)

    print("\n" + "=" * 88)
    print(f"{'Model':<15} {'MSE (€²)':>12} {'RMSE (€)':>10} {'Orig (€)':>14} {'Opt (€)':>14} {'Lift (€)':>12}")
    print("-" * 88)
    for name, r in results.items():
        if name.startswith("_"):
            continue
        mse_val = r.get("mse_eur2", r.get("mse"))
        mse_str = f"{mse_val:.2f}" if isinstance(mse_val, (int, float)) else "N/A"
        rmse_val = r.get("rmse_eur")
        rmse_str = f"{rmse_val:.2f}" if isinstance(rmse_val, (int, float)) else "N/A"
        orig = r.get("original_profit", 0)
        opt = r.get("optimized_profit", 0)
        print(f"{name:<15} {mse_str:>12} {rmse_str:>10} {orig:>14.2f} {opt:>14.2f} {opt - orig:>+12.2f}")
    print("=" * 88)


if __name__ == "__main__":
    main()

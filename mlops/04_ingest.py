#!/usr/bin/env python3
"""
04_ingest.py — Parallel Onboarding Pipeline
=============================================
Ingests monthly sales CSVs into the system:
  Thread 1: Generates embeddings -> stores in ChromaDB knowledge base
  Thread 2: Trains MLP + Transformer models on the uploaded data

Usage:
    python 04_ingest.py data/monthly/                 # Ingest all CSVs
    python 04_ingest.py data/monthly/ --epochs 100    # Custom training epochs
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
EPOCHS = 80
BATCH = 128
LR = 5e-4
NUM_SHELVES = 7
SHELF_WIDTH_CM = 300

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
# Progress tracker (thread-safe)
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Thread-safe progress tracking for parallel tasks."""

    def __init__(self):
        self._lock = threading.Lock()
        self.embedding_status = "pending"
        self.training_status = "pending"
        self.embedding_progress = ""
        self.training_progress = ""
        self.start_time = time.time()

    def update_embedding(self, status: str, progress: str = ""):
        with self._lock:
            self.embedding_status = status
            self.embedding_progress = progress

    def update_training(self, status: str, progress: str = ""):
        with self._lock:
            self.training_status = status
            self.training_progress = progress

    def print_status(self):
        with self._lock:
            elapsed = time.time() - self.start_time
            e_icon = {"pending": "..", "running": ">>", "done": "OK",
                      "error": "!!"}.get(self.embedding_status, "?")
            t_icon = {"pending": "..", "running": ">>", "done": "OK",
                      "error": "!!"}.get(self.training_status, "?")
            print(f"\r  [{e_icon}] Embeddings: {self.embedding_progress:<40} "
                  f"| [{t_icon}] Models: {self.training_progress:<30} "
                  f"[{elapsed:.0f}s]", end="", flush=True)

    @property
    def all_done(self):
        with self._lock:
            return (self.embedding_status in ("done", "error") and
                    self.training_status in ("done", "error"))


# ---------------------------------------------------------------------------
# Thread 1: Embedding generation → ChromaDB
# ---------------------------------------------------------------------------

def run_embedding_thread(csv_dir: Path, tracker: ProgressTracker):
    """Embed monthly CSVs into the knowledge base."""
    try:
        tracker.update_embedding("running", "Loading knowledge base…")
        from utils.knowledge_base import ShelfKnowledgeBase
        kb = ShelfKnowledgeBase()

        def on_progress(current, total, filename, n_chunks):
            tracker.update_embedding(
                "running",
                f"{current}/{total} files ({filename}: {n_chunks} chunks)"
            )

        tracker.update_embedding("running", "Scanning CSV files…")
        results = kb.ingest_directory(csv_dir, callback=on_progress)

        stats = kb.stats()
        tracker.update_embedding(
            "done",
            f"{stats['total_chunks']} chunks, "
            f"{stats['n_months']} months, "
            f"{stats['n_categories']} categories"
        )

    except Exception as e:
        tracker.update_embedding("error", str(e)[:60])
        raise


# ---------------------------------------------------------------------------
# Thread 2: LSTM Training
# ---------------------------------------------------------------------------

def _load_all_monthly_csvs(csv_dir: Path) -> pd.DataFrame:
    """Load and concatenate all monthly CSVs into a single DataFrame."""
    csv_files = sorted(csv_dir.glob("sales_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No sales_*.csv files found in {csv_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Extract month/year from filename for context
        from utils.knowledge_base import parse_month_from_filename
        meta = parse_month_from_filename(f.name)
        if meta:
            df["_year"] = meta[0]
            df["_month"] = meta[1]
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def _generate_training_data(df: pd.DataFrame, n_samples: int = 20000,
                            seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training data by simulating shelf reassignments.
    Replicates the logic from utils/retail_physics.py but works with
    any user's data.
    """
    from utils.retail_physics import compute_rack_profit

    rng = np.random.RandomState(seed)
    records = []

    rack_ids = df["rack_id"].unique()

    for _ in range(n_samples):
        rack_id = rng.choice(rack_ids)
        rack_df = df[df["rack_id"] == rack_id].copy()
        if len(rack_df) < 3:
            continue

        # Pick a random product
        idx = rng.choice(rack_df.index)
        row = rack_df.loc[idx]
        original_shelf = int(row["shelf_level"])

        # Try a random new shelf
        candidate_shelf = rng.randint(1, NUM_SHELVES + 1)

        # Compute profit before/after
        original_profit = compute_rack_profit(rack_df)
        modified_df = rack_df.copy()
        modified_df.at[idx, "shelf_level"] = candidate_shelf
        new_profit = compute_rack_profit(modified_df)

        profit_lift = new_profit - original_profit

        # Shelf context
        shelf_counts = rack_df["shelf_level"].value_counts()
        n_shelves_used = len(shelf_counts)

        records.append({
            "price_numeric": row["price_numeric"],
            "profit_margin_percentage": row["profit_margin_percentage"],
            "estimated_monthly_sales": row["estimated_monthly_sales"],
            "product_width_cm": row["product_width_cm"],
            "original_shelf": original_shelf,
            "new_shelf": candidate_shelf,
            "n_products_on_original_shelf": int(shelf_counts.get(original_shelf, 0)),
            "n_products_on_new_shelf": int(shelf_counts.get(candidate_shelf, 0)),
            "n_shelves_used": n_shelves_used,
            "rack_product_count": len(rack_df),
            "profit_lift": profit_lift,
        })

    return pd.DataFrame(records)


def run_training_thread(csv_dir: Path, tracker: ProgressTracker,
                        epochs: int = EPOCHS):
    """Train MLP + Transformer on the user's uploaded data."""
    try:
        tracker.update_training("running", "Loading CSV data...")
        df = _load_all_monthly_csvs(csv_dir)
        tracker.update_training("running",
                                f"Loaded {len(df)} products from CSVs")

        # Check required columns
        needed = ["price_numeric", "profit_margin_percentage",
                  "estimated_monthly_sales", "product_width_cm",
                  "rack_id", "shelf_level"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            tracker.update_training("error", f"Missing cols: {missing}")
            return

        # Generate training data
        tracker.update_training("running", "Generating training samples...")
        train_df = _generate_training_data(df, n_samples=max(20000, len(df) * 3),
                                           seed=42)
        test_df = _generate_training_data(df, n_samples=max(3000, len(df)),
                                          seed=99)
        tracker.update_training(
            "running",
            f"Train: {len(train_df)}, Test: {len(test_df)} samples"
        )

        if len(train_df) < 100:
            tracker.update_training("error",
                                    "Not enough training data. Need more CSVs.")
            return

        input_dim = len(FEATURE_COLS)
        seq_len = 10
        criterion = nn.MSELoss()

        # Prepare flat tensors (for MLP)
        train_X_flat = torch.FloatTensor(
            train_df[FEATURE_COLS].values.astype(np.float32))
        train_y_flat = torch.FloatTensor(
            train_df["profit_lift"].values.astype(np.float32))
        test_X_flat = torch.FloatTensor(
            test_df[FEATURE_COLS].values.astype(np.float32))
        test_y_flat = torch.FloatTensor(
            test_df["profit_lift"].values.astype(np.float32))

        # Prepare sequence tensors (for Transformer)
        def make_sequences(data_df, sl):
            X = data_df[FEATURE_COLS].values.astype(np.float32)
            y = data_df["profit_lift"].values.astype(np.float32)
            n_seq = len(X) // sl
            if n_seq == 0:
                return None, None
            X_seq = X[:n_seq * sl].reshape(n_seq, sl, -1)
            y_seq = y[:n_seq * sl].reshape(n_seq, sl)
            return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)

        train_X_seq, train_y_seq = make_sequences(train_df, seq_len)
        test_X_seq, test_y_seq = make_sequences(test_df, seq_len)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # ---- Train MLP ----
        tracker.update_training("running", "Training MLP...")
        from models.mlp import build_mlp
        mlp = build_mlp(input_dim=input_dim)
        mlp_opt = optim.Adam(mlp.parameters(), lr=LR)
        mlp_sched = optim.lr_scheduler.ReduceLROnPlateau(
            mlp_opt, patience=10, factor=0.5)
        mlp_dataset = TensorDataset(train_X_flat, train_y_flat)
        mlp_loader = DataLoader(mlp_dataset, batch_size=BATCH, shuffle=True)

        mlp.train()
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            for bx, by in mlp_loader:
                pred = mlp(bx)
                loss = criterion(pred, by)
                mlp_opt.zero_grad()
                loss.backward()
                mlp_opt.step()
                total_loss += loss.item() * len(bx)
                count += len(bx)
            avg_loss = total_loss / max(count, 1)
            mlp_sched.step(avg_loss)
            if (epoch + 1) % 20 == 0 or epoch == 0:
                tracker.update_training(
                    "running",
                    f"MLP Epoch {epoch+1}/{epochs} Loss: {avg_loss:.2f}"
                )

        mlp.eval()
        with torch.no_grad():
            mlp_mse = criterion(mlp(test_X_flat), test_y_flat).item()
        torch.save(mlp.state_dict(), RESULTS_DIR / "mlp.pth")

        # ---- Train Transformer ----
        if train_X_seq is not None:
            tracker.update_training("running", "Training Transformer...")
            from models.transformer_model import build_transformer
            transformer = build_transformer(input_dim=input_dim)
            trans_epochs = int(epochs * 1.5)  # Transformer needs more epochs
            trans_opt = optim.Adam(transformer.parameters(), lr=1e-4)
            trans_sched = optim.lr_scheduler.ReduceLROnPlateau(
                trans_opt, patience=10, factor=0.5)
            trans_dataset = TensorDataset(train_X_seq, train_y_seq)
            trans_loader = DataLoader(trans_dataset, batch_size=BATCH, shuffle=True)

            transformer.train()
            for epoch in range(trans_epochs):
                total_loss = 0
                count = 0
                for bx, by in trans_loader:
                    pred = transformer(bx)
                    loss = criterion(pred, by)
                    trans_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                    trans_opt.step()
                    total_loss += loss.item() * len(bx)
                    count += len(bx)
                avg_loss = total_loss / max(count, 1)
                trans_sched.step(avg_loss)
                if (epoch + 1) % 20 == 0 or epoch == 0:
                    tracker.update_training(
                        "running",
                        f"Transformer Epoch {epoch+1}/{trans_epochs} "
                        f"Loss: {avg_loss:.2f}"
                    )

            transformer.eval()
            with torch.no_grad():
                trans_mse = criterion(
                    transformer(test_X_seq), test_y_seq).item()
            torch.save(transformer.state_dict(),
                       RESULTS_DIR / "transformer.pth")

            tracker.update_training(
                "done",
                f"MLP MSE: {mlp_mse:.1f} | Trans MSE: {trans_mse:.1f}"
            )
        else:
            tracker.update_training(
                "done",
                f"MLP MSE: {mlp_mse:.1f} (not enough data for Transformer)"
            )

    except Exception as e:
        tracker.update_training("error", str(e)[:60])
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest monthly sales CSVs (embeddings + model training)")
    parser.add_argument("input_dir", type=str,
                        help="Directory containing sales_*.csv files")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--sequential", action="store_true",
                        help="Run embedding and training sequentially "
                             "(for debugging)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: Directory not found: {input_dir}")
        sys.exit(1)

    csv_files = list(input_dir.glob("sales_*.csv"))
    if not csv_files:
        print(f"ERROR: No sales_*.csv files found in {input_dir}")
        sys.exit(1)

    print("=" * 65)
    print("  SHELF OPTIMIZER -- DATA INGESTION")
    print("=" * 65)
    print(f"\n  Input directory: {input_dir}")
    print(f"  CSV files found: {len(csv_files)}")
    for f in sorted(csv_files):
        print(f"    {f.name}")
    print(f"  Training epochs: {args.epochs}")
    print()

    tracker = ProgressTracker()

    if args.sequential:
        # --- Sequential mode (for debugging) ---
        print("Running sequentially (debug mode)\n")

        print("  Phase 1: Embedding generation")
        run_embedding_thread(input_dir, tracker)
        tracker.print_status()
        print()

        print("\n  Phase 2: MLP + Transformer training")
        run_training_thread(input_dir, tracker, epochs=args.epochs)
        tracker.print_status()
        print()
    else:
        # --- Parallel mode ---
        print("  Running embedding + training in parallel...\n")

        t1 = threading.Thread(
            target=run_embedding_thread,
            args=(input_dir, tracker),
            daemon=True,
        )
        t2 = threading.Thread(
            target=run_training_thread,
            args=(input_dir, tracker, args.epochs),
            daemon=True,
        )

        t1.start()
        t2.start()

        # Print progress while threads are running
        while not tracker.all_done:
            tracker.print_status()
            time.sleep(1)

        # Final status
        tracker.print_status()
        print()

    # Summary
    print("\n" + "=" * 65)
    if tracker.embedding_status == "done" and tracker.training_status == "done":
        print("  INGESTION COMPLETE")
        print(f"  Embeddings: {tracker.embedding_progress}")
        print(f"  Models:     {tracker.training_progress}")
        print("\n  The system is ready. Run 05_predict.py to optimize shelves.")
    else:
        print("  INGESTION FINISHED WITH ISSUES")
        if tracker.embedding_status == "error":
            print(f"  Embeddings ERROR: {tracker.embedding_progress}")
        if tracker.training_status == "error":
            print(f"  Training ERROR: {tracker.training_progress}")
    print("=" * 65)


if __name__ == "__main__":
    main()

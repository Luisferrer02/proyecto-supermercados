#!/usr/bin/env python3
"""
04_ingest.py — Parallel Onboarding Pipeline
=============================================
Ingests monthly sales CSVs into the system:
  Thread 1: Generates embeddings → stores in ChromaDB knowledge base
  Thread 2: Trains MLP + Transformer models on the uploaded data

Usage:
    python 04_ingest.py data/monthly/                 # Ingest all CSVs
    python 04_ingest.py data/monthly/ --epochs 100    # Custom training epochs
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.mlp import build_mlp
from models.transformer_model import build_transformer
from utils.csv_schema import all_ok, format_report, validate_directory
from utils.data_io import load_monthly_csvs
from utils.knowledge_base import ShelfKnowledgeBase
from utils.model_persistence import save_model
from utils.retail_physics import generate_synthetic_training_data
from utils.training import EPOCHS, FEATURE_COLS, make_sequences, train_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


# ---------------------------------------------------------------------------
# Progress tracker (thread-safe)
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Thread-safe progress tracking for parallel tasks."""

    _ICONS = {"pending": "..", "running": ">>", "done": "OK", "error": "!!"}

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
            e = self._ICONS.get(self.embedding_status, "?")
            t = self._ICONS.get(self.training_status, "?")
            print(
                f"\r  [{e}] Embeddings: {self.embedding_progress:<40} "
                f"| [{t}] Models: {self.training_progress:<30} "
                f"[{elapsed:.0f}s]",
                end="", flush=True,
            )

    @property
    def all_done(self):
        with self._lock:
            return (
                self.embedding_status in ("done", "error")
                and self.training_status in ("done", "error")
            )


# ---------------------------------------------------------------------------
# Thread 1: Embedding generation → ChromaDB
# ---------------------------------------------------------------------------

def run_embedding_thread(csv_dir: Path, tracker: ProgressTracker):
    """Embed monthly CSVs into the knowledge base."""
    try:
        tracker.update_embedding("running", "Loading knowledge base…")
        kb = ShelfKnowledgeBase()

        def on_progress(current, total, filename, n_chunks):
            tracker.update_embedding(
                "running", f"{current}/{total} files ({filename}: {n_chunks} chunks)"
            )

        tracker.update_embedding("running", "Scanning CSV files…")
        kb.ingest_directory(csv_dir, callback=on_progress)

        stats = kb.stats()
        tracker.update_embedding(
            "done",
            f"{stats['total_chunks']} chunks, {stats['n_months']} months, {stats['n_categories']} categories",
        )
    except Exception as e:
        tracker.update_embedding("error", str(e)[:60])
        raise


# ---------------------------------------------------------------------------
# Thread 2: Model training
# ---------------------------------------------------------------------------

def run_training_thread(csv_dir: Path, tracker: ProgressTracker, epochs: int = EPOCHS):
    """Train MLP + Transformer on the user's uploaded data."""
    try:
        tracker.update_training("running", "Loading CSV data...")
        df = load_monthly_csvs(csv_dir, add_month_cols=True)
        tracker.update_training("running", f"Loaded {len(df)} products from CSVs")

        needed = ["price_numeric", "profit_margin_percentage",
                  "estimated_monthly_sales", "product_width_cm", "rack_id", "shelf_level"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            tracker.update_training("error", f"Missing cols: {missing}")
            return

        tracker.update_training("running", "Generating training samples...")
        train_df = generate_synthetic_training_data(df, n_samples=max(20000, len(df) * 3), seed=42)
        test_df  = generate_synthetic_training_data(df, n_samples=max(3000, len(df)), seed=99)
        tracker.update_training("running", f"Train: {len(train_df)}, Test: {len(test_df)} samples")

        if len(train_df) < 100:
            tracker.update_training("error", "Not enough training data. Need more CSVs.")
            return

        input_dim = len(FEATURE_COLS)
        criterion = nn.MSELoss()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # ---- MLP (flat tensors) ----
        tracker.update_training("running", "Training MLP...")
        train_X = torch.FloatTensor(train_df[FEATURE_COLS].to_numpy().astype(np.float32))
        train_y = torch.FloatTensor(train_df["profit_lift"].to_numpy().astype(np.float32))
        test_X  = torch.FloatTensor(test_df[FEATURE_COLS].to_numpy().astype(np.float32))
        test_y  = torch.FloatTensor(test_df["profit_lift"].to_numpy().astype(np.float32))

        mlp = build_mlp(input_dim=input_dim)

        def mlp_progress(epoch, total, loss):
            if epoch % 20 == 0 or epoch == 1:
                tracker.update_training("running", f"MLP Epoch {epoch}/{total} Loss: {loss:.2f}")

        mlp_metrics = train_model(mlp, train_X, train_y, test_X, test_y,
                                  name="MLP", epochs=epochs, on_epoch=mlp_progress)
        save_model(mlp, "mlp", RESULTS_DIR,
                   metadata={"origin": "04_ingest.py", "epochs": epochs, **mlp_metrics})

        # ---- Transformer (sequence tensors) ----
        train_X_seq, train_y_seq = make_sequences(train_df)
        test_X_seq,  test_y_seq  = make_sequences(test_df)

        if train_X_seq is not None:
            tracker.update_training("running", "Training Transformer...")
            transformer = build_transformer(input_dim=input_dim)
            trans_epochs = int(epochs * 1.5)

            def trans_progress(epoch, total, loss):
                if epoch % 20 == 0 or epoch == 1:
                    tracker.update_training("running", f"Transformer Epoch {epoch}/{total} Loss: {loss:.2f}")

            trans_metrics = train_model(
                transformer, train_X_seq, train_y_seq, test_X_seq, test_y_seq,
                name="Transformer", epochs=trans_epochs, lr=1e-4, clip_grad=1.0,
                on_epoch=trans_progress,
            )
            save_model(transformer, "transformer", RESULTS_DIR,
                       metadata={"origin": "04_ingest.py", "epochs": trans_epochs, **trans_metrics})

            mlp.eval()
            transformer.eval()
            with torch.no_grad():
                mlp_mse = criterion(mlp(test_X), test_y).item()
                trans_mse = criterion(transformer(test_X_seq), test_y_seq).item()
            tracker.update_training("done", f"MLP MSE: {mlp_mse:.1f} | Trans MSE: {trans_mse:.1f}")
        else:
            mlp.eval()
            with torch.no_grad():
                mlp_mse = criterion(mlp(test_X), test_y).item()
            tracker.update_training("done", f"MLP MSE: {mlp_mse:.1f} (not enough data for Transformer)")

    except Exception as e:
        tracker.update_training("error", str(e)[:60])
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest monthly sales CSVs (embeddings + model training)")
    parser.add_argument("input_dir", type=str, help="Directory containing sales_*.csv files")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--sequential", action="store_true",
                        help="Run embedding and training sequentially (for debugging)")
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
    print(f"  Training epochs: {args.epochs}\n")

    validation = validate_directory(input_dir)
    if not all_ok(validation):
        print(format_report(validation))
        print("\nAborting: at least one CSV failed schema validation.")
        sys.exit(2)
    n_warn = sum(len(r.warnings) for r in validation)
    print(f"  Schema validation: {len(validation)} files OK ({n_warn} warnings)\n")

    tracker = ProgressTracker()

    if args.sequential:
        print("Running sequentially (debug mode)\n")
        print("  Phase 1: Embedding generation")
        run_embedding_thread(input_dir, tracker)
        tracker.print_status()
        print("\n  Phase 2: MLP + Transformer training")
        run_training_thread(input_dir, tracker, epochs=args.epochs)
        tracker.print_status()
        print()
    else:
        print("  Running embedding + training in parallel...\n")
        t1 = threading.Thread(target=run_embedding_thread, args=(input_dir, tracker), daemon=True)
        t2 = threading.Thread(target=run_training_thread, args=(input_dir, tracker, args.epochs), daemon=True)
        t1.start()
        t2.start()
        while not tracker.all_done:
            tracker.print_status()
            time.sleep(1)
        tracker.print_status()
        print()

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
            print(f"  Training ERROR:   {tracker.training_progress}")
    print("=" * 65)


if __name__ == "__main__":
    main()

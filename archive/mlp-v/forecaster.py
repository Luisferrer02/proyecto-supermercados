"""
forecaster.py
─────────────────────────────────────────────────────────────────────────────
XGBoost-based Demand Forecaster for the Shelf-Space Optimization System.

Role in the pipeline
────────────────────
Given a (product, shelf-position) pair plus contextual features, the
forecaster outputs a *lift multiplier* — a scalar that scales the product's
baseline demand up or down depending on where it is placed.

  actual_demand = base_demand × lift_multiplier × ε,  ε ~ LogNormal(0, σ)

The model is trained OFFLINE on historical planogram / POS data and then
FROZEN during PPO training to keep the RL reward signal stationary.

Feature Schema (11 features)
─────────────────────────────
 0  shelf_tier          int  {0=bottom, 1=mid-low, 2=mid-high, 3=eye-level}
 1  facing_count        int  number of product facings (1-6)
 2  is_end_cap          bin  1 if position is a promotional end-cap
 3  is_checkout         bin  1 if position is near checkout
 4  adjacency_promo     bin  1 if adjacent to a promotional display
 5  product_category    int  category ID (0-9)
 6  base_price          flt  unit retail price in dollars
 7  margin_pct          flt  gross margin as a fraction (0-1)
 8  days_since_restock  int  shelf freshness signal
 9  day_of_week         int  0=Monday … 6=Sunday
10  week_of_year        int  1-52 seasonality signal

Target: lift_multiplier ∈ (0, ∞), typically [0.5, 2.5]
        (ratio of actual_sales to baseline_demand for a given position)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "shelf_tier",
    "facing_count",
    "is_end_cap",
    "is_checkout",
    "adjacency_promo",
    "product_category",
    "base_price",
    "margin_pct",
    "days_since_restock",
    "day_of_week",
    "week_of_year",
]

N_FEATURES = len(FEATURE_NAMES)

DEFAULT_XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,   # L1 — encourages sparse feature use
    "reg_lambda": 1.0,  # L2 — prevents extreme lift predictions
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

# ─────────────────────────────────────────────────────────────────────────────
# Data Generation (synthetic, mirrors real planogram experiment structure)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    n_samples: int = 20_000,
    noise_std: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data that encodes domain knowledge about
    how shelf placement affects sales lift.

    Ground-truth rules baked in
    ───────────────────────────
    • Eye-level (+tier 3) → +60% lift vs bottom shelf
    • Each additional facing → ~12% lift (diminishing returns via sqrt)
    • End-cap → +50% lift
    • Checkout adjacency → +30% lift (impulse category dependent)
    • Promo adjacency → +20% lift
    • Day-of-week: weekends +15%
    • Freshness penalty: stale shelves lose lift
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    shelf_tier = rng.integers(0, 4, size=n)          # 0-3
    facing_count = rng.integers(1, 7, size=n)         # 1-6
    is_end_cap = rng.binomial(1, 0.15, size=n)
    is_checkout = rng.binomial(1, 0.10, size=n)
    adjacency_promo = rng.binomial(1, 0.20, size=n)
    product_category = rng.integers(0, 10, size=n)
    base_price = rng.uniform(0.99, 49.99, size=n)
    margin_pct = rng.uniform(0.10, 0.65, size=n)
    days_since_restock = rng.integers(0, 15, size=n)
    day_of_week = rng.integers(0, 7, size=n)
    week_of_year = rng.integers(1, 53, size=n)

    # --- deterministic lift components ---
    tier_lift = np.array([0.70, 0.85, 1.00, 1.60])[shelf_tier]
    facing_lift = 1.0 + 0.12 * np.sqrt(facing_count - 1)
    endcap_lift = 1.0 + 0.50 * is_end_cap
    checkout_lift = 1.0 + 0.30 * is_checkout * (product_category < 4).astype(float)
    promo_adj_lift = 1.0 + 0.20 * adjacency_promo
    weekend_lift = 1.0 + 0.15 * (day_of_week >= 5).astype(float)
    freshness_penalty = 1.0 - 0.03 * np.minimum(days_since_restock, 10)

    lift = (
        tier_lift
        * facing_lift
        * endcap_lift
        * checkout_lift
        * promo_adj_lift
        * weekend_lift
        * freshness_penalty
    )

    # multiplicative log-normal noise
    noise = rng.lognormal(mean=0.0, sigma=noise_std, size=n)
    lift = np.clip(lift * noise, 0.3, 4.0)

    X = np.column_stack([
        shelf_tier, facing_count, is_end_cap, is_checkout, adjacency_promo,
        product_category, base_price, margin_pct, days_since_restock,
        day_of_week, week_of_year,
    ]).astype(np.float32)

    return X, lift.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Forecaster Class
# ─────────────────────────────────────────────────────────────────────────────

class DemandForecaster:
    """
    Wraps an XGBoost regressor that predicts sales-lift multipliers.

    Usage
    ─────
    # Train from scratch (first time)
    forecaster = DemandForecaster()
    forecaster.train()
    forecaster.save("models/forecaster.ubj")

    # Load pretrained (during RL training)
    forecaster = DemandForecaster.load("models/forecaster.ubj")
    lift = forecaster.predict(feature_matrix)  # shape (N,)
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        self.params = params or DEFAULT_XGB_PARAMS
        self.model: Optional[xgb.XGBRegressor] = None
        self._is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        n_synthetic_samples: int = 20_000,
        val_size: float = 0.15,
        verbose: bool = True,
    ) -> dict:
        """
        Train the XGBoost lift model.

        If X and y are None, generates a synthetic dataset so the system
        can run end-to-end without real POS data.

        Returns a dict of validation metrics.
        """
        if X is None or y is None:
            logger.info("No data supplied — generating synthetic dataset (%d samples).", n_synthetic_samples)
            X, y = generate_synthetic_dataset(n_samples=n_synthetic_samples)

        if X.shape[1] != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {X.shape[1]}.")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=42
        )

        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=50 if verbose else False,
        )
        self._is_trained = True

        # Validation metrics
        y_pred = self.model.predict(X_val)
        metrics = {
            "val_mae": float(mean_absolute_error(y_val, y_pred)),
            "val_r2": float(r2_score(y_val, y_pred)),
            "val_mape": float(np.mean(np.abs((y_val - y_pred) / (y_val + 1e-6)))),
        }
        if verbose:
            logger.info("Validation metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict lift multipliers for a batch of feature vectors.

        Parameters
        ----------
        X : np.ndarray of shape (N, 11) or (11,)

        Returns
        -------
        lift : np.ndarray of shape (N,), values clipped to [0.3, 4.0]
        """
        self._check_trained()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        raw = self.model.predict(X.astype(np.float32))
        return np.clip(raw, 0.3, 4.0)

    def predict_single(
        self,
        shelf_tier: int,
        facing_count: int,
        is_end_cap: int,
        is_checkout: int,
        adjacency_promo: int,
        product_category: int,
        base_price: float,
        margin_pct: float,
        days_since_restock: int,
        day_of_week: int,
        week_of_year: int,
    ) -> float:
        """Convenience wrapper for a single feature vector."""
        x = np.array([[
            shelf_tier, facing_count, is_end_cap, is_checkout, adjacency_promo,
            product_category, base_price, margin_pct, days_since_restock,
            day_of_week, week_of_year,
        ]], dtype=np.float32)
        return float(self.predict(x)[0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model to disk in XGBoost's native binary format (.ubj)."""
        self._check_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info("Forecaster saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "DemandForecaster":
        """Load a previously saved forecaster."""
        if not Path(path).exists():
            raise FileNotFoundError(f"No model found at {path}")
        obj = cls()
        obj.model = xgb.XGBRegressor()
        obj.model.load_model(str(path))
        obj._is_trained = True
        logger.info("Forecaster loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self._is_trained or self.model is None:
            raise RuntimeError("Forecaster has not been trained. Call .train() or .load() first.")

    def feature_importances(self) -> dict[str, float]:
        """Return a dict of feature_name → importance score."""
        self._check_trained()
        scores = self.model.feature_importances_
        return dict(zip(FEATURE_NAMES, scores.tolist()))

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return f"DemandForecaster(status={status}, n_features={N_FEATURES})"


# ─────────────────────────────────────────────────────────────────────────────
# CLI: quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    forecaster = DemandForecaster()
    metrics = forecaster.train(n_synthetic_samples=25_000, verbose=True)

    print("\n── Validation Metrics ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    print("\n── Feature Importances ─────────────────────────")
    for feat, score in sorted(forecaster.feature_importances().items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 40)
        print(f"  {feat:22s}: {bar} ({score:.3f})")

    # Single-prediction demo
    lift = forecaster.predict_single(
        shelf_tier=3, facing_count=4, is_end_cap=1, is_checkout=0,
        adjacency_promo=1, product_category=2, base_price=5.99,
        margin_pct=0.35, days_since_restock=1, day_of_week=6, week_of_year=48,
    )
    print(f"\n── Example Prediction ──────────────────────────")
    print(f"  Eye-level, 4 facings, end-cap, Sunday, week 48:")
    print(f"  Predicted lift multiplier = {lift:.3f}x")

    forecaster.save("models/forecaster.ubj")

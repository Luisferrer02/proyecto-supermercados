"""
utils/data_io.py — Shared data loading, parsing, and shelf assignment
=====================================================================
Consolidates helpers that were duplicated across 01_augment_data.py,
01_generate_monthly_sales.py, 02_train_models.py, 03_evaluate.py, and
04_ingest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils.retail_physics import NUM_SHELVES

# ---------------------------------------------------------------------------
# Price parsing
# ---------------------------------------------------------------------------

def parse_eur_price(price_str: str) -> float:
    """Convert '0,36 €' or '32,52 €' → float."""
    if pd.isna(price_str) or not str(price_str).strip():
        return 0.0
    cleaned = str(price_str).replace("€", "").replace("\xa0", "").strip()
    cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_monthly_csvs(
    csv_dir: str | Path,
    *,
    add_month_cols: bool = False,
    required: bool = True,
) -> pd.DataFrame:
    """Load and concatenate all ``sales_*.csv`` files from *csv_dir*.

    Parameters
    ----------
    csv_dir : path
        Directory containing ``sales_YYYY_MM_monthname.csv`` files.
    add_month_cols : bool
        If True, add ``_year`` and ``_month`` columns parsed from filenames.
    required : bool
        If True and no CSVs are found, print an error and ``sys.exit(1)``.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all monthly CSVs.
    """
    from utils.knowledge_base import parse_month_from_filename

    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob("sales_*.csv"))

    if not csv_files:
        if required:
            print(f"ERROR: No sales_*.csv files found in {csv_dir}")
            print("       Run 01_generate_monthly_sales.py first.")
            sys.exit(1)
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        print(f"   {f.name} ({len(df)} products)")
        if add_month_cols:
            meta = parse_month_from_filename(f.name)
            if meta:
                df["_year"], df["_month"] = meta[0], meta[1]
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Shelf assignment
# ---------------------------------------------------------------------------

def assign_shelves(df: pd.DataFrame, rng: np.random.Generator | np.random.RandomState) -> pd.DataFrame:
    """Assign ``rack_id`` (one per category) and random ``shelf_level`` 1-7."""
    categories = df["Category"].unique().tolist()
    df["rack_id"] = df["Category"].map({cat: i for i, cat in enumerate(categories)})
    if isinstance(rng, np.random.Generator):
        df["shelf_level"] = rng.integers(1, NUM_SHELVES + 1, size=len(df))
    else:
        df["shelf_level"] = [rng.randint(1, NUM_SHELVES + 1) for _ in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# Category profiles (margin & width ranges by keyword)
# ---------------------------------------------------------------------------

# (keywords, margin_range, width_range). First match wins.
CATEGORY_PROFILES: list[tuple[list[str], tuple[float, float], tuple[float, float]]] = [
    (["fruta", "verdura", "lechuga"],                                      (25, 45), (8,  25)),
    (["pescado", "marisco", "salaz"],                                      (20, 40), (10, 30)),
    (["cerdo", "pollo", "vacuno", "cordero", "ave", "carne", "hamburguesa"], (15, 35), (12, 28)),
    (["chocolate", "galleta", "cereal", "turron", "bolleria"],             (30, 55), (5,  20)),
    (["leche", "yogur", "queso", "mantequilla"],                           (20, 40), (6,  18)),
    (["cerveza", "vino", "licor", "agua", "refres", "zumo"],               (25, 50), (6,  12)),
    (["higiene", "cuidado", "gel", "champu", "desodorante", "jabon"],      (35, 60), (4,  10)),
    (["perfume", "colonia", "maquillaje", "labio", "ojo"],                 (40, 70), (3,   8)),
    (["conserva", "atun", "aceite", "vinagre"],                            (25, 45), (5,  15)),
    (["pasta", "arroz", "legumbre", "harina"],                             (20, 40), (6,  15)),
    (["congelad", "hielo"],                                                (25, 45), (8,  22)),
    (["pan", "pico", "tostada"],                                           (30, 50), (8,  20)),
    (["jamon", "embutido", "bacon", "chopped", "mortadela"],               (25, 45), (8,  18)),
]
_DEFAULT_PROFILE = ((20, 50), (5, 20))


def profile_for(category: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (margin_range, width_range) for a category string."""
    cat_lower = category.lower()
    for keywords, margin_range, width_range in CATEGORY_PROFILES:
        if any(k in cat_lower for k in keywords):
            return margin_range, width_range
    return _DEFAULT_PROFILE


# ---------------------------------------------------------------------------
# Seasonal factors (canonical, used by 01_generate and 05_predict)
# ---------------------------------------------------------------------------

SEASONAL_FACTORS: dict[int, dict[str, float]] = {
    1:  {"fruta": 0.8, "verdura": 0.9, "chocolate": 1.3, "galleta": 1.1,
         "conserva": 1.1, "default": 0.95},
    2:  {"fruta": 0.8, "verdura": 0.9, "chocolate": 1.5, "galleta": 1.1,
         "default": 0.95},
    3:  {"fruta": 0.9, "verdura": 1.0, "cerveza": 1.1, "default": 1.0},
    4:  {"fruta": 1.0, "verdura": 1.1, "cerveza": 1.1, "helado": 1.1,
         "default": 1.0},
    5:  {"fruta": 1.1, "verdura": 1.1, "cerveza": 1.2, "helado": 1.3,
         "agua": 1.2, "refres": 1.2, "default": 1.0},
    6:  {"fruta": 1.3, "verdura": 1.2, "helado": 1.5, "agua": 1.4,
         "refres": 1.4, "cerveza": 1.4, "default": 1.0},
    7:  {"fruta": 1.4, "verdura": 1.3, "helado": 1.8, "agua": 1.6,
         "refres": 1.5, "cerveza": 1.5, "default": 1.0},
    8:  {"fruta": 1.5, "verdura": 1.3, "helado": 1.9, "agua": 1.7,
         "refres": 1.6, "cerveza": 1.6, "default": 0.95},
    9:  {"fruta": 1.1, "verdura": 1.1, "cereal": 1.2, "galleta": 1.1,
         "leche": 1.1, "default": 1.05},
    10: {"fruta": 0.9, "verdura": 1.0, "chocolate": 1.2, "galleta": 1.2,
         "conserva": 1.1, "default": 1.0},
    11: {"chocolate": 1.4, "galleta": 1.3, "vino": 1.2, "licor": 1.3,
         "conserva": 1.2, "turron": 1.8, "default": 1.05},
    12: {"chocolate": 1.8, "galleta": 1.5, "vino": 1.6, "licor": 1.8,
         "marisco": 1.9, "turron": 2.5, "jamon": 1.5, "embutido": 1.4,
         "carne": 1.3, "default": 1.15},
}


def get_seasonal_mult(month: int, category: str) -> float:
    """Return the seasonal sales multiplier for a given month/category."""
    factors = SEASONAL_FACTORS.get(month, {"default": 1.0})
    cat_lower = category.lower()
    return next(
        (mult for key, mult in factors.items() if key != "default" and key in cat_lower),
        factors.get("default", 1.0),
    )

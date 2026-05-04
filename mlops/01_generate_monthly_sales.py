#!/usr/bin/env python3
"""
01_generate_monthly_sales.py — Monthly Sales Dataset Generator
================================================================
Reads `products_macro.csv` (Category, name, subtitle, price, discount_price)
and creates 6 monthly sales CSVs (July-December 2023), each simulating a
different month's shelf activity.

Output columns per CSV:
  Category, name, subtitle, price, discount_price,
  price_numeric, discount_price_numeric,
  estimated_monthly_sales, profit_margin_percentage,
  product_width_cm, rack_id, shelf_level

Not all products appear in every month — a random 60-90 % subset is chosen
each month, with seasonal variation in sales figures.

Usage:
    python 01_generate_monthly_sales.py                  # Heuristic mode (no API)
    python 01_generate_monthly_sales.py --use-llm        # OpenRouter LLM mode
    python 01_generate_monthly_sales.py --output-dir data/monthly
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# Ensure the mlops/ directory is on the path for utils imports

from utils.data_io import (
    assign_shelves,
    get_seasonal_mult,
    parse_eur_price,
    profile_for,
)
from utils.retail_physics import enforce_shelf_constraint

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "products_macro.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "monthly"

BATCH_SIZE = 50

# Months to generate
MONTHS = [
    (2023, 1, "january"),
    (2023, 2, "february"),
    (2023, 3, "march"),
    (2023, 4, "april"),
    (2023, 5, "may"),
    (2023, 6, "june"),
    (2023, 7, "july"),
    (2023, 8, "august"),
    (2023, 9, "september"),
    (2023, 10, "october"),
    (2023, 11, "november"),
    (2023, 12, "december"),
]

MONTH_NAMES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}


# ---------------------------------------------------------------------------
# Sales / margin / width generators (category-aware heuristics)
# ---------------------------------------------------------------------------

def generate_sales_data(row: pd.Series, rng: np.random.RandomState,
                        month: int) -> dict:
    """
    Generate estimated_monthly_sales, profit_margin_percentage,
    and product_width_cm for a single product in a given month.
    """
    price = row["price_numeric"]
    category = row["Category"]

    base_sales = max(10, int(300 / (price + 0.1)))
    noise = rng.uniform(0.7, 1.3)
    sales = int(base_sales * noise)

    seasonal_mult = get_seasonal_mult(month, category)
    sales = max(1, int(sales * seasonal_mult))

    margin_range, width_range = profile_for(category)
    margin = round(rng.uniform(*margin_range), 1)
    width = round(rng.uniform(*width_range), 1)

    # ±5% month-to-month noise on margin
    margin = round(margin * rng.uniform(0.95, 1.05), 1)

    return {
        "estimated_monthly_sales": sales,
        "profit_margin_percentage": margin,
        "product_width_cm": width,
    }


# ---------------------------------------------------------------------------
# OpenRouter LLM augmentation
# ---------------------------------------------------------------------------

def llm_augment_batch(batch_df: pd.DataFrame, month: int, year: int,
                      model: str | None = None) -> list[dict]:
    """Send a batch of products to the LLM and parse CSV estimates."""
    from utils.llm_client import chat_with_failover, resolve_models

    month_es = MONTH_NAMES_ES.get(month, str(month))

    products_text = "\n".join(
        f"{i+1}. {row['Category']} | {row['name']} | {row['price']}"
        for i, (_, row) in enumerate(batch_df.iterrows())
    )

    prompt = f"""You are a retail data analyst for a Spanish supermarket.
The month is {month_es.capitalize()} {year}.

For each product below, estimate:
- estimated_monthly_sales (integer): units sold this month. Consider seasonal demand.
- profit_margin_percentage (float 5-70): profit margin for this category.
- product_width_cm (float 2-60): shelf width in cm.

Products:
{products_text}

Respond with EXACTLY {len(batch_df)} lines of CSV, one per product, in order.
Format: sales,margin,width
Example: 200,30.5,10.0

No headers, no explanation, no markdown. Just the CSV lines."""

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    cascade = resolve_models()
    if model and model not in cascade:
        cascade = [model, *cascade]
    elif model:
        cascade = [model, *(m for m in cascade if m != model)]

    content = chat_with_failover(
        prompt, api_key=api_key, models=cascade, max_tokens=3000,
    )

    defaults = {"estimated_monthly_sales": 50, "profit_margin_percentage": 30.0, "product_width_cm": 12.0}

    if content:
        results = []
        for line in content.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip leading number prefix like "1. " or "1,"
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    results.append({
                        "estimated_monthly_sales": max(1, int(float(parts[0].strip()))),
                        "profit_margin_percentage": max(1.0, min(70.0, float(parts[1].strip()))),
                        "product_width_cm": max(2.0, min(60.0, float(parts[2].strip()))),
                    })
                    continue
                except (ValueError, IndexError):
                    pass
            results.append(dict(defaults))

        # Pad if model returned fewer lines than expected
        while len(results) < len(batch_df):
            results.append(dict(defaults))
        return results[:len(batch_df)]

    print(" [using defaults]", end="")
    return [dict(defaults) for _ in range(len(batch_df))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all = _load_catalogue(Path(args.input))

    if args.use_llm:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: Set OPENROUTER_API_KEY in your .env or environment.")
            sys.exit(1)
        print("LLM mode enabled (OpenRouter API)")
    else:
        print("Heuristic mode (no API calls)")

    for year, month, month_name in MONTHS:
        _generate_month(args, df_all, output_dir, year, month, month_name)

    print(f"\nDone. Generated {len(MONTHS)} monthly datasets in {output_dir}/")
    print("   Files:")
    for year, month, month_name in MONTHS:
        filename = f"sales_{year}_{month:02d}_{month_name}.csv"
        print(f"     {filename}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 12 monthly sales datasets (Jan-Dec 2023)")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="Path to products_macro.csv")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory where monthly CSVs are written")
    parser.add_argument("--min-pct", type=float, default=0.60,
                        help="Minimum %% of products to include per month")
    parser.add_argument("--max-pct", type=float, default=0.90,
                        help="Maximum %% of products to include per month")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use OpenRouter LLM instead of local heuristics "
                             "(requires OPENROUTER_API_KEY)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip months that already have a CSV file")
    return parser.parse_args()


def _load_catalogue(input_path: Path) -> pd.DataFrame:
    print(f"Loading {input_path}...")
    df_all = pd.read_csv(input_path)
    print(f"   {len(df_all)} products loaded, "
          f"{df_all['Category'].nunique()} categories")
    df_all["price_numeric"] = df_all["price"].apply(parse_eur_price)
    df_all["discount_price_numeric"] = df_all["discount_price"].apply(parse_eur_price)
    return df_all


def _generate_month(args, df_all, output_dir, year, month, month_name):
    print(f"\n{'='*60}")
    print(f"Generating {month_name.capitalize()} {year}...")
    print(f"{'='*60}")

    filename = f"sales_{year}_{month:02d}_{month_name}.csv"
    out_path = output_dir / filename
    if args.skip_existing and out_path.exists():
        print(f"   Skipping (already exists: {filename})")
        return

    month_seed = args.seed + year * 100 + month
    rng = np.random.RandomState(month_seed)

    df_month = _sample_products(df_all, rng, args.min_pct, args.max_pct)

    if args.use_llm:
        _augment_via_llm(df_month, rng, month, year)
    else:
        _augment_via_heuristic(df_month, rng, month)

    df_month = assign_shelves(df_month, rng)
    df_month = enforce_shelf_constraint(df_month)

    final_cols = [
        "Category", "name", "subtitle", "price", "discount_price",
        "price_numeric", "discount_price_numeric",
        "estimated_monthly_sales", "profit_margin_percentage",
        "product_width_cm", "rack_id", "shelf_level",
    ]
    df_month = df_month[final_cols]
    df_month.to_csv(out_path, index=False)

    print(f"   Saved -> {out_path}")
    print(f"   Products:     {len(df_month)}")
    print(f"   Categories:   {df_month['Category'].nunique()}")
    print(f"   Avg sales:    {df_month['estimated_monthly_sales'].mean():.0f}")
    print(f"   Avg margin:   {df_month['profit_margin_percentage'].mean():.1f}%")
    print(f"   Avg width:    {df_month['product_width_cm'].mean():.1f} cm")


def _sample_products(df_all, rng, min_pct, max_pct):
    pct = rng.uniform(min_pct, max_pct)
    n_products = max(10, int(len(df_all) * pct))
    sample_idx = rng.choice(df_all.index, size=n_products, replace=False)
    df_month = df_all.loc[sorted(sample_idx)].copy().reset_index(drop=True)
    print(f"   Selected {len(df_month)} / {len(df_all)} products ({pct:.0%})")
    return df_month


def _augment_via_heuristic(df_month, rng, month):
    aug_records = [generate_sales_data(df_month.loc[idx], rng, month)
                   for idx in df_month.index]
    aug_df = pd.DataFrame(aug_records)
    for col in aug_df.columns:
        df_month[col] = aug_df[col].to_numpy()


def _augment_via_llm(df_month, rng, month, year):
    from utils.llm_client import _is_local_mode

    aug_cols = ["estimated_monthly_sales", "profit_margin_percentage",
                "product_width_cm"]
    for c in aug_cols:
        df_month[c] = 0.0

    n_batches = (len(df_month) + BATCH_SIZE - 1) // BATCH_SIZE
    # Local models can only process one request at a time
    use_threads = not _is_local_mode()
    max_workers = 4 if use_threads else 1

    def process_batch(b):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(df_month))
        batch = df_month.iloc[start:end]
        results = llm_augment_batch(batch, month, year)
        return b, start, end, results

    def _apply_results(start, end, results):
        for col in aug_cols:
            df_month.iloc[start:end, df_month.columns.get_loc(col)] = [r[col] for r in results]

    completed = 0
    try:
        if use_threads:
            for chunk_start in range(0, n_batches, max_workers):
                chunk_end = min(chunk_start + max_workers, n_batches)
                batch_indices = list(range(chunk_start, chunk_end))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_batch, b): b for b in batch_indices}
                    for future in as_completed(futures):
                        _, start, end, results = future.result()
                        _apply_results(start, end, results)
                        completed += 1
                        print(f"   Batch {completed}/{n_batches} "
                              f"(products {start+1}-{end}) ... OK")
                time.sleep(1)
        else:
            for b in range(n_batches):
                _, start, end, results = process_batch(b)
                _apply_results(start, end, results)
                completed += 1
                print(f"   Batch {completed}/{n_batches} "
                      f"(products {start+1}-{end}) ... OK")
    except KeyboardInterrupt:
        print(f"\n\nWARNING: Interrupted at batch {completed}/{n_batches}. "
              f"Filling remaining with heuristics...")
        for idx in df_month.index:
            if df_month.loc[idx, "estimated_monthly_sales"] == 0:
                aug = generate_sales_data(df_month.loc[idx], rng, month)
                for k, v in aug.items():
                    df_month.loc[idx, k] = v


if __name__ == "__main__":
    main()
    os._exit(0)  # Force exit — kills lingering HTTP threads

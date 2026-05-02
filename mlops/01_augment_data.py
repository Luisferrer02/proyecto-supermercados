#!/usr/bin/env python3
"""
01_augment_data.py — Smart Data Augmentor
==========================================
Enriches the Mercadona product CSV using the OpenRouter API (LLM) to estimate:
  - estimated_monthly_sales
  - profit_margin_percentage
  - product_width_cm

Then assigns rack_id, random shelf_level, and enforces the 300 cm constraint.

Usage:
    python 01_augment_data.py                 # Full run (requires OPENROUTER_API_KEY)
    python 01_augment_data.py --dry-run       # Uses deterministic mock data
    python 01_augment_data.py --limit 20      # Only augment first 20 products
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.retail_physics import enforce_shelf_constraint, validate_all_shelves  # noqa: I001
from utils.data_io import assign_shelves, parse_eur_price, profile_for

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_INPUT = BASE_DIR / "products_macro.csv"
CSV_OUTPUT = BASE_DIR / "data" / "products_augmented.csv"
BATCH_SIZE = 10
MODEL_NAME = "arcee-ai/trinity-large-preview:free"
FALLBACK_MODEL = "stepfun/step-3.5-flash:free"


# ---------------------------------------------------------------------------
# Mock augmentation (for --dry-run)
# ---------------------------------------------------------------------------

def mock_augment(row: pd.Series, rng: np.random.Generator) -> dict:
    """Deterministic mock estimates based on simple heuristics."""
    price = row["price_numeric"]
    category = str(row["Category"])

    base_sales = max(10, int(300 / (price + 0.1)))
    sales = int(rng.integers(max(1, base_sales - 20), base_sales + 51))

    margin_range, width_range = profile_for(category)

    return {
        "estimated_monthly_sales": sales,
        "profit_margin_percentage": round(float(rng.uniform(*margin_range)), 1),
        "product_width_cm": round(float(rng.uniform(*width_range)), 1),
    }


# ---------------------------------------------------------------------------
# OpenRouter API augmentation
# ---------------------------------------------------------------------------

def llm_augment_batch(batch_df: pd.DataFrame, client, model: str = MODEL_NAME) -> list[dict]:
    """Send a batch of products to the LLM and parse JSON estimates."""
    _DEFAULTS = {"estimated_monthly_sales": 50, "profit_margin_percentage": 30.0, "product_width_cm": 12.0}
    products_text = "\n".join(
        f"{i+1}. Category: {row['Category']} | Name: {row['name']} | "
        f"Subtitle: {row['subtitle']} | Price: {row['price']}"
        for i, (_, row) in enumerate(batch_df.iterrows())
    )

    prompt = f"""You are a retail data analyst. For each product below, estimate three values:
1. estimated_monthly_sales: realistic units sold per month in a typical Spanish supermarket.
2. profit_margin_percentage: realistic profit margin % for this category (value between 5 and 70).
3. product_width_cm: realistic shelf width in centimeters the product occupies on a shelf.

Products:
{products_text}

Respond ONLY with a JSON array. Each element must have exactly these keys:
"estimated_monthly_sales" (integer), "profit_margin_percentage" (float), "product_width_cm" (float).
No markdown, no explanation, just the JSON array."""

    models_to_try = [model, FALLBACK_MODEL] if model != FALLBACK_MODEL else [model]

    for current_model in models_to_try:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000,
                )
                content = response.choices[0].message.content.strip()
                json_match = re.search(r"\[.*\]", content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                results = json.loads(content)

                # Pad or trim to match batch size
                n = len(batch_df)
                while len(results) < n:
                    results.append(dict(_DEFAULTS))
                results = results[:n]

                for r in results:
                    r["estimated_monthly_sales"] = max(1, int(r.get("estimated_monthly_sales", 50)))
                    r["profit_margin_percentage"] = max(1.0, min(70.0, float(r.get("profit_margin_percentage", 30))))
                    r["product_width_cm"] = max(2.0, min(60.0, float(r.get("product_width_cm", 12))))

                return results

            except Exception as e:
                err_str = str(e)
                wait = 2 ** (attempt + 1)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower()
                if is_rate_limit:
                    wait = max(wait, 5)
                if attempt < 2:
                    print(f"\n      Error: {e}, retrying in {wait}s …", end="")
                    time.sleep(wait)
                elif current_model != models_to_try[-1]:
                    print("\n      Switching to fallback model …", end="")

    print(" [using defaults]", end="")
    return [dict(_DEFAULTS) for _ in range(len(batch_df))]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smart Data Augmentor")
    parser.add_argument("--dry-run", action="store_true", help="Use mock data instead of API calls")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N products")
    args = parser.parse_args()

    print(f" Loading {CSV_INPUT} …")
    df = pd.read_csv(CSV_INPUT)
    if args.limit:
        df = df.head(args.limit)
    print(f"   {len(df)} products loaded, {df['Category'].nunique()} categories")

    df["price_numeric"] = df["price"].apply(parse_eur_price)
    df["discount_price_numeric"] = df["discount_price"].apply(parse_eur_price)

    for col in ["estimated_monthly_sales", "profit_margin_percentage", "product_width_cm"]:
        df[col] = 0.0

    if args.dry_run:
        print(" DRY RUN — using mock augmentation …")
        rng = np.random.default_rng(42)
        aug_data = [mock_augment(df.loc[idx], rng) for idx in df.index]
        for col in ["estimated_monthly_sales", "profit_margin_percentage", "product_width_cm"]:
            df[col] = [r[col] for r in aug_data]
    else:
        print(" Calling OpenRouter API …")
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print(" Set OPENROUTER_API_KEY environment variable first.")
            sys.exit(1)
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
        progress_path = BASE_DIR / "data" / "products_progress.csv"
        try:
            for b in range(n_batches):
                start, end = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, len(df))
                print(f"   Batch {b+1}/{n_batches} (products {start+1}-{end}) …", end=" ")
                results = llm_augment_batch(df.iloc[start:end], client)
                for col in ["estimated_monthly_sales", "profit_margin_percentage", "product_width_cm"]:
                    df.iloc[start:end, df.columns.get_loc(col)] = [r[col] for r in results]
                print()
                if (b + 1) % 50 == 0:
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(progress_path, index=False)
                    print(f"    Progress saved ({end} products so far)")
                time.sleep(3)
        except KeyboardInterrupt:
            print(f"\n\n  Interrupted at batch {b+1}/{n_batches}. Saving progress …")
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(progress_path, index=False)
            print(f"    Partial results saved to {progress_path}")
            sys.exit(0)

    # Assign rack_id and shelf_level
    rng_shelves = np.random.default_rng(42)
    df = assign_shelves(df, rng_shelves)
    print(f"  Assigned {df['Category'].nunique()} racks (one per category)")

    # Enforce 300cm constraint
    violations_before = validate_all_shelves(df)
    if violations_before:
        print(f"  {len(violations_before)} shelf violations detected. Fixing …")
        df = enforce_shelf_constraint(df)
        print(f"   After fix: {len(validate_all_shelves(df))} violations remaining.")
    else:
        print(" No shelf width violations.")

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUTPUT, index=False)
    print(f" Saved augmented data to {CSV_OUTPUT}")
    print(f"   Shape: {df.shape}")
    print("\n--- Quick Stats ---")
    print(f"   Avg monthly sales  : {df['estimated_monthly_sales'].mean():.0f}")
    print(f"   Avg profit margin  : {df['profit_margin_percentage'].mean():.1f}%")
    print(f"   Avg product width  : {df['product_width_cm'].mean():.1f} cm")


if __name__ == "__main__":
    main()

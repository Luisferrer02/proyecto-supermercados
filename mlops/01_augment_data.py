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

import os
import re
import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()  # Auto-load .env file

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
# Price parser
# ---------------------------------------------------------------------------

def parse_eur_price(price_str: str) -> float:
    """Convert  '0,36 €' or '32,52 €'  → float."""
    if pd.isna(price_str) or not str(price_str).strip():
        return 0.0
    cleaned = str(price_str).replace("€", "").replace("\xa0", "").strip()
    cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Mock augmentation (for --dry-run)
# ---------------------------------------------------------------------------

def mock_augment(row: pd.Series, rng: np.random.RandomState) -> Dict:
    """Deterministic mock estimates based on simple heuristics."""
    price = row["price_numeric"]
    category = str(row["Category"]).lower()

    # Heuristic sales: cheaper → more sales
    base_sales = max(10, int(300 / (price + 0.1)))
    sales = rng.randint(max(1, base_sales - 20), base_sales + 50)

    # Margin by rough category grouping
    if any(k in category for k in ["fruta", "verdura", "lechuga"]):
        margin = round(rng.uniform(25, 45), 1)
        width = round(rng.uniform(8, 25), 1)
    elif any(k in category for k in ["pescado", "marisco", "salaz"]):
        margin = round(rng.uniform(20, 40), 1)
        width = round(rng.uniform(10, 30), 1)
    elif any(k in category for k in ["cerdo", "pollo", "vacuno", "cordero", "ave"]):
        margin = round(rng.uniform(15, 35), 1)
        width = round(rng.uniform(12, 28), 1)
    elif any(k in category for k in ["chocolate", "galleta", "cereal"]):
        margin = round(rng.uniform(30, 55), 1)
        width = round(rng.uniform(5, 20), 1)
    elif any(k in category for k in ["leche", "yogur", "queso"]):
        margin = round(rng.uniform(20, 40), 1)
        width = round(rng.uniform(6, 18), 1)
    elif any(k in category for k in ["cerveza", "vino", "licor", "agua", "refres"]):
        margin = round(rng.uniform(25, 50), 1)
        width = round(rng.uniform(6, 12), 1)
    elif any(k in category for k in ["higiene", "cuidado", "gel", "champu", "desodorante"]):
        margin = round(rng.uniform(35, 60), 1)
        width = round(rng.uniform(4, 10), 1)
    elif any(k in category for k in ["perfume", "colonia", "maquillaje", "labio", "ojo"]):
        margin = round(rng.uniform(40, 70), 1)
        width = round(rng.uniform(3, 8), 1)
    else:
        margin = round(rng.uniform(20, 50), 1)
        width = round(rng.uniform(5, 20), 1)

    return {
        "estimated_monthly_sales": sales,
        "profit_margin_percentage": margin,
        "product_width_cm": width,
    }


# ---------------------------------------------------------------------------
# OpenRouter API augmentation
# ---------------------------------------------------------------------------

def llm_augment_batch(batch_df: pd.DataFrame, client, model: str = MODEL_NAME) -> List[Dict]:
    """
    Send a batch of products to the LLM and parse JSON estimates.
    Includes retry logic with exponential backoff and model fallback.
    """
    products_desc = []
    for i, (_, row) in enumerate(batch_df.iterrows()):
        products_desc.append(
            f"{i+1}. Category: {row['Category']} | "
            f"Name: {row['name']} | "
            f"Subtitle: {row['subtitle']} | "
            f"Price: {row['price']}"
        )
    products_text = "\n".join(products_desc)

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
    max_retries = 3

    for current_model in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000,
                )
                content = response.choices[0].message.content.strip()

                # Try to extract JSON from possible markdown fences
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                results = json.loads(content)

                # Validate length
                if len(results) != len(batch_df):
                    while len(results) < len(batch_df):
                        results.append({
                            "estimated_monthly_sales": 50,
                            "profit_margin_percentage": 30.0,
                            "product_width_cm": 12.0,
                        })
                    results = results[:len(batch_df)]

                # Sanitize values
                for r in results:
                    r["estimated_monthly_sales"] = max(1, int(r.get("estimated_monthly_sales", 50)))
                    r["profit_margin_percentage"] = max(1.0, min(70.0, float(r.get("profit_margin_percentage", 30))))
                    r["product_width_cm"] = max(2.0, min(60.0, float(r.get("product_width_cm", 12))))

                return results

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower()
                wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds

                if is_rate_limit:
                    wait = max(wait, 5)  # at least 5s for rate limits
                    if attempt < max_retries - 1:
                        print(f"\n     ⏳ Rate limited ({current_model}), retrying in {wait}s (attempt {attempt+2}/{max_retries}) …", end="")
                        time.sleep(wait)
                        continue
                    else:
                        if current_model != models_to_try[-1]:
                            print(f"\n     🔄 Switching to fallback model …", end="")
                            break  # try next model
                        # else fall through to defaults
                else:
                    print(f"\n     ✗ Error: {e}", end="")
                    if attempt < max_retries - 1:
                        time.sleep(wait)
                        continue

    # All retries and models exhausted — return defaults
    print(" [using defaults]", end="")
    return [
        {"estimated_monthly_sales": 50, "profit_margin_percentage": 30.0, "product_width_cm": 12.0}
        for _ in range(len(batch_df))
    ]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smart Data Augmentor")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock data instead of API calls")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N products")
    args = parser.parse_args()

    # 1. Load CSV
    print(f"📂 Loading {CSV_INPUT} …")
    df = pd.read_csv(CSV_INPUT)
    if args.limit:
        df = df.head(args.limit)
    print(f"   {len(df)} products loaded, {df['Category'].nunique()} categories")

    # 2. Parse prices
    df["price_numeric"] = df["price"].apply(parse_eur_price)
    df["discount_price_numeric"] = df["discount_price"].apply(parse_eur_price)

    # 3. Augment
    aug_cols = ["estimated_monthly_sales", "profit_margin_percentage", "product_width_cm"]
    for c in aug_cols:
        df[c] = 0.0

    if args.dry_run:
        print("🧪 DRY RUN — using mock augmentation …")
        rng = np.random.RandomState(42)
        for idx in df.index:
            mock = mock_augment(df.loc[idx], rng)
            for k, v in mock.items():
                df.at[idx, k] = v
    else:
        print("🌐 Calling OpenRouter API …")
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("❌ Set OPENROUTER_API_KEY environment variable first.")
            sys.exit(1)
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
        try:
            for b in range(n_batches):
                start = b * BATCH_SIZE
                end = min(start + BATCH_SIZE, len(df))
                batch = df.iloc[start:end]
                print(f"   Batch {b+1}/{n_batches} (products {start+1}-{end}) …", end=" ")

                results = llm_augment_batch(batch, client)
                for i, res in enumerate(results):
                    idx = df.index[start + i]
                    for k, v in res.items():
                        df.at[idx, k] = v
                print("✓")

                # Save progress every 50 batches
                if (b + 1) % 50 == 0:
                    progress_path = BASE_DIR / "data" / "products_progress.csv"
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(progress_path, index=False)
                    print(f"   💾 Progress saved ({end} products so far)")

                time.sleep(3)  # rate-limit courtesy for free tier
        except KeyboardInterrupt:
            print(f"\n\n⚠  Interrupted at batch {b+1}/{n_batches}. Saving progress …")
            progress_path = BASE_DIR / "data" / "products_progress.csv"
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(progress_path, index=False)
            print(f"   💾 Partial results saved to {progress_path}")
            print("   Re-run the script to resume (already-augmented products keep defaults for merge).")
            sys.exit(0)

    # 4. Assign rack_id by category
    categories = df["Category"].unique().tolist()
    cat_to_rack = {cat: i for i, cat in enumerate(categories)}
    df["rack_id"] = df["Category"].map(cat_to_rack)
    print(f"🗄  Assigned {len(categories)} racks (one per category)")

    # 5. Random baseline shelf_level 1-7
    random.seed(42)
    df["shelf_level"] = [random.randint(1, 7) for _ in range(len(df))]

    # 6. Enforce 300cm constraint
    from utils.retail_physics import enforce_shelf_constraint, validate_all_shelves
    violations_before = validate_all_shelves(df)
    if violations_before:
        print(f"⚠  {len(violations_before)} shelf violations detected. Fixing …")
        df = enforce_shelf_constraint(df)
        violations_after = validate_all_shelves(df)
        print(f"   After fix: {len(violations_after)} violations remaining.")
    else:
        print("✅ No shelf width violations.")

    # 7. Save
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"💾 Saved augmented data to {CSV_OUTPUT}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Shape: {df.shape}")
    print()
    # Quick stats
    print("--- Quick Stats ---")
    print(f"   Avg monthly sales  : {df['estimated_monthly_sales'].mean():.0f}")
    print(f"   Avg profit margin  : {df['profit_margin_percentage'].mean():.1f}%")
    print(f"   Avg product width  : {df['product_width_cm'].mean():.1f} cm")


if __name__ == "__main__":
    main()

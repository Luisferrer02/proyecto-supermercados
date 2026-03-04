#!/usr/bin/env python3
"""
01_generate_monthly_sales.py — Monthly Sales Dataset Generator
================================================================
Reads `products_macro.csv` (Category, name, subtitle, price, discount_price)
and creates 6 monthly sales CSVs (July–December 2025), each simulating a
different month's shelf activity.

Output columns per CSV:
  Category, name, subtitle, price, discount_price,
  price_numeric, discount_price_numeric,
  estimated_monthly_sales, profit_margin_percentage,
  product_width_cm, rack_id, shelf_level

Not all products appear in every month — a random 60-90% subset is chosen
each month, with seasonal variation in sales figures.

Usage:
    python 01_generate_monthly_sales.py                  # Heuristic mode (no API)
    python 01_generate_monthly_sales.py --use-llm        # OpenRouter LLM mode
    python 01_generate_monthly_sales.py --output-dir data/monthly
"""

import argparse
import json
import os
import re
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # Auto-load .env file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "products_macro.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "monthly"

NUM_SHELVES = 7
SHELF_WIDTH_CM = 300
BATCH_SIZE = 30
MODEL_NAME = "arcee-ai/trinity-large-preview:free"
FALLBACK_MODEL = "stepfun/step-3.5-flash:free"

# Months to generate
MONTHS = [
    (2025, 1, "january"),
    (2025, 2, "february"),
    (2025, 3, "march"),
    (2025, 4, "april"),
    (2025, 5, "may"),
    (2025, 6, "june"),
    (2025, 7, "july"),
    (2025, 8, "august"),
    (2025, 9, "september"),
    (2025, 10, "october"),
    (2025, 11, "november"),
    (2025, 12, "december"),
]

# Seasonal multipliers: how much sales shift per month
# (summer months = more fresh food, December = holidays/treats)
SEASONAL_FACTORS = {
    1: {"fruta": 0.8, "verdura": 0.9, "chocolate": 1.3, "galleta": 1.1,
        "conserva": 1.1, "default": 0.95},
    2: {"fruta": 0.8, "verdura": 0.9, "chocolate": 1.5, "galleta": 1.1,
        "default": 0.95},
    3: {"fruta": 0.9, "verdura": 1.0, "cerveza": 1.1, "default": 1.0},
    4: {"fruta": 1.0, "verdura": 1.1, "cerveza": 1.1, "helado": 1.1,
        "default": 1.0},
    5: {"fruta": 1.1, "verdura": 1.1, "cerveza": 1.2, "helado": 1.3,
        "agua": 1.2, "refres": 1.2, "default": 1.0},
    6: {"fruta": 1.3, "verdura": 1.2, "helado": 1.5, "agua": 1.4,
        "refres": 1.4, "cerveza": 1.4, "default": 1.0},
    7: {"fruta": 1.4, "verdura": 1.3, "helado": 1.8, "agua": 1.6,
        "refres": 1.5, "cerveza": 1.5, "default": 1.0},
    8: {"fruta": 1.5, "verdura": 1.3, "helado": 1.9, "agua": 1.7,
        "refres": 1.6, "cerveza": 1.6, "default": 0.95},
    9: {"fruta": 1.1, "verdura": 1.1, "cereal": 1.2, "galleta": 1.1,
        "leche": 1.1, "default": 1.05},
    10: {"fruta": 0.9, "verdura": 1.0, "chocolate": 1.2, "galleta": 1.2,
         "conserva": 1.1, "default": 1.0},
    11: {"chocolate": 1.4, "galleta": 1.3, "vino": 1.2, "licor": 1.3,
         "conserva": 1.2, "turrón": 1.8, "default": 1.05},
    12: {"chocolate": 1.8, "galleta": 1.5, "vino": 1.6, "licor": 1.8,
         "marisco": 1.9, "turrón": 2.5, "jamón": 1.5, "embutido": 1.4,
         "carne": 1.3, "default": 1.15},
}


MONTH_NAMES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}


# ---------------------------------------------------------------------------
# Price parser (same as 01_augment_data.py)
# ---------------------------------------------------------------------------

def parse_eur_price(price_str) -> float:
    """Convert '0,36 €' or '32,52 €' → float."""
    if pd.isna(price_str) or str(price_str).strip() == "":
        return 0.0
    s = str(price_str).replace("€", "").replace("\xa0", "").strip()
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Sales / margin / width generators (category-aware heuristics)
# ---------------------------------------------------------------------------

def _category_key(category: str) -> str:
    """Lowercase category for matching."""
    return str(category).lower()


def _get_seasonal_mult(month: int, category: str) -> float:
    """Return the seasonal sales multiplier for a given month/category."""
    factors = SEASONAL_FACTORS.get(month, {})
    cat = _category_key(category)
    for key, mult in factors.items():
        if key != "default" and key in cat:
            return mult
    return factors.get("default", 1.0)


def generate_sales_data(row: pd.Series, rng: np.random.RandomState,
                        month: int) -> dict:
    """
    Generate estimated_monthly_sales, profit_margin_percentage,
    and product_width_cm for a single product in a given month.
    """
    price = row["price_numeric"]
    category = _category_key(row["Category"])

    # --- Base sales: cheaper products sell more ---
    base_sales = max(10, int(300 / (price + 0.1)))
    # Add random noise (±30%)
    noise = rng.uniform(0.7, 1.3)
    sales = int(base_sales * noise)

    # Apply seasonal multiplier
    seasonal_mult = _get_seasonal_mult(month, row["Category"])
    sales = max(1, int(sales * seasonal_mult))

    # --- Profit margin by category group ---
    if any(k in category for k in ["fruta", "verdura", "lechuga"]):
        margin = round(rng.uniform(25, 45), 1)
        width = round(rng.uniform(8, 25), 1)
    elif any(k in category for k in ["pescado", "marisco", "salaz"]):
        margin = round(rng.uniform(20, 40), 1)
        width = round(rng.uniform(10, 30), 1)
    elif any(k in category for k in ["cerdo", "pollo", "vacuno", "cordero",
                                      "ave", "carne", "hamburguesa"]):
        margin = round(rng.uniform(15, 35), 1)
        width = round(rng.uniform(12, 28), 1)
    elif any(k in category for k in ["chocolate", "galleta", "cereal",
                                      "turrón", "bollería"]):
        margin = round(rng.uniform(30, 55), 1)
        width = round(rng.uniform(5, 20), 1)
    elif any(k in category for k in ["leche", "yogur", "queso", "mantequilla"]):
        margin = round(rng.uniform(20, 40), 1)
        width = round(rng.uniform(6, 18), 1)
    elif any(k in category for k in ["cerveza", "vino", "licor", "agua",
                                      "refres", "zumo"]):
        margin = round(rng.uniform(25, 50), 1)
        width = round(rng.uniform(6, 12), 1)
    elif any(k in category for k in ["higiene", "cuidado", "gel", "champu",
                                      "desodorante", "jabón"]):
        margin = round(rng.uniform(35, 60), 1)
        width = round(rng.uniform(4, 10), 1)
    elif any(k in category for k in ["perfume", "colonia", "maquillaje",
                                      "labio", "ojo"]):
        margin = round(rng.uniform(40, 70), 1)
        width = round(rng.uniform(3, 8), 1)
    elif any(k in category for k in ["conserva", "atún", "aceite", "vinagre"]):
        margin = round(rng.uniform(25, 45), 1)
        width = round(rng.uniform(5, 15), 1)
    elif any(k in category for k in ["pasta", "arroz", "legumbre", "harina"]):
        margin = round(rng.uniform(20, 40), 1)
        width = round(rng.uniform(6, 15), 1)
    elif any(k in category for k in ["congelad", "hielo"]):
        margin = round(rng.uniform(25, 45), 1)
        width = round(rng.uniform(8, 22), 1)
    elif any(k in category for k in ["pan", "pico", "tostada"]):
        margin = round(rng.uniform(30, 50), 1)
        width = round(rng.uniform(8, 20), 1)
    elif any(k in category for k in ["jamón", "embutido", "bacón",
                                      "chopped", "mortadela"]):
        margin = round(rng.uniform(25, 45), 1)
        width = round(rng.uniform(8, 18), 1)
    else:
        margin = round(rng.uniform(20, 50), 1)
        width = round(rng.uniform(5, 20), 1)

    # Add slight month-to-month noise to margin (±5%)
    margin = round(margin * rng.uniform(0.95, 1.05), 1)

    return {
        "estimated_monthly_sales": sales,
        "profit_margin_percentage": margin,
        "product_width_cm": width,
    }


# ---------------------------------------------------------------------------
# OpenRouter LLM augmentation
# ---------------------------------------------------------------------------

def llm_augment_batch(batch_df: pd.DataFrame, client, month: int, year: int,
                      model: str = MODEL_NAME) -> List[Dict]:
    """
    Send a batch of products to the LLM and parse JSON estimates.
    Includes retry logic with exponential backoff and model fallback.
    The prompt includes month context for seasonal awareness.
    """
    month_es = MONTH_NAMES_ES.get(month, str(month))

    products_desc = []
    for i, (_, row) in enumerate(batch_df.iterrows()):
        products_desc.append(
            f"{i+1}. Category: {row['Category']} | "
            f"Name: {row['name']} | "
            f"Subtitle: {row['subtitle']} | "
            f"Price: {row['price']}"
        )
    products_text = "\n".join(products_desc)

    prompt = f"""You are a retail data analyst for a Spanish supermarket.
The month is {month_es.capitalize()} {year}.

For each product below, estimate three values considering the time of year:
1. estimated_monthly_sales: realistic units sold THIS month in a typical Spanish supermarket.
   Consider seasonal demand (e.g. more fresh fruit in summer, more sweets/marisco in December).
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
                    max_tokens=5000,
                    timeout=30,
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
                    r["estimated_monthly_sales"] = max(1, int(
                        r.get("estimated_monthly_sales", 50)))
                    r["profit_margin_percentage"] = max(1.0, min(70.0, float(
                        r.get("profit_margin_percentage", 30))))
                    r["product_width_cm"] = max(2.0, min(60.0, float(
                        r.get("product_width_cm", 12))))

                return results

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower()
                wait = 2 ** (attempt + 1) + random.uniform(0, 2)  # jitter

                if is_rate_limit:
                    wait = max(wait, 5)
                    if attempt < max_retries - 1:
                        print(f"\n     Rate limited ({current_model}), "
                              f"retrying in {wait:.0f}s "
                              f"(attempt {attempt+2}/{max_retries})...",
                              end="")
                        time.sleep(wait)
                        continue
                    else:
                        if current_model != models_to_try[-1]:
                            print(f"\n     Switching to fallback model...",
                                  end="")
                            break  # try next model
                else:
                    print(f"\n     ✗ Error: {e}", end="")
                    if attempt < max_retries - 1:
                        time.sleep(wait)
                        continue

    # All retries exhausted — return defaults
    print(" [using defaults]", end="")
    return [
        {"estimated_monthly_sales": 50,
         "profit_margin_percentage": 30.0,
         "product_width_cm": 12.0}
        for _ in range(len(batch_df))
    ]


# ---------------------------------------------------------------------------
# Shelf assignment & constraint enforcement
# ---------------------------------------------------------------------------

def assign_shelves(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    """Assign rack_id (by category) and random shelf_level 1-7."""
    categories = df["Category"].unique().tolist()
    cat_to_rack = {cat: i for i, cat in enumerate(categories)}
    df["rack_id"] = df["Category"].map(cat_to_rack)
    df["shelf_level"] = [rng.randint(1, NUM_SHELVES + 1) for _ in range(len(df))]
    return df


def enforce_shelf_constraint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Move products off shelves that exceed 300cm total width.
    Overflow products are moved to the shelf with the most remaining space.
    """
    df = df.copy()
    for (rack_id, shelf_level), group in df.groupby(["rack_id", "shelf_level"]):
        total_w = group["product_width_cm"].sum()
        if total_w <= SHELF_WIDTH_CM:
            continue

        # Sort by width descending — move widest products first
        sorted_idx = group.sort_values("product_width_cm", ascending=False).index
        current_w = total_w
        for idx in sorted_idx:
            if current_w <= SHELF_WIDTH_CM:
                break
            # Find the shelf (same rack) with the most free space
            rack_df = df[df["rack_id"] == rack_id]
            space_by_shelf = {}
            for s in range(1, NUM_SHELVES + 1):
                used = rack_df[rack_df["shelf_level"] == s]["product_width_cm"].sum()
                space_by_shelf[s] = SHELF_WIDTH_CM - used
            # Pick shelf with most space (excluding current shelf)
            best_shelf = max(
                (s for s in space_by_shelf if s != shelf_level),
                key=lambda s: space_by_shelf[s],
                default=shelf_level,
            )
            pw = df.at[idx, "product_width_cm"]
            if space_by_shelf.get(best_shelf, 0) >= pw:
                df.at[idx, "shelf_level"] = best_shelf
                current_w -= pw
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate 12 monthly sales datasets (Jan-Dec 2025)")
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
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load base catalogue
    print(f"Loading {input_path}...")
    df_all = pd.read_csv(input_path)
    print(f"   {len(df_all)} products loaded, "
          f"{df_all['Category'].nunique()} categories")

    # 2. Parse prices once
    df_all["price_numeric"] = df_all["price"].apply(parse_eur_price)
    df_all["discount_price_numeric"] = df_all["discount_price"].apply(parse_eur_price)

    master_rng = np.random.RandomState(args.seed)

    # Set up LLM client if requested
    client = None
    if args.use_llm:
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: Install openai package: pip install openai")
            sys.exit(1)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: Set OPENROUTER_API_KEY in your .env or environment.")
            sys.exit(1)
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=30,
        )
        print("LLM mode enabled (OpenRouter API)")
    else:
        print("Heuristic mode (no API calls)")

    # 3. Generate one CSV per month
    for year, month, month_name in MONTHS:
        print(f"\n{'='*60}")
        print(f"Generating {month_name.capitalize()} {year}...")
        print(f"{'='*60}")

        # Skip if file already exists
        filename = f"sales_{year}_{month:02d}_{month_name}.csv"
        out_path = output_dir / filename
        if args.skip_existing and out_path.exists():
            print(f"   Skipping (already exists: {filename})")
            continue

        # Seed for this month (deterministic but different each month)
        month_seed = args.seed + year * 100 + month
        rng = np.random.RandomState(month_seed)

        # Random subset: 60-90% of products
        pct = rng.uniform(args.min_pct, args.max_pct)
        n_products = max(10, int(len(df_all) * pct))
        sample_idx = rng.choice(df_all.index, size=n_products, replace=False)
        df_month = df_all.loc[sorted(sample_idx)].copy().reset_index(drop=True)
        print(f"   Selected {len(df_month)} / {len(df_all)} products ({pct:.0%})")

        # Generate sales data
        if args.use_llm and client is not None:
            # ---- LLM mode: batch API calls ----
            aug_cols = ["estimated_monthly_sales", "profit_margin_percentage",
                        "product_width_cm"]
            for c in aug_cols:
                df_month[c] = 0.0

            n_batches = (len(df_month) + BATCH_SIZE - 1) // BATCH_SIZE
            max_workers = 4  # parallel API calls

            def process_batch(b):
                """Process a single batch — called from thread pool."""
                start = b * BATCH_SIZE
                end = min(start + BATCH_SIZE, len(df_month))
                batch = df_month.iloc[start:end]
                results = llm_augment_batch(batch, client, month, year)
                return b, start, end, results

            try:
                completed = 0
                for chunk_start in range(0, n_batches, max_workers):
                    chunk_end = min(chunk_start + max_workers, n_batches)
                    batch_indices = list(range(chunk_start, chunk_end))

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(process_batch, b): b
                            for b in batch_indices
                        }

                        for future in as_completed(futures):
                            b, start, end, results = future.result()
                            for i, res in enumerate(results):
                                idx = df_month.index[start + i]
                                for k, v in res.items():
                                    df_month.at[idx, k] = v
                            completed += 1
                            print(f"   Batch {completed}/{n_batches} "
                                  f"(products {start+1}-{end}) ... OK")

                    time.sleep(1)  # brief pause between parallel chunks

            except KeyboardInterrupt:
                print(f"\n\nWARNING: Interrupted at batch {completed}/{n_batches}. "
                      f"Filling remaining with heuristics...")
                # Fill remaining products with heuristic data
                for idx in df_month.index:
                    if df_month.at[idx, "estimated_monthly_sales"] == 0:
                        aug = generate_sales_data(df_month.loc[idx], rng, month)
                        for k, v in aug.items():
                            df_month.at[idx, k] = v
        else:
            # ---- Heuristic mode: local generation ----
            aug_records = []
            for idx in df_month.index:
                aug = generate_sales_data(df_month.loc[idx], rng, month)
                aug_records.append(aug)
            aug_df = pd.DataFrame(aug_records)
            for col in aug_df.columns:
                df_month[col] = aug_df[col].values

        # Assign racks and shelves
        df_month = assign_shelves(df_month, rng)

        # Enforce 300cm constraint
        df_month = enforce_shelf_constraint(df_month)

        # Select and order final columns
        final_cols = [
            "Category", "name", "subtitle", "price", "discount_price",
            "price_numeric", "discount_price_numeric",
            "estimated_monthly_sales", "profit_margin_percentage",
            "product_width_cm", "rack_id", "shelf_level",
        ]
        df_month = df_month[final_cols]

        # Save
        filename = f"sales_{year}_{month:02d}_{month_name}.csv"
        out_path = output_dir / filename
        df_month.to_csv(out_path, index=False)

        # Stats
        print(f"   Saved -> {out_path}")
        print(f"   Products:     {len(df_month)}")
        print(f"   Categories:   {df_month['Category'].nunique()}")
        print(f"   Avg sales:    {df_month['estimated_monthly_sales'].mean():.0f}")
        print(f"   Avg margin:   {df_month['profit_margin_percentage'].mean():.1f}%")
        print(f"   Avg width:    {df_month['product_width_cm'].mean():.1f} cm")

    print(f"\nDone. Generated {len(MONTHS)} monthly datasets in {output_dir}/")
    print("   Files:")
    for year, month, month_name in MONTHS:
        filename = f"sales_{year}_{month:02d}_{month_name}.csv"
        print(f"     {filename}")


if __name__ == "__main__":
    main()
    os._exit(0)  # Force exit — kills lingering HTTP threads

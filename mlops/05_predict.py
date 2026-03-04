#!/usr/bin/env python3
"""
05_predict.py — RAG-Powered Shelf Optimization
================================================
End-to-end prediction pipeline:
  1. RAG retrieval: pulls relevant months from the knowledge base
  2. LLM forecast: reasons about trends and generates sales predictions
  3. Ensemble optimization: MLP proposes, Transformer validates
  4. Output: CSV + profit comparison

Usage:
    python 05_predict.py --month 2026-01
    python 05_predict.py --month 2026-01 --category "Verdura"
    python 05_predict.py --month 2026-01 --dry-run   # Skip LLM, use heuristics
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"
MONTHLY_DIR = DATA_DIR / "monthly"
NUM_SHELVES = 7
SHELF_WIDTH_CM = 300

MODEL_NAME = "arcee-ai/trinity-large-preview:free"
FALLBACK_MODEL = "stepfun/step-3.5-flash:free"

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

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


# ---------------------------------------------------------------------------
# Step 1: RAG Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(target_year: int, target_month: int,
                     category: Optional[str] = None) -> dict:
    """
    Query the knowledge base for relevant historical months.
    Returns structured context for the LLM.
    """
    from utils.knowledge_base import ShelfKnowledgeBase

    kb = ShelfKnowledgeBase()
    context = kb.retrieve_context(target_year, target_month, category)

    if not context:
        print("   ⚠  No data found in knowledge base. "
              "Run 04_ingest.py first.")
        return {}

    print(f"   📚 Retrieved context from {len(context)} month(s):")
    for month_key, data in context.items():
        n_docs = len(data["documents"])
        print(f"      • {month_key}: {n_docs} category summaries")

    return context


# ---------------------------------------------------------------------------
# Step 2: LLM Forecast
# ---------------------------------------------------------------------------

def build_forecast_prompt(target_year: int, target_month: int,
                          context: dict,
                          base_products: pd.DataFrame) -> str:
    """
    Build the LLM prompt with RAG context + product list.
    Asks the LLM to forecast sales adjustments for the target month.
    """
    month_name = MONTH_NAMES.get(target_month, str(target_month))

    # Build context section
    context_sections = []
    for month_key, data in sorted(context.items()):
        docs = data["documents"]
        meta = data["metadatas"]
        section = f"### Data from {month_key}\n"
        for doc in docs:
            section += f"- {doc}\n"
        context_sections.append(section)

    context_text = "\n".join(context_sections)

    # Product list (truncate to manageable size)
    products = base_products[["Category", "name", "price_numeric",
                              "estimated_monthly_sales",
                              "profit_margin_percentage"]].copy()
    if len(products) > 100:
        # Sample key products per category
        products = (products.sort_values(["Category", "estimated_monthly_sales"], 
                                         ascending=[True, False])
                            .groupby("Category")
                            .head(5))

    product_lines = []
    for _, r in products.iterrows():
        product_lines.append(
            f"  {r['Category']} | {r['name']} | "
            f"€{r['price_numeric']:.2f} | "
            f"Sales: {int(r['estimated_monthly_sales'])} | "
            f"Margin: {r['profit_margin_percentage']:.1f}%"
        )
    products_text = "\n".join(product_lines[:150])

    prompt = f"""You are a retail analyst predicting sales for a Spanish supermarket.

## Target Month: {month_name} {target_year}

## Historical Context (from knowledge base)
{context_text}

## Current Product Base (representative sample)
{products_text}

## Task
Based on the historical data above, forecast how sales will change for \
{month_name} {target_year} compared to the most recent month's data.

Consider:
1. **Seasonal trends**: What categories sell more/less in {month_name}?
2. **Recent momentum**: Are any categories trending up or down in recent months?
3. **Year-over-year**: If we have data from {month_name} last year, \
what patterns repeat?

Respond ONLY with a JSON object mapping category names to a sales \
multiplier (float). A value of 1.0 means no change, 1.3 means +30% \
sales, 0.7 means -30% sales.

Example format:
{{"Fruta": 1.2, "Verdura": 1.1, "Marisco": 0.8, "Chocolate": 1.5}}

Include ALL categories present in the data. Be realistic — most \
multipliers should be between 0.7 and 1.5.
Respond with ONLY the JSON object, no explanation."""

    return prompt


def llm_forecast(target_year: int, target_month: int,
                 context: dict,
                 base_products: pd.DataFrame) -> dict:
    """
    Call the LLM to forecast sales multipliers per category.
    Returns {category: multiplier} dict.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("   ❌ OPENROUTER_API_KEY not set. Use --dry-run for heuristics.")
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    prompt = build_forecast_prompt(target_year, target_month,
                                   context, base_products)

    models_to_try = [MODEL_NAME, FALLBACK_MODEL]
    for model in models_to_try:
        try:
            print(f"   🤖 Querying LLM ({model.split('/')[-1]})…")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000,
            )
            content = response.choices[0].message.content.strip()

            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            multipliers = json.loads(content)

            # Sanitize
            clean = {}
            for cat, mult in multipliers.items():
                clean[cat] = max(0.3, min(2.5, float(mult)))

            print(f"   ✅ Got forecasts for {len(clean)} categories")
            return clean

        except Exception as e:
            print(f"   ⚠  Error with {model}: {e}")
            continue

    print("   ⚠  All LLM attempts failed. Using neutral multipliers.")
    return {}


def heuristic_forecast(target_month: int,
                       categories: list) -> dict:
    """Fallback: seasonal multipliers without LLM."""
    seasonal = {
        1: {"fruta": 0.8, "verdura": 0.9, "chocolate": 1.3, "default": 0.95},
        2: {"fruta": 0.8, "chocolate": 1.5, "default": 0.95},
        3: {"fruta": 0.9, "verdura": 1.0, "default": 1.0},
        4: {"fruta": 1.0, "verdura": 1.1, "default": 1.0},
        5: {"fruta": 1.1, "verdura": 1.1, "default": 1.0},
        6: {"fruta": 1.3, "verdura": 1.2, "helado": 1.5, "default": 1.0},
        7: {"fruta": 1.4, "verdura": 1.3, "helado": 1.8, "agua": 1.5,
            "default": 1.0},
        8: {"fruta": 1.5, "verdura": 1.3, "helado": 1.9, "agua": 1.6,
            "default": 0.95},
        9: {"fruta": 1.1, "cereal": 1.2, "default": 1.05},
        10: {"fruta": 0.9, "chocolate": 1.2, "default": 1.0},
        11: {"chocolate": 1.4, "vino": 1.2, "turrón": 1.8, "default": 1.05},
        12: {"chocolate": 1.8, "marisco": 1.9, "turrón": 2.5, "jamón": 1.5,
             "vino": 1.6, "default": 1.15},
    }
    factors = seasonal.get(target_month, {"default": 1.0})

    result = {}
    for cat in categories:
        cat_lower = cat.lower()
        matched = False
        for key, mult in factors.items():
            if key != "default" and key in cat_lower:
                result[cat] = mult
                matched = True
                break
        if not matched:
            result[cat] = factors.get("default", 1.0)

    return result


# ---------------------------------------------------------------------------
# Step 3: Apply forecast + Ensemble optimization
# ---------------------------------------------------------------------------

def apply_forecast(df: pd.DataFrame, multipliers: dict) -> pd.DataFrame:
    """Apply sales multipliers from the LLM forecast."""
    df = df.copy()
    for cat, mult in multipliers.items():
        mask = df["Category"] == cat
        df.loc[mask, "estimated_monthly_sales"] = (
            df.loc[mask, "estimated_monthly_sales"] * mult
        ).astype(int).clip(lower=1)
    return df


def _mlp_optimize_rack(rack_df, mlp_model, noise_scale=0.0):
    """
    Use MLP to greedily assign each product to the best shelf.
    noise_scale > 0 adds randomness to create diverse candidates.
    Returns a copy of rack_df with updated shelf_level.
    """
    result = rack_df.copy()

    # Sort by profit potential (highest first)
    result["_profit_potential"] = (
        result["price_numeric"]
        * (result["profit_margin_percentage"] / 100.0)
        * result["estimated_monthly_sales"]
    )
    result = result.sort_values("_profit_potential", ascending=False)

    capacity = {s: SHELF_WIDTH_CM for s in range(1, NUM_SHELVES + 1)}
    shelf_counts = {s: 0 for s in range(1, NUM_SHELVES + 1)}

    for idx in result.index:
        row = result.loc[idx]
        original_shelf = int(row["shelf_level"])
        w = row["product_width_cm"]
        best_shelf = original_shelf
        best_lift = -float("inf")

        n_shelves_used = sum(1 for c in shelf_counts.values() if c > 0)

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
                shelf_counts.get(original_shelf, 0),
                shelf_counts.get(candidate_shelf, 0),
                n_shelves_used,
                len(result),
            ]])

            with torch.no_grad():
                pred_lift = mlp_model(features).item()

            # Add noise for diversity in candidate generation
            if noise_scale > 0:
                pred_lift += np.random.normal(0, noise_scale)

            if pred_lift > best_lift:
                best_lift = pred_lift
                best_shelf = candidate_shelf

        result.at[idx, "shelf_level"] = best_shelf
        capacity[best_shelf] -= w
        shelf_counts[best_shelf] += 1

    result.drop(columns=["_profit_potential"], inplace=True)
    return result


def _transformer_score_rack(rack_df, transformer_model):
    """
    Use the Transformer to predict total profit lift for a rack layout.
    Returns the sum of per-product predicted profit lifts.
    """
    n = len(rack_df)
    shelf_counts = rack_df["shelf_level"].value_counts().to_dict()
    n_shelves_used = len(shelf_counts)

    features = []
    for _, row in rack_df.iterrows():
        shelf = int(row["shelf_level"])
        features.append([
            row["price_numeric"],
            row["profit_margin_percentage"],
            row["estimated_monthly_sales"],
            row["product_width_cm"],
            shelf,                           # original_shelf (keeping it as current)
            shelf,                           # new_shelf (same = staying put)
            shelf_counts.get(shelf, 0),      # n_products_on_original_shelf
            shelf_counts.get(shelf, 0),      # n_products_on_new_shelf
            n_shelves_used,
            n,
        ])

    # Transformer expects (batch, seq_len, features)
    X = torch.FloatTensor([features])  # (1, n_products, 10)
    with torch.no_grad():
        scores = transformer_model(X)  # (1, n_products)
    return scores.sum().item()


def optimize_ensemble(df: pd.DataFrame,
                      mlp_path: Path,
                      transformer_path: Path,
                      n_candidates: int = 5) -> pd.DataFrame:
    """
    Ensemble optimization: MLP proposes, Transformer validates.

    For each rack:
      1. MLP generates N candidate layouts (greedy + noise)
      2. Transformer scores each candidate
      3. Best-scored candidate is selected
    """
    from models.mlp import build_mlp
    from models.transformer_model import build_transformer

    input_dim = len(FEATURE_COLS)

    # Load MLP
    mlp = build_mlp(input_dim=input_dim)
    if not mlp_path.exists():
        print(f"   ERROR: No MLP model at {mlp_path}")
        print("   Run 02_train_models.py first.")
        sys.exit(1)
    mlp.load_state_dict(torch.load(mlp_path, weights_only=True))
    mlp.eval()

    # Load Transformer
    transformer = build_transformer(input_dim=input_dim)
    if not transformer_path.exists():
        print(f"   ERROR: No Transformer model at {transformer_path}")
        print("   Run 02_train_models.py first.")
        sys.exit(1)
    transformer.load_state_dict(torch.load(transformer_path, weights_only=True))
    transformer.eval()

    optimized_dfs = []
    total_racks = df["rack_id"].nunique()

    for i, rack_id in enumerate(df["rack_id"].unique()):
        rack_df = df[df["rack_id"] == rack_id].copy()
        if len(rack_df) < 2:
            optimized_dfs.append(rack_df)
            continue

        # MLP generates candidates:
        # Candidate 0 = pure greedy (no noise)
        # Candidates 1..N-1 = greedy + increasing noise
        candidates = []
        noise_levels = [0.0] + [5.0 * (i + 1) for i in range(n_candidates - 1)]

        for noise in noise_levels:
            candidate = _mlp_optimize_rack(rack_df, mlp, noise_scale=noise)
            candidates.append(candidate)

        # Transformer scores each candidate
        best_score = -float("inf")
        best_candidate = candidates[0]

        for candidate in candidates:
            score = _transformer_score_rack(candidate, transformer)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        optimized_dfs.append(best_candidate)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"   Rack {i+1}/{total_racks}...")

    return pd.concat(optimized_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 4: Output
# ---------------------------------------------------------------------------

def save_results(original_df: pd.DataFrame, optimized_df: pd.DataFrame,
                 target_year: int, target_month: int,
                 multipliers: dict):
    """Save optimized layout and generate summary."""
    from utils.retail_physics import compute_rack_profit

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    month_name = MONTH_NAMES.get(target_month, str(target_month)).lower()

    # Save optimized CSV
    out_csv = RESULTS_DIR / f"optimized_{target_year}_{target_month:02d}_{month_name}.csv"
    optimized_df.to_csv(out_csv, index=False)
    print(f"\n   💾 Optimized layout saved → {out_csv}")

    # Save forecast multipliers
    forecast_path = RESULTS_DIR / f"forecast_{target_year}_{target_month:02d}.json"
    with open(forecast_path, "w") as f:
        json.dump(multipliers, f, indent=2, ensure_ascii=False)
    print(f"   💾 Forecast multipliers → {forecast_path}")

    # Compute profit comparison per rack
    print(f"\n   {'='*60}")
    print(f"   OPTIMIZATION RESULTS — {MONTH_NAMES[target_month]} {target_year}")
    print(f"   {'='*60}")

    total_orig_profit = 0
    total_opt_profit = 0
    rack_results = []

    for rack_id in sorted(optimized_df["rack_id"].unique()):
        orig_rack = original_df[original_df["rack_id"] == rack_id]
        opt_rack = optimized_df[optimized_df["rack_id"] == rack_id]

        if len(orig_rack) == 0 or len(opt_rack) == 0:
            continue

        orig_profit = compute_rack_profit(orig_rack)
        opt_profit = compute_rack_profit(opt_rack)
        lift = opt_profit - orig_profit
        category = orig_rack["Category"].iloc[0]

        total_orig_profit += orig_profit
        total_opt_profit += opt_profit
        rack_results.append({
            "rack_id": rack_id,
            "category": category,
            "orig_profit": orig_profit,
            "opt_profit": opt_profit,
            "lift": lift,
        })

    # Print top improving racks
    rack_results.sort(key=lambda x: x["lift"], reverse=True)
    print(f"\n   {'Category':<35} {'Original':>10} {'Optimized':>10} {'Lift':>10}")
    print(f"   {'-'*65}")
    for r in rack_results[:15]:
        print(f"   {r['category']:<35} €{r['orig_profit']:>8.0f} "
              f"€{r['opt_profit']:>8.0f} {r['lift']:>+8.0f}")

    total_lift = total_opt_profit - total_orig_profit
    pct = (total_lift / total_orig_profit * 100) if total_orig_profit > 0 else 0
    print(f"   {'-'*65}")
    print(f"   {'TOTAL':<35} €{total_orig_profit:>8.0f} "
          f"€{total_opt_profit:>8.0f} {total_lift:>+8.0f} ({pct:+.1f}%)")
    print(f"   {'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG-powered shelf optimization for a target month")
    parser.add_argument("--month", type=str, required=True,
                        help="Target month in YYYY-MM format (e.g. 2026-01)")
    parser.add_argument("--category", type=str, default=None,
                        help="Optimize only a specific category")
    parser.add_argument("--data-dir", type=str,
                        default=str(MONTHLY_DIR),
                        help="Directory with monthly sales CSVs")
    parser.add_argument("--n-candidates", type=int, default=5,
                        help="Number of candidate layouts MLP generates per rack")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use heuristic forecast instead of LLM")
    args = parser.parse_args()

    # Parse target month
    try:
        parts = args.month.split("-")
        target_year = int(parts[0])
        target_month = int(parts[1])
        assert 1 <= target_month <= 12
    except (ValueError, IndexError, AssertionError):
        print("❌ Invalid month format. Use YYYY-MM (e.g. 2026-01)")
        sys.exit(1)

    month_name = MONTH_NAMES[target_month]
    data_dir = Path(args.data_dir)
    mlp_path = RESULTS_DIR / "mlp.pth"
    transformer_path = RESULTS_DIR / "transformer.pth"

    print("=" * 65)
    print(f"  SHELF OPTIMIZATION -- {month_name} {target_year}")
    print("=" * 65)

    # Step 1: RAG Retrieval
    print(f"\n  Step 1: Retrieving historical context...")
    context = retrieve_context(target_year, target_month, args.category)

    if not context:
        print("   No context available. Continuing with heuristics only.")

    # Step 2: Load base product data (most recent month)
    print(f"\n  Step 2: Loading base product data...")
    from utils.knowledge_base import ShelfKnowledgeBase
    kb = ShelfKnowledgeBase()
    base_df = kb.get_latest_month_data(data_dir, args.category)

    if base_df is None or len(base_df) == 0:
        print("   ERROR: No product data found. Upload CSVs and run 04_ingest.py.")
        sys.exit(1)

    print(f"   Loaded {len(base_df)} products, "
          f"{base_df['Category'].nunique()} categories")

    # Step 3: Forecast
    print(f"\n  Step 3: Forecasting sales for {month_name} {target_year}...")
    if args.dry_run or not context:
        print("   Using heuristic seasonal forecast")
        categories = base_df["Category"].unique().tolist()
        multipliers = heuristic_forecast(target_month, categories)
    else:
        multipliers = llm_forecast(target_year, target_month,
                                   context, base_df)

    # Show top adjustments
    if multipliers:
        sorted_mults = sorted(multipliers.items(),
                              key=lambda x: abs(x[1] - 1.0), reverse=True)
        print(f"\n   Top forecast adjustments:")
        for cat, mult in sorted_mults[:10]:
            direction = "UP" if mult > 1.0 else "DOWN" if mult < 1.0 else "--"
            print(f"     [{direction}] {cat}: x{mult:.2f}")

    # Step 4: Apply forecast
    print(f"\n  Step 4: Applying forecast to product data...")
    forecasted_df = apply_forecast(base_df, multipliers)
    original_df = base_df.copy()

    print(f"   Avg sales before: "
          f"{base_df['estimated_monthly_sales'].mean():.0f}")
    print(f"   Avg sales after:  "
          f"{forecasted_df['estimated_monthly_sales'].mean():.0f}")

    # Step 5: Ensemble Optimization (MLP proposes, Transformer validates)
    print(f"\n  Step 5: Running ensemble optimization (MLP + Transformer)...")
    optimized_df = optimize_ensemble(
        forecasted_df, mlp_path, transformer_path,
        n_candidates=args.n_candidates,
    )
    print(f"   Optimized {len(optimized_df)} products across "
          f"{optimized_df['rack_id'].nunique()} racks")

    # Step 6: Save results
    save_results(original_df, optimized_df,
                 target_year, target_month, multipliers)

    print(f"\n  Done! Check results/ for output files.")


if __name__ == "__main__":
    main()

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
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv


from models.mlp import build_mlp
from models.transformer_model import build_transformer
from utils.training import FEATURE_COLS
from utils.data_io import get_seasonal_mult
from utils.explainability import explain_all
from utils.knowledge_base import ShelfKnowledgeBase
from utils.llm_client import chat_with_failover
from utils.retail_physics import (
    NUM_SHELVES,
    SHELF_WIDTH_CM,
    compute_rack_profit,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MONTHLY_DIR = BASE_DIR / "data" / "monthly"

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


# ---------------------------------------------------------------------------
# Step 1: RAG Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(target_year: int, target_month: int, category: str | None = None) -> dict:
    """Query the knowledge base for relevant historical months."""
    kb = ShelfKnowledgeBase()
    context = kb.retrieve_context(target_year, target_month, category)

    if not context:
        print("     No data found in knowledge base. Run 04_ingest.py first.")
        return {}

    print(f"    Retrieved context from {len(context)} month(s):")
    for month_key, data in context.items():
        print(f"      - {month_key}: {len(data['documents'])} category summaries")
    return context


# ---------------------------------------------------------------------------
# Step 2: LLM Forecast
# ---------------------------------------------------------------------------

def _build_forecast_prompt(target_year: int, target_month: int,
                            context: dict, base_products: pd.DataFrame) -> str:
    month_name = MONTH_NAMES.get(target_month, str(target_month))

    # Limit RAG context to top 20 categories per month to keep prompt ~4K tokens
    context_lines = []
    for month_key, data in sorted(context.items()):
        docs = data["documents"]
        scored = []
        for doc in docs:
            try:
                sales = float(doc.split("Total sales: ")[1].split(" ")[0])
            except (IndexError, ValueError):
                sales = 0
            scored.append((sales, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        context_lines.append(f"### {month_key}")
        context_lines.extend(f"- {doc}" for _, doc in scored[:20])
    context_text = "\n".join(context_lines)

    categories = sorted(base_products["Category"].unique())
    products_text = "\n".join(f"  {cat}" for cat in categories)

    return f"""You are a retail analyst predicting sales for a Spanish supermarket.

## Target Month: {month_name} {target_year}

## Historical Context (from knowledge base)
{context_text}

## Categories to forecast
{products_text}

## Task
Predict a sales multiplier for EACH category above for {month_name} {target_year}.

Key seasonal patterns for a Spanish supermarket:
- **January**: Post-holiday dip. Less marisco/turron/vino (Christmas is over). More soups, legumes, diet products. Fruit and vegetables drop (winter).
- **Summer (Jun-Aug)**: More agua, refrescos, helados, cerveza, fruta, verdura. Less chocolate, soups, hot drinks.
- **November-December**: Christmas surge — marisco, turron, chocolate, vino, jamon, embutido all spike. General spending up 10-15%.
- **September**: Back to school — cereals, leche, galletas up. Summer products drop.

Rules:
- Most categories should be between 0.85 and 1.15 (subtle changes)
- Only strongly seasonal categories should go beyond that range (0.7-1.5)
- Use the historical data above to identify actual trends, not just generic patterns

Respond with ONLY a JSON object. Keys = exact category names from the list above. Values = float multiplier.
Example: {{"Fruta": 0.85, "Chocolate": 1.3, "Agua": 0.8}}
No markdown fences, no explanation."""


def llm_forecast(target_year: int, target_month: int,
                 context: dict, base_products: pd.DataFrame) -> dict:
    """Call the LLM to forecast sales multipliers per category."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    prompt = _build_forecast_prompt(target_year, target_month, context, base_products)
    content = chat_with_failover(prompt, api_key=api_key)
    if not content:
        return {}

    try:
        # Strip markdown fences
        cleaned = re.sub(r"```\w*\n?", "", content).strip()
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        raw = json_match.group(0) if json_match else cleaned
        # Fix common LLM JSON issues
        raw = raw.replace("'", '"')
        raw = re.sub(r",{2,}", ",", raw)       # double/triple commas → single
        raw = re.sub(r",\s*}", "}", raw)        # trailing comma before }
        raw = re.sub(r",\s*]", "]", raw)        # trailing comma before ]
        multipliers = json.loads(raw)
    except Exception as exc:
        print(f"     Could not parse JSON from LLM response: {exc}")
        print(f"     Raw response (first 500 chars): {content[:500]}")
        return {}

    clean = {}
    for cat, mult in multipliers.items():
        try:
            clean[cat] = max(0.3, min(2.5, float(mult)))
        except (TypeError, ValueError):
            continue

    print(f"    Got forecasts for {len(clean)} categories")
    return clean


def heuristic_forecast(target_month: int, categories: list) -> dict:
    """Fallback: seasonal multipliers without LLM."""
    return {cat: get_seasonal_mult(target_month, cat) for cat in categories}


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


def _transformer_score_rack(rack_df: pd.DataFrame, original_rack_df: pd.DataFrame, transformer_model) -> float:
    """Use the Transformer to score a rack layout (sum of predicted profit lifts vs original)."""
    n = len(rack_df)
    shelf_counts_new = rack_df["shelf_level"].value_counts().to_dict()
    shelf_counts_orig = original_rack_df["shelf_level"].value_counts().to_dict()
    n_shelves_used = len(shelf_counts_new)

    features = []
    for (_, row), (_, orig_row) in zip(rack_df.iterrows(), original_rack_df.iterrows()):
        orig_shelf = int(orig_row["shelf_level"])
        new_shelf = int(row["shelf_level"])
        features.append([
            row["price_numeric"], row["profit_margin_percentage"],
            row["estimated_monthly_sales"], row["product_width_cm"],
            orig_shelf, new_shelf,
            shelf_counts_orig.get(orig_shelf, 0),
            shelf_counts_new.get(new_shelf, 0),
            n_shelves_used, n,
        ])
    X = torch.FloatTensor([features])  # (1, n_products, 10)
    with torch.no_grad():
        return transformer_model(X).sum().item()


def optimize_ensemble(df: pd.DataFrame, mlp_path: Path, transformer_path: Path,
                      n_candidates: int = 5) -> pd.DataFrame:
    """Ensemble: MLP proposes N candidate layouts, Transformer picks the best."""
    from utils.training import FeatureNormalizer, optimize_rack_mlp

    input_dim = len(FEATURE_COLS)

    # Load MLP (lift-based)
    mlp = build_mlp(input_dim=input_dim)
    if not mlp_path.exists():
        print(f"   ERROR: No MLP model at {mlp_path}. Run 02_train_models.py first.")
        sys.exit(1)
    mlp.load_state_dict(torch.load(mlp_path, weights_only=True))
    mlp.eval()

    # Load normalizer
    normalizer = None
    normalizer_path = mlp_path.parent / "normalizer.pth"
    if normalizer_path.exists():
        data = torch.load(normalizer_path, weights_only=True)
        normalizer = FeatureNormalizer()
        normalizer.mean = data["mean"]
        normalizer.std = data["std"]

    # Load Transformer (for scoring candidates)
    transformer = build_transformer(input_dim=input_dim)
    if not transformer_path.exists():
        print(f"   ERROR: No Transformer model at {transformer_path}. Run 02_train_models.py first.")
        sys.exit(1)
    transformer.load_state_dict(torch.load(transformer_path, weights_only=True))
    transformer.eval()

    optimized_dfs = []
    rack_ids = df["rack_id"].unique()

    for i, rack_id in enumerate(rack_ids):
        rack_df = df[df["rack_id"] == rack_id].copy()
        if len(rack_df) < 2:
            optimized_dfs.append(rack_df)
            continue

        # MLP generates N candidate layouts (greedy + noise)
        noise_levels = [0.0] + [5.0 * (j + 1) for j in range(n_candidates - 1)]
        candidates = [
            optimize_rack_mlp(rack_df, mlp, NUM_SHELVES, SHELF_WIDTH_CM,
                              noise_scale=noise, normalizer=normalizer)
            for noise in noise_levels
        ]

        # Transformer scores each candidate vs original — pick the best
        best = max(candidates, key=lambda c: _transformer_score_rack(c, rack_df, transformer))
        optimized_dfs.append(best)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"   Rack {i+1}/{len(rack_ids)}...")

    return pd.concat(optimized_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 4: Output
# ---------------------------------------------------------------------------

def save_results(original_df: pd.DataFrame, optimized_df: pd.DataFrame,
                 target_year: int, target_month: int,
                 multipliers: dict, forecast_source: str = "unknown"):
    """Save optimized layout and generate summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    month_name = MONTH_NAMES.get(target_month, str(target_month)).lower()

    out_csv = RESULTS_DIR / f"optimized_{target_year}_{target_month:02d}_{month_name}.csv"
    optimized_df.to_csv(out_csv, index=False)
    print(f"\n    Optimized layout saved → {out_csv}")

    try:
        explanations = explain_all(original_df, optimized_df)
        n_moved = sum(len(v) for v in explanations.values())
        expl_path = RESULTS_DIR / f"explanations_{target_year}_{target_month:02d}.json"
        with open(expl_path, "w", encoding="utf-8") as f:
            json.dump({
                "_target_year": target_year, "_target_month": target_month,
                "n_products_moved": n_moved, "by_rack": explanations,
            }, f, indent=2, ensure_ascii=False)
        print(f"    Explanations ({n_moved} products moved) → {expl_path}")
    except Exception as exc:
        print(f"     Could not generate explanations: {exc}")

    forecast_path = RESULTS_DIR / f"forecast_{target_year}_{target_month:02d}.json"
    with open(forecast_path, "w") as f:
        json.dump({
            "_source": forecast_source,
            "_target_year": target_year, "_target_month": target_month,
            "multipliers": multipliers,
        }, f, indent=2, ensure_ascii=False)
    print(f"    Forecast multipliers → {forecast_path}  (source: {forecast_source})")

    print(f"\n   {'='*60}")
    print(f"   OPTIMIZATION RESULTS — {MONTH_NAMES[target_month]} {target_year}")
    print(f"   {'='*60}")

    rack_results = []
    total_orig = total_opt = 0.0
    for rack_id in sorted(optimized_df["rack_id"].unique()):
        orig_rack = original_df[original_df["rack_id"] == rack_id]
        opt_rack  = optimized_df[optimized_df["rack_id"] == rack_id]
        if orig_rack.empty or opt_rack.empty:
            continue
        orig_p = compute_rack_profit(orig_rack)
        opt_p  = compute_rack_profit(opt_rack)
        total_orig += orig_p
        total_opt  += opt_p
        rack_results.append({
            "category": orig_rack["Category"].iloc[0],
            "orig": orig_p, "opt": opt_p, "lift": opt_p - orig_p,
        })

    rack_results.sort(key=lambda x: x["lift"], reverse=True)
    print(f"\n   {'Category':<35} {'Original':>10} {'Optimized':>10} {'Lift':>10}")
    print(f"   {'-'*65}")
    for r in rack_results[:15]:
        print(f"   {r['category']:<35} €{r['orig']:>8.0f} €{r['opt']:>8.0f} {r['lift']:>+8.0f}")

    total_lift = total_opt - total_orig
    pct = (total_lift / total_orig * 100) if total_orig > 0 else 0
    print(f"   {'-'*65}")
    print(f"   {'TOTAL':<35} €{total_orig:>8.0f} €{total_opt:>8.0f} {total_lift:>+8.0f} ({pct:+.1f}%)")
    print(f"   {'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG-powered shelf optimization for a target month")
    parser.add_argument("--month", type=str, required=True, help="Target month YYYY-MM (e.g. 2026-01)")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=str(MONTHLY_DIR))
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Use heuristic forecast instead of LLM")
    args = parser.parse_args()

    try:
        year_s, month_s = args.month.split("-")
        target_year, target_month = int(year_s), int(month_s)
        if not (1 <= target_month <= 12):
            raise ValueError("month out of range")
    except (ValueError, IndexError):
        print(" Invalid month format. Use YYYY-MM (e.g. 2026-01)")
        sys.exit(1)

    month_name = MONTH_NAMES[target_month]
    data_dir = Path(args.data_dir)
    mlp_path = RESULTS_DIR / "mlp.pth"
    transformer_path = RESULTS_DIR / "transformer.pth"

    print("=" * 65)
    print(f"  SHELF OPTIMIZATION -- {month_name} {target_year}")
    print("=" * 65)

    print("\n  Step 1: Retrieving historical context...")
    context = retrieve_context(target_year, target_month, args.category)
    if not context:
        print("   No context available. Continuing with heuristics only.")

    print("\n  Step 2: Loading base product data...")
    kb = ShelfKnowledgeBase()
    base_df = kb.get_latest_month_data(data_dir, args.category)
    if base_df is None or len(base_df) == 0:
        print("   ERROR: No product data found. Upload CSVs and run 04_ingest.py.")
        sys.exit(1)
    print(f"   Loaded {len(base_df)} products, {base_df['Category'].nunique()} categories")

    print(f"\n  Step 3: Forecasting sales for {month_name} {target_year}...")
    categories = base_df["Category"].unique().tolist()
    forecast_source = "heuristic"

    if args.dry_run or not context:
        print("   Using heuristic seasonal forecast (dry-run or no RAG context)")
        multipliers = heuristic_forecast(target_month, categories)
    else:
        multipliers = llm_forecast(target_year, target_month, context, base_df)
        if multipliers:
            forecast_source = "llm"
        else:
            print("     LLM unavailable — falling back to heuristic forecast.")
            multipliers = heuristic_forecast(target_month, categories)

    if multipliers:
        top = sorted(multipliers.items(), key=lambda x: abs(x[1] - 1.0), reverse=True)[:10]
        print("\n   Top forecast adjustments:")
        for cat, mult in top:
            direction = "UP" if mult > 1.0 else "DOWN" if mult < 1.0 else "--"
            print(f"     [{direction}] {cat}: x{mult:.2f}")

    print("\n  Step 4: Applying forecast to product data...")
    forecasted_df = apply_forecast(base_df, multipliers)
    print(f"   Avg sales before: {base_df['estimated_monthly_sales'].mean():.0f}")
    print(f"   Avg sales after:  {forecasted_df['estimated_monthly_sales'].mean():.0f}")

    print("\n  Step 5: Running ensemble optimization (MLP + Transformer)...")
    optimized_df = optimize_ensemble(forecasted_df, mlp_path, transformer_path,
                                     n_candidates=args.n_candidates)
    print(f"   Optimized {len(optimized_df)} products across {optimized_df['rack_id'].nunique()} racks")

    save_results(base_df, optimized_df, target_year, target_month, multipliers,
                 forecast_source=forecast_source)

    print("\n  Done! Check results/ for output files.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
03_evaluate.py — Evaluation & Visualization
=============================================
Generates:
  1. Comparison table (MSE + Total Predicted Profit per model)
  2. Matplotlib plots: rack comparison, MSE, profit, shelf bars, alluvial

Usage:
    python 03_evaluate.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — must be before pyplot import

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from utils.data_io import load_monthly_csvs
from utils.retail_physics import NUM_SHELVES, compute_rack_profit, optimize_rack_greedy

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MONTHLY_DIR = BASE_DIR / "data" / "monthly"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "training_results.json"

SHELF_LABELS = [
    "1 (Bottom)", "2 (Bottom)", "3 (Eye)", "4 (Eye)", "5 (Eye)", "6 (Top)", "7 (Top)"
]
SHELF_CMAP = {
    1: "#3498db", 2: "#2980b9",
    3: "#f39c12", 4: "#e67e22", 5: "#d35400",
    6: "#1abc9c", 7: "#16a085",
}
SHELF_NAMES = {
    1: "Shelf 1 (Bottom)", 2: "Shelf 2 (Bottom)",
    3: "Shelf 3 (Eye)", 4: "Shelf 4 (Eye)", 5: "Shelf 5 (Eye)",
    6: "Shelf 6 (Top)", 7: "Shelf 7 (Top)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load monthly CSVs and training results."""
    df = load_monthly_csvs(MONTHLY_DIR)
    print(f"Loaded {len(df)} product-month records from {MONTHLY_DIR}")

    results = {}
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            results = json.load(f)
    return df, results


def _load_layouts() -> dict[str, pd.DataFrame]:
    """Load all rack layout CSVs from results/."""
    layouts = {}
    for csv_file in RESULTS_DIR.glob("rack_layout_*.csv"):
        name = csv_file.stem.replace("rack_layout_", "").upper()
        if name == "ORIGINAL":
            name = "Original"
        layouts[name] = pd.read_csv(csv_file)
    return layouts


def _pick_best_ml(layouts: dict) -> tuple[str, pd.DataFrame]:
    """Return (name, df) of the best non-greedy ML layout by profit."""
    best_name, best_profit = None, -float("inf")
    for name, layout in layouts.items():
        if name in {"Original", "GREEDY"}:
            continue
        p = compute_rack_profit(layout)
        if p > best_profit:
            best_profit = p
            best_name = name
    return (best_name or "GREEDY"), layouts[best_name or "GREEDY"]


def _profit_by_shelf(rack_df: pd.DataFrame) -> list[float]:
    """Return monthly profit per shelf level (list of NUM_SHELVES floats)."""
    return [
        compute_rack_profit(rack_df[rack_df["shelf_level"] == s]) if len(rack_df[rack_df["shelf_level"] == s]) else 0
        for s in range(1, NUM_SHELVES + 1)
    ]


# ---------------------------------------------------------------------------
# 1. Comparison Table
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict):
    """Pretty-print the model comparison table."""
    print("\n" + "=" * 75)
    print("                    MODEL COMPARISON TABLE")
    print("=" * 75)
    print(f"{'Model':<18} {'MSE':>12} {'Orig Profit (€)':>18} {'Opt Profit (€)':>18} {'Lift (€)':>12}")
    print("-" * 75)

    for name, r in results.items():
        mse_val = r.get("mse")
        mse_str = f"{mse_val:.4f}" if mse_val is not None else "  N/A"
        orig = r.get("original_profit", 0)
        opt = r.get("optimized_profit", 0)
        print(f"{name:<18} {mse_str:>12} {orig:>18.2f} {opt:>18.2f} {opt - orig:>+12.2f}")

    print("=" * 75)

    best_profit_model = max(results.items(), key=lambda x: x[1].get("optimized_profit", 0))
    best_mse_models = {k: v for k, v in results.items() if "mse" in v}
    if best_mse_models:
        best_mse_model = min(best_mse_models.items(), key=lambda x: x[1]["mse"])
        print(f"\n Lowest MSE    : {best_mse_model[0]} ({best_mse_model[1]['mse']:.4f})")
    print(f" Highest Profit: {best_profit_model[0]} (€{best_profit_model[1].get('optimized_profit', 0):.2f})")


# ---------------------------------------------------------------------------
# 2. Rack Visualization
# ---------------------------------------------------------------------------

def visualize_rack(df: pd.DataFrame, rack_id: int | None = None):
    """Side-by-side bar chart: Original vs Optimized rack profit by shelf."""
    if rack_id is None:
        rack_id = df.groupby("rack_id").size().idxmax()

    rack_df = df[df["rack_id"] == rack_id].head(40).copy()
    category = rack_df["Category"].iloc[0] if "Category" in rack_df.columns else f"Rack {rack_id}"

    orig_profits = _profit_by_shelf(rack_df)
    greedy_df = optimize_rack_greedy(rack_df)
    # Only show optimized if it actually improves
    if compute_rack_profit(greedy_df) > compute_rack_profit(rack_df):
        opt_profits = _profit_by_shelf(greedy_df)
    else:
        opt_profits = orig_profits

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(f"Shelf Profit Distribution — Category: {category}",
                 fontsize=16, fontweight="bold", y=0.98)

    shelves = list(range(1, NUM_SHELVES + 1))
    colors_orig = ["#4a90d9"] * 2 + ["#f5a623"] * 3 + ["#50c8c8"] * 2
    colors_opt  = ["#3a7bc8"] * 2 + ["#e8961e"] * 3 + ["#40b0b0"] * 2

    for ax, profits, colors, title, title_color in [
        (axes[0], orig_profits, colors_orig, "[Original] Layout", "#c0392b"),
        (axes[1], opt_profits,  colors_opt,  "[Optimized] Layout", "#27ae60"),
    ]:
        bars = ax.barh(shelves, profits, color=colors, edgecolor="white", linewidth=1.2, height=0.7)
        ax.set_yticks(shelves)
        ax.set_yticklabels(SHELF_LABELS, fontsize=11)
        ax.set_xlabel("Monthly Profit (€)", fontsize=12)
        ax.set_title(title, fontsize=14, pad=12, color=title_color, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        max_val = max(profits) if profits else 1
        for bar, val in zip(bars, profits, strict=True):
            if val > 0:
                ax.text(bar.get_width() + max_val * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"€{val:.0f}", va="center", fontsize=9)

    orig_total, opt_total = sum(orig_profits), sum(opt_profits)
    lift_pct = ((opt_total - orig_total) / orig_total * 100) if orig_total else 0
    fig.text(0.5, 0.02,
             f"Original Total: €{orig_total:,.0f}  →  Optimized Total: €{opt_total:,.0f}  "
             f"(Profit Lift: {lift_pct:+.1f}%)",
             ha="center", fontsize=13, fontweight="bold",
             bbox={"boxstyle": "round,pad=0.4", "facecolor": "#e8f5e9", "alpha": 0.8})

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plot_path = RESULTS_DIR / "rack_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n Rack visualization saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 3. Model MSE Bar Plot
# ---------------------------------------------------------------------------

def plot_mse_comparison(results: dict):
    """Bar chart comparing MSE across supervised models."""
    mse_models = {k: v["mse"] for k, v in results.items() if "mse" in v}
    if not mse_models:
        print("   No MSE data available to plot.")
        return

    names, values = list(mse_models.keys()), list(mse_models.values())
    colors = ["#4a90d9", "#50c878", "#f5a623"][:len(names)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=13)
    ax.set_title("Model Prediction Accuracy Comparison", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    max_val = max(values)
    for bar, val in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "mse_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f" MSE comparison saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Profit Comparison Bar Plot
# ---------------------------------------------------------------------------

def plot_profit_comparison(results: dict):
    """Grouped bar chart: original vs optimized profit per model."""
    models = {k: v for k, v in results.items()
              if "original_profit" in v and "optimized_profit" in v}
    if not models:
        print("   No profit data available to plot.")
        return

    names = list(models.keys())
    orig_vals = [v["original_profit"] for v in models.values()]
    opt_vals  = [v["optimized_profit"] for v in models.values()]
    x, w = np.arange(len(names)), 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - w/2, orig_vals, w, label="Original Layout",
                   color="#e74c3c", edgecolor="white", linewidth=1.2)
    bars2 = ax.bar(x + w/2, opt_vals,  w, label="Optimized Layout",
                   color="#2ecc71", edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Monthly Profit (€)", fontsize=13)
    ax.set_title("Total Predicted Profit by Model Strategy", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, orig_vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"€{val:.0f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, opt_vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"€{val:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "profit_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f" Profit comparison saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Grouped Vertical Bar: items per shelf (Original vs Best Model)
# ---------------------------------------------------------------------------

def visualize_shelf_comparison():
    """Grouped vertical bar chart: products per shelf — Original vs Best Model."""
    layouts = _load_layouts()
    if "Original" not in layouts or len(layouts) < 2:
        print("   Not enough rack layouts. Run 02_train_models.py first.")
        return

    orig = layouts["Original"]
    best_name, best_df = _pick_best_ml(layouts)
    category = orig["Category"].iloc[0] if "Category" in orig.columns else ""
    shelves = list(range(1, NUM_SHELVES + 1))
    labels = ["1\n(Bottom)", "2\n(Bottom)", "3\n(Eye)", "4\n(Eye)",
              "5\n(Eye)", "6\n(Top)", "7\n(Top)"]

    orig_counts = [len(orig[orig["shelf_level"] == s]) for s in shelves]
    best_counts = [len(best_df[best_df["shelf_level"] == s]) for s in shelves]
    x, w = np.arange(len(shelves)), 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - w/2, orig_counts, w, label="Original",
                   color="#e74c3c", edgecolor="white", linewidth=1.2)
    bars2 = ax.bar(x + w/2, best_counts, w, label=best_name,
                   color="#2ecc71", edgecolor="white", linewidth=1.2)
    ax.set_xlabel("Shelf Level", fontsize=13)
    ax.set_ylabel("Number of Products", fontsize=13)
    ax.set_title(f"Products per Shelf — Original vs {best_name}\n{category}",
                 fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, orig_counts, strict=True):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar, val in zip(bars2, best_counts, strict=True):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = RESULTS_DIR / "shelf_comparison_bars.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f" Shelf comparison bars saved to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Alluvial / Flow diagram (product shelf movements)
# ---------------------------------------------------------------------------

def _build_shelf_positions(shelf_counts: dict) -> dict:
    """Compute normalised y-start/y-end for each shelf bar."""
    positions = {}
    y_cursor = 0.0
    for shelf in range(1, NUM_SHELVES + 1):
        count = shelf_counts.get(shelf, 0)
        height = max(count * 0.3, 0.15)
        positions[shelf] = {"y_start": y_cursor, "y_end": y_cursor + height, "count": count}
        y_cursor += height + 0.15
    max_y = max(p["y_end"] for p in positions.values())
    for pos in positions.values():
        pos["y_start"] /= max_y / NUM_SHELVES
        pos["y_end"]   /= max_y / NUM_SHELVES
    return positions


def _draw_shelf_bars(ax, side_x: float, positions: dict, title: str, title_color: str, bar_w: float):
    ax.text(side_x + bar_w / 2, NUM_SHELVES + 0.3, title,
            ha="center", va="bottom", fontsize=15, fontweight="bold", color=title_color)
    for shelf in range(1, NUM_SHELVES + 1):
        p = positions[shelf]
        color = SHELF_CMAP.get(shelf, "#999")
        ax.add_patch(plt.Rectangle(
            (side_x, p["y_start"]), bar_w, p["y_end"] - p["y_start"],
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85,
        ))
        label_x = side_x - 0.02 if side_x == 0.0 else side_x + bar_w + 0.02
        ax.text(label_x, (p["y_start"] + p["y_end"]) / 2,
                f"{SHELF_NAMES[shelf]}\n({p['count']} items)",
                ha="right" if side_x == 0.0 else "left",
                va="center", fontsize=8, fontweight="bold", color=color)


def visualize_alluvial():
    """Alluvial diagram: product movements between original and optimised shelves."""
    layouts = _load_layouts()
    if "Original" not in layouts or len(layouts) < 2:
        print("   Not enough rack layouts. Run 02_train_models.py first.")
        return

    orig = layouts["Original"]
    best_name, best_df = _pick_best_ml(layouts)
    category = orig["Category"].iloc[0] if "Category" in orig.columns else ""

    orig_sorted = orig.sort_values("name").reset_index(drop=True)
    best_sorted = best_df.sort_values("name").reset_index(drop=True)
    products = [
        {
            "name": orig_sorted.iloc[i]["name"],
            "orig_shelf": int(orig_sorted.iloc[i]["shelf_level"]),
            "new_shelf": int(best_sorted.iloc[i]["shelf_level"]),
        }
        for i in range(len(orig_sorted))
    ]

    orig_counts = defaultdict(int)
    new_counts  = defaultdict(int)
    for p in products:
        orig_counts[p["orig_shelf"]] += 1
        new_counts[p["new_shelf"]]   += 1

    left_pos  = _build_shelf_positions(dict(orig_counts))
    right_pos = _build_shelf_positions(dict(new_counts))

    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.5, NUM_SHELVES + 0.5)
    ax.set_axis_off()

    bar_w = 0.08
    _draw_shelf_bars(ax, 0.0, left_pos,  "ORIGINAL",  "#c0392b", bar_w)
    _draw_shelf_bars(ax, 1.0, right_pos, best_name,   "#27ae60", bar_w)

    left_cursors  = {s: left_pos[s]["y_start"]  for s in range(1, NUM_SHELVES + 1)}
    right_cursors = {s: right_pos[s]["y_start"] for s in range(1, NUM_SHELVES + 1)}

    products_sorted = sorted(products, key=lambda p: (p["orig_shelf"] != p["new_shelf"], p["orig_shelf"], p["new_shelf"]))

    def short(n, mx=22):
        return n[:mx] + "…" if len(n) > mx else n

    for p in products_sorted:
        os_, ns_ = p["orig_shelf"], p["new_shelf"]
        left_h  = (left_pos[os_]["y_end"]  - left_pos[os_]["y_start"])  / max(orig_counts[os_], 1)
        right_h = (right_pos[ns_]["y_end"] - right_pos[ns_]["y_start"]) / max(new_counts[ns_], 1)

        ylb, ylt = left_cursors[os_],  left_cursors[os_]  + left_h
        yrb, yrt = right_cursors[ns_], right_cursors[ns_] + right_h
        left_cursors[os_]  = ylt
        right_cursors[ns_] = yrt

        color = SHELF_CMAP.get(os_, "#999")
        cx = 0.5
        verts = [
            (bar_w, ylb), (cx, ylb), (cx, yrb), (1.0, yrb),
            (1.0, yrt),   (cx, yrt), (cx, ylt), (bar_w, ylt), (bar_w, ylb),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        ax.add_patch(mpatches.PathPatch(
            MplPath(verts, codes), facecolor=color,
            alpha=0.5 if os_ != ns_ else 0.25,
            edgecolor=color, linewidth=0.3,
        ))
        ax.text(bar_w + 0.015, (ylb + ylt) / 2, short(p["name"]),
                ha="left", va="center", fontsize=5.5, color="#222",
                fontweight="bold", clip_on=True)

    orig_profit = compute_rack_profit(orig)
    best_profit = compute_rack_profit(layouts[best_name])
    lift = best_profit - orig_profit
    pct = (lift / orig_profit * 100) if orig_profit > 0 else 0
    fig.suptitle(f"Product Shelf Movements — {category}\nOriginal → {best_name}",
                 fontsize=18, fontweight="bold", y=0.97)
    ax.text(0.5, -0.3,
            f"Original: €{orig_profit:,.0f}  →  {best_name}: €{best_profit:,.0f}  "
            f"(Profit Lift: {lift:+,.0f} / {pct:+.1f}%)",
            ha="center", va="center", fontsize=13, fontweight="bold",
            transform=ax.transData,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#e8f5e9", "alpha": 0.9})

    path = RESULTS_DIR / "alluvial_diagram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f" Alluvial diagram saved to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df, results = load_data()

    if not results:
        print("No training_results.json found. Running evaluation with fresh data...")
        rack_df = df[df["rack_id"] == df.groupby("rack_id").size().idxmax()].head(40).copy()
        results = {"Greedy": {
            "original_profit": compute_rack_profit(rack_df),
            "optimized_profit": compute_rack_profit(optimize_rack_greedy(rack_df)),
        }}

    print_comparison_table(results)
    visualize_rack(df)
    plot_mse_comparison(results)
    plot_profit_comparison(results)
    visualize_shelf_comparison()
    visualize_alluvial()

    print("\nEvaluation complete! Check results/ for plots.")


if __name__ == "__main__":
    main()

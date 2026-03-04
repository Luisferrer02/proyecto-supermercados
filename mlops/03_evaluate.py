#!/usr/bin/env python3
"""
03_evaluate.py — Evaluation & Visualization
=============================================
Generates:
  1. Comparison table (MSE + Total Predicted Profit per model)
  2. Matplotlib plot: "Optimized Rack" vs "Original Rack"

Usage:
    python 03_evaluate.py
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.retail_physics import (
    compute_rack_profit,
    optimize_rack_greedy,
    get_shelf_multiplier,
    NUM_SHELVES,
    SHELF_WIDTH_CM,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MONTHLY_DIR = BASE_DIR / "data" / "monthly"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "training_results.json"


def load_data():
    """Load monthly CSVs and training results."""
    csv_files = sorted(MONTHLY_DIR.glob("sales_*.csv"))
    if not csv_files:
        print(f"ERROR: No sales_*.csv files found in {MONTHLY_DIR}")
        print("       Run 01_generate_monthly_sales.py first.")
        sys.exit(1)

    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} product-month records from {len(csv_files)} files")

    results = {}
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            results = json.load(f)
    return df, results


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
        mse_val = r.get("mse", None)
        mse_str = f"{mse_val:.4f}" if mse_val is not None else "  N/A"
        orig = r.get("original_profit", 0)
        opt = r.get("optimized_profit", 0)
        lift = opt - orig
        print(f"{name:<18} {mse_str:>12} {orig:>18.2f} {opt:>18.2f} {lift:>+12.2f}")

    print("=" * 75)

    # Find best model
    best_profit_model = max(results.items(), key=lambda x: x[1].get("optimized_profit", 0))
    best_mse_models = {k: v for k, v in results.items() if "mse" in v}
    if best_mse_models:
        best_mse_model = min(best_mse_models.items(), key=lambda x: x[1]["mse"])
        print(f"\n🏆 Lowest MSE    : {best_mse_model[0]} ({best_mse_model[1]['mse']:.4f})")
    print(f"🏆 Highest Profit: {best_profit_model[0]} (€{best_profit_model[1].get('optimized_profit', 0):.2f})")


# ---------------------------------------------------------------------------
# 2. Rack Visualization
# ---------------------------------------------------------------------------

def visualize_rack(df: pd.DataFrame, results: dict, rack_id: int = None):
    """
    Create a side-by-side bar chart: Original vs Optimized rack profit
    broken down by shelf level.
    """
    if rack_id is None:
        # Use the rack with most products
        rack_counts = df.groupby("rack_id").size()
        rack_id = rack_counts.idxmax()

    rack_df = df[df["rack_id"] == rack_id].copy()
    if len(rack_df) > 40:
        rack_df = rack_df.head(40)

    category_name = rack_df["Category"].iloc[0] if "Category" in rack_df.columns else f"Rack {rack_id}"

    # Original profit per shelf
    orig_profit_by_shelf = []
    for shelf in range(1, NUM_SHELVES + 1):
        shelf_df = rack_df[rack_df["shelf_level"] == shelf]
        if len(shelf_df) == 0:
            orig_profit_by_shelf.append(0)
        else:
            orig_profit_by_shelf.append(compute_rack_profit(shelf_df))

    # Optimized rack (greedy)
    opt_df = optimize_rack_greedy(rack_df)
    opt_profit_by_shelf = []
    for shelf in range(1, NUM_SHELVES + 1):
        shelf_df = opt_df[opt_df["shelf_level"] == shelf]
        if len(shelf_df) == 0:
            opt_profit_by_shelf.append(0)
        else:
            opt_profit_by_shelf.append(compute_rack_profit(shelf_df))

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(f"Shelf Profit Distribution — Category: {category_name}",
                 fontsize=16, fontweight="bold", y=0.98)

    shelves = list(range(1, NUM_SHELVES + 1))
    shelf_labels = [
        "1 (Bottom)", "2 (Bottom)", "3 (Eye ★)", "4 (Eye ★)", "5 (Eye ★)",
        "6 (Top)", "7 (Top)"
    ]

    # Colors: eye level = gold, bottom = blue, top = teal
    colors_orig = ["#4a90d9"] * 2 + ["#f5a623"] * 3 + ["#50c8c8"] * 2
    colors_opt = ["#3a7bc8"] * 2 + ["#e8961e"] * 3 + ["#40b0b0"] * 2

    # -- Original rack --
    ax1 = axes[0]
    bars1 = ax1.barh(shelves, orig_profit_by_shelf, color=colors_orig,
                     edgecolor="white", linewidth=1.2, height=0.7)
    ax1.set_yticks(shelves)
    ax1.set_yticklabels(shelf_labels, fontsize=11)
    ax1.set_xlabel("Monthly Profit (€)", fontsize=12)
    ax1.set_title("[Original] Layout", fontsize=14, pad=12, color="#c0392b", fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars1, orig_profit_by_shelf):
        if val > 0:
            ax1.text(bar.get_width() + max(orig_profit_by_shelf) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"€{val:.0f}", va="center", fontsize=9)

    # -- Optimized rack --
    ax2 = axes[1]
    bars2 = ax2.barh(shelves, opt_profit_by_shelf, color=colors_opt,
                     edgecolor="white", linewidth=1.2, height=0.7)
    ax2.set_yticks(shelves)
    ax2.set_yticklabels(shelf_labels, fontsize=11)
    ax2.set_xlabel("Monthly Profit (€)", fontsize=12)
    ax2.set_title("[Optimized] Layout", fontsize=14, pad=12, color="#27ae60", fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars2, opt_profit_by_shelf):
        if val > 0:
            ax2.text(bar.get_width() + max(opt_profit_by_shelf) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"€{val:.0f}", va="center", fontsize=9)

    # Totals
    orig_total = sum(orig_profit_by_shelf)
    opt_total = sum(opt_profit_by_shelf)
    lift_pct = ((opt_total - orig_total) / orig_total * 100) if orig_total else 0

    fig.text(0.5, 0.02,
             f"Original Total: €{orig_total:,.0f}  →  Optimized Total: €{opt_total:,.0f}  "
             f"(Profit Lift: {lift_pct:+.1f}%)",
             ha="center", fontsize=13, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    plot_path = RESULTS_DIR / "rack_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Rack visualization saved to {plot_path}")
    plt.close()

    return plot_path


# ---------------------------------------------------------------------------
# 3. Model MSE Bar Plot
# ---------------------------------------------------------------------------

def plot_mse_comparison(results: dict):
    """Bar chart comparing MSE across supervised models."""
    mse_models = {k: v["mse"] for k, v in results.items() if "mse" in v}
    if not mse_models:
        print("   No MSE data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(mse_models.keys())
    values = list(mse_models.values())
    colors = ["#4a90d9", "#50c878", "#f5a623"][:len(names)]

    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=13)
    ax.set_title("Model Prediction Accuracy Comparison", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "mse_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"📊 MSE comparison saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Profit Comparison Bar Plot
# ---------------------------------------------------------------------------

def plot_profit_comparison(results: dict):
    """Grouped bar chart: original vs optimized profit per model."""
    models_with_profit = {k: v for k, v in results.items()
                          if "original_profit" in v and "optimized_profit" in v}
    if not models_with_profit:
        print("   No profit data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(models_with_profit.keys())
    orig_vals = [v["original_profit"] for v in models_with_profit.values()]
    opt_vals = [v["optimized_profit"] for v in models_with_profit.values()]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, orig_vals, width, label="Original Layout",
                   color="#e74c3c", edgecolor="white", linewidth=1.2)
    bars2 = ax.bar(x + width / 2, opt_vals, width, label="Optimized Layout",
                   color="#2ecc71", edgecolor="white", linewidth=1.2)

    ax.set_ylabel("Monthly Profit (€)", fontsize=13)
    ax.set_title("Total Predicted Profit by Model Strategy", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars1, orig_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"€{val:.0f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, opt_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"€{val:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "profit_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"📊 Profit comparison saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Grouped Vertical Bar: items per shelf (Original vs Optimized)
# ---------------------------------------------------------------------------

def _load_layouts():
    """Load Original + best ML model layout CSVs."""
    layouts = {}
    for csv_file in RESULTS_DIR.glob("rack_layout_*.csv"):
        model_name = csv_file.stem.replace("rack_layout_", "").upper()
        if model_name == "ORIGINAL":
            model_name = "Original"
        layouts[model_name] = pd.read_csv(csv_file)
    return layouts


def _pick_best_ml(layouts):
    """Return (name, df) of the best non-greedy ML layout."""
    best_name, best_profit = None, -float("inf")
    for name, layout in layouts.items():
        if name in ["Original", "GREEDY"]:
            continue
        p = compute_rack_profit(layout)
        if p > best_profit:
            best_profit = p
            best_name = name
    if best_name is None:
        best_name = "GREEDY"
    return best_name, layouts[best_name]


def visualize_shelf_comparison(df: pd.DataFrame):
    """Grouped vertical bar chart: products per shelf — Original vs Best Model."""
    layouts = _load_layouts()
    if "Original" not in layouts or len(layouts) < 2:
        print("   ⏭ Not enough rack layouts. Run 02_train_models.py first.")
        return

    orig = layouts["Original"]
    best_name, best_df = _pick_best_ml(layouts)
    category = orig["Category"].iloc[0] if "Category" in orig.columns else ""

    shelves = list(range(1, NUM_SHELVES + 1))
    labels = ["1\n(Bottom)", "2\n(Bottom)", "3\n(Eye ★)", "4\n(Eye ★)",
              "5\n(Eye ★)", "6\n(Top)", "7\n(Top)"]

    orig_counts = [len(orig[orig["shelf_level"] == s]) for s in shelves]
    best_counts = [len(best_df[best_df["shelf_level"] == s]) for s in shelves]

    x = np.arange(len(shelves))
    w = 0.35

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

    for bar, val in zip(bars1, orig_counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar, val in zip(bars2, best_counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = RESULTS_DIR / "shelf_comparison_bars.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"📊 Shelf comparison bars saved to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Alluvial / Flow diagram (product shelf movements)
# ---------------------------------------------------------------------------

def visualize_alluvial(df: pd.DataFrame):
    """
    Alluvial diagram: left = original shelves, right = optimized shelves.
    Each flow is a product moving between shelves. Flow width proportional
    to count of items making this transition. Product names inside flows.
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as MplPath
    import matplotlib.patches as mpatches

    layouts = _load_layouts()
    if "Original" not in layouts or len(layouts) < 2:
        print("   ⏭ Not enough rack layouts. Run 02_train_models.py first.")
        return

    orig = layouts["Original"]
    best_name, best_df = _pick_best_ml(layouts)
    category = orig["Category"].iloc[0] if "Category" in orig.columns else ""

    # Merge original and optimized shelves per product
    # Use index alignment — both DFs should have the same products in order
    orig_sorted = orig.sort_values("name").reset_index(drop=True)
    best_sorted = best_df.sort_values("name").reset_index(drop=True)

    products = []
    for i in range(len(orig_sorted)):
        products.append({
            "name": orig_sorted.iloc[i]["name"],
            "orig_shelf": int(orig_sorted.iloc[i]["shelf_level"]),
            "new_shelf": int(best_sorted.iloc[i]["shelf_level"]),
            "width_cm": orig_sorted.iloc[i]["product_width_cm"],
        })

    # Group by (orig_shelf → new_shelf) transitions
    from collections import defaultdict
    transitions = defaultdict(list)
    for p in products:
        key = (p["orig_shelf"], p["new_shelf"])
        transitions[key].append(p["name"])

    # Shelf colors
    shelf_cmap = {
        1: "#3498db", 2: "#2980b9",
        3: "#f39c12", 4: "#e67e22", 5: "#d35400",
        6: "#1abc9c", 7: "#16a085",
    }
    shelf_labels = {
        1: "Shelf 1 (Bottom)", 2: "Shelf 2 (Bottom)",
        3: "Shelf 3 (Eye ★)", 4: "Shelf 4 (Eye ★)", 5: "Shelf 5 (Eye ★)",
        6: "Shelf 6 (Top)", 7: "Shelf 7 (Top)",
    }

    # --- Layout geometry ---
    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.5, NUM_SHELVES + 0.5)
    ax.set_axis_off()

    left_x = 0.0     # original column
    right_x = 1.0    # optimized column
    bar_w = 0.08     # width of shelf bars

    # Compute stacking positions for left and right
    # Each shelf has a vertical extent based on how many products it holds
    total_products = len(products)
    shelf_height_unit = NUM_SHELVES / (total_products + NUM_SHELVES)  # spacing

    def build_shelf_positions(shelf_counts):
        """Compute y-start/y-end for each shelf bar based on product count."""
        positions = {}
        y_cursor = 0
        for shelf in range(1, NUM_SHELVES + 1):
            count = shelf_counts.get(shelf, 0)
            height = max(count * 0.3, 0.15)  # min height for empty shelves
            positions[shelf] = {"y_start": y_cursor, "y_end": y_cursor + height, "count": count}
            y_cursor += height + 0.15  # gap between shelves
        return positions

    orig_counts = defaultdict(int)
    new_counts = defaultdict(int)
    for p in products:
        orig_counts[p["orig_shelf"]] += 1
        new_counts[p["new_shelf"]] += 1

    left_pos = build_shelf_positions(dict(orig_counts))
    right_pos = build_shelf_positions(dict(new_counts))

    # Scale to fit in y range
    max_y = max(
        max(p["y_end"] for p in left_pos.values()),
        max(p["y_end"] for p in right_pos.values()),
    )
    for pos in [left_pos, right_pos]:
        for shelf in pos:
            pos[shelf]["y_start"] = pos[shelf]["y_start"] / max_y * NUM_SHELVES
            pos[shelf]["y_end"] = pos[shelf]["y_end"] / max_y * NUM_SHELVES

    # Draw shelf bars
    for side_x, positions, title, title_color in [
        (left_x, left_pos, "ORIGINAL", "#c0392b"),
        (right_x, right_pos, f"{best_name}", "#27ae60"),
    ]:
        # Title
        ax.text(side_x + bar_w/2, NUM_SHELVES + 0.3, title,
                ha="center", va="bottom", fontsize=15, fontweight="bold", color=title_color)

        for shelf in range(1, NUM_SHELVES + 1):
            p = positions[shelf]
            color = shelf_cmap.get(shelf, "#999")
            rect = plt.Rectangle(
                (side_x, p["y_start"]), bar_w, p["y_end"] - p["y_start"],
                facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85,
            )
            ax.add_patch(rect)

            # Shelf label
            label_x = side_x - 0.02 if side_x == left_x else side_x + bar_w + 0.02
            ha = "right" if side_x == left_x else "left"
            mid_y = (p["y_start"] + p["y_end"]) / 2
            ax.text(label_x, mid_y,
                    f"{shelf_labels[shelf]}\n({p['count']} items)",
                    ha=ha, va="center", fontsize=8, fontweight="bold",
                    color=color)

    # Draw one flow per product (individual bezier curves)
    left_cursors = {s: left_pos[s]["y_start"] for s in range(1, NUM_SHELVES + 1)}
    right_cursors = {s: right_pos[s]["y_start"] for s in range(1, NUM_SHELVES + 1)}

    # Sort: same-shelf stays first, then by origin shelf
    products_sorted = sorted(products, key=lambda p: (p["orig_shelf"] != p["new_shelf"], p["orig_shelf"], p["new_shelf"]))

    def short(n, mx=22):
        return n[:mx] + "…" if len(n) > mx else n

    for p in products_sorted:
        orig_shelf = p["orig_shelf"]
        new_shelf = p["new_shelf"]

        # Each product gets 1 unit of height in its shelf
        left_total_h = left_pos[orig_shelf]["y_end"] - left_pos[orig_shelf]["y_start"]
        right_total_h = right_pos[new_shelf]["y_end"] - right_pos[new_shelf]["y_start"]

        left_h = left_total_h / max(orig_counts[orig_shelf], 1)
        right_h = right_total_h / max(new_counts[new_shelf], 1)

        y_left_bottom = left_cursors[orig_shelf]
        y_left_top = y_left_bottom + left_h
        y_right_bottom = right_cursors[new_shelf]
        y_right_top = y_right_bottom + right_h

        left_cursors[orig_shelf] = y_left_top
        right_cursors[new_shelf] = y_right_top

        # Color by origin shelf; highlight moves
        color = shelf_cmap.get(orig_shelf, "#999")
        alpha = 0.5 if orig_shelf != new_shelf else 0.25

        # Bezier path
        cx = 0.5
        verts = [
            (left_x + bar_w, y_left_bottom),
            (cx, y_left_bottom),
            (cx, y_right_bottom),
            (right_x, y_right_bottom),
            (right_x, y_right_top),
            (cx, y_right_top),
            (cx, y_left_top),
            (left_x + bar_w, y_left_top),
            (left_x + bar_w, y_left_bottom),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        path = MplPath(verts, codes)
        patch = mpatches.PathPatch(path, facecolor=color, alpha=alpha,
                                   edgecolor=color, linewidth=0.3)
        ax.add_patch(patch)

        # Product name at the start of the flow
        label_x = left_x + bar_w + 0.015
        label_y = (y_left_bottom + y_left_top) / 2
        ax.text(label_x, label_y, short(p["name"]),
                ha="left", va="center",
                fontsize=5.5, color="#222",
                fontweight="bold",
                clip_on=True)

    # Title
    fig.suptitle(f"Product Shelf Movements — {category}\nOriginal → {best_name}",
                 fontsize=18, fontweight="bold", y=0.97)

    # Profit annotation
    orig_profit = compute_rack_profit(orig)
    best_profit = compute_rack_profit(layouts[best_name])
    lift = best_profit - orig_profit
    pct = (lift / orig_profit * 100) if orig_profit > 0 else 0
    ax.text(0.5, -0.3,
            f"Original: €{orig_profit:,.0f}  →  {best_name}: €{best_profit:,.0f}  "
            f"(Profit Lift: {lift:+,.0f} / {pct:+.1f}%)",
            ha="center", va="center", fontsize=13, fontweight="bold",
            transform=ax.transData,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", alpha=0.9))

    path = RESULTS_DIR / "alluvial_diagram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"📊 Alluvial diagram saved to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df, results = load_data()

    if not results:
        print("No training_results.json found. Running evaluation with fresh data...")
        from utils.retail_physics import optimize_rack_greedy, compute_rack_profit
        rack_counts = df.groupby("rack_id").size()
        target_rack = rack_counts.idxmax()
        rack_df = df[df["rack_id"] == target_rack].head(40).copy()
        orig_profit = compute_rack_profit(rack_df)
        opt_df = optimize_rack_greedy(rack_df)
        opt_profit = compute_rack_profit(opt_df)
        results = {
            "Greedy": {
                "original_profit": orig_profit,
                "optimized_profit": opt_profit,
            }
        }

    # 1. Print comparison table
    print_comparison_table(results)

    # 2. Rack visualization (original vs optimized profit by shelf)
    visualize_rack(df, results)

    # 3. MSE bar plot
    plot_mse_comparison(results)

    # 4. Profit comparison
    plot_profit_comparison(results)

    # 5. Shelf comparison bars (grouped vertical)
    visualize_shelf_comparison(df)

    # 6. Alluvial diagram (product movements)
    visualize_alluvial(df)

    print("\nEvaluation complete! Check results/ for plots.")


if __name__ == "__main__":
    main()


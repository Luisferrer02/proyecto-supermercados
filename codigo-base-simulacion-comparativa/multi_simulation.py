"""
Supermarket Digital Twin - Multi-Run Simulation Experiment
==========================================================
Runs the sales simulation 10 times with different random seeds,
identifies top-5 products per run, then analyzes common factors
among the top performers across all runs.

Same formula as simulation.py:
  sales = BASE_DEMAND × popularity × shelf_factor × adjacency_factor × noise
"""

import numpy as np
import pandas as pd
from collections import Counter

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_SIMULATIONS = 10
NUM_DAYS = 30
BASE_DEMAND = 20
TOP_N = 5
BASE_SEED = 100  # seeds will be 100, 101, ..., 109

SHELF_FACTOR = {
    1: 0.50,  # bottom shelf
    2: 0.75,  # below eye level
    3: 1.00,  # eye level (best)
    4: 0.65,  # top shelf
}

# ── Product Catalog (same as simulation.py) ────────────────────────────────────
catalog = [
    (1,  "Whole Milk 1L",        "dairy",     1.20, 0.15, 0.90, "S1", 3),
    (2,  "Greek Yogurt",         "dairy",     2.50, 0.30, 0.70, "S1", 2),
    (3,  "Cheddar Cheese",       "dairy",     3.80, 0.25, 0.60, "S1", 1),
    (4,  "White Bread",          "bakery",    1.00, 0.20, 0.85, "S2", 3),
    (5,  "Croissants x4",       "bakery",    2.20, 0.35, 0.55, "S2", 2),
    (6,  "Baguette",             "bakery",    1.50, 0.25, 0.50, "S2", 4),
    (7,  "Coca-Cola 2L",         "beverages", 1.80, 0.20, 0.80, "S3", 3),
    (8,  "Orange Juice 1L",      "beverages", 2.50, 0.22, 0.65, "S3", 2),
    (9,  "Sparkling Water 1.5L", "beverages", 0.90, 0.18, 0.55, "S3", 1),
    (10, "Spaghetti 500g",       "pasta",     1.10, 0.28, 0.75, "S4", 3),
    (11, "Penne 500g",           "pasta",     1.10, 0.28, 0.60, "S4", 2),
    (12, "Tomato Sauce 400g",    "pasta",     1.60, 0.32, 0.70, "S4", 1),
    (13, "Olive Oil 500ml",      "condiments",4.50, 0.20, 0.50, "S4", 4),
    (14, "Chicken Breast 500g",  "meat",      5.00, 0.18, 0.72, "S5", 3),
    (15, "Ground Beef 400g",     "meat",      4.20, 0.16, 0.68, "S5", 2),
    (16, "Salmon Fillet 300g",   "meat",      6.50, 0.22, 0.45, "S5", 1),
    (17, "Bananas 1kg",          "produce",   1.30, 0.25, 0.88, "S6", 3),
    (18, "Apples 1kg",           "produce",   2.00, 0.22, 0.75, "S6", 2),
    (19, "Tomatoes 500g",        "produce",   1.80, 0.20, 0.65, "S6", 4),
    (20, "Potato Chips 150g",    "snacks",    2.80, 0.40, 0.70, "S7", 3),
]

columns = [
    "product_id", "name", "category", "price", "margin",
    "popularity", "shelf_id", "rack_level",
]
products_df = pd.DataFrame(catalog, columns=columns)


# ── Adjacency Factors (computed once — layout is fixed) ────────────────────────
def compute_adjacency_factors(df):
    """Returns {product_id: factor}. +5% per same-category shelf neighbour, max +15%."""
    factors = {}
    for _, row in df.iterrows():
        neighbours = df[
            (df["shelf_id"] == row["shelf_id"]) &
            (df["category"] == row["category"]) &
            (df["product_id"] != row["product_id"])
        ]
        bonus = min(len(neighbours) * 0.05, 0.15)
        factors[row["product_id"]] = 1.0 + bonus
    return factors

adjacency = compute_adjacency_factors(products_df)


# ── Single Simulation Run ──────────────────────────────────────────────────────
def run_simulation(seed):
    """Simulate 30 days of sales. Returns a DataFrame aggregated per product."""
    rng = np.random.default_rng(seed)  # isolated RNG per run
    records = []

    for day in range(1, NUM_DAYS + 1):
        for _, prod in products_df.iterrows():
            pid = prod["product_id"]
            shelf_f = SHELF_FACTOR[prod["rack_level"]]
            adj_f = adjacency[pid]
            noise = rng.uniform(0.85, 1.15)

            raw = BASE_DEMAND * prod["popularity"] * shelf_f * adj_f * noise
            units = max(1, int(round(raw)))

            records.append({
                "product_id": pid,
                "name":       prod["name"],
                "category":   prod["category"],
                "price":      prod["price"],
                "margin":     prod["margin"],
                "popularity": prod["popularity"],
                "shelf_id":   prod["shelf_id"],
                "rack_level": prod["rack_level"],
                "units_sold": units,
                "revenue":    round(units * prod["price"], 2),
                "profit":     round(units * prod["price"] * prod["margin"], 2),
            })

    daily_df = pd.DataFrame(records)

    # Aggregate per product for this simulation
    agg = daily_df.groupby(
        ["product_id", "name", "category", "price", "margin",
         "popularity", "shelf_id", "rack_level"]
    ).agg(
        total_units=("units_sold", "sum"),
        total_revenue=("revenue", "sum"),
        total_profit=("profit", "sum"),
    ).reset_index()

    return agg.sort_values("total_profit", ascending=False)


# ── Run All Simulations ───────────────────────────────────────────────────────
all_top = []

print("=" * 70)
print("  SUPERMARKET DIGITAL TWIN – MULTI-RUN EXPERIMENT (10 SIMULATIONS)")
print("=" * 70)

for i in range(NUM_SIMULATIONS):
    seed = BASE_SEED + i
    result = run_simulation(seed)
    top = result.head(TOP_N).copy()
    top["sim_id"] = i + 1
    top["seed"] = seed
    all_top.append(top)

    # Per-simulation summary
    print(f"\n  Simulation {i+1:>2}  (seed={seed})")
    print(f"  {'Product':<25} {'Category':<12} {'Shelf':>5} {'Rack':>5} {'Profit':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*5} {'-'*5} {'-'*10}")
    for _, r in top.iterrows():
        print(f"  {r['name']:<25} {r['category']:<12} {r['shelf_id']:>5} "
              f"{r['rack_level']:>5} ${r['total_profit']:>8,.2f}")

top_df = pd.concat(all_top, ignore_index=True)

# ── Export combined top-products data ──────────────────────────────────────────
top_df.to_csv("multi_simulation_top_products.csv", index=False)

# ── Cross-Simulation Analysis ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  CROSS-SIMULATION ANALYSIS – COMMON FACTORS AMONG TOP PRODUCTS")
print("=" * 70)

# A) Frequency counts
print("\n  A) APPEARANCE FREQUENCY (how many times a product reached top 5)")
print("  " + "-" * 55)
name_counts = Counter(top_df["name"])
for name, count in name_counts.most_common():
    bar = "#" * count
    print(f"    {name:<25} {count:>2}/10  {bar}")

print(f"\n  Category frequency:")
cat_counts = Counter(top_df["category"])
for cat, count in cat_counts.most_common():
    print(f"    {cat:<15} {count:>2} appearances")

print(f"\n  Shelf frequency:")
shelf_counts = Counter(top_df["shelf_id"])
for shelf, count in shelf_counts.most_common():
    print(f"    {shelf:<6} {count:>2} appearances")

print(f"\n  Rack level frequency:")
rack_counts = Counter(top_df["rack_level"])
for rack, count in sorted(rack_counts.items()):
    label = {1: "bottom", 2: "below eye", 3: "eye level", 4: "top"}[rack]
    print(f"    Level {rack} ({label:<10}) {count:>2} appearances")

# B) Average attribute values
print(f"\n  B) AVERAGE ATTRIBUTES OF TOP PERFORMERS")
print("  " + "-" * 55)
avg_price = top_df["price"].mean()
avg_margin = top_df["margin"].mean()
avg_popularity = top_df["popularity"].mean()
avg_profit = top_df["total_profit"].mean()
print(f"    Average price:       ${avg_price:.2f}")
print(f"    Average margin:       {avg_margin:.1%}")
print(f"    Average popularity:   {avg_popularity:.2f}")
print(f"    Average profit/run:  ${avg_profit:,.2f}")

# C) Pattern identification
print(f"\n  C) IDENTIFIED PATTERNS")
print("  " + "-" * 55)

most_common_cat = cat_counts.most_common(1)[0]
most_common_rack = rack_counts.most_common(1)[0]
most_stable = name_counts.most_common(1)[0]

# Products that appeared in ALL 10 simulations
stable_products = [name for name, c in name_counts.items() if c == NUM_SIMULATIONS]
# Rack level 3 dominance
rack3_pct = rack_counts.get(3, 0) / len(top_df) * 100

print(f"    Most common category:    {most_common_cat[0]} "
      f"({most_common_cat[1]} appearances)")
print(f"    Most common rack level:  {most_common_rack[0]} "
      f"({most_common_rack[1]} appearances)")
print(f"    Eye-level (rack 3) share: {rack3_pct:.0f}% of all top entries")
print(f"    Price range of top prods: ${top_df['price'].min():.2f} – "
      f"${top_df['price'].max():.2f}")
print(f"    Margin range:             {top_df['margin'].min():.0%} – "
      f"{top_df['margin'].max():.0%}")

if stable_products:
    print(f"\n    Products in top 5 in ALL {NUM_SIMULATIONS} simulations:")
    for name in stable_products:
        row = products_df[products_df["name"] == name].iloc[0]
        print(f"      - {name:<25} (pop={row['popularity']:.2f}, "
              f"margin={row['margin']:.0%}, rack={row['rack_level']})")
else:
    print(f"    Most stable product: {most_stable[0]} ({most_stable[1]}/{NUM_SIMULATIONS} runs)")

print("\n" + "=" * 70)
print(f"  Exported {len(top_df)} top-product entries to "
      f"multi_simulation_top_products.csv")
print("=" * 70)

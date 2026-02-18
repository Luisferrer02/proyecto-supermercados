"""
Supermarket Digital Twin - Sales Simulation
============================================
Simulates daily sales for a small product catalog over 30 days.
Sales depend on product popularity, shelf position, category adjacency, and random noise.
All data is synthetic. Uses a fixed seed for reproducibility.
"""

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
NUM_DAYS = 30
BASE_DEMAND = 20  # max units/day for a perfect-scoring product

np.random.seed(RANDOM_SEED)

# Shelf visibility factor by rack level (1=bottom, 4=top)
# Level 3 (eye-level) is the prime spot in retail.
SHELF_FACTOR = {
    1: 0.50,  # bottom – hard to see, low traffic
    2: 0.75,  # below eye level – decent
    3: 1.00,  # eye level – best visibility
    4: 0.65,  # top shelf – out of easy reach
}

# ── Product Catalog ────────────────────────────────────────────────────────────
catalog = [
    # (product_id, name, category, price, margin, popularity, shelf_id, rack_level)
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

# ── Adjacency / Category Bonus ────────────────────────────────────────────────
# Products on the same shelf benefit from category coherence.
# We give a small bonus when >=2 products of the same category share a shelf.
def compute_adjacency_factors(df):
    """Returns a dict {product_id: adjacency_factor}."""
    factors = {}
    for _, row in df.iterrows():
        same_shelf_same_cat = df[
            (df["shelf_id"] == row["shelf_id"]) &
            (df["category"] == row["category"]) &
            (df["product_id"] != row["product_id"])
        ]
        # +5% per adjacent same-category neighbour, capped at +15%
        bonus = min(len(same_shelf_same_cat) * 0.05, 0.15)
        factors[row["product_id"]] = 1.0 + bonus
    return factors

adjacency = compute_adjacency_factors(products_df)

# ── Daily Sales Simulation ─────────────────────────────────────────────────────
records = []

for day in range(1, NUM_DAYS + 1):
    for _, prod in products_df.iterrows():
        pid = prod["product_id"]
        shelf_f = SHELF_FACTOR[prod["rack_level"]]
        adj_f = adjacency[pid]
        noise = np.random.uniform(0.85, 1.15)  # ±15% daily variation

        raw_sales = BASE_DEMAND * prod["popularity"] * shelf_f * adj_f * noise
        units_sold = max(1, int(round(raw_sales)))  # at least 1 unit

        revenue = round(units_sold * prod["price"], 2)
        profit = round(units_sold * prod["price"] * prod["margin"], 2)

        records.append({
            "day": day,
            "product_id": pid,
            "name": prod["name"],
            "category": prod["category"],
            "shelf_id": prod["shelf_id"],
            "rack_level": prod["rack_level"],
            "units_sold": units_sold,
            "revenue": revenue,
            "profit": profit,
        })

sales_df = pd.DataFrame(records)

# ── Export ─────────────────────────────────────────────────────────────────────
sales_df.to_csv("sales_simulation.csv", index=False)
sales_df.to_json("sales_simulation.json", orient="records", indent=2)

# ── Summary ────────────────────────────────────────────────────────────────────
total_revenue = sales_df["revenue"].sum()
total_profit = sales_df["profit"].sum()

profit_by_product = (
    sales_df.groupby(["product_id", "name"])["profit"]
    .sum()
    .sort_values(ascending=False)
)
top5 = profit_by_product.head(5)

print("=" * 60)
print("  SUPERMARKET DIGITAL TWIN – 30-DAY SIMULATION SUMMARY")
print("=" * 60)
print(f"  Total Revenue:  ${total_revenue:>10,.2f}")
print(f"  Total Profit:   ${total_profit:>10,.2f}")
print(f"  Profit Margin:  {total_profit / total_revenue * 100:>9.1f}%")
print("-" * 60)
print("  TOP 5 PRODUCTS BY PROFIT")
print("-" * 60)
for (pid, name), value in top5.items():
    print(f"    #{pid:<3} {name:<25} ${value:>8,.2f}")
print("=" * 60)
print(f"\nExported {len(sales_df)} rows to sales_simulation.csv and sales_simulation.json")

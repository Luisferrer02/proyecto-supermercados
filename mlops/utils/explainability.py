"""
Explainability for shelf-placement decisions
============================================
Produces rule-based, human-readable reasons for every product that changed
shelf during optimization. Not a SHAP/LIME surrogate — designed to be
understood by a supermarket manager.

For each moved product, we surface at most 3 reasons chosen among:

  - margin_premium      product's margin is above rack median
  - sales_volume        product's sales are above rack median
  - eye_level_promotion moved into shelves 3–5 (prime location)
  - relegated_low_return moved out of shelves 3–5 (low margin×sales)
  - crowding_relief     old shelf was overcrowded
  - space_fit           width of the product suits the new shelf

Each reason has a label, an explanation template, and produces a record
like:
    {
      "product": "Aceite de Oliva",
      "old_shelf": 2,
      "new_shelf": 4,
      "expected_lift_eur": 12.4,
      "reasons": [
        {"code": "margin_premium",
         "text": "Margen del 40 % — por encima de la media de la estantería (28 %)"},
        {"code": "eye_level_promotion",
         "text": "Promovido a balda a la altura de los ojos (nivel 4)"}
      ]
    }

The output is deterministic and depends only on the DataFrames, so it is
safe to re-run.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


EYE_LEVEL = {3, 4, 5}
CROWDING_THRESHOLD = 6  # products on a single shelf


def _product_profit_score(row: pd.Series) -> float:
    """Quick profit potential for a product (proxy, not real €)."""
    price = float(row.get("price_numeric", 0) or 0)
    margin = float(row.get("profit_margin_percentage", 0) or 0) / 100.0
    sales = float(row.get("estimated_monthly_sales", 0) or 0)
    return price * margin * sales


def explain_rack(original_rack: pd.DataFrame,
                 optimized_rack: pd.DataFrame) -> List[Dict]:
    """Return explanations for every product that changed shelf in a rack.

    Both DataFrames must contain the same products (identified by `name`)
    and the columns: name, shelf_level, price_numeric, profit_margin_percentage,
    estimated_monthly_sales, product_width_cm.
    """
    if "name" not in original_rack.columns or "name" not in optimized_rack.columns:
        return []

    original_by_name = original_rack.set_index("name")
    optimized_by_name = optimized_rack.set_index("name")

    # Reference stats computed on the rack (not on the full catalogue) —
    # reasoning is always local so the manager gets context-specific
    # explanations ("above THIS rack's median", not global).
    margin_series = pd.to_numeric(
        optimized_rack["profit_margin_percentage"], errors="coerce"
    )
    sales_series = pd.to_numeric(
        optimized_rack["estimated_monthly_sales"], errors="coerce"
    )
    margin_median = float(margin_series.median(skipna=True) or 0)
    sales_median = float(sales_series.median(skipna=True) or 0)

    # Count products per shelf in ORIGINAL layout for crowding check
    orig_shelf_counts = (
        pd.to_numeric(original_rack["shelf_level"], errors="coerce")
        .value_counts()
        .to_dict()
    )

    explanations: List[Dict] = []
    common_names = original_by_name.index.intersection(optimized_by_name.index)
    for name in common_names:
        orig_row = original_by_name.loc[name]
        new_row = optimized_by_name.loc[name]

        orig_shelf = int(float(orig_row["shelf_level"]))
        new_shelf = int(float(new_row["shelf_level"]))
        if orig_shelf == new_shelf:
            continue  # didn't move

        margin = float(new_row.get("profit_margin_percentage", 0) or 0)
        sales = float(new_row.get("estimated_monthly_sales", 0) or 0)
        score = _product_profit_score(new_row)
        old_crowd = int(orig_shelf_counts.get(orig_shelf, 0))

        reasons: List[Dict[str, str]] = []

        # Eye-level promotion / relegation
        if new_shelf in EYE_LEVEL and orig_shelf not in EYE_LEVEL:
            reasons.append({
                "code": "eye_level_promotion",
                "text": f"Promovido a balda a la altura de los ojos "
                        f"(nivel {new_shelf}) porque es un producto con "
                        f"buen retorno (margen {margin:.0f} %, "
                        f"{int(sales)} ventas/mes).",
            })
        elif orig_shelf in EYE_LEVEL and new_shelf not in EYE_LEVEL:
            reasons.append({
                "code": "relegated_low_return",
                "text": f"Liberó espacio en balda a la altura de los ojos "
                        f"(dejó el nivel {orig_shelf}) para otros productos "
                        f"con mejor ratio margen × ventas.",
            })
        else:
            # Within-tier move
            if new_shelf < orig_shelf and new_shelf in EYE_LEVEL:
                reasons.append({
                    "code": "eye_level_promotion",
                    "text": f"Reubicado al nivel {new_shelf} (zona premium) "
                            f"para aprovechar mejor su visibilidad.",
                })

        # Margin premium
        if margin > margin_median * 1.15 and margin > 15:
            reasons.append({
                "code": "margin_premium",
                "text": f"Margen del {margin:.0f} % — por encima de la media "
                        f"de la estantería ({margin_median:.0f} %).",
            })

        # Sales volume
        if sales > sales_median * 1.25 and sales > 20:
            reasons.append({
                "code": "sales_volume",
                "text": f"Volumen de ventas elevado "
                        f"({int(sales)} unidades/mes, "
                        f"frente a {int(sales_median)} de media).",
            })

        # Crowding relief
        if old_crowd > CROWDING_THRESHOLD and \
                int(orig_shelf_counts.get(new_shelf, 0)) <= CROWDING_THRESHOLD:
            reasons.append({
                "code": "crowding_relief",
                "text": f"La balda original tenía {old_crowd} productos "
                        f"apretados; se ha movido a una balda con más "
                        f"espacio para no perder visibilidad.",
            })

        if not reasons:
            reasons.append({
                "code": "reoptimization",
                "text": "Reubicación óptima según el modelo "
                        "(combinación de margen, ventas y espacio).",
            })

        # Keep at most three reasons, ordered as they were appended so the
        # most decisive (shelf-tier change) shows first.
        explanations.append({
            "product": str(name),
            "old_shelf": orig_shelf,
            "new_shelf": new_shelf,
            "margin_pct": round(margin, 1),
            "monthly_sales": int(sales),
            "profit_score": round(score, 2),
            "reasons": reasons[:3],
        })

    # Sort by biggest profit potential first (managers care most about these)
    explanations.sort(key=lambda e: e["profit_score"], reverse=True)
    return explanations


def explain_all(original_df: pd.DataFrame,
                optimized_df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Explain every rack in the prediction output.

    Returns a dict keyed by rack_id (as string) whose values are lists of
    per-product explanations. Racks whose products did not move at all are
    omitted.
    """
    out: Dict[str, List[Dict]] = {}
    if "rack_id" not in original_df.columns or "rack_id" not in optimized_df.columns:
        return out
    for rack_id in optimized_df["rack_id"].unique():
        orig_rack = original_df[original_df["rack_id"] == rack_id]
        opt_rack = optimized_df[optimized_df["rack_id"] == rack_id]
        if orig_rack.empty or opt_rack.empty:
            continue
        rack_expl = explain_rack(orig_rack, opt_rack)
        if rack_expl:
            out[str(rack_id)] = rack_expl
    return out

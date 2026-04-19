"""
CSV Schema Validation
=====================
Validates that monthly sales CSVs (sales_YYYY_MM_monthname.csv) match the
expected schema before ingestion. Produces user-friendly error reports that
explain exactly which columns are missing, which have wrong types, and which
values are out of range.

Used by 04_ingest.py (pre-flight check) and the web /upload route.

The validator never mutates or writes — it only reports. It is safe to call
from multiple threads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "Category":                   "string",
    "name":                       "string",
    "price_numeric":              "numeric",
    "profit_margin_percentage":   "numeric",
    "estimated_monthly_sales":    "numeric",
    "product_width_cm":           "numeric",
    "rack_id":                    "numeric",
    "shelf_level":                "numeric",
}

OPTIONAL_COLUMNS = {
    "subtitle":                   "string",
    "price":                      "string",
    "discount_price":             "string",
    "discount_price_numeric":     "numeric",
}

VALUE_RANGES = {
    "price_numeric":              (0.01, 1000.0),
    "profit_margin_percentage":   (0.0, 100.0),
    "estimated_monthly_sales":    (0.0, 100000.0),
    "product_width_cm":           (1.0, 100.0),
    "shelf_level":                (1, 7),
    "rack_id":                    (0, 10000),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of validating one CSV file."""
    path: str
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_count: int = 0

    def format(self) -> str:
        lines = [f"[{'OK' if self.ok else 'FAIL'}] {self.path}"]
        lines.append(f"  rows: {self.row_count}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_csv(path: str | Path) -> ValidationResult:
    """Validate a single monthly sales CSV file.

    Returns a ValidationResult whose `ok` field is True if the file can be
    safely ingested. Does not raise; callers decide how to react.
    """
    p = Path(path)
    result = ValidationResult(path=str(p), ok=False)

    if not p.exists():
        result.errors.append(f"File does not exist")
        return result

    # Filename pattern check (soft — warning only)
    if not p.name.startswith("sales_") or not p.name.endswith(".csv"):
        result.warnings.append(
            f"Filename '{p.name}' does not match pattern 'sales_*.csv'"
        )

    # Read the CSV
    try:
        df = pd.read_csv(p)
    except Exception as exc:
        result.errors.append(f"Cannot parse CSV: {exc}")
        return result

    result.row_count = len(df)
    if len(df) == 0:
        result.errors.append("CSV has no data rows")
        return result

    # Required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        result.errors.append(
            f"Missing required columns: {', '.join(missing)}"
        )

    # Type checks for numeric columns present
    for col, kind in {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}.items():
        if col not in df.columns:
            continue
        if kind == "numeric":
            coerced = pd.to_numeric(df[col], errors="coerce")
            n_bad = coerced.isna().sum() - df[col].isna().sum()
            if n_bad > 0:
                result.errors.append(
                    f"Column '{col}' has {n_bad} non-numeric values"
                )

    # Value range checks (warnings, not errors — clamping handles them)
    for col, (lo, hi) in VALUE_RANGES.items():
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        out_of_range = ((coerced < lo) | (coerced > hi)).sum()
        if out_of_range > 0:
            result.warnings.append(
                f"Column '{col}' has {out_of_range} values outside [{lo}, {hi}]"
            )

    # Null checks on critical columns
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                result.warnings.append(
                    f"Column '{col}' has {n_null} null values"
                )

    result.ok = len(result.errors) == 0
    return result


def validate_directory(csv_dir: str | Path) -> List[ValidationResult]:
    """Validate every sales_*.csv in a directory."""
    d = Path(csv_dir)
    if not d.exists():
        return [ValidationResult(path=str(d), ok=False,
                                 errors=["Directory does not exist"])]
    results = []
    for p in sorted(d.glob("sales_*.csv")):
        results.append(validate_csv(p))
    if not results:
        results.append(ValidationResult(
            path=str(d), ok=False,
            errors=["No files matching 'sales_*.csv' found in directory"],
        ))
    return results


def format_report(results: List[ValidationResult]) -> str:
    lines = []
    n_ok = sum(1 for r in results if r.ok)
    lines.append(f"=== CSV Schema Validation Report ===")
    lines.append(f"Total files: {len(results)}  |  OK: {n_ok}  "
                 f"|  Failed: {len(results) - n_ok}")
    lines.append("")
    for r in results:
        lines.append(r.format())
        lines.append("")
    return "\n".join(lines)


def all_ok(results: List[ValidationResult]) -> bool:
    return all(r.ok for r in results) and len(results) > 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate monthly sales CSVs against the expected schema"
    )
    parser.add_argument("path", help="CSV file or directory to validate")
    args = parser.parse_args(argv)

    target = Path(args.path)
    if target.is_dir():
        results = validate_directory(target)
    else:
        results = [validate_csv(target)]

    print(format_report(results))
    return 0 if all_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(_main())

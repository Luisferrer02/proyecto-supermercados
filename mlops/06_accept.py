#!/usr/bin/env python3
"""
06_accept.py — Accept an optimized month into the knowledge base
=================================================================
Copies the optimized CSV into data/monthly/ and ingests it into ChromaDB
so future predictions can use it as context.

Usage:
    python 06_accept.py --month 2026-01
"""

import argparse
import sys
from pathlib import Path

from utils.knowledge_base import ShelfKnowledgeBase

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MONTHLY_DIR = BASE_DIR / "data" / "monthly"

MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}


def main():
    parser = argparse.ArgumentParser(description="Accept optimized month into knowledge base")
    parser.add_argument("--month", type=str, required=True, help="Month to accept (YYYY-MM)")
    args = parser.parse_args()

    try:
        year_s, month_s = args.month.split("-")
        year, month = int(year_s), int(month_s)
        if not (1 <= month <= 12):
            raise ValueError
    except (ValueError, IndexError):
        print("ERROR: Invalid month format. Use YYYY-MM")
        sys.exit(1)

    month_name = MONTH_NAMES[month]
    optimized_csv = RESULTS_DIR / f"optimized_{year}_{month:02d}_{month_name}.csv"

    if not optimized_csv.exists():
        print(f"ERROR: No optimized CSV found at {optimized_csv}")
        print("       Run 05_predict.py first.")
        sys.exit(1)

    # Copy to monthly dir with standard naming
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    target = MONTHLY_DIR / f"sales_{year}_{month:02d}_{month_name}.csv"
    target.write_bytes(optimized_csv.read_bytes())
    print(f"Saved → {target}")

    # Ingest into RAG
    kb = ShelfKnowledgeBase()
    n_chunks = kb.ingest_csv(target)
    print(f"Ingested into RAG: {n_chunks} category summaries")
    print(f"\n✓ {year}-{month:02d} accepted. Future predictions will use it as context.")


if __name__ == "__main__":
    main()

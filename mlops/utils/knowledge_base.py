"""
Knowledge Base — ChromaDB Vector Store for Monthly Sales
=========================================================
Embeds monthly sales CSVs as per-category summaries and provides
retrieval of relevant months for prediction (last 2 months + same
month last year).

Uses sentence-transformers for embedding and ChromaDB for storage.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"

COLLECTION_NAME = "monthly_sales"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_month_from_filename(filename: str) -> Optional[Tuple[int, int, str]]:
    """
    Extract (year, month, month_name) from filenames like:
      sales_2025_07_july.csv → (2025, 7, 'july')
    """
    match = re.match(r"sales_(\d{4})_(\d{2})_(\w+)\.csv", filename)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3)
    return None


def _summarize_category(cat_df: pd.DataFrame, category: str,
                        year: int, month: int) -> str:
    """
    Create a natural-language summary of a category's sales for embedding.
    This is what gets stored in the vector DB and retrieved by the LLM.
    """
    n_products = len(cat_df)
    avg_sales = cat_df["estimated_monthly_sales"].mean()
    total_sales = cat_df["estimated_monthly_sales"].sum()
    avg_price = cat_df["price_numeric"].mean()
    avg_margin = cat_df["profit_margin_percentage"].mean()
    avg_width = cat_df["product_width_cm"].mean()

    top_sellers = cat_df.nlargest(5, "estimated_monthly_sales")
    top_list = ", ".join(
        f"{r['name']} ({int(r['estimated_monthly_sales'])} units)"
        for _, r in top_sellers.iterrows()
    )

    # Shelf distribution
    shelf_dist = cat_df["shelf_level"].value_counts().sort_index()
    shelf_str = ", ".join(f"S{s}: {c}" for s, c in shelf_dist.items())

    return (
        f"Category: {category} | Month: {month:02d}/{year} | "
        f"Products: {n_products} | "
        f"Total sales: {total_sales:.0f} units | "
        f"Avg sales/product: {avg_sales:.0f} | "
        f"Avg price: €{avg_price:.2f} | "
        f"Avg margin: {avg_margin:.1f}% | "
        f"Avg width: {avg_width:.1f}cm | "
        f"Shelf distribution: [{shelf_str}] | "
        f"Top sellers: {top_list}"
    )


# ---------------------------------------------------------------------------
# Knowledge Base class
# ---------------------------------------------------------------------------

class ShelfKnowledgeBase:
    """ChromaDB-backed knowledge base for monthly sales data."""

    def __init__(self, persist_dir: Optional[str] = None):
        import chromadb

        self.persist_dir = persist_dir or str(KB_DIR)
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = None

    @property
    def embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
        return self._embedder

    def ingest_csv(self, csv_path: str | Path) -> int:
        """
        Ingest a monthly sales CSV into the knowledge base.
        Returns the number of chunks (categories) embedded.
        """
        csv_path = Path(csv_path)
        meta = parse_month_from_filename(csv_path.name)
        if meta is None:
            raise ValueError(
                f"Cannot parse month from filename: {csv_path.name}. "
                f"Expected format: sales_YYYY_MM_monthname.csv"
            )
        year, month, month_name = meta

        df = pd.read_csv(csv_path)
        required_cols = [
            "Category", "name", "estimated_monthly_sales",
            "price_numeric", "profit_margin_percentage",
            "product_width_cm", "shelf_level",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

        # Remove existing data for this month (allow re-ingestion)
        existing = self.collection.get(
            where={"month_key": f"{year}_{month:02d}"}
        )
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        # Chunk by category
        documents = []
        metadatas = []
        ids = []

        for category, cat_df in df.groupby("Category"):
            summary = _summarize_category(cat_df, category, year, month)
            doc_id = f"{year}_{month:02d}_{category}"

            # Store raw data as metadata for later retrieval
            cat_stats = {
                "year": year,
                "month": month,
                "month_name": month_name,
                "month_key": f"{year}_{month:02d}",
                "category": str(category),
                "n_products": len(cat_df),
                "total_sales": float(cat_df["estimated_monthly_sales"].sum()),
                "avg_sales": float(cat_df["estimated_monthly_sales"].mean()),
                "avg_price": float(cat_df["price_numeric"].mean()),
                "avg_margin": float(cat_df["profit_margin_percentage"].mean()),
                "source_file": csv_path.name,
            }

            documents.append(summary)
            metadatas.append(cat_stats)
            ids.append(doc_id)

        # Generate embeddings and store
        embeddings = self.embedder.encode(documents).tolist()
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return len(documents)

    def ingest_directory(self, dir_path: str | Path,
                         callback=None) -> Dict[str, int]:
        """
        Ingest all sales_*.csv files from a directory.
        Returns {filename: n_chunks} dict.
        """
        dir_path = Path(dir_path)
        results = {}
        csv_files = sorted(dir_path.glob("sales_*.csv"))

        for i, csv_file in enumerate(csv_files):
            n_chunks = self.ingest_csv(csv_file)
            results[csv_file.name] = n_chunks
            if callback:
                callback(i + 1, len(csv_files), csv_file.name, n_chunks)

        return results

    def retrieve_context(self, target_year: int, target_month: int,
                         category: Optional[str] = None,
                         n_recent: int = 2) -> Dict[str, pd.DataFrame]:
        """
        Retrieve relevant months for prediction:
          - Last n_recent months (trend context)
          - Same month last year (seasonal context)

        Returns dict of {month_key: summary_text_list}.
        """
        # Compute which months we want
        wanted_keys = set()

        # Last n_recent months
        y, m = target_year, target_month
        for _ in range(n_recent):
            m -= 1
            if m < 1:
                m = 12
                y -= 1
            wanted_keys.add(f"{y}_{m:02d}")

        # Same month last year
        wanted_keys.add(f"{target_year - 1}_{target_month:02d}")

        # Query for each wanted month
        context = {}
        for month_key in sorted(wanted_keys):
            where_filter = {"month_key": month_key}
            if category:
                where_filter = {
                    "$and": [
                        {"month_key": month_key},
                        {"category": category},
                    ]
                }

            results = self.collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )

            if results["documents"]:
                context[month_key] = {
                    "documents": results["documents"],
                    "metadatas": results["metadatas"],
                }

        return context

    def get_latest_month_data(self, csv_dir: str | Path,
                              category: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load the most recent month's raw CSV data (for use as the base
        for prediction). Returns the DataFrame or None.
        """
        csv_dir = Path(csv_dir)
        csv_files = sorted(csv_dir.glob("sales_*.csv"))
        if not csv_files:
            return None

        latest = csv_files[-1]
        df = pd.read_csv(latest)
        if category:
            df = df[df["Category"] == category].copy()
        return df

    def get_all_months_data(self, csv_dir: str | Path) -> List[pd.DataFrame]:
        """Load all monthly CSVs and return as list of DataFrames."""
        csv_dir = Path(csv_dir)
        csv_files = sorted(csv_dir.glob("sales_*.csv"))
        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            meta = parse_month_from_filename(f.name)
            if meta:
                df["_year"] = meta[0]
                df["_month"] = meta[1]
            dfs.append(df)
        return dfs

    def stats(self) -> Dict:
        """Return knowledge base statistics."""
        count = self.collection.count()
        all_data = self.collection.get(include=["metadatas"])
        months = set()
        categories = set()
        for m in all_data["metadatas"]:
            months.add(m.get("month_key", "?"))
            categories.add(m.get("category", "?"))

        return {
            "total_chunks": count,
            "months": sorted(months),
            "n_months": len(months),
            "n_categories": len(categories),
        }

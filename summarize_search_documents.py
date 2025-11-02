"""Generate a CSV report summarizing claims text and vector data.

This script reads `multiquery_summary.csv`, fetches the corresponding documents
from Elasticsearch for both the source patent (`syutugan`) and the evaluated
patent (`ax_docs`), and writes a compact report highlighting the claims text
and the first 30 characters of the vectorized data for each. The intent is to
make it easy for teammates to spot missing/garbled text without re-running
manual queries.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from dotenv import dotenv_values


def load_env() -> None:
    """Load environment variables from project `.env` files."""
    root = Path(__file__).parent

    env: Dict[str, Optional[str]] = {}
    for env_path in (root / ".env", root / "elastic-search-test/.env"):
        if env_path.exists():
            env.update({
                k: v for k, v in dotenv_values(env_path).items()
                if v is not None
            })

    os.environ.update(env)  # type: ignore[arg-type]


def get_indexer():
    """Lazy import helper so sys.path edits happen after env load."""
    root = Path(__file__).parent / "elastic-search-test"
    sys.path.append(str(root))
    from data_sync.elasticsearch_indexer import \
        ElasticsearchIndexer  # type: ignore

    return ElasticsearchIndexer()


def preview(text: Optional[str], length: int = 400) -> str:
    if not text:
        return ""
    snippet = text.strip().replace("\n", " ")
    return snippet[:length]


def get_vector_preview(vector_data: Optional[list], length: int = 30) -> str:
    """Get a preview of vector data as a string."""
    if not vector_data or not isinstance(vector_data, list):
        return ""

    # Convert vector to string and take first 30 characters
    vector_str = str(vector_data)
    return vector_str[:length]


def main(
    input_csv: Path = Path("multiquery_summary.csv"),
    output_csv: Path = Path("multiquery_claims_report.csv"),
    preview_length: int = 400,
    vector_preview_length: int = 30,
) -> None:
    load_env()
    indexer = get_indexer()

    fieldnames = [
        "case_id",
        "contains_ax_docs",
        "syutugan_claims_preview",
        "syutugan_vector_preview",
        "ax_claims_preview",
        "ax_vector_preview",
    ]

    with input_csv.open("r", encoding="utf-8") as infile, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            syutugan_id = row.get("syutugan", "").strip()
            ax_id = row.get("ax_docs", "").strip()

            syutugan_doc = indexer.get_document_by_id(syutugan_id)
            ax_doc = indexer.get_document_by_id(ax_id)

            writer.writerow(
                {
                    "case_id": row.get("case_id", ""),
                    "contains_ax_docs": row.get("contains_ax_docs", ""),
                    "syutugan_claims_preview": preview(
                        syutugan_doc.get("claims_text")
                        if syutugan_doc else "",
                        preview_length,
                    ),
                    "syutugan_vector_preview": get_vector_preview(
                        syutugan_doc.get("claims_vector")
                        if syutugan_doc else None,
                        vector_preview_length,
                    ),
                    "ax_claims_preview": preview(
                        ax_doc.get("claims_text") if ax_doc else "",
                        preview_length,
                    ),
                    "ax_vector_preview": get_vector_preview(
                        ax_doc.get("claims_vector") if ax_doc else None,
                        vector_preview_length,
                    ),
                }
            )

    indexer.close()


if __name__ == "__main__":
    main()


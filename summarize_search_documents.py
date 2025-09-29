"""Generate a CSV report summarizing source/ax patent texts for multiquery results.

This script reads `multiquery_summary.csv`, fetches the corresponding documents
from Elasticsearch for both the source patent (`syutugan`) and the evaluated
patent (`ax_docs`), and writes a compact report highlighting the summary length
and leading snippet for each. The intent is to make it easy for teammates to
spot missing/garbled text without re-running manual queries.
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
            env.update({k: v for k, v in dotenv_values(env_path).items() if v is not None})

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


def main(
    input_csv: Path = Path("multiquery_summary.csv"),
    output_csv: Path = Path("multiquery_text_report.csv"),
    preview_length: int = 400,
) -> None:
    load_env()
    indexer = get_indexer()

    fieldnames = [
        "case_id",
        "contains_ax_docs",
        "syutugan_summary_preview",
        "ax_summary_preview",
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
                    "syutugan_summary_preview": preview(
                        syutugan_doc.get("summary") if syutugan_doc else "",
                        preview_length,
                    ),
                    "ax_summary_preview": preview(
                        ax_doc.get("summary") if ax_doc else "",
                        preview_length,
                    ),
                }
            )

    indexer.close()


if __name__ == "__main__":
    main()


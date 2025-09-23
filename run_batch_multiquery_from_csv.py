#!/usr/bin/env python3
"""
Batch runner: For each row in a CSV (syutugan, ax_docs), run the multi-query
vector search pipeline and summarize results to a CSV.

Input CSV columns: case_id, pattern, syutugan, ax_docs, ay_docs

Output CSV columns:
  - case_id, pattern, syutugan, ax_docs
  - unique_results: int (deduplicated result count)
  - contains_ax_docs: bool
  - match_index: 0-based index in merged results (or -1 if absent)
  - max_score: float (if matched)
  - times_hit: int (if matched)
  - best_rank: int (if matched)
  - matched_title: str (if matched)
  - per_query_hits: semicolon-separated counts per generated query
  - error: error message if failed

Usage:
  python run_batch_multiquery_from_csv.py \
    --input syutugan_ax_exist_only.csv \
    --output multiquery_summary.csv \
    [--k 100 --num-queries 10 --min-score 0.0 --exclude-source]

Requires environment for embeddings and generation:
  - Azure embeddings: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
  - Generation: either AZURE_OPENAI_CHAT_DEPLOYMENT or OPENAI_API_KEY (and optional OPENAI_CHAT_MODEL)
"""
import argparse
import asyncio
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load .env files (root then project-specific)
load_dotenv(override=False)
proj_env = Path("elastic-search-test/.env")
if proj_env.exists():
    load_dotenv(proj_env, override=True)

# Import pipeline function
import sys as _sys
_sys.path.append(str(Path(__file__).parent / "elastic-search-test"))
from run_multiquery_vector_search import multiquery_vector_search  # type: ignore


def normalize_id(x: str) -> str:
    return re.sub(r"\D", "", x or "")


async def process_row(row: Dict[str, str], k: int, num_queries: int, min_score: float, exclude_source: bool) -> Dict[str, Any]:
    case_id = row.get("case_id", "")
    pattern = row.get("pattern", "")
    syutugan = normalize_id(row.get("syutugan", ""))
    ax_docs = normalize_id(row.get("ax_docs", ""))

    result: Dict[str, Any] = {
        "case_id": case_id,
        "pattern": pattern,
        "syutugan": syutugan,
        "ax_docs": ax_docs,
        "unique_results": 0,
        "contains_ax_docs": False,
        "match_index": -1,
        "max_score": "",
        "times_hit": "",
        "best_rank": "",
        "matched_title": "",
        "per_query_hits": "",
        "error": "",
    }

    try:
        res = await multiquery_vector_search(
            patent_id_raw=syutugan,
            k=k,
            num_queries=num_queries,
            min_score=min_score,
            exclude_source=exclude_source,
        )

        per_query_hits = res.get("per_query_hits", [])
        merged = res.get("results", [])
        ids = [str(r.get("patent_id")) for r in merged]

        result["unique_results"] = int(res.get("unique_results", len(merged)))
        result["per_query_hits"] = ";".join(str(x) for x in per_query_hits)

        if ax_docs in ids:
            idx = ids.index(ax_docs)
            match = merged[idx]
            result["contains_ax_docs"] = True
            result["match_index"] = idx
            result["max_score"] = match.get("max_score", "")
            result["times_hit"] = match.get("times_hit", "")
            result["best_rank"] = match.get("best_rank", "")
            # Try to pull title
            doc = match.get("document", {})
            result["matched_title"] = doc.get("title", "")

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch multi-query vector search from CSV")
    parser.add_argument("--input", default="syutugan_ax_exist_only.csv", help="Input CSV path")
    parser.add_argument("--output", default="multiquery_summary.csv", help="Output CSV path")
    parser.add_argument("--k", type=int, default=100, help="Top-K per query")
    parser.add_argument("--num-queries", type=int, default=10, help="Generated passages per row")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum similarity score")
    parser.add_argument("--exclude-source", action="store_true", help="Exclude syutugan from results")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    rows: List[Dict[str, str]] = []
    with in_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Process sequentially to respect rate limits
    results: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] case_id={row.get('case_id')} syutugan={row.get('syutugan')} ax_docs={row.get('ax_docs')}")
        res = asyncio.run(
            process_row(
                row,
                k=args.k,
                num_queries=args.num_queries,
                min_score=args.min_score,
                exclude_source=args.exclude_source,
            )
        )
        results.append(res)

    # Write output CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "pattern",
        "syutugan",
        "ax_docs",
        "unique_results",
        "contains_ax_docs",
        "match_index",
        "max_score",
        "times_hit",
        "best_rank",
        "matched_title",
        "per_query_hits",
        "error",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Filter rows from syutugan_ax_test_data_converted.csv to only those whose
syutugan and ax_docs exist in Elasticsearch index `patent_vectors`.

Usage:
  python filter_syutugan_ax_in_es.py \
      --input syutugan_ax_test_data_converted.csv \
      --output syutugan_ax_exist_only.csv \
      [--mode both|any]

Default mode is `both` (syutugan AND ax_docs must exist). Mode `any` keeps
rows where either exists.
"""
import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Set

from dotenv import load_dotenv

# Load env from root and elastic-search-test if present
load_dotenv(override=False)
es_env = Path("elastic-search-test/.env")
if es_env.exists():
    # Prefer per-project settings to override root .env values
    load_dotenv(es_env, override=True)

# Reuse project indexer for efficient mget
import sys as _sys
proj_dir = Path(__file__).parent / "elastic-search-test"
_sys.path.append(str(proj_dir))
from data_sync.elasticsearch_indexer import ElasticsearchIndexer  # type: ignore


def normalize_id(s: str) -> str:
    """Keep only digits in patent id (e.g., 'JP2015163142A' -> '2015163142')."""
    return re.sub(r"\D", "", s or "")


def main():
    parser = argparse.ArgumentParser(description="Filter CSV rows by ES existence")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--mode",
        choices=["both", "any"],
        default="both",
        help="Keep rows where both or any of (syutugan, ax_docs) exist",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Read CSV
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Collect unique IDs to check (syutugan + ax_docs)
    ids: Set[str] = set()
    for r in rows:
        sid = normalize_id(r.get("syutugan", ""))
        aid = normalize_id(r.get("ax_docs", ""))
        if sid:
            ids.add(sid)
        if aid:
            ids.add(aid)

    # Query ES existence in bulk
    indexer = ElasticsearchIndexer()
    existence: Dict[str, bool] = indexer.bulk_check_existence(list(ids))

    # Filter rows by mode
    kept: List[dict] = []
    kept_count_both = 0
    kept_count_any = 0

    for r in rows:
        sid = normalize_id(r.get("syutugan", ""))
        aid = normalize_id(r.get("ax_docs", ""))
        s_exists = existence.get(sid, False)
        a_exists = existence.get(aid, False)

        if args.mode == "both":
            keep = s_exists and a_exists
        else:
            keep = s_exists or a_exists

        if keep:
            kept.append(r)
            if s_exists and a_exists:
                kept_count_both += 1
            if s_exists or a_exists:
                kept_count_any += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(kept)

    print(f"Input rows: {len(rows)}")
    print(f"Unique IDs checked: {len(ids)}")
    print(f"Kept rows ({args.mode}): {len(kept)}")
    print(f"  of kept, BOTH existed: {kept_count_both}")
    print(f"  of kept, ANY existed:  {kept_count_any}")


if __name__ == "__main__":
    main()

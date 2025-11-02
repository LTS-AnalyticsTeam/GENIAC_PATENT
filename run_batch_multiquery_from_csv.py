#!/usr/bin/env python3
"""
Batch runner: For each row in a CSV (syutugan, ax_docs), run the multi-query
vector search pipeline and summarize results to a CSV.

Input CSV columns: case_id, pattern, syutugan, ax_docs, ay_docs

Output CSV columns:
  - case_id, pattern, syutugan, ax_docs
  - unique_results: deduplicated result count
  - total_hits: 上位集合として取得したユニーク件数
  - contains_ax_docs: bool
  - match_index: 0-based index in merged results (or -1 if absent)
  - hit_rank: 1-based rank in the merged result list where ax_docs appears
  - hit_score: cosine score at the first appearance of ax_docs
  - max_score: maximum cosine score observed for ax_docs
  - summary_score: 最大 summary ベクトル由来スコア
  - claims_score: 最大 claims ベクトル由来スコア
  - times_hit: number of query pairs that retrieved ax_docs
  - best_rank: best (lowest) rank recorded across query pairs
  - matched_title: str (if matched)
  - per_query_hits_pair: semicolon-separated "summary/claims/combined" counts per generated query pair
  - top10_hits: 上位10件の doc_id とスコア (rank:doc_id:score)
  - match_cosine_max: 最大コサイン類似度（ax_docs がヒットした場合）
  - match_cosine_pairs: pair_index:cosine をセミコロン区切りで列挙
  - error: error message if failed

Usage:
  python run_batch_multiquery_from_csv.py \
    --input syutugan_ax_exist_only.csv \
    --output multiquery_summary.csv \
    [--k 1000 --num-queries 10 --min-score 0.0 --exclude-source]

Requires environment for embeddings and generation:
  - Azure embeddings: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
  - Generation: either AZURE_OPENAI_CHAT_DEPLOYMENT or OPENAI_API_KEY (and optional OPENAI_CHAT_MODEL)
"""
import argparse
import asyncio
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

from dotenv import load_dotenv

MAX_K = 1000  # Elasticsearch負荷対策のための上限

# Load .env files (root then project-specific)
load_dotenv(override=False)
proj_env = Path("elastic-search-test/.env")
if proj_env.exists():
    load_dotenv(proj_env, override=True)

# Import pipeline function
import sys as _sys

_sys.path.append(str(Path(__file__).parent / "elastic-search-test"))
from data_sync.elasticsearch_indexer import \
    ElasticsearchIndexer  # type: ignore
from run_multiquery_vector_search import \
    multiquery_vector_search  # type: ignore


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
        "total_hits": 0,
        "contains_ax_docs": False,
        "match_index": -1,
        "hit_rank": "",
        "hit_score": "",
        "max_score": "",
        "times_hit": "",
        "best_rank": "",
        "summary_score": "",
        "claims_score": "",
        "matched_title": "",
        "match_cosine_max": "",
        "match_cosine_pairs": "",
        "per_query_hits_pair": "",
        "top10_hits": "",
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

        merged = res.get("results", [])
        ids = [str(r.get("patent_id")) for r in merged]

        result["unique_results"] = int(res.get("unique_results", len(merged)))
        result["total_hits"] = len(merged)
        pair_hits = res.get("per_query_hits_pair", [])
        result["per_query_hits_pair"] = ";".join(str(x) for x in pair_hits)

        top_hits = []
        for rank_idx, entry in enumerate(merged[:10], start=1):
            pid = str(entry.get("patent_id", ""))
            try:
                score_val = float(entry.get("max_score", ""))
                score_str = f"{score_val:.6f}"
            except (TypeError, ValueError):
                score_str = ""
            top_hits.append(f"{rank_idx}:{pid}:{score_str}")
        result["top10_hits"] = ";".join(top_hits)

        if ax_docs in ids:
            idx = ids.index(ax_docs)
            match = merged[idx]
            result["contains_ax_docs"] = True
            result["match_index"] = idx
            result["hit_rank"] = idx + 1
            result["max_score"] = match.get("max_score", "")
            result["summary_score"] = match.get("summary_score", "")
            result["claims_score"] = match.get("claims_score", "")
            result["times_hit"] = match.get("times_hit", "")
            result["best_rank"] = match.get("best_rank", "")
            # Try to pull title
            doc = match.get("document", {})
            result["matched_title"] = doc.get("title", "")

            match_sources = match.get("match_sources", []) or []
            query_embeddings = res.get("query_embeddings", []) or []
            first_pair_index = match.get("first_pair_index")
            first_rank = match.get("first_rank")
            first_vector_field = match.get("first_vector_field")
            first_cosine = None
            first_score_raw = match.get("first_score")
            cosine_pairs: List[str] = []
            cosine_values: List[float] = []

            if match_sources and query_embeddings:
                es_indexer = None
                doc_vectors: Dict[str, List[float]] = {}
                vector_norms: Dict[str, float] = {}
                try:
                    es_indexer = ElasticsearchIndexer()
                    doc_full = es_indexer.get_document_by_id(ax_docs)
                    if doc_full:
                        for field_name in {
                            es_indexer.summary_vector_field,
                            es_indexer.claims1_vector_field,
                            es_indexer.vector_field,
                        }:
                            vector = doc_full.get(field_name)
                            if vector:
                                doc_vectors[field_name] = vector
                                vector_norms[field_name] = math.sqrt(
                                    sum(float(x) * float(x) for x in vector)
                                )
                finally:
                    if es_indexer:
                        es_indexer.close()

                if doc_vectors:
                    for source in match_sources:
                        pair_idx = source.get("pair_index")
                        if pair_idx is None or not (0 <= pair_idx < len(query_embeddings)):
                            continue
                        field_name = source.get("vector_field") or es_indexer.summary_vector_field if es_indexer else ""
                        doc_vector = doc_vectors.get(field_name) or doc_vectors.get(es_indexer.summary_vector_field if es_indexer else "")
                        if not doc_vector:
                            continue
                        doc_norm = vector_norms.get(field_name) or 0.0
                        if doc_norm == 0.0:
                            continue
                        query_vec = query_embeddings[pair_idx]
                        if not query_vec:
                            continue
                        query_norm = math.sqrt(sum(float(x) * float(x) for x in query_vec))
                        if query_norm == 0:
                            continue
                        dot_prod = sum(
                            float(a) * float(b)
                            for a, b in zip(query_vec, doc_vector)
                        )
                        cosine = dot_prod / (query_norm * doc_norm)
                        cosine_values.append(cosine)
                        cosine_pairs.append(f"{pair_idx}:{field_name}:{cosine:.6f}")
                        if (
                            first_cosine is None
                            and first_pair_index is not None
                            and pair_idx == first_pair_index
                        ):
                            rank_val = source.get("rank")
                            try:
                                rank_int = int(rank_val) if rank_val is not None else None
                            except (TypeError, ValueError):
                                rank_int = None
                            if (
                                (first_rank is None or rank_int == first_rank)
                                and (first_vector_field is None or field_name == first_vector_field)
                            ):
                                first_cosine = cosine

            if cosine_values:
                result["match_cosine_max"] = f"{max(cosine_values):.6f}"
                result["match_cosine_pairs"] = ";".join(cosine_pairs)
            if first_cosine is not None:
                result["hit_score"] = f"{first_cosine:.6f}"
            elif first_score_raw is not None:
                try:
                    result["hit_score"] = f"{float(first_score_raw):.6f}"
                except (TypeError, ValueError):
                    pass

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch multi-query vector search from CSV")
    parser.add_argument("--input", default="syutugan_ax_exist_only.csv", help="Input CSV path")
    parser.add_argument("--output", default="multiquery_summary.csv", help="Output CSV path")
    parser.add_argument("--k", type=int, default=1000, help="Top-K per query (上限1000に自動制限)")
    parser.add_argument("--num-queries", type=int, default=10, help="Generated passages per row")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum similarity score")
    parser.add_argument("--exclude-source", action="store_true", help="Exclude syutugan from results")
    parser.add_argument("--no-timestamp", action="store_true", help="Do not append timestamp suffix to output filename")
    args = parser.parse_args()

    in_path = Path(args.input)
    base_out_path = Path(args.output)
    if args.no_timestamp:
        out_path = base_out_path
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = base_out_path.with_name(f"{base_out_path.stem}_{timestamp}{base_out_path.suffix}")
    rows: List[Dict[str, str]] = []
    with in_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Process sequentially to respect rate limits
    effective_k = min(args.k, MAX_K)
    if effective_k < args.k:
        print(f"指定されたk={args.k}は上限{MAX_K}を超えています。k={effective_k}で実行します。")

    results: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] case_id={row.get('case_id')} syutugan={row.get('syutugan')} ax_docs={row.get('ax_docs')}")
        res = asyncio.run(
            process_row(
                row,
                k=effective_k,
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
        "total_hits",
        "contains_ax_docs",
        "match_index",
        "hit_rank",
        "hit_score",
        "max_score",
        "summary_score",
        "claims_score",
        "times_hit",
        "best_rank",
        "matched_title",
        "match_cosine_max",
        "match_cosine_pairs",
        "top10_hits",
        "per_query_hits_pair",
        "error",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved summary: {out_path}")

    hit_scores: List[float] = []
    for res in results:
        score_str = res.get("hit_score")
        if not score_str:
            continue
        try:
            hit_scores.append(float(score_str))
        except (TypeError, ValueError):
            continue

    if hit_scores:
        avg_score = statistics.mean(hit_scores)
        median_score = statistics.median(hit_scores)
        min_score = min(hit_scores)
        print(f"Hit score count={len(hit_scores)} avg={avg_score:.6f} median={median_score:.6f} min={min_score:.6f}")
        print("最小スコア閾値 (recall=100%): {:.6f}".format(min_score))
    else:
        print("No hit scores recorded; unable to compute statistics.")


if __name__ == "__main__":
    main()

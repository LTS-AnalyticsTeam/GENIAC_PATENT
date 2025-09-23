#!/usr/bin/env python3
"""
Multi-query vector search from a single patent document.

Flow:
- Fetch a patent by patent_id from Cosmos DB
- Generate 10 ~200-char Japanese passages from summary/claims (OpenAI if available; fallback otherwise)
- Embed the 10 passages with Azure OpenAI embeddings
- For each embedding, run KNN vector search on Elasticsearch (top-100)
- Merge and de-duplicate results across queries

Usage:
  python run_multiquery_vector_search.py --patent JP2011050607 --k 100 --num-queries 10 --exclude-source

Environment variables (from elastic-search-test/.env):
- COSMOS_* for Cosmos DB
- ELASTICSEARCH_HOST, ELASTICSEARCH_INDEX
- AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
- AZURE_OPENAI_DEPLOYMENT (embeddings)
- Optional for text generation: AZURE_OPENAI_CHAT_DEPLOYMENT
"""
import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Local imports
import sys as _sys
_sys.path.append(str(Path(__file__).parent))

from data_sync.cosmos_client import CosmosDBClient
from data_sync.elasticsearch_indexer import ElasticsearchIndexer
from data_sync.embedding_processor import EmbeddingProcessor

try:
    from openai import AzureOpenAI, OpenAI
except Exception:  # pragma: no cover
    AzureOpenAI = None  # type: ignore
    OpenAI = None  # type: ignore

load_dotenv()

logger = logging.getLogger("multiquery_search")


def normalize_patent_id(raw: str) -> str:
    """Normalize patent id by stripping non-digits (e.g., 'JP2011050607A' -> '2011050607')."""
    digits = re.sub(r"\D", "", str(raw or ""))
    return digits


def safe_text(x: Any) -> str:
    return x if isinstance(x, str) else (" ".join(x) if isinstance(x, list) else str(x or ""))


def truncate_chars(text: str, max_chars: int) -> str:
    if not text:
        return ""
    return text[:max_chars]


class PassageGenerator:
    """Generate ~200-char Japanese passages from a Cosmos patent doc using OpenAI if available."""

    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        # Required chat/text deployment for generation (Azure)
        self.chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("AZURE_OPENAI_TEXT_DEPLOYMENT")

        # Optional: standard OpenAI credentials for generation
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        self.provider: Optional[str] = None
        self.client = None

        # Prefer Azure if fully configured, otherwise use OpenAI if available
        if AzureOpenAI and self.api_key and self.endpoint and self.chat_deployment:
            try:
                self.client = AzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                )
                self.provider = "azure"
                logger.info("Azure OpenAI chat client initialized for passage generation")
            except Exception as e:
                raise RuntimeError(f"生成AIクライアント(Azure)の初期化に失敗しました: {e}")
        elif OpenAI and self.openai_key:
            try:
                self.client = OpenAI(api_key=self.openai_key)
                self.provider = "openai"
                logger.info("OpenAI chat client initialized for passage generation")
            except Exception as e:
                raise RuntimeError(f"生成AIクライアント(OpenAI)の初期化に失敗しました: {e}")
        else:
            # Strict: no heuristic fallback; require one of the providers
            raise RuntimeError(
                "生成AIが未設定です。Azure を使う場合は AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT を、"
                "OpenAI を使う場合は OPENAI_API_KEY (必要に応じて OPENAI_CHAT_MODEL) を設定してください。"
            )

    async def generate(self, doc: Dict[str, Any], num_passages: int = 10) -> List[str]:
        """Generate passages from summary/claims/title using OpenAI or heuristics."""
        title = safe_text(doc.get("title", ""))
        summary = safe_text(doc.get("summary", ""))
        claims = doc.get("claims", [])
        if isinstance(claims, list):
            claims_text = "\n".join(safe_text(c) for c in claims[:5])
        else:
            claims_text = safe_text(claims)

        # Some datasets may lack description; rely on summary + claims
        context = (
            f"タイトル: {truncate_chars(title, 200)}\n\n"
            f"要約: {truncate_chars(summary, 1500)}\n\n"
            f"請求項(抜粋): {truncate_chars(claims_text, 2000)}\n"
        )

        # Strictly require OpenAI chat generation (no fallback)
        try:
            system_msg = (
                "あなたは日本語の特許理解に長けたアシスタントです。"
                "入力の要約/請求項に基づき、主題・課題・解決手段・効果・用途など"
                "異なる観点で200文字前後の短文を10本作成してください。"
                "各短文は互いに焦点が重ならないようにしてください。"
                "出力はJSON配列（各要素は文字列）で、余分な説明やコードブロックを含めないでください。"
            )
            user_msg = (
                "以下の特許情報からバリエーションのある検索文を作成してください。\n\n" + context
            )
            # Use chat.completions API for the selected provider
            if self.provider == "azure":
                resp = self.client.chat.completions.create(
                    model=self.chat_deployment,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.5,
                    max_tokens=1200,
                )
            else:  # openai
                resp = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.5,
                    max_tokens=1200,
                )

            content = resp.choices[0].message.content.strip()
            passages = json.loads(content)
            # Clean and enforce count
            cleaned: List[str] = []
            for p in passages:
                s = safe_text(p).strip()
                if s:
                    cleaned.append(truncate_chars(s, 260))
                if len(cleaned) >= num_passages:
                    break
            if len(cleaned) < num_passages:
                raise RuntimeError(f"生成AIの出力数が不足しています: {len(cleaned)}/{num_passages}")
            return cleaned[:num_passages]
        except Exception as e:
            raise RuntimeError(f"生成AIでの文章生成に失敗しました: {e}")


async def multiquery_vector_search(
    patent_id_raw: str,
    k: int = 100,
    num_queries: int = 10,
    min_score: float = 0.0,
    exclude_source: bool = False,
) -> Dict[str, Any]:
    """Execute the multi-query vector search pipeline for a given patent_id."""
    # Initialize clients
    cosmos = CosmosDBClient()
    es = ElasticsearchIndexer()
    embedder = EmbeddingProcessor()
    generator = PassageGenerator()

    # Normalize patent id and fetch document
    patent_id = normalize_patent_id(patent_id_raw)
    doc = cosmos.get_document_by_id(patent_id)
    if not doc:
        raise ValueError(f"Patent not found in Cosmos DB: {patent_id_raw} -> {patent_id}")

    # Generate passages
    passages = await generator.generate(doc, num_passages=num_queries)

    # Embed passages
    embeddings = await embedder.generate_batch_embeddings(passages)

    # Run vector searches and merge
    aggregated: Dict[str, Dict[str, Any]] = {}
    per_query_hits: List[int] = []
    for qi, vec in enumerate(embeddings):
        if not vec:
            per_query_hits.append(0)
            continue
        hits = es.search_similar_documents(query_vector=vec, k=k, min_score=min_score)
        per_query_hits.append(len(hits))
        for rank, h in enumerate(hits, start=1):
            doc_id = h.get("_id") or h.get("patent_id")
            if not doc_id:
                continue
            if exclude_source and str(doc_id) == patent_id:
                continue

            entry = aggregated.get(doc_id)
            score = h.get("_score", 0.0)
            if entry is None:
                aggregated[doc_id] = {
                    "patent_id": doc_id,
                    "max_score": score,
                    "first_query_index": qi,
                    "times_hit": 1,
                    "best_rank": rank,
                    "source_example": passages[qi],
                    "document": h,
                }
            else:
                entry["times_hit"] += 1
                if score > entry["max_score"]:
                    entry["max_score"] = score
                    entry["first_query_index"] = qi
                    entry["best_rank"] = rank
                    entry["source_example"] = passages[qi]
                    entry["document"] = h

    # Sort by max_score desc, break ties by times_hit desc, then best_rank asc
    merged = list(aggregated.values())
    merged.sort(key=lambda x: (-float(x.get("max_score", 0.0)), -int(x.get("times_hit", 0)), int(x.get("best_rank", 1e9))))

    result = {
        "query_patent_id": patent_id,
        "generated_queries": passages,
        "per_query_hits": per_query_hits,
        "unique_results": len(merged),
        "results": merged,
    }

    # Cleanup
    try:
        cosmos.close()
        es.close()
    except Exception:
        pass

    return result


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description="Run multi-query vector search from a patent doc")
    parser.add_argument("--patent", required=True, help="Patent number (e.g., JP2011050607A or 2011050607)")
    parser.add_argument("--k", type=int, default=100, help="Top-K per query")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of generated passages")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum similarity score filter")
    parser.add_argument("--exclude-source", action="store_true", help="Exclude the source patent from results")
    parser.add_argument("--output", default="", help="Optional output JSON filepath")
    args = parser.parse_args()

    res = asyncio.run(
        multiquery_vector_search(
            patent_id_raw=args.patent,
            k=args.k,
            num_queries=args.num_queries,
            min_score=args.min_score,
            exclude_source=args.exclude_source,
        )
    )

    print("\n==== Multi-Query Vector Search Result (summary) ====")
    print(f"Query patent_id: {res['query_patent_id']}")
    print(f"Generated queries: {len(res['generated_queries'])} -> per-query hits: {res['per_query_hits']}")
    print(f"Unique merged results: {res['unique_results']}")

    # Show top 20
    for i, r in enumerate(res["results"][:20], 1):
        print(
            f"{i:>3}. id={r['patent_id']} score={r['max_score']:.4f} hits={r['times_hit']} "
            f"best_rank={r['best_rank']} title={r['document'].get('title','')}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\nSaved detailed JSON to: {args.output}")


if __name__ == "__main__":
    main()

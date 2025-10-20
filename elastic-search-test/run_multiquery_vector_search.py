#!/usr/bin/env python3
"""Multi-query hybrid search from a single patent document.

Flow:
- Fetch a patent by patent_id from Cosmos DB
- Generate two sets of queries (vector-oriented passages & keyword-oriented lists)
- Embed the vector passages with Azure OpenAI embeddings
- Run vector KNN searches and keyword text searches on Elasticsearch
- Merge and de-duplicate results across all queries
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
# ローカルモジュール
import sys as _sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

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
    """Normalize patent id by stripping non-digits."""
    digits = re.sub(r"\D", "", str(raw or ""))
    return digits


def safe_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(str(e) for e in x if e)
    if x is None:
        return ""
    return str(x)


def truncate_chars(text: str, max_chars: int) -> str:
    return text[:max_chars] if text else ""


class PassageGenerator:
    """Generate passages for vector search and keyword-oriented queries."""

    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        self.chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv(
            "AZURE_OPENAI_TEXT_DEPLOYMENT"
        )
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        self.provider: Optional[str] = None
        self.client = None

        if AzureOpenAI and self.api_key and self.endpoint and self.chat_deployment:
            try:
                self.client = AzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                )
                self.provider = "azure"
                logger.info("Azure OpenAI chat client initialized for query generation")
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"生成AIクライアント(Azure)の初期化に失敗しました: {exc}")
        elif OpenAI and self.openai_key:
            try:
                self.client = OpenAI(api_key=self.openai_key)
                self.provider = "openai"
                logger.info("OpenAI chat client initialized for query generation")
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"生成AIクライアント(OpenAI)の初期化に失敗しました: {exc}")
        else:
            raise RuntimeError(
                "生成AIが未設定です。Azure を使う場合は AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_CHAT_DEPLOYMENT を、OpenAI を使う場合は OPENAI_API_KEY を設定してください。"
            )

    def _call_chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> str:
        try:
            if self.provider == "azure":
                resp = self.client.chat.completions.create(
                    model=self.chat_deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.5,
                    max_tokens=max_tokens,
                )
            else:
                resp = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.5,
                    max_tokens=max_tokens,
                )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"生成AIでの文章生成に失敗しました: {exc}")

    async def generate(
        self,
        doc: Dict[str, Any],
        num_items: int = 10,
    ) -> Dict[str, List[str]]:
        """Generate vector passages and keyword queries."""

        title = safe_text(doc.get("title"))
        claims = doc.get("claims", [])
        if isinstance(claims, list) and claims:
            first_claim_entry = claims[0]
            if isinstance(first_claim_entry, dict):
                first_claim = safe_text(first_claim_entry.get("text", ""))
            else:
                first_claim = safe_text(first_claim_entry)
        elif isinstance(claims, dict):
            first_claim = safe_text(claims.get("text", ""))
        else:
            first_claim = ""

        if not first_claim:
            first_claim = safe_text(doc.get("claims_text", ""))

        summary = ""  # 明示的に要約を使わない

        context = (
            f"タイトル: {truncate_chars(title, 200)}\n\n"
            f"請求項1: {truncate_chars(first_claim, 2000)}\n"
        )

        vector_system = (
            "あなたは日本語の特許理解に長けたアシスタントです。"
            "入力の要約/請求項に基づき、主題・課題・解決手段・効果・用途など"
            "異なる観点で200文字前後の短文を10本作成してください。"
            "各短文は互いに焦点が重ならないようにしてください。"
            "必ずJSON配列（要素数10の文字列配列）のみを返し、前後に空行・説明文・コードブロックを出力してはいけません。"
        )
        vector_user = "以下の特許情報からバリエーションのある検索文を作成してください。\n\n" + context
        vector_output = self._call_chat(vector_system, vector_user)
        try:
            vector_passages_raw = json.loads(vector_output)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise RuntimeError(
                f"ベクトル検索用クエリがJSONではありません: {exc}; output={vector_output!r}"
            )
        vector_passages: List[str] = []
        for passage in vector_passages_raw:
            text = safe_text(passage).strip()
            if text:
                vector_passages.append(truncate_chars(text, 260))
            if len(vector_passages) >= num_items:
                break
        if len(vector_passages) < num_items:
            raise RuntimeError(
                f"ベクトル検索用の生成文が不足しています: {len(vector_passages)}/{num_items}"
            )

        keyword_system = (
            "あなたは日本語の特許理解に長けたアシスタントです。"
            "入力テキストに実在する固有語・専門フレーズを抽出し、検索向けのキーワード列を生成してください。"
            "必ずJSON配列（要素数10の文字列配列）のみを返し、前後に余分な文字・説明文・コードブロックを出力してはいけません。"
        )
        keyword_user = (
            "以下の指示に従ってキーワード列を10本生成してください。\n"
            "# 入力\n" + context +
            "\n# 条件\n"
            "- 出力は JSON 配列（文字列10要素）のみ\n"
            "- 各要素は空白区切りで4〜6語程度\n"
            "- 各列に入力テキストに実在する固有語・専門フレーズを2つ以上含める\n"
            "- 固有語とは、部品名・処理名・表示態様・状態遷移などの一般的でない名詞句を指す\n"
            "- 助詞や一般語（例: する、できる、方法）は含めない\n"
            "- 説明文・ラベル・コードブロックは禁止\n"
        )
        keyword_output = self._call_chat(keyword_system, keyword_user)
        try:
            keyword_raw = json.loads(keyword_output)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise RuntimeError(
                f"キーワード検索用クエリがJSONではありません: {exc}; output={keyword_output!r}"
            )
        keyword_queries: List[str] = []
        for kw in keyword_raw:
            text = safe_text(kw).strip()
            if text:
                keyword_queries.append(truncate_chars(text, 200))
            if len(keyword_queries) >= num_items:
                break
        if len(keyword_queries) < num_items:
            raise RuntimeError(
                f"キーワード検索用の生成文が不足しています: {len(keyword_queries)}/{num_items}"
            )

        return {
            "vector_passages": vector_passages,
            "keyword_queries": keyword_queries,
        }


async def multiquery_vector_search(
    patent_id_raw: str,
    k: int = 100,
    num_queries: int = 10,
    min_score: float = 0.0,
    exclude_source: bool = False,
) -> Dict[str, Any]:
    """Execute the multi-query search pipeline for a given patent_id."""

    cosmos = CosmosDBClient()
    es = ElasticsearchIndexer()
    embedder = EmbeddingProcessor()
    generator = PassageGenerator()

    patent_id = normalize_patent_id(patent_id_raw)
    doc = cosmos.get_document_by_id(patent_id)
    if not doc:
        raise ValueError(f"Patent not found in Cosmos DB: {patent_id_raw} -> {patent_id}")

    generated = await generator.generate(doc, num_items=num_queries)
    vector_passages = generated["vector_passages"]
    keyword_queries = generated["keyword_queries"]

    embeddings = await embedder.generate_batch_embeddings(vector_passages)

    aggregated: Dict[str, Dict[str, Any]] = {}
    per_pair_hits: List[int] = []

    def process_hits(
        hits: List[Dict[str, Any]],
        pair_index: int,
        vector_passage: str,
    ) -> None:
        for rank, hit in enumerate(hits, start=1):
            doc_id = hit.get("patent_id") or hit.get("_id")
            if not doc_id:
                continue
            if exclude_source and str(doc_id) == patent_id:
                continue

            score = float(hit.get("_score", 0.0))
            vector_field = hit.get("_vector_field") or es.summary_vector_field
            document_source = hit.get("document", {})

            entry = aggregated.get(doc_id)
            if entry is None:
                entry = {
                    "patent_id": doc_id,
                    "max_score": score,
                    "times_hit": 1,
                    "best_rank": rank,
                    "first_score": None,
                    "first_rank": rank,
                    "first_pair_index": pair_index,
                    "first_vector_field": vector_field,
                    "document": document_source,
                    "summary_score": score if vector_field == es.summary_vector_field else None,
                    "claims_score": score if vector_field == es.claims1_vector_field else None,
                    "summary_rank": rank if vector_field == es.summary_vector_field else None,
                    "claims_rank": rank if vector_field == es.claims1_vector_field else None,
                    "match_sources": [],
                }
                aggregated[doc_id] = entry
            else:
                entry["times_hit"] += 1
                if score > entry.get("max_score", 0.0):
                    entry["max_score"] = score
                    entry["best_rank"] = rank
                    entry["document"] = document_source
                if rank < entry.get("best_rank", rank):
                    entry["best_rank"] = rank

                if vector_field == es.summary_vector_field:
                    existing_summary = entry.get("summary_score")
                    if existing_summary is None or score > existing_summary:
                        entry["summary_score"] = score
                        entry["summary_rank"] = rank
                elif vector_field == es.claims1_vector_field:
                    existing_claims = entry.get("claims_score")
                    if existing_claims is None or score > existing_claims:
                        entry["claims_score"] = score
                        entry["claims_rank"] = rank

            entry["match_sources"].append(
                {
                    "pair_index": pair_index,
                    "vector_passage": vector_passage,
                    "rank": rank,
                    "score": score,
                    "vector_field": vector_field,
                }
            )

            if (
                entry["first_score"] is None
                or rank < entry.get("first_rank", 1_000_000)
            ):
                entry["first_rank"] = rank
                entry["first_score"] = score
                entry["first_pair_index"] = pair_index
                entry["first_vector_field"] = vector_field
                entry["document"] = document_source

    for idx, (vec, keyword_query) in enumerate(zip(embeddings, keyword_queries)):
        if not vec:
            per_pair_hits.append("0/0/0")
            continue

        summary_hits = es.hybrid_search(
            query_text=keyword_query,
            query_vector=vec,
            k=k,
            text_weight=0.6,
            vector_weight=0.4,
            min_score=min_score,
            vector_field=es.summary_vector_field,
        )
        claims_hits = es.hybrid_search(
            query_text=keyword_query,
            query_vector=vec,
            k=k,
            text_weight=0.6,
            vector_weight=0.4,
            min_score=min_score,
            vector_field=es.claims1_vector_field,
        )

        unique_ids = {hit.get("patent_id") for hit in summary_hits + claims_hits if hit.get("patent_id")}
        per_pair_hits.append(f"{len(summary_hits)}/{len(claims_hits)}/{len(unique_ids)}")

        vector_passage = vector_passages[idx]
        process_hits(summary_hits, idx, vector_passage)
        process_hits(claims_hits, idx, vector_passage)

    merged = list(aggregated.values())
    merged.sort(
        key=lambda x: (
            -float(x.get("max_score", 0.0)),
            -int(x.get("times_hit", 0)),
            int(x.get("best_rank", 1_000_000)),
        )
    )

    result = {
        "query_patent_id": patent_id,
        "vector_passages": vector_passages,
        "keyword_queries": keyword_queries,
        "per_query_hits_pair": per_pair_hits,
        "unique_results": len(merged),
        "results": merged,
        "query_embeddings": embeddings,
        "vector_fields": {
            "summary": es.summary_vector_field,
            "claims": es.claims1_vector_field,
        },
        "vector_weights": {
            "summary": es.summary_vector_weight,
            "claims": es.claims_vector_weight,
        },
        "vector_min_score": min_score,
    }

    try:
        cosmos.close()
        es.close()
    except Exception:  # pragma: no cover
        pass

    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run multi-query hybrid search from a patent doc")
    parser.add_argument("--patent", required=True, help="Patent number (e.g., JP2011050607A or 2011050607)")
    parser.add_argument("--k", type=int, default=100, help="Top-K per query")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of generated passages/keywords")
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

    print("\n==== Multi-Query Hybrid Search Result (summary) ====")
    print(f"Query patent_id: {res['query_patent_id']}")
    print(
        f"Generated vector queries: {len(res['vector_passages'])} -> per-query hits: {res['per_query_hits_vector']}"
    )
    print(
        f"Generated keyword queries: {len(res['keyword_queries'])} -> per-query hits: {res['per_query_hits_keyword']}"
    )
    print(f"Unique merged results: {res['unique_results']}")

    for i, entry in enumerate(res["results"][:20], 1):
        doc = entry.get("document", {})
        title = doc.get("title", "")
        print(
            f"{i:>3}. id={entry['patent_id']} score={entry['max_score']:.4f} hits={entry['times_hit']} "
            f"best_rank={entry['best_rank']} title={title}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\nSaved detailed JSON to: {args.output}")


if __name__ == "__main__":
    main()

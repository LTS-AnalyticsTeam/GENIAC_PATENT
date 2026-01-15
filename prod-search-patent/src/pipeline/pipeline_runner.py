from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from analysis import AnalysisService, PatentWorkItem

from .cohere_reranker import CohereReranker
from .config import PipelineConfig
from .cosmos_client import CosmosPatentClient
from .elasticsearch_stage1 import Stage1ElasticsearchIndexer
from .embedding_service import EmbeddingService
from .exceptions import IngestionError, JobCancelledError, PipelineStageError
from .graph_rag import GraphRAGService
from .job_manager import JobManager, JobState
from .parsing_service import load_json_document, parse_text_document
from .patent_search_pipeline import run_patent_search_from_json
from .query_generation import QueryGenerator
from .trimming import sort_and_trim
from .stage2_indexer import Stage2Indexer, StopwordCleaner
from .web_search_service import search_web_references

logger = logging.getLogger(__name__)
QUERY_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "query"
STOPWORDS_PATH = Path(__file__).resolve().parents[2] / "stopwords.txt"
RERANK_TOP_K = 1000
FUSION_TOP_K = 50
GRAPH_CANDIDATE_LIMIT = 1000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_detail(detail: Dict | None) -> Dict:
    if detail is None:
        detail = {}
    detail.setdefault("stages", {})
    detail.setdefault("completed_stages", [])
    return detail

def _build_rerank_query(parsed: Dict[str, Any]) -> str:
    parts: List[str] = []
    title = parsed.get("title")
    summary = parsed.get("summary")
    claim1 = parsed.get("claim1")
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")
    if claim1:
        parts.append(f"Claim1: {claim1}")
    text = "\n\n".join(part for part in parts if part)
    return text or parsed.get("title") or parsed.get("summary") or "Patent query"


@lru_cache(maxsize=1)
def _load_stopwords() -> List[str]:
    try:
        with STOPWORDS_PATH.open("r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]
    except FileNotFoundError:
        logger.warning("Stopwords file not found at %s", STOPWORDS_PATH)
        return []


def _textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_textify(item) for item in value if item is not None]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "claim_text", "value", "content"):
            if key in value:
                return _textify(value.get(key))
        return ""
    return str(value).strip()


def _extract_description_section(description: Any, key: str) -> str:
    if isinstance(description, dict):
        entries = description.get(key) or description.get(key.replace("_", "-"))
    else:
        entries = description
    return _textify(entries)


def _extract_claim_one(claims: Any) -> str:
    if not isinstance(claims, list):
        return _textify(claims)
    fallback = ""
    for claim in claims:
        if isinstance(claim, dict):
            num_candidate = ""
            for key in ("num", "claim-num", "claim_number", "claimNo"):
                if claim.get(key) is not None:
                    num_candidate = str(claim.get(key))
                    break
            norm = re.sub(r"\D", "", num_candidate or "")
            text = _textify(claim)
            if norm == "1":
                return text
            if not fallback and text:
                fallback = text
        else:
            text = _textify(claim)
            if text and not fallback:
                fallback = text
    return fallback


def _extract_patent_text_fields(cosmos_doc: Dict | None, fallback: Dict | None = None) -> Dict[str, str]:
    result = {"title": "", "abstract": "", "technical_field": "", "claim1": ""}
    if isinstance(cosmos_doc, dict):
        biblio = cosmos_doc.get("bibliographic") or {}
        title_value = cosmos_doc.get("title") or biblio.get("title") or biblio.get("invention_title")
        if isinstance(title_value, list):
            for entry in title_value:
                if isinstance(entry, dict) and entry.get("lang") in {"ja", "JP"}:
                    result["title"] = _textify(entry)
                    break
            if not result["title"]:
                result["title"] = _textify(title_value)
        else:
            result["title"] = _textify(title_value)

        abstract_value = cosmos_doc.get("abstract")
        if not abstract_value:
            abstract_value = cosmos_doc.get("summary")
        result["abstract"] = _textify(abstract_value)

        description = cosmos_doc.get("description") or {}
        result["technical_field"] = _extract_description_section(description, "technical-field")

        claims = cosmos_doc.get("claims") or []
        result["claim1"] = _extract_claim_one(claims)

    if fallback:
        if not result["title"]:
            result["title"] = _textify(fallback.get("title"))
        if not result["abstract"]:
            result["abstract"] = _textify(fallback.get("summary"))
        if not result["technical_field"]:
            result["technical_field"] = _textify(fallback.get("technical_field"))
        if not result["claim1"]:
            result["claim1"] = _textify(fallback.get("claim1"))

    return result


class StageTracker:
    """Keep JobState detail mutations uniform across pipeline stages."""

    def __init__(self, job_manager: JobManager, job_id: str) -> None:
        self.job_manager = job_manager
        self.job_id = job_id
        self._timers: Dict[str, float] = {}

    def _ensure_active(self) -> JobState | None:
        if self.job_manager.is_cancelled(self.job_id):
            raise JobCancelledError(self.job_id)
        state = self.job_manager.get_state(self.job_id)
        if state and state.status == "cancelled":
            raise JobCancelledError(self.job_id)
        return state

    def _detail(self) -> Dict:
        state = self._ensure_active()
        return _ensure_detail(state.detail if state else {})

    def start(self, stage: str, extra_detail: Dict | None = None) -> None:
        detail = self._detail()
        if extra_detail:
            detail.update(extra_detail)
        stage_entry = detail["stages"].setdefault(stage, {})
        stage_entry.setdefault("started_at", _now_iso())
        stage_entry["status"] = "in_progress"
        stage_entry["updated_at"] = _now_iso()
        detail["current_stage"] = stage
        self._timers[stage] = time.perf_counter()
        self.job_manager.set_state(self.job_id, JobState(status=stage, detail=detail))

    def update(self, stage: str, values: Dict) -> None:
        if not values:
            return
        detail = self._detail()
        stage_entry = detail["stages"].setdefault(stage, {})
        stage_entry.update(values)
        stage_entry["updated_at"] = _now_iso()
        self.job_manager.set_state(self.job_id, JobState(status=stage, detail=detail))

    def complete(self, stage: str) -> None:
        detail = self._detail()
        stage_entry = detail["stages"].setdefault(stage, {})
        stage_entry["status"] = "completed"
        stage_entry["completed_at"] = _now_iso()
        stage_entry["updated_at"] = stage_entry["completed_at"]
        start_time = self._timers.pop(stage, None)
        if start_time is not None:
            stage_entry["duration_seconds"] = round(time.perf_counter() - start_time, 3)
        if stage not in detail["completed_stages"]:
            detail["completed_stages"].append(stage)
        detail["current_stage"] = stage
        self.job_manager.set_state(self.job_id, JobState(status=stage, detail=detail))

    def finalize(self, pipeline_stats: Dict) -> None:
        detail = self._detail()
        detail["pipeline_stats"] = pipeline_stats
        detail["current_stage"] = "completed"
        detail["completed_at"] = _now_iso()
        self.job_manager.set_state(self.job_id, JobState(status="completed", detail=detail))

    def get_stage_duration(self, stage: str) -> float | None:
        detail = self._detail()
        entry = detail.get("stages", {}).get(stage)
        if not entry:
            return None
        return entry.get("duration_seconds")


def _persist_queries(job_id: str, queries: List[str]) -> None:
    try:
        QUERY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": job_id,
            "generated_at": _now_iso(),
            "queries": queries,
        }
        with (QUERY_OUTPUT_DIR / f"{job_id}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to persist queries for job %s", job_id)


def _build_alpha_source_json(parsed: Dict[str, Any]) -> Dict[str, Any]:
    source_json = parsed.get("source_json")
    if isinstance(source_json, dict):
        return source_json

    claims = parsed.get("claims") or []
    claim_entries = []
    for claim in claims:
        text = (claim or {}).get("text")
        if text:
            claim_entries.append({"text": text})
    if not claim_entries and parsed.get("claim1"):
        claim_entries.append({"text": parsed.get("claim1")})

    # Ensure patent_id is properly set
    patent_id = parsed.get("patent_id") or ""

    return {
        "bibliographic": {
            "title": parsed.get("title") or "",
            "publication": {"doc_number": patent_id},
        },
        "abstract": parsed.get("summary") or "",
        "claims": claim_entries,
        "description": parsed.get("description"),
    }


def _patent_id_matches(alpha_id: str, candidate: Any) -> bool:
    if not alpha_id or not candidate:
        return False
    a_norm = re.sub(r"\s+", "", str(alpha_id)).upper()
    b_norm = re.sub(r"\s+", "", str(candidate)).upper()
    if a_norm == b_norm:
        return True
    a_digits = re.sub(r"\D", "", a_norm)
    b_digits = re.sub(r"\D", "", b_norm)
    return bool(a_digits and a_digits == b_digits)


def _canonical_patent_id(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", "", str(value)).upper()
    return text


async def run_pipeline(
    config: PipelineConfig,
    job_manager: JobManager,
    job_id: str,
    payload_bytes: bytes,
) -> Dict:
    """Execute full ingestion pipeline and update job state along the way."""
    try:
        config.ensure_embedding_credentials()
        config.ensure_rerank_credentials()
    except RuntimeError as exc:
        raise PipelineStageError("configuration", str(exc)) from exc

    tracker = StageTracker(job_manager, job_id)

    # Stage 1: JSON loading
    tracker.start("parsing")
    try:
        parsed = load_json_document(payload_bytes)
    except IngestionError:
        parsed = parse_text_document(payload_bytes)
    alpha_source_json = _build_alpha_source_json(parsed)
    alpha_patent_id_raw = parsed.get("patent_id") or alpha_source_json.get("bibliographic", {}).get("publication", {}).get("doc_number")
    alpha_patent_id = str(alpha_patent_id_raw).strip() if alpha_patent_id_raw else ""
    tracker.complete("parsing")

    ipc_codes = parsed["classification_ipc"]
    logger.info("Job %s: IPC codes from parsed document: %s", job_id, ipc_codes[:10] if ipc_codes else [])
    logger.info("Job %s: es_stage1_limit = %d", job_id, config.es_stage1_limit)

    # Stage 2: Keyword search (IPC filtering + keyword scoring)
    # Note: cosmos_query was removed as keyword_search already includes IPC filtering in STAGE1
    tracker.start("keyword_search", {"ipc_codes": ipc_codes})

    # Build Cosmos-format JSON for keyword search
    # Use source_json if available, otherwise build from parsed data
    import re

    source_json = parsed.get("source_json")
    if source_json and isinstance(source_json, dict):
        # source_jsonがある場合、それをベースにipc_prefixを追加
        cosmos_format_json = dict(source_json)

        # ipc_prefixを計算して追加（patent_search_pipelineが期待する形式）
        ipc_prefixes = []
        for ipc in ipc_codes:
            ipc_str = ipc if isinstance(ipc, str) else ""
            if ipc_str and len(ipc_str) >= 4:
                s = ipc_str.upper()
                if "/" in s:
                    s = s.split("/", 1)[0]
                cleaned = re.sub(r"\s+", "", s)[:5]
                if cleaned and cleaned not in ipc_prefixes:
                    ipc_prefixes.append(cleaned)

        # bibliographic.classification.ipc_prefixを設定
        if "bibliographic" not in cosmos_format_json:
            cosmos_format_json["bibliographic"] = {}
        if "classification" not in cosmos_format_json["bibliographic"]:
            cosmos_format_json["bibliographic"]["classification"] = {}
        cosmos_format_json["bibliographic"]["classification"]["ipc_prefix"] = ipc_prefixes

        # claimsをclaim_text形式に変換（patent_search_pipelineが期待する形式）
        original_claims = cosmos_format_json.get("claims", [])
        converted_claims = []
        for claim in original_claims:
            if isinstance(claim, dict):
                text = claim.get("text") or claim.get("claim_text", "")
                if text:
                    converted_claims.append({"claim_text": text})
        if not converted_claims and parsed.get("claim1"):
            converted_claims.append({"claim_text": parsed.get("claim1")})
        cosmos_format_json["claims"] = converted_claims

        # invention_titleを設定（titleがある場合）
        if "bibliographic" in cosmos_format_json:
            if "title" in cosmos_format_json["bibliographic"] and "invention_title" not in cosmos_format_json["bibliographic"]:
                cosmos_format_json["bibliographic"]["invention_title"] = cosmos_format_json["bibliographic"]["title"]

        logger.info("Job %s: Using source_json for keyword search, ipc_prefixes=%s", job_id, ipc_prefixes[:3])
    else:
        # source_jsonがない場合、parsedデータから構築
        claims = parsed.get("claims") or []
        claim_entries = []
        for claim in claims:
            text = (claim or {}).get("text")
            if text:
                claim_entries.append({"claim_text": text})
        if not claim_entries and parsed.get("claim1"):
            claim_entries.append({"claim_text": parsed.get("claim1")})

        # Get IPC prefixes
        ipc_prefixes = []
        for ipc in ipc_codes:
            ipc_str = ipc if isinstance(ipc, str) else ""
            if ipc_str and len(ipc_str) >= 4:
                s = ipc_str.upper()
                if "/" in s:
                    s = s.split("/", 1)[0]
                cleaned = re.sub(r"\s+", "", s)[:5]
                if cleaned and cleaned not in ipc_prefixes:
                    ipc_prefixes.append(cleaned)

        cosmos_format_json = {
            "bibliographic": {
                "invention_title": parsed.get("title", ""),
                "publication": {"doc_number": parsed.get("patent_id", "")},
                "classification": {
                    "ipc_prefix": ipc_prefixes
                }
            },
            "abstract": parsed.get("summary", ""),
            "claims": claim_entries,
            "description": None,
        }

        logger.info("Job %s: Built cosmos_format_json from parsed data, ipc_prefixes=%s", job_id, ipc_prefixes[:3])

    # Run keyword search pipeline
    search_result = run_patent_search_from_json(cosmos_format_json, job_manager, job_id)

    narrowed_patent_ids = search_result.get("keyword_search_results", [])
    keyword_score_map: Dict[str, float] = {}
    for entry in search_result.get("search_results", []) or []:
        doc_number = entry.get("doc_number")
        if not doc_number:
            continue
        canon = _canonical_patent_id(doc_number)
        if not canon:
            continue
        try:
            score = float(entry.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        keyword_score_map[canon] = score
    removed_self_id = 0
    if alpha_patent_id:
        before = len(narrowed_patent_ids)
        narrowed_patent_ids = [pid for pid in narrowed_patent_ids if not _patent_id_matches(alpha_patent_id, pid)]
        removed_self_id = before - len(narrowed_patent_ids)
    logger.info("Job %s: Keyword search returned %d patents", job_id, len(narrowed_patent_ids))

    tracker.update("keyword_search", {
        "narrowed_count": len(narrowed_patent_ids),
        "stage1_IPC_candidates": search_result.get("pipeline_stats", {}).get("stage1_IPC_candidates", 0),
        "stage2_keyword_filter_results": search_result.get("pipeline_stats", {}).get("stage2_keyword_filter_results", 0),
        "removed_self_patent": removed_self_id,
    })
    tracker.complete("keyword_search")

    # Use narrowed patent_ids for subsequent processing
    # Fetch documents from Cosmos for the narrowed patents
    cosmos_client = CosmosPatentClient(config)
    try:
        narrowed_cosmos_map = cosmos_client.fetch_by_patent_ids(narrowed_patent_ids[:config.es_stage1_limit])
    finally:
        cosmos_client.close()

    trimmed_docs: List[Dict] = []
    keyword_patent_ids: set[str] = set()
    for patent_id in narrowed_patent_ids[:config.es_stage1_limit]:
        if patent_id in keyword_patent_ids:
            continue
        keyword_patent_ids.add(patent_id)
        canon_id = _canonical_patent_id(patent_id)
        kw_score = keyword_score_map.get(canon_id, 0.0)
        cosmos_doc = narrowed_cosmos_map.get(patent_id, {})
        trimmed_docs.append(
            {
                "job_id": job_id,
                "patent_id": patent_id,
                "title": cosmos_doc.get("title") or parsed.get("title"),
                "summary": cosmos_doc.get("abstract") or cosmos_doc.get("summary") or parsed.get("summary"),
                "claim1": cosmos_doc.get("claim1") or parsed.get("claim1"),
                "keyword_score": kw_score,
            }
        )

    stage1 = Stage1ElasticsearchIndexer(config)
    stage1.ensure_index()

    # Stage 4: Embedding generation
    tracker.start("embedding", {"candidate_count": len(trimmed_docs)})

    existing_sources = stage1.fetch_existing(doc.get("patent_id") for doc in trimmed_docs)
    docs_to_embed: List[Dict] = []
    reused_embeddings = 0
    vector_field = config.es_vector_field
    for doc in trimmed_docs:
        patent_id = doc.get("patent_id")
        existing = existing_sources.get(patent_id) if patent_id else None
        vector = (existing or {}).get(vector_field) if existing else None
        if vector:
            doc[vector_field] = vector
            reused_embeddings += 1
        else:
            docs_to_embed.append(doc)

    embedder = EmbeddingService(config)
    try:
        if docs_to_embed:
            await embedder.embed_documents(
                docs_to_embed,
                vector_field,
            )
    finally:
        await embedder.aclose()

    tracker.update(
        "embedding",
        {
            "embedded_documents": len(docs_to_embed),
            "reused_embeddings": reused_embeddings,
        },
    )

    enriched_docs = trimmed_docs
    tracker.complete("embedding")

    # Stage 5: Stage1 Elasticsearch indexing
    tracker.start("stage1_indexing")
    if enriched_docs:
        success, failures = stage1.bulk_index(enriched_docs)
    else:
        success, failures = 0, 0
    logger.info(
        "Stage1 indexed=%s failures=%s reused_vectors=%s",
        success,
        failures,
        reused_embeddings,
    )
    tracker.update(
        "stage1_indexing",
        {
            "indexed_documents": success,
            "failed_documents": failures,
        },
    )
    tracker.complete("stage1_indexing")

    # Stage 6: Vector search over Stage1 index
    tracker.start("vector_search")
    vector_docs: List[Dict] = []
    keyword_max_score = 0.0
    vector_max_score = 0.0
    generated_queries: List[str] = []
    hits_per_query: List[int] = []
    try:
        query_generator = QueryGenerator(config)
        generated_queries = await query_generator.generate_queries(
            parsed.get("title", ""),
            parsed.get("summary", ""),
            parsed.get("claim1", ""),
            count=10,
        )
    except PipelineStageError as exc:
        logger.warning("Job %s: query generation failed: %s", job_id, exc)
        generated_queries = []

    _persist_queries(job_id, generated_queries)

    allowed_patent_ids = {str(doc.get("patent_id")) for doc in trimmed_docs if doc.get("patent_id")}
    allowed_candidate_count = len(allowed_patent_ids)
    knn_k = config.es_vector_k
    knn_num_candidates = config.es_num_candidates
    if allowed_candidate_count:
        knn_k = max(1, min(config.es_vector_k, allowed_candidate_count))
        knn_num_candidates = max(knn_k, min(config.es_num_candidates, allowed_candidate_count))

    aggregated_hits: Dict[str, Dict] = {}
    if generated_queries:
        query_vectors: List[List[float]] = []
        query_embedder = EmbeddingService(config)
        try:
            query_vectors = await query_embedder.embed_texts(generated_queries)
        finally:
            await query_embedder.aclose()

        for query_text, vector in zip(generated_queries, query_vectors):
            hits = stage1.knn_search(
                vector,
                k=knn_k,
                num_candidates=knn_num_candidates,
                job_id=job_id,
            )
            hits_per_query.append(len(hits))
            for rank, hit in enumerate(hits):
                source = hit.get("_source") or {}
                patent_id = source.get("patent_id") or hit.get("_id")
                if not patent_id:
                    continue
                patent_id_str = str(patent_id)
                if allowed_patent_ids and patent_id_str not in allowed_patent_ids:
                    continue
                entry = aggregated_hits.setdefault(
                    patent_id_str,
                    {
                        "source": source,
                        "score": hit.get("_score", 0.0),
                        "best_rank": rank,
                        "matches": [],
                    },
                )
                score = hit.get("_score", 0.0)
                if score > entry["score"]:
                    entry["score"] = score
                    entry["source"] = source
                entry["best_rank"] = min(entry["best_rank"], rank)
                entry["matches"].append(
                    {
                        "query": query_text,
                        "rank": rank,
                        "score": score,
                    }
                )
                entry["source"].setdefault("patent_id", patent_id_str)

        sorted_hits = sorted(
            aggregated_hits.values(),
            key=lambda item: (-item["score"], item["best_rank"]),
        )
        max_candidates = max(1, min(config.es_vector_k, len(sorted_hits)))
        for item in sorted_hits[:max_candidates]:
            doc = dict(item["source"])
            doc["vector_score"] = item["score"]
            doc["vector_best_rank"] = item["best_rank"]
            doc["vector_matched_queries"] = [match["query"] for match in item["matches"][:3]]
            vector_docs.append(doc)

    if vector_docs:
        for doc in vector_docs:
            try:
                kw_score = float(doc.get("keyword_score") or 0.0)
            except (TypeError, ValueError):
                kw_score = 0.0
            try:
                vec_score = float(doc.get("vector_score") or 0.0)
            except (TypeError, ValueError):
                vec_score = 0.0
            if kw_score > keyword_max_score:
                keyword_max_score = kw_score
            if vec_score > vector_max_score:
                vector_max_score = vec_score

    tracker.update(
        "vector_search",
        {
            "generated_queries": generated_queries,
            "hits_per_query": hits_per_query,
            "unique_hits": len(vector_docs),
            "keyword_candidate_count": allowed_candidate_count,
            "top_patent_ids": [doc.get("patent_id") for doc in vector_docs[:10]],
            # 全ヒットを UI で一覧表示できるように保持
            "vector_patent_ids": [doc.get("patent_id") for doc in vector_docs],
        },
    )
    tracker.complete("vector_search")

    # Stage 7: Graph reset + indexing with Cosmos payloads
    tracker.start("graph_ingest", {"vector_candidates": len(vector_docs)})
    cleaner = StopwordCleaner(_load_stopwords())
    graph_documents: List[Dict] = []
    graph_scores: Dict[str, float] = {}
    seen_graph_ids: set[str] = set()
    graph_candidate_ids = []
    for candidate in vector_docs:
        pid = candidate.get("patent_id")
        if not pid or pid in seen_graph_ids:
            continue
        seen_graph_ids.add(pid)
        graph_candidate_ids.append(pid)
    cosmos_graph_map: Dict[str, Dict] = {}
    if graph_candidate_ids:
        cosmos_graph = CosmosPatentClient(config)
        try:
            cosmos_graph_map = cosmos_graph.fetch_by_patent_ids(graph_candidate_ids[:GRAPH_CANDIDATE_LIMIT])
        finally:
            cosmos_graph.close()
    stage2_indexer = Stage2Indexer(config, cleaner=cleaner)
    alpha_graph_document: Dict[str, Any] | None = None
    if alpha_patent_id:
        alpha_fields = _extract_patent_text_fields(alpha_source_json, parsed)
        alpha_graph_document = {
            "patent_id": alpha_patent_id,
            "title": alpha_fields.get("title"),
            "abstract": alpha_fields.get("abstract"),
            "technical_field": alpha_fields.get("technical_field"),
            "claim1": alpha_fields.get("claim1"),
            "vector_score": None,
        }
    indexed_documents = 0
    try:
        stage2_indexer.reset_graph()
        documents_to_index: List[Dict] = []
        if alpha_graph_document:
            documents_to_index.append(alpha_graph_document)
        if graph_candidate_ids:
            processed_ids: set[str] = set()
            for doc in vector_docs[:GRAPH_CANDIDATE_LIMIT]:
                pid = doc.get("patent_id")
                if not pid or pid in processed_ids:
                    continue
                processed_ids.add(pid)
                source_doc = cosmos_graph_map.get(pid)
                extracted = _extract_patent_text_fields(source_doc, doc)
                cleaned_fields = {
                    "title": cleaner.clean_text(extracted.get("title")),
                    "abstract": cleaner.clean_text(extracted.get("abstract")),
                    "technical_field": cleaner.clean_text(extracted.get("technical_field")),
                    "claim1": cleaner.clean_text(extracted.get("claim1")),
                }
                if cleaned_fields["title"]:
                    doc["title"] = cleaned_fields["title"]
                if cleaned_fields["abstract"]:
                    doc["summary"] = cleaned_fields["abstract"]
                if cleaned_fields["claim1"]:
                    doc["claim1"] = cleaned_fields["claim1"]
                candidate_payload = {
                    "patent_id": pid,
                    "title": extracted.get("title"),
                    "abstract": extracted.get("abstract"),
                    "technical_field": extracted.get("technical_field"),
                    "claim1": extracted.get("claim1"),
                    "vector_score": doc.get("vector_score"),
                }
                graph_documents.append(candidate_payload)
                documents_to_index.append(candidate_payload)
        if documents_to_index:
            indexed_documents = stage2_indexer.index_documents(documents_to_index)
    finally:
        stage2_indexer.close()
    tracker.update(
        "graph_ingest",
        {
            "graph_candidates": len(graph_documents),
            "graph_indexed": indexed_documents,
            "alpha_indexed": bool(alpha_graph_document),
        },
    )
    tracker.complete("graph_ingest")

    # Stage 8: Graph-RAG scoring from alpha patent
    tracker.start("graph_rag", {"graph_candidates": len(graph_candidate_ids)})
    graph_ranked: List[Dict] = []
    graph_max_score = 0.0
    if graph_candidate_ids:
        graph_service = GraphRAGService(config)
        try:
            graph_ranked = graph_service.top_k(
                [pid for pid in graph_candidate_ids[:GRAPH_CANDIDATE_LIMIT] if pid],
                alpha_patent_id or None,
                k=min(GRAPH_CANDIDATE_LIMIT, len(graph_candidate_ids)),
            )
        finally:
            graph_service.close()
        for entry in graph_ranked:
            pid = entry.get("patent_id")
            if not pid:
                continue
            raw_score = entry.get("graph_score") or 0.0
            graph_scores[pid] = raw_score
            if raw_score > graph_max_score:
                graph_max_score = raw_score
    tracker.update(
        "graph_rag",
        {
            "graph_ranked": len(graph_ranked),
            "graph_scored_patent_ids": [entry.get("patent_id") for entry in graph_ranked[:10]],
            "graph_score_max": graph_max_score,
        },
    )
    tracker.complete("graph_rag")

    # Stage 9: Cohere reranking
    tracker.start("rerank", {"vector_candidates": len(vector_docs)})
    reranked_docs: List[Dict] = []
    rerank_max_score = 0.0
    rerank_query = _build_rerank_query(parsed)
    rerank_details: Dict[str, Any] = {
        "rerank_results": 0,
        "rerank_top_patent_ids": [],
        "rerank_top_results": [],
        "rerank_patent_results": [],
        "rerank_query_preview": rerank_query[:200],
    }
    if vector_docs:
        reranker = CohereReranker(config)
        try:
            reranked_docs = reranker.rerank(rerank_query, vector_docs, top_n=RERANK_TOP_K)
        except Exception as exc:
            logger.warning("Job %s: Cohere rerank failed, falling back to vector order: %s", job_id, exc)
            fallback_docs: List[Dict] = []
            for doc in vector_docs[:RERANK_TOP_K]:
                copied = dict(doc)
                copied.setdefault("rerank_score", doc.get("vector_score"))
                fallback_docs.append(copied)
            reranked_docs = fallback_docs
            rerank_details["rerank_error"] = str(exc)
        rerank_details.update(
            {
                "rerank_results": len(reranked_docs),
                "rerank_top_patent_ids": [doc.get("patent_id") for doc in reranked_docs[:10]],
                "rerank_top_results": [
                    {
                        "patent_id": doc.get("patent_id"),
                        "title": doc.get("title"),
                        "rerank_score": doc.get("rerank_score"),
                        "vector_score": doc.get("vector_score"),
                    }
                    for doc in reranked_docs[:10]
                    if doc.get("patent_id")
                ],
                "rerank_patent_results": [
                    {
                        "patent_id": doc.get("patent_id"),
                        "title": doc.get("title"),
                        "rerank_score": doc.get("rerank_score"),
                        "vector_score": doc.get("vector_score"),
                    }
                    for doc in reranked_docs
                    if doc.get("patent_id")
                ],
            }
        )
    if reranked_docs:
        for doc in reranked_docs:
            try:
                rerank_val = float(doc.get("rerank_score") or 0.0)
            except (TypeError, ValueError):
                rerank_val = 0.0
            if rerank_val > rerank_max_score:
                rerank_max_score = rerank_val
    tracker.update("rerank", rerank_details)
    tracker.complete("rerank")

    # Stage 10: Fuse rerank + graph scores and keep top 50
    tracker.start(
        "fusion",
        {
            "rerank_results": len(reranked_docs),
            "graph_scored": len(graph_scores),
        },
    )
    fused_candidates: List[Dict] = []
    fusion_serializable: List[Dict] = []
    if reranked_docs:
        for doc in reranked_docs:
            pid = doc.get("patent_id")
            graph_raw = graph_scores.get(pid, 0.0) if pid else 0.0
            if graph_max_score > 0:
                graph_score = graph_raw / graph_max_score
            else:
                graph_score = 0.0
            rerank_score = doc.get("rerank_score")
            try:
                rerank_value = float(rerank_score) if rerank_score is not None else 0.0
            except (TypeError, ValueError):
                rerank_value = 0.0
            rerank_norm = rerank_value / rerank_max_score if rerank_max_score > 0 else 0.0
            keyword_raw = doc.get("keyword_score") or 0.0
            try:
                keyword_value = float(keyword_raw)
            except (TypeError, ValueError):
                keyword_value = 0.0
            keyword_norm = keyword_value / keyword_max_score if keyword_max_score > 0 else 0.0
            vector_raw = doc.get("vector_score") or 0.0
            try:
                vector_value = float(vector_raw)
            except (TypeError, ValueError):
                vector_value = 0.0
            vector_norm = vector_value / vector_max_score if vector_max_score > 0 else 0.0
            doc["graph_score_raw"] = graph_raw
            doc["graph_score"] = graph_score
            doc["keyword_score"] = keyword_value
            doc["keyword_score_norm"] = keyword_norm
            doc["vector_score_norm"] = vector_norm
            doc["rerank_score_norm"] = rerank_norm
            doc["fusion_score"] = (
                keyword_norm * 0.2
                + rerank_norm * 0.4
                + vector_norm * 0.3
                + graph_score * 0.1
            )
            fused_candidates.append(doc)
        fused_candidates.sort(key=lambda item: item.get("fusion_score", 0.0), reverse=True)
    selected_patent_candidates = fused_candidates[:FUSION_TOP_K]
    fusion_serializable = []
    for doc in selected_patent_candidates:
        pid = doc.get("patent_id")
        if not pid:
            continue
        fusion_serializable.append(
            {
                "patent_id": pid,
                "title": doc.get("title"),
                "summary": doc.get("summary") or doc.get("abstract"),
                "graph_score": doc.get("graph_score"),
                "graph_score_raw": doc.get("graph_score_raw"),
                "keyword_score": doc.get("keyword_score"),
                "keyword_score_norm": doc.get("keyword_score_norm"),
                "rerank_score": doc.get("rerank_score"),
                "rerank_score_norm": doc.get("rerank_score_norm"),
                "fusion_score": doc.get("fusion_score"),
                "vector_score": doc.get("vector_score"),
                "vector_score_norm": doc.get("vector_score_norm"),
            }
        )
    tracker.update(
        "fusion",
        {
            "fusion_candidates": len(fused_candidates),
            "fusion_top": len(selected_patent_candidates),
            "fusion_top_patent_ids": [doc.get("patent_id") for doc in selected_patent_candidates[:10]],
            "fusion_patent_results": fusion_serializable,
        },
    )
    tracker.complete("fusion")

    # Stage 11: Web search for additional references
    tracker.start("web_search")
    web_results: List[Dict] = []
    claim1_text = parsed.get("claim1", "")

    if claim1_text:
        try:
            import os
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            web_results = await search_web_references(claim1_text, azure_deployment)
            logger.info(f"Job %s: Web search returned %d results", job_id, len(web_results))
        except Exception as e:
            logger.warning(f"Job %s: Web search failed: %s", job_id, e)
            web_results = []
    else:
        logger.warning(f"Job %s: No claim1 text available for web search", job_id)

    tracker.update("web_search", {
        "web_results_count": len(web_results),
        "web_result_ids": [r.get("patent_id") for r in web_results[:5]]
    })
    tracker.complete("web_search")

    # Stage 12: Merge all candidates (50 patents + web results) for analysis
    tracker.start("merge_candidates")
    combined_results = list(selected_patent_candidates) + list(web_results)
    logger.info(
        f"Job %s: Combined results: {len(selected_patent_candidates)} patents + {len(web_results)} web = {len(combined_results)} total",
        job_id,
    )

    tracker.update(
        "merge_candidates",
        {
            "combined_count": len(combined_results),
            "patent_count": len(selected_patent_candidates),
            "web_count": len(web_results),
        },
    )
    tracker.complete("merge_candidates")

    # Stage 13: Analyze ALL combined candidates (50+ candidates)
    tracker.start("analysis", {"analysis_candidates": len(combined_results)})
    analysis_service = AnalysisService()
    all_analysis_results: List[Dict] = []  # 全候補の分析結果を保存
    missing_candidates: List[str] = []
    analysis_payload: Dict[str, Any] | None = None

    if combined_results:
        # Separate patent results and web results from ALL combined results
        patent_results = [entry for entry in combined_results if not entry.get("is_web_result")]
        web_only_results = [entry for entry in combined_results if entry.get("is_web_result")]

        # Fetch Cosmos data only for patent results
        analysis_cosmos = CosmosPatentClient(config)
        try:
            candidate_docs_map = analysis_cosmos.fetch_by_patent_ids(
                [entry["patent_id"] for entry in patent_results if entry.get("patent_id")]
            )
        finally:
            analysis_cosmos.close()

        ordered_candidates: List[Dict] = []

        # Process patent results first (priority for evaluation)
        for entry in patent_results:
            pid = entry.get("patent_id")
            if not pid:
                continue
            candidate_doc = candidate_docs_map.get(pid)
            if candidate_doc:
                ordered_candidates.append(candidate_doc)
            else:
                missing_candidates.append(pid)

        # Add web results AFTER patent results (lower priority)
        for entry in web_only_results:
            pid = entry.get("patent_id")
            if not pid:
                continue

            # Create a minimal candidate_json for web results
            web_candidate_json = {
                "bibliographic": {
                    "invention_title": entry.get("title", ""),
                    "publication": {"doc_number": pid},
                },
                "abstract": entry.get("summary", ""),
                "claims": [],
                "source_url": entry.get("source_url", ""),
                "is_web_result": True,
                "page_content": entry.get("page_content", ""),  # URL本文内容を追加
            }
            ordered_candidates.append(web_candidate_json)

        tracker.update(
            "analysis",
            {
                "ordered_candidates": len(ordered_candidates),
                "missing_candidates": missing_candidates,
            },
        )

        if ordered_candidates:
            try:
                analysis_response = await analysis_service.analyze_candidates(
                    alpha_json=alpha_source_json,
                    candidates_json=ordered_candidates,
                )
                analysis_payload = analysis_response.model_dump()

                # 分析結果を各候補に紐付け
                for entry in combined_results:
                    entry["analysis_status"] = "completed"
                    entry["analysis"] = analysis_payload

            except Exception as exc:  # pragma: no cover
                logger.warning("Analysis failed: %s", exc)
                analysis_payload = None
                for entry in combined_results:
                    entry.setdefault("analysis_status", "failed")
                    entry.setdefault("analysis_error", "解析結果を生成できませんでした")
        else:
            analysis_payload = None
            for entry in combined_results:
                entry.setdefault("analysis_status", "failed")
                entry.setdefault("analysis_error", "候補データが取得できませんでした")

    if missing_candidates:
        tracker.update(
            "analysis",
            {
                "error": "Cosmos DBに該当のデータがありませんでした",
            },
        )

    tracker.complete("analysis")

    # Stage 14: Select top 10 from analyzed candidates using LLM
    tracker.start("final_selection")

    # AnalysisServiceを使って候補選択
    alpha_info = {
        "title": parsed.get("title", ""),
        "summary": parsed.get("summary", ""),
        "claim1": claim1_text,
    }

    try:
        final_candidates = await analysis_service.select_top_candidates(
            combined_candidates=combined_results,
            analysis_result=analysis_payload,
            alpha_info=alpha_info,
            top_n=10,
        )
        logger.info(f"Job %s: Selected {len(final_candidates)} final candidates from {len(combined_results)} analyzed", job_id)
    except Exception as e:
        logger.warning(f"Job %s: Candidate selection failed: %s, using all candidates", job_id, e)
        final_candidates = combined_results[:10] if len(combined_results) > 10 else combined_results

    tracker.update("final_selection", {
        "analyzed_count": len(combined_results),
        "final_count": len(final_candidates),
        "patent_count": sum(1 for c in final_candidates if not c.get("is_web_result")),
        "web_count": sum(1 for c in final_candidates if c.get("is_web_result")),
    })
    tracker.complete("final_selection")

    # Extract assessments from analysis_payload and attach to each final candidate
    if analysis_payload:
        claim1_candidates = analysis_payload.get("claim1_candidates", [])
        rest_claim_candidates = analysis_payload.get("rest_claim_candidates", [])

        # Create a mapping from patent_id to assessments
        assessments_map = {}
        for cand in claim1_candidates:
            doc_id = cand.get("doc_id")
            assessments_list = cand.get("assessments", [])
            if doc_id:
                assessments_map[doc_id] = assessments_list

        for cand in rest_claim_candidates:
            doc_id = cand.get("doc_id")
            assessments_list = cand.get("assessments", [])
            if doc_id:
                # Merge with existing assessments if any
                if doc_id in assessments_map:
                    assessments_map[doc_id].extend(assessments_list)
                else:
                    assessments_map[doc_id] = assessments_list

        # Attach assessments to final_candidates
        for candidate in final_candidates:
            patent_id = candidate.get("patent_id")
            if patent_id in assessments_map:
                assessments = assessments_map[patent_id]
                # If this is a web result, force all evidence sections to "web"
                if candidate.get("is_web_result"):
                    for assessment in assessments:
                        for evidence in assessment.get("evidence", []):
                            evidence["section"] = "web"
                candidate["assessments"] = assessments
            else:
                candidate["assessments"] = []
    else:
        # No analysis_payload, set empty assessments for all
        for candidate in final_candidates:
            candidate["assessments"] = []

    # Prepare full web search results with title and URL
    # Include ALL web results (not just the ones in final_candidates)
    web_search_details = [
        {
            "patent_id": r.get("patent_id", ""),
            "title": r.get("title", ""),
            "source_url": r.get("source_url", ""),
            "summary": r.get("summary", ""),
        }
        for r in web_results
    ]

    logger.info(f"Job %s: Created web_search_details with {len(web_search_details)} items", job_id)

    final_payload = {
        "results": final_candidates,  # 選択された上位10件を返す
        "fusion_results": fusion_serializable,
        "keyword_search_results": narrowed_patent_ids,
        "search_results": search_result.get("search_results", []),
        "web_search_results": [r.get("patent_id") for r in web_results],
        "web_search_details": web_search_details,
        "pipeline_stats": {
            "trimmed": len(trimmed_docs),
            "stage1_indexed": success,
            "stage1_failed": failures,
            "stage1_reused": reused_embeddings,
            "stage1_embedded": len(docs_to_embed),
            "vector_search_results": len(vector_docs),
            "vector_queries": len(generated_queries),
            "graph_candidates": len(graph_documents),
            "graph_indexed": indexed_documents,
            "graph_ranked": len(graph_scores),
            "rerank_results": len(reranked_docs),
            "fusion_candidates": len(fused_candidates),
            "fusion_top": len(selected_patent_candidates),
            "web_search_results": len(web_results),
            "analysis_candidates": len(combined_results),  # 分析した候補数（50+）
            "final_candidates": len(final_candidates),  # 最終選択数（10）
            "stage1_IPC_candidates": search_result.get("pipeline_stats", {}).get("stage1_IPC_candidates", 0),
            "stage2_keyword_filter_results": len(narrowed_patent_ids),
            "analysis_completed": 1 if analysis_payload else 0,
        },
    }

    tracker.finalize(final_payload["pipeline_stats"])
    job_manager.store_result(job_id, final_payload)
    return final_payload


async def run_test_pipeline_from_patent_id(
    job_id: str,
    patent_id: str,
    job_manager: JobManager,
    config: PipelineConfig,
) -> Dict[str, Any]:
    """
    テスト用パイプライン: 特許番号からCosmosDBのJSONを取得し、既存のパイプラインを実行
    """
    logger.info(f"Job %s: Starting test pipeline for patent_id=%s", job_id, patent_id)

    # Fetch JSON from Cosmos DB
    cosmos_client = CosmosPatentClient(config)
    try:
        docs_map = cosmos_client.fetch_by_patent_ids([patent_id])
        if not docs_map or patent_id not in docs_map:
            raise PipelineStageError(
                "fetching_from_cosmos",
                f"Patent ID {patent_id} not found in Cosmos DB"
            )

        patent_json = docs_map[patent_id]
        logger.info(f"Job %s: Successfully fetched patent JSON from Cosmos DB", job_id)

        # Convert JSON to bytes for existing pipeline
        json_bytes = json.dumps(patent_json, ensure_ascii=False).encode("utf-8")

    finally:
        cosmos_client.close()

    # Run existing pipeline with the fetched JSON
    logger.info(f"Job %s: Running standard pipeline with fetched JSON", job_id)
    return await run_pipeline(config, job_manager, job_id, json_bytes)


def _extract_parsed_from_cosmos_json(cosmos_json: Dict) -> Dict:
    """CosmosDBのJSON構造からパース済みデータを抽出"""
    bibliographic = cosmos_json.get("bibliographic", {})

    # Title
    title = ""
    if isinstance(bibliographic.get("invention_title"), list):
        for t in bibliographic["invention_title"]:
            if isinstance(t, dict) and t.get("lang") == "ja":
                title = t.get("text", "")
                break
    elif isinstance(bibliographic.get("invention_title"), str):
        title = bibliographic["invention_title"]

    # Summary/Abstract
    abstract = cosmos_json.get("abstract", {})
    summary = ""
    if isinstance(abstract, str):
        summary = abstract
    elif isinstance(abstract, dict):
        abstract_text = abstract.get("text", "")
        if isinstance(abstract_text, list):
            summary = " ".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in abstract_text])
        elif isinstance(abstract_text, str):
            summary = abstract_text

    # Claim1
    claims = cosmos_json.get("claims", [])
    claim1 = ""
    if claims and len(claims) > 0:
        first_claim = claims[0]
        if isinstance(first_claim, dict):
            claim_text = first_claim.get("text", "")
            if isinstance(claim_text, list):
                claim1 = " ".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in claim_text])
            else:
                claim1 = str(claim_text)

    # IPC
    ipc_list = []
    classification = bibliographic.get("classification", {})
    ipc_data = classification.get("ipc", [])
    for ipc_entry in ipc_data:
        if isinstance(ipc_entry, dict):
            ipc_text = ipc_entry.get("text", "")
            if ipc_text:
                ipc_list.append(ipc_text)

    return {
        "title": title.strip(),
        "summary": summary.strip(),
        "claim1": claim1.strip(),
        "ipc": ipc_list,
    }

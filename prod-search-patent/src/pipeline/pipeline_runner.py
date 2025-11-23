from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from analysis import AnalysisService, PatentWorkItem

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
from .stage2_indexer import Stage2Indexer
from .trimming import sort_and_trim

logger = logging.getLogger(__name__)
QUERY_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "query"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_detail(detail: Dict | None) -> Dict:
    if detail is None:
        detail = {}
    detail.setdefault("stages", {})
    detail.setdefault("completed_stages", [])
    return detail


def _deduplicate_by_patent_id(documents: List[Dict]) -> List[Dict]:
    seen: set[str] = set()
    deduped: List[Dict] = []
    for doc in documents:
        patent_id = doc.get("patent_id") or doc.get("id")
        if not patent_id:
            continue
        patent_id_str = str(patent_id)
        if patent_id_str in seen:
            continue
        seen.add(patent_id_str)
        deduped.append(doc)
    return deduped


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

    return {
        "bibliographic": {
            "title": parsed.get("title"),
            "publication": {"doc_number": parsed.get("patent_id")},
        },
        "abstract": parsed.get("summary"),
        "claims": claim_entries,
        "description": None,
    }


async def run_pipeline(
    config: PipelineConfig,
    job_manager: JobManager,
    job_id: str,
    payload_bytes: bytes,
) -> Dict:
    """Execute full ingestion pipeline and update job state along the way."""
    try:
        config.ensure_embedding_credentials()
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
    logger.info("Job %s: Keyword search returned %d patents", job_id, len(narrowed_patent_ids))

    tracker.update("keyword_search", {
        "narrowed_count": len(narrowed_patent_ids),
        "stage1_IPC_candidates": search_result.get("pipeline_stats", {}).get("stage1_IPC_candidates", 0),
        "stage2_keyword_filter_results": search_result.get("pipeline_stats", {}).get("stage2_keyword_filter_results", 0),
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
    seen_trimmed: set[str] = set()
    for patent_id in narrowed_patent_ids[:config.es_stage1_limit]:
        if patent_id in seen_trimmed:
            continue
        seen_trimmed.add(patent_id)
        cosmos_doc = narrowed_cosmos_map.get(patent_id, {})
        trimmed_docs.append(
            {
                "patent_id": patent_id,
                "title": cosmos_doc.get("title") or parsed.get("title"),
                "summary": cosmos_doc.get("abstract") or cosmos_doc.get("summary") or parsed.get("summary"),
                "claim1": cosmos_doc.get("claim1") or parsed.get("claim1"),
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
    if docs_to_embed:
        success, failures = stage1.bulk_index(docs_to_embed)
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
                k=config.es_vector_k,
                num_candidates=config.es_num_candidates,
            )
            hits_per_query.append(len(hits))
            for rank, hit in enumerate(hits):
                source = hit.get("_source") or {}
                patent_id = source.get("patent_id") or hit.get("_id")
                if not patent_id:
                    continue
                entry = aggregated_hits.setdefault(
                    patent_id,
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
                entry["source"].setdefault("patent_id", patent_id)

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

    tracker.update(
        "vector_search",
        {
            "generated_queries": generated_queries,
            "hits_per_query": hits_per_query,
            "unique_hits": len(vector_docs),
            "top_patent_ids": [doc.get("patent_id") for doc in vector_docs[:10]],
        },
    )
    tracker.complete("vector_search")

    # Stage 7: Neo4j enrichment
    tracker.start("stage2_indexing")
    stage2 = Stage2Indexer(config)
    stage2.reset_graph()
    tracker.update("stage2_indexing", {"graph_reset": True})
    stage2_docs = vector_docs if vector_docs else enriched_docs
    stage2_docs = _deduplicate_by_patent_id(stage2_docs)
    cosmos_client = CosmosPatentClient(config)
    try:
        stage2_cosmos_map = cosmos_client.fetch_by_patent_ids(
            [doc.get("patent_id") for doc in stage2_docs if doc.get("patent_id")]
        )
    finally:
        cosmos_client.close()

    def _extract_classification(doc: Dict) -> List[str]:
        biblio = doc.get("bibliographic") or {}
        classification = biblio.get("classification") or {}
        ipc_entries = classification.get("ipc") or []
        codes: List[str] = []
        for entry in ipc_entries:
            if isinstance(entry, dict):
                text = entry.get("text") or entry.get("value")
            else:
                text = entry
            if text:
                text = str(text).strip()
                if text and text not in codes:
                    codes.append(text)
        top_level = doc.get("classification_ipc")
        if isinstance(top_level, list):
            for value in top_level:
                if not value:
                    continue
                text = str(value).strip()
                if text and text not in codes:
                    codes.append(text)
        return codes

    stage2_graph_docs: List[Dict] = []
    for doc in stage2_docs:
        pid = doc.get("patent_id")
        if not pid:
            continue
        cosmos_doc = stage2_cosmos_map.get(pid)
        graph_doc = {
            "patent_id": pid,
            "title": None,
            "summary": None,
            "classification_ipc": None,
        }
        if cosmos_doc:
            graph_doc["title"] = cosmos_doc.get("title") or doc.get("title")
            graph_doc["summary"] = (
                cosmos_doc.get("abstract")
                or cosmos_doc.get("summary")
                or doc.get("summary")
            )
            classification = _extract_classification(cosmos_doc)
            graph_doc["classification_ipc"] = classification or None
        else:
            graph_doc["title"] = doc.get("title")
            graph_doc["summary"] = doc.get("summary")
        stage2_graph_docs.append(graph_doc)

    stage2.upsert_graph(stage2_graph_docs)
    tracker.update(
        "stage2_indexing",
        {
            "ingested_documents": len(stage2_graph_docs),
            "source": "vector_search" if vector_docs else "cosmos_trimmed",
        },
    )
    tracker.complete("stage2_indexing")

    # Stage 8: Graph-RAG ranking
    tracker.start("graph_rag")
    rag = GraphRAGService(config)
    top_results = rag.top_k([doc.get("patent_id") for doc in stage2_docs], k=10)
    top_results = _deduplicate_by_patent_id(top_results)
    rag.close()
    stage2.close()
    tracker.complete("graph_rag")

    tracker.start("analysis", {"analysis_candidates": len(top_results)})
    analysis_service = AnalysisService()
    analysis_payload: Dict | None = None
    missing_candidates: List[str] = []

    if top_results:
        analysis_cosmos = CosmosPatentClient(config)
        try:
            candidate_docs_map = analysis_cosmos.fetch_by_patent_ids(
                [entry["patent_id"] for entry in top_results if entry.get("patent_id")]
            )
        finally:
            analysis_cosmos.close()

        ordered_candidates: List[Dict] = []
        for entry in top_results:
            pid = entry.get("patent_id")
            if not pid:
                continue
            candidate_doc = candidate_docs_map.get(pid)
            if candidate_doc:
                ordered_candidates.append(candidate_doc)
            else:
                missing_candidates.append(pid)

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
            except Exception as exc:  # pragma: no cover
                logger.warning("Analysis failed: %s", exc)
                analysis_payload = None

    for entry in top_results:
        if analysis_payload:
            entry["analysis_status"] = "completed"
            entry["analysis"] = analysis_payload
        else:
            entry.setdefault("analysis_status", "failed")
            entry.setdefault("analysis_error", "解析結果を生成できませんでした")

    if missing_candidates:
        tracker.update(
            "analysis",
            {
                "error": "Cosmos DBに該当のデータがありませんでした",
            },
        )

    tracker.complete("analysis")

    final_payload = {
        "results": top_results,
        "keyword_search_results": narrowed_patent_ids,
        "pipeline_stats": {
            "trimmed": len(trimmed_docs),
            "stage1_indexed": success,
            "stage1_failed": failures,
            "stage1_reused": reused_embeddings,
            "stage1_embedded": len(docs_to_embed),
            "vector_search_results": len(vector_docs),
            "vector_queries": len(generated_queries),
            "analysis_candidates": len(top_results),
            "stage1_IPC_candidates": search_result.get("pipeline_stats", {}).get("stage1_IPC_candidates", 0),
            "stage2_keyword_filter_results": len(narrowed_patent_ids),
            "analysis_completed": 1 if analysis_payload else 0,
        },
    }

    tracker.finalize(final_payload["pipeline_stats"])
    job_manager.store_result(job_id, final_payload)
    return final_payload

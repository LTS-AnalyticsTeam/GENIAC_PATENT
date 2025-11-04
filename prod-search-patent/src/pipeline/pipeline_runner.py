from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List
from pathlib import Path

from .config import PipelineConfig
from .cosmos_client import CosmosPatentClient
from .embedding_service import EmbeddingService
from .elasticsearch_stage1 import Stage1ElasticsearchIndexer
from .exceptions import IngestionError, PipelineStageError
from .graph_rag import GraphRAGService
from .job_manager import JobManager, JobState
from .parsing_service import parse_xml_document
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


class StageTracker:
    """Keep JobState detail mutations uniform across pipeline stages."""

    def __init__(self, job_manager: JobManager, job_id: str) -> None:
        self.job_manager = job_manager
        self.job_id = job_id

    def _detail(self) -> Dict:
        state = self.job_manager.get_state(self.job_id)
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


async def run_pipeline(
    config: PipelineConfig,
    job_manager: JobManager,
    job_id: str,
    xml_bytes: bytes,
) -> Dict:
    """Execute full ingestion pipeline and update job state along the way."""
    try:
        config.ensure_embedding_credentials()
    except RuntimeError as exc:
        raise PipelineStageError("configuration", str(exc)) from exc

    tracker = StageTracker(job_manager, job_id)

    # Stage 1: XML parsing
    tracker.start("parsing")
    parsed = parse_xml_document(xml_bytes)
    tracker.complete("parsing")

    ipc_codes = parsed["classification_ipc"]

    # Stage 2: Cosmos query
    tracker.start("cosmos_query", {"ipc_codes": ipc_codes})
    cosmos_client = CosmosPatentClient(config)
    try:
        cosmos_docs = cosmos_client.query_by_ipc_prefixes(ipc_codes, config.es_stage1_limit)
    finally:
        cosmos_client.close()

    if not cosmos_docs:
        logger.warning("Job %s: no Cosmos matches for IPC %s; falling back to XML content", job_id, ipc_codes)
        fallback_doc = {
            "id": parsed.get("patent_id") or job_id,
            "patent_id": parsed.get("patent_id") or job_id,
            "title": parsed.get("title"),
            "summary": parsed.get("summary"),
            "claim1": parsed.get("claim1"),
            "claims_text": parsed.get("claims_text"),
            "claims": parsed.get("claims"),
            "classification_ipc": ipc_codes,
            "source": {"origin": "xml_fallback"},
        }
        cosmos_docs = [fallback_doc]

        tracker.update(
            "cosmos_query",
            {
                "note": "fallback_xml_only",
                "matched_documents": 0,
            },
        )
    else:
        tracker.update(
            "cosmos_query",
            {
                "matched_documents": len(cosmos_docs),
            },
        )

    tracker.complete("cosmos_query")

    # Stage 3: Trimming to Stage1 limit
    tracker.start("trimming")
    trimmed_docs_raw = sort_and_trim(cosmos_docs, config.es_stage1_limit)

    trimmed_docs: List[Dict] = []
    for doc in trimmed_docs_raw:
        trimmed_docs.append(
            {
                "patent_id": doc.get("patent_id") or doc.get("id"),
                "title": doc.get("title") or parsed.get("title"),
                "summary": doc.get("summary") or parsed.get("summary"),
                "claim1": doc.get("claim1") or parsed.get("claim1"),
                "claims_text": doc.get("claims_text") or parsed.get("claims_text"),
                "claims": doc.get("claims") or parsed.get("claims"),
                "classification_ipc": doc.get("classification_ipc") or ipc_codes,
                "source": doc,
            }
        )
    tracker.complete("trimming")

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
    stage2.upsert_graph(stage2_docs)
    tracker.update(
        "stage2_indexing",
        {
            "ingested_documents": len(stage2_docs),
            "source": "vector_search" if vector_docs else "cosmos_trimmed",
        },
    )
    tracker.complete("stage2_indexing")

    # Stage 8: Graph-RAG ranking
    tracker.start("graph_rag")
    rag = GraphRAGService(config)
    top_results = rag.top_k([doc.get("patent_id") for doc in stage2_docs], k=10)
    rag.close()
    stage2.close()
    tracker.complete("graph_rag")

    final_payload = {
        "results": top_results,
        "pipeline_stats": {
            "trimmed": len(trimmed_docs),
            "stage1_indexed": success,
            "stage1_failed": failures,
            "stage1_reused": reused_embeddings,
            "stage1_embedded": len(docs_to_embed),
            "vector_search_results": len(vector_docs),
            "vector_queries": len(generated_queries),
        },
    }

    tracker.finalize(final_payload["pipeline_stats"])
    job_manager.store_result(job_id, final_payload)
    return final_payload

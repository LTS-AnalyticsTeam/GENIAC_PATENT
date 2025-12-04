from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
from .web_search_service import search_web_references

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


def _to_ipc_prefixes(ipc_list: Sequence[str]) -> List[str]:
    prefixes: List[str] = []
    for ipc in ipc_list:
        ipc_str = ipc if isinstance(ipc, str) else ""
        if ipc_str and len(ipc_str) >= 4:
            s = ipc_str.upper()
            if "/" in s:
                s = s.split("/", 1)[0]
            cleaned = re.sub(r"\\s+", "", s)[:5]
            if cleaned and cleaned not in prefixes:
                prefixes.append(cleaned)
    return prefixes


def _extract_section_texts(description: Dict | None) -> List[Dict[str, str]]:
    """Extract known sections (technical-field, background-art, summary-of-invention, industrial-applicability)."""
    sections: List[Dict[str, str]] = []
    if not description:
        return sections
    for key in ["technical-field", "background-art", "summary-of-invention", "industrial-applicability"]:
        value = description.get(key)
        if not value:
            continue
        text = ""
        if isinstance(value, list):
            parts = []
            for entry in value:
                if isinstance(entry, dict):
                    parts.append(str(entry.get("text") or ""))
                else:
                    parts.append(str(entry))
            text = " ".join([p for p in parts if p]).strip()
        elif isinstance(value, dict):
            text = str(value.get("text") or "")
        else:
            text = str(value)
        text = (text or "").strip()
        if text:
            sections.append({"type": key, "text": text})
    return sections


def _extract_claim_texts(doc: Dict | None, fallback_claim1: str | None = None) -> List[str]:
    claims: List[str] = []
    if doc:
        raw_claims = doc.get("claims") or []
        if isinstance(raw_claims, list):
            for entry in raw_claims:
                if isinstance(entry, dict):
                    text = entry.get("text") or entry.get("claim_text")
                    if text:
                        claims.append(str(text))
                elif isinstance(entry, str):
                    claims.append(entry)
    if not claims and fallback_claim1:
        claims.append(fallback_claim1)
    return claims[:3]


def _extract_citation_texts(description: Dict | None) -> List[str]:
    citations: List[str] = []
    if not description:
        return citations
    citation_entries = description.get("citation-list") or []
    if isinstance(citation_entries, list):
        for entry in citation_entries:
            if isinstance(entry, dict):
                txt = entry.get("text")
                if txt:
                    citations.append(str(txt))
            elif isinstance(entry, str):
                citations.append(entry)
    return citations


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
            # 全ヒットを UI で一覧表示できるように保持
            "vector_patent_ids": [doc.get("patent_id") for doc in vector_docs],
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

    alpha_id = parsed.get("patent_id") or alpha_source_json.get("bibliographic", {}).get("publication", {}).get("doc_number")
    alpha_description = alpha_source_json.get("description") if isinstance(alpha_source_json, dict) else None
    alpha_sections = _extract_section_texts(alpha_description)
    if not alpha_sections:
        # フォールバック: タイトル・サマリーをセクションとして扱う
        if parsed.get("title"):
            alpha_sections.append({"type": "technical-field", "text": str(parsed.get("title"))})
        if parsed.get("summary"):
            alpha_sections.append({"type": "summary-of-invention", "text": str(parsed.get("summary"))})
    alpha_claims = _extract_claim_texts(alpha_source_json, fallback_claim1=parsed.get("claim1"))
    alpha_citations = _extract_citation_texts(alpha_description)
    alpha_ipc_codes = parsed.get("classification_ipc") or []

    stage2_graph_docs: List[Dict] = []
    if alpha_id:
        stage2_graph_docs.append(
            {
                "patent_id": alpha_id,
                "title": parsed.get("title"),
                "summary": parsed.get("summary"),
                "classification_ipc": alpha_ipc_codes,
                "ipc_prefixes": _to_ipc_prefixes(alpha_ipc_codes),
                "sections": alpha_sections,
                "claims": alpha_claims,
                "citation_texts": alpha_citations,
                "vector_score": None,
            }
        )

    for doc in stage2_docs:
        pid = doc.get("patent_id")
        if not pid:
            continue
        cosmos_doc = stage2_cosmos_map.get(pid) or {}
        biblio = cosmos_doc.get("bibliographic") or {}
        classification = _extract_classification(cosmos_doc)
        description = cosmos_doc.get("description") if isinstance(cosmos_doc, dict) else None
        sections = _extract_section_texts(description)
        claims = _extract_claim_texts(cosmos_doc, fallback_claim1=doc.get("claim1"))
        citations = _extract_citation_texts(description)

        graph_doc = {
            "patent_id": pid,
            "title": cosmos_doc.get("title") or doc.get("title"),
            "summary": cosmos_doc.get("abstract") or cosmos_doc.get("summary") or doc.get("summary"),
            "classification_ipc": classification or doc.get("classification_ipc"),
            "ipc_prefixes": _to_ipc_prefixes(classification or doc.get("classification_ipc") or []),
            "sections": sections,
            "claims": claims,
            "citation_texts": citations,
            "vector_score": doc.get("vector_score"),
            "publication": biblio.get("publication"),
        }
        stage2_graph_docs.append(graph_doc)

    stage2_graph_docs = _deduplicate_by_patent_id(stage2_graph_docs)
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
    top_results = rag.top_k([doc.get("patent_id") for doc in stage2_docs], alpha_id=alpha_id or "", k=30)
    top_results = _deduplicate_by_patent_id(top_results)
    tracker.update(
        "graph_rag",
        {
            "graph_results": len(top_results),
            "graph_top_patent_ids": [entry.get("patent_id") for entry in top_results[:10]],
            "graph_top_results": [
                {
                    "patent_id": entry.get("patent_id"),
                    "title": entry.get("title"),
                    "graph_score": entry.get("graph_score"),
                    "vector_score": entry.get("vector_score"),
                }
                for entry in top_results[:10]
            ],
            # 全件（k件）を UI で一覧表示できるように保持
            "graph_patent_results": [
                {
                    "patent_id": entry.get("patent_id"),
                    "title": entry.get("title"),
                    "graph_score": entry.get("graph_score"),
                    "vector_score": entry.get("vector_score"),
                }
                for entry in top_results
                if entry.get("patent_id")
            ],
        },
    )
    rag.close()
    stage2.close()
    tracker.complete("graph_rag")

    # Stage 9: Web search for additional references
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

    # Stage 10: Merge and select final 10 candidates from 30 patents + web results
    tracker.start("final_selection")
    combined_results = list(top_results) + list(web_results)
    logger.info(f"Job %s: Combined results: {len(top_results)} patents + {len(web_results)} web = {len(combined_results)} total", job_id)

    # If combined results <= 10, use all
    if len(combined_results) <= 10:
        final_candidates = combined_results
        logger.info(f"Job %s: Using all {len(final_candidates)} combined results (<=10)", job_id)
    else:
        # Use LLM to select top 10 from combined results
        try:
            import os
            from openai import AzureOpenAI
            import json as json_module

            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            )
            deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT")

            # Prepare candidate list for LLM
            candidates_for_llm = []
            for idx, candidate in enumerate(combined_results):
                candidates_for_llm.append({
                    "index": idx,
                    "patent_id": candidate.get("patent_id"),
                    "title": candidate.get("title", ""),
                    "summary": (candidate.get("summary") or "")[:500],
                    "is_web_result": candidate.get("is_web_result", False),
                    "source": candidate.get("source", "patent"),
                })

            user_prompt = f"""あなたは特許の新規性・進歩性を判断する専門家です。

【タスク】
以下の特許出願に対して、最も関連性の高い先行技術候補を10件選択してください。

【出願内容】
タイトル: {parsed.get("title", "")}
要約: {parsed.get("summary", "")[:500]}
請求項1: {claim1_text[:500]}

【候補一覧】
{json_module.dumps(candidates_for_llm, ensure_ascii=False, indent=2)}

【選択基準】
1. 技術的な関連性が高いもの
2. 新規性・進歩性の判断に重要なもの
3. 特許データベースとWeb検索結果をバランスよく含める（可能であれば）

【出力形式】
以下のJSON形式で、選択した候補のindexリストを返してください：
{{"selected_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}}

必ず有効なJSONのみを出力し、余計なテキストは含めないでください。
"""

            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
            )

            content = resp.choices[0].message.content or ""
            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_text = content[start:end+1]
                data = json_module.loads(json_text)
                selected_indices = data.get("selected_indices", [])

                # Validate and select
                final_candidates = []
                for idx in selected_indices[:10]:
                    if 0 <= idx < len(combined_results):
                        final_candidates.append(combined_results[idx])

                if len(final_candidates) < 10:
                    # Fill with remaining candidates if LLM didn't select enough
                    for idx, candidate in enumerate(combined_results):
                        if idx not in selected_indices and len(final_candidates) < 10:
                            final_candidates.append(candidate)

                logger.info(f"Job %s: LLM selected {len(final_candidates)} final candidates", job_id)
            else:
                raise ValueError("No valid JSON found in LLM response")

        except Exception as e:
            logger.warning(f"Job %s: Final selection with LLM failed: %s, using simple selection", job_id, e)
            # Fallback: simple selection (first 10 from combined)
            final_candidates = combined_results[:10]

    tracker.update("final_selection", {
        "combined_count": len(combined_results),
        "final_count": len(final_candidates),
        "patent_count": sum(1 for c in final_candidates if not c.get("is_web_result")),
        "web_count": sum(1 for c in final_candidates if c.get("is_web_result")),
    })
    tracker.complete("final_selection")

    # Use final_candidates for analysis instead of top_results
    top_results = final_candidates

    tracker.start("analysis", {"analysis_candidates": len(top_results)})
    analysis_service = AnalysisService()
    analysis_payload: Dict | None = None
    missing_candidates: List[str] = []

    if top_results:
        # Separate patent results and web results
        patent_results = [entry for entry in top_results if not entry.get("is_web_result")]
        web_only_results = [entry for entry in top_results if entry.get("is_web_result")]

        # Fetch Cosmos data only for patent results
        analysis_cosmos = CosmosPatentClient(config)
        try:
            candidate_docs_map = analysis_cosmos.fetch_by_patent_ids(
                [entry["patent_id"] for entry in patent_results if entry.get("patent_id")]
            )
        finally:
            analysis_cosmos.close()

        ordered_candidates: List[Dict] = []

        # Process patent results (need Cosmos data)
        for entry in patent_results:
            pid = entry.get("patent_id")
            if not pid:
                continue
            candidate_doc = candidate_docs_map.get(pid)
            if candidate_doc:
                ordered_candidates.append(candidate_doc)
            else:
                missing_candidates.append(pid)

        # For web results, create simplified candidate_json
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
        "results": top_results,
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
            "web_search_results": len(web_results),
            "analysis_candidates": len(top_results),
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

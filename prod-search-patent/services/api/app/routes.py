from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from pipeline import IngestionError, JobManager, PipelineConfig
from pipeline.job_manager import JobState

from .models import GraphResult, IngestResponse, JobCancelResponse, JobStatusResponse, PipelineResultResponse, KeywordSearchResultResponse, PatentIdTestRequest, WebSearchResultResponse, WebSearchDetail
from .deps import get_job_manager, get_pipeline_config

logger = logging.getLogger(__name__)


router = APIRouter()


def _initial_job_detail() -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "current_stage": "queued",
        "stages": {
            "queued": {
                "status": "completed",
                "started_at": now,
                "completed_at": now,
            }
        },
        "completed_stages": ["queued"],
        "created_at": now,
    }


@router.post("/ingest-json", response_model=IngestResponse, status_code=202)
async def ingest_patent_json(
    request: Request,
    file: UploadFile = File(...),
    job_manager: JobManager = Depends(get_job_manager),
    config: PipelineConfig = Depends(get_pipeline_config),
) -> IngestResponse:
    json_bytes = await file.read()
    if not json_bytes:
        raise HTTPException(status_code=400, detail="Empty JSON payload")

    initial_detail = _initial_job_detail()
    job_id = job_manager.create_job(JobState(status="queued", detail=initial_detail))
    job_manager.store_payload(job_id, {"json": json_bytes.decode("utf-8")})
    job_manager.enqueue_job(job_id)

    status_url = str(request.url_for("get_job_status", job_id=job_id))
    result_url = str(request.url_for("get_job_result", job_id=job_id))
    return IngestResponse(job_id=job_id, status_url=status_url, result_url=result_url, detail=initial_detail)


@router.post("/ingest-text", response_model=IngestResponse, status_code=202)
async def ingest_patent_text(
    request: Request,
    file: UploadFile = File(...),
    job_manager: JobManager = Depends(get_job_manager),
    config: PipelineConfig = Depends(get_pipeline_config),
) -> IngestResponse:
    text_bytes = await file.read()
    if not text_bytes:
        raise HTTPException(status_code=400, detail="Empty text payload")

    initial_detail = _initial_job_detail()
    job_id = job_manager.create_job(JobState(status="queued", detail=initial_detail))
    job_manager.store_payload(job_id, {"text": text_bytes.decode("utf-8")})
    job_manager.enqueue_job(job_id)

    status_url = str(request.url_for("get_job_status", job_id=job_id))
    result_url = str(request.url_for("get_job_result", job_id=job_id))
    return IngestResponse(job_id=job_id, status_url=status_url, result_url=result_url, detail=initial_detail)


@router.post("/ingest-patent-id", response_model=IngestResponse, status_code=202)
async def ingest_patent_id(
    request: Request,
    payload: PatentIdTestRequest,
    job_manager: JobManager = Depends(get_job_manager),
    config: PipelineConfig = Depends(get_pipeline_config),
) -> IngestResponse:
    """テスト用エンドポイント: 特許番号を指定してCosmosDBから取得し、パイプラインを実行"""
    patent_id = payload.patent_id.strip()
    if not patent_id:
        raise HTTPException(status_code=400, detail="Patent ID is required")

    initial_detail = _initial_job_detail()
    job_id = job_manager.create_job(JobState(status="queued", detail=initial_detail))
    job_manager.store_payload(job_id, {"patent_id": patent_id})
    job_manager.enqueue_job(job_id)

    status_url = str(request.url_for("get_job_status", job_id=job_id))
    result_url = str(request.url_for("get_job_result", job_id=job_id))
    return IngestResponse(job_id=job_id, status_url=status_url, result_url=result_url, detail=initial_detail)



@router.get("/status/{job_id}", name="get_job_status", response_model=JobStatusResponse)
def get_status(job_id: str, job_manager: JobManager = Depends(get_job_manager)) -> JobStatusResponse:
    state = job_manager.get_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(job_id=job_id, status=state.status, detail=state.detail)


@router.get("/result/{job_id}", name="get_job_result", response_model=PipelineResultResponse, response_model_exclude_unset=False, response_model_exclude_none=False)
def get_result(job_id: str, job_manager: JobManager = Depends(get_job_manager)) -> PipelineResultResponse:
    state = job_manager.get_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.status != "completed":
        raise HTTPException(status_code=202, detail="Job still in progress")
    result_payload = job_manager.fetch_result(job_id)
    if not result_payload:
        raise HTTPException(status_code=404, detail="Result unavailable")
    results = [GraphResult(**item) for item in result_payload.get("results", [])]
    fusion_results = [GraphResult(**item) for item in result_payload.get("fusion_results", [])]
    web_details_raw = result_payload.get("web_search_details", [])
    web_details = [WebSearchDetail(**item) for item in web_details_raw]
    return PipelineResultResponse(
        job_id=job_id,
        completed_at=datetime.now(timezone.utc),
        results=results,
        fusion_results=fusion_results,
        pipeline_stats=result_payload.get("pipeline_stats", {}),
        web_search_details=web_details,
    )


@router.post("/cancel/{job_id}", response_model=JobCancelResponse, status_code=202)
def cancel_job(job_id: str, job_manager: JobManager = Depends(get_job_manager)) -> JobCancelResponse:
    state = job_manager.get_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.status in {"completed", "failed", "cancelled"}:
        return JobCancelResponse(job_id=job_id, status=state.status, queue_entries_removed=0)

    removed = job_manager.cancel_job(job_id, reason="ユーザー操作によりキャンセルされました")
    return JobCancelResponse(job_id=job_id, status="cancelled", queue_entries_removed=removed)


@router.get("/keyword-search-result/{job_id}", response_model=KeywordSearchResultResponse)
def get_keyword_search_result(job_id: str, job_manager: JobManager = Depends(get_job_manager)) -> KeywordSearchResultResponse:
    """キーワード検索結果（10,000件の特許番号リスト）を取得"""
    state = job_manager.get_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    result_payload = job_manager.fetch_result(job_id)
    if not result_payload:
        raise HTTPException(status_code=404, detail="Keyword search result not available")

    patent_ids = result_payload.get("keyword_search_results", [])
    pipeline_stats = result_payload.get("pipeline_stats", {})

    # search_results (with scores) を取得
    search_results_raw = result_payload.get("search_results", [])
    from app.models import SearchResultItem
    search_results = []
    if search_results_raw:
        try:
            search_results = [
                SearchResultItem(doc_number=item["doc_number"], score=item["score"])
                for item in search_results_raw
            ]
            logger.info(f"Loaded {len(search_results)} search results with scores")
        except Exception as e:
            logger.error(f"Failed to parse search_results: {e}")
            # Fallback: search_results が空のままになる

    return KeywordSearchResultResponse(
        job_id=job_id,
        patent_ids=patent_ids,
        search_results=search_results,
        pipeline_stats=pipeline_stats,
        total_count=len(patent_ids)
    )


@router.get("/web-search-result/{job_id}", response_model=WebSearchResultResponse)
def get_web_search_result(job_id: str, job_manager: JobManager = Depends(get_job_manager)) -> WebSearchResultResponse:
    """Web検索結果（タイトルとURLを含む）を取得"""
    state = job_manager.get_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    result_payload = job_manager.fetch_result(job_id)
    if not result_payload:
        raise HTTPException(status_code=404, detail="Web search result not available")

    web_details_raw = result_payload.get("web_search_details", [])
    web_details = [WebSearchDetail(**item) for item in web_details_raw]

    return WebSearchResultResponse(
        job_id=job_id,
        web_results=web_details,
        total_count=len(web_details)
    )


@router.get("/healthz")
async def healthz(config: PipelineConfig = Depends(get_pipeline_config)) -> Dict[str, Any]:
    es = Elasticsearch(config.es_host)
    es_ok = es.ping()

    with GraphDatabase.driver(
        config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
    ) as driver:
        try:
            driver.verify_connectivity()
            neo4j_ok = True
        except Exception:
            neo4j_ok = False

    redis_ok = True
    try:
        JobManager(config.redis_url).client.ping()
    except Exception:
        redis_ok = False

    async with httpx.AsyncClient(timeout=5.0) as client:
        vectorizer_ok = False
        try:
            resp = await client.get(f"{config.vectorizer_url}/health")
            vectorizer_ok = resp.status_code == 200
        except httpx.HTTPError:
            vectorizer_ok = False

    status = es_ok and neo4j_ok and redis_ok and vectorizer_ok
    if not status:
        raise HTTPException(
            status_code=503,
            detail={
                "elasticsearch": es_ok,
                "neo4j": neo4j_ok,
                "redis": redis_ok,
                "vectorizer": vectorizer_ok,
            },
        )
    return {
        "status": "ok",
        "elasticsearch": es_ok,
        "neo4j": neo4j_ok,
        "redis": redis_ok,
        "vectorizer": vectorizer_ok,
    }

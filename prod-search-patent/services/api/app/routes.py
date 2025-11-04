from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import httpx
from elasticsearch import Elasticsearch
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from neo4j import GraphDatabase

from pipeline import IngestionError, JobManager, PipelineConfig
from pipeline.job_manager import JobState

from .models import GraphResult, IngestResponse, JobStatusResponse, PipelineResultResponse
from .deps import get_job_manager, get_pipeline_config

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


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest_patent(
    request: Request,
    file: UploadFile = File(...),
    job_manager: JobManager = Depends(get_job_manager),
    config: PipelineConfig = Depends(get_pipeline_config),
) -> IngestResponse:
    xml_bytes = await file.read()
    if not xml_bytes:
        raise HTTPException(status_code=400, detail="Empty XML payload")

    initial_detail = _initial_job_detail()
    job_id = job_manager.create_job(JobState(status="queued", detail=initial_detail))
    job_manager.store_payload(job_id, {"xml": xml_bytes.decode("utf-8")})
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


@router.get("/result/{job_id}", name="get_job_result", response_model=PipelineResultResponse)
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
    return PipelineResultResponse(
        job_id=job_id,
        completed_at=datetime.now(timezone.utc),
        results=results,
        pipeline_stats=result_payload.get("pipeline_stats", {}),
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

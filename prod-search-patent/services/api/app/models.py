from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl, BaseModel, Field


class IngestResponse(BaseModel):
    job_id: str = Field(..., description="Background job identifier")
    status_url: AnyHttpUrl
    result_url: AnyHttpUrl
    detail: Dict[str, Any] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    detail: Dict[str, Any] = Field(default_factory=dict)


class GraphResult(BaseModel):
    patent_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    classification_ipc: Optional[List[str]] = None
    graph_score: Optional[float] = None
    analysis_status: Optional[str] = None
    analysis_error: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None


class PipelineResultResponse(BaseModel):
    job_id: str
    completed_at: datetime
    results: List[GraphResult]
    pipeline_stats: Dict[str, Any]


class JobCancelResponse(BaseModel):
    job_id: str
    status: str
    queue_entries_removed: int = Field(0, description="Number of queued tasks removed")


class KeywordSearchResultResponse(BaseModel):
    job_id: str
    patent_ids: List[str] = Field(default_factory=list)
    pipeline_stats: Dict[str, Any] = Field(default_factory=dict)
    total_count: int = 0

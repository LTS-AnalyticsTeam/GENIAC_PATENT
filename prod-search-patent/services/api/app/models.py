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
    model_config = {
        "use_enum_values": True,
        "json_schema_extra": {"example": {"patent_id": "123"}},
    }

    patent_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    classification_ipc: Optional[List[str]] = None
    graph_score: Optional[float] = None
    analysis_status: Optional[str] = None
    analysis_error: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    # Fields for Web search results and assessments
    is_web_result: bool = Field(default=False)
    source_url: Optional[str] = Field(default=None)
    pub_number: Optional[str] = Field(default=None)
    assessments: List[Dict[str, Any]] = Field(default_factory=list)


class PipelineResultResponse(BaseModel):
    job_id: str
    completed_at: datetime
    results: List[GraphResult]
    pipeline_stats: Dict[str, Any]
    web_search_details: List[WebSearchDetail] = Field(default_factory=list)


class JobCancelResponse(BaseModel):
    job_id: str
    status: str
    queue_entries_removed: int = Field(0, description="Number of queued tasks removed")


class SearchResultItem(BaseModel):
    doc_number: str = Field(..., description="Patent document number")
    score: float = Field(..., description="Keyword search score")


class KeywordSearchResultResponse(BaseModel):
    job_id: str
    patent_ids: List[str] = Field(default_factory=list)
    search_results: List[SearchResultItem] = Field(default_factory=list)
    pipeline_stats: Dict[str, Any] = Field(default_factory=dict)
    total_count: int = 0


class PatentIdTestRequest(BaseModel):
    patent_id: str = Field(..., description="Patent ID to fetch from Cosmos DB for testing")


class WebSearchDetail(BaseModel):
    patent_id: str = Field(..., description="Unique identifier for the web result")
    title: str = Field(default="", description="Title of the web result")
    source_url: str = Field(default="", description="URL of the web result")
    summary: str = Field(default="", description="Summary/abstract of the web result")


class WebSearchResultResponse(BaseModel):
    job_id: str
    web_results: List[WebSearchDetail] = Field(default_factory=list)
    total_count: int = 0

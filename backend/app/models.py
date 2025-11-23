from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Snippet(BaseModel):
    section: str
    claim_no: int = Field(..., ge=0)
    text: str
    offset: int = Field(..., ge=0)
    len: int = Field(..., ge=0)
    match_type: str
    score: float = Field(..., ge=0.0, le=1.0)


class Explanation(BaseModel):
    summary: str
    why_match: List[str]
    examiner_hints: List[str]


class Candidate(BaseModel):
    doc_id: str
    title: str
    pub_number: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=0)
    ipc: List[str] = Field(default_factory=list)
    score: float = Field(..., ge=0.0, le=1.0)
    snippets: List[Snippet]
    explanation: Explanation
    source_url: Optional[str] = None


class AlphaInfo(BaseModel):
    title: str
    pub_number: str
    claim1: str
    claims_rest: List[str]


class RunLimits(BaseModel):
    max_total: int
    Ay_min: int = Field(..., ge=0)


class AnalysisResponse(BaseModel):
    run_id: str
    alpha: AlphaInfo
    Ax: Candidate
    Ay: List[Candidate]
    limits: RunLimits

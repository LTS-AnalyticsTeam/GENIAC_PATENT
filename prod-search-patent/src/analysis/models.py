from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# ---- 基本スキーマ ----

SectionName = Literal[
    "technical-field",
    "background-art",
    "citation-list",
    "summary-of-invention",
    "description-of-drawings",
    "description-of-embodiments",
    "advantageous-effects",
]

class DescriptionItem(BaseModel):
    num: Optional[str] = None
    text: str

class Description(BaseModel):
    # alias を使うので JSON のハイフン名をそのまま受け取れる
    technical_field: List[DescriptionItem] = Field(default_factory=list, alias="technical-field")
    background_art: List[DescriptionItem] = Field(default_factory=list, alias="background-art")
    citation_list: List[Dict] = Field(default_factory=list, alias="citation-list")
    summary_of_invention: List[Dict] = Field(default_factory=list, alias="summary-of-invention")
    description_of_drawings: List[DescriptionItem] = Field(default_factory=list, alias="description-of-drawings")
    description_of_embodiments: List[Dict] = Field(default_factory=list, alias="description-of-embodiments")
    advantageous_effects: List[Dict] = Field(default_factory=list, alias="advantageous-effects")

class PatentDoc(BaseModel):
    title: Optional[str] = None
    pub_number: Optional[str] = None
    claim1: str
    description: Optional[Description] = None

class Evidence(BaseModel):
    section: str
    quote: str
    why: str
    offset: Optional[int] = None
    length: Optional[int] = None
    alpha_fragment: Optional[str] = None
    candidate_quote: Optional[str] = None

class ClaimAssessment(BaseModel):
    claim_no: int
    novelty: Literal["denied", "uncertain", "supported"]
    inventive_step: Optional[Literal["denied", "uncertain", "supported"]] = None
    evidence: List[Evidence] = Field(default_factory=list)
    examiner_hints: List[str] = Field(default_factory=list)

class AssessmentCandidate(BaseModel):
    doc_id: str
    title: str
    pub_number: Optional[str] = None
    score: float = Field(..., ge=0.0, le=1.0)
    assessments: List[ClaimAssessment]
    ipc: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    source_url: Optional[str] = None
    is_web_result: bool = False
    year: Optional[int] = Field(default=None, ge=0)

class AlphaInfo(BaseModel):
    title: str
    pub_number: Optional[str] = None
    claim1: str
    claims_rest: List[str]

class RunLimits(BaseModel):
    max_total: int
    Ay_min: int = Field(..., ge=0)

class AnalysisResponse(BaseModel):
    run_id: str
    alpha: AlphaInfo
    claim1_candidates: List[AssessmentCandidate]
    rest_claim_candidates: List[AssessmentCandidate]
    limits: RunLimits

# ---- Legacy types (for stub generators and internal heuristics) ----

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

class Span(BaseModel):
    start: int
    end: int

class EvidenceHit(BaseModel):
    section: SectionName
    num: Optional[str] = None
    text: str
    spans: List[Span] = Field(default_factory=list)
    support: int = 0
    score: float = 0.0

class ModelVote(BaseModel):
    model_name: str
    score: float

class EnsembleDecision(BaseModel):
    votes: List[ModelVote]
    final_score: float
    rationale: List[str]

class JudgmentBasis(BaseModel):
    reasoning: List[str]
    decision: EnsembleDecision

class Candidate(BaseModel):
    doc_id: str
    title: str
    pub_number: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=0)
    ipc: List[str] = Field(default_factory=list)
    score: float = Field(..., ge=0.0, le=1.0)
    snippets: List[Snippet] = Field(default_factory=list)
    explanation: Explanation
    source_url: Optional[str] = None
    evidence_hits: List[EvidenceHit] = Field(default_factory=list)
    judgment_basis: Optional[JudgmentBasis] = None

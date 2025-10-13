from __future__ import annotations

from uuid import uuid4

from ..data.dummy import build_dummy_alpha, build_dummy_ax, build_dummy_ay_candidates
from ..models import AnalysisResponse, RunLimits


class AnalyzerService:
    """Service layer responsible for orchestrating mock prior-art analysis."""

    def analyze(self) -> AnalysisResponse:
        run_id = str(uuid4())
        alpha = build_dummy_alpha()
        ax = build_dummy_ax()
        ay_candidates = build_dummy_ay_candidates()
        limits = RunLimits(max_total=len(ay_candidates), Ay_min=1)
        return AnalysisResponse(run_id=run_id, alpha=alpha, Ax=ax, Ay=ay_candidates, limits=limits)

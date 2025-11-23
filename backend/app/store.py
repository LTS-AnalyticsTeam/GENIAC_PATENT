from __future__ import annotations

from typing import Dict, Optional

from .models import AnalysisResponse


class RunStore:
    """In-memory run storage (placeholder for persistence)."""

    def __init__(self) -> None:
        self._store: Dict[str, AnalysisResponse] = {}

    def save(self, response: AnalysisResponse) -> None:
        self._store[response.run_id] = response

    def get(self, run_id: str) -> Optional[AnalysisResponse]:
        return self._store.get(run_id)

    def clear(self) -> None:
        self._store.clear()


run_store = RunStore()

from __future__ import annotations

from typing import Optional


class IngestionError(Exception):
    """Custom error raised when ingestion cannot proceed (client-visible)."""

    def __init__(self, message: str, *, status_code: int = 400, detail: Optional[dict] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail or {}


class PipelineStageError(Exception):
    """Generic failure inside pipeline stages (server-side)."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


class JobCancelledError(Exception):
    """Raised when a job cancellation request is observed during processing."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Job {job_id} cancelled")
        self.job_id = job_id

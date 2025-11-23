from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

import redis


@dataclass
class JobState:
    status: str
    detail: Dict[str, object] = field(default_factory=dict)


class JobManager:
    """Simple Redis-backed job state store."""

    def __init__(self, redis_url: str) -> None:
        self.client = redis.Redis.from_url(redis_url)
        self.queue_key = "pipeline:jobs"

    def _cancel_flag_key(self, job_id: str) -> str:
        return f"job_cancelled:{job_id}"

    def create_job(self, initial: Optional[JobState] = None) -> str:
        job_id = uuid.uuid4().hex
        self.set_state(job_id, initial or JobState(status="queued"))
        return job_id

    def set_state(self, job_id: str, state: JobState) -> None:
        payload = {"status": state.status, "detail": state.detail}
        self.client.set(f"job:{job_id}", json.dumps(payload), ex=60 * 60 * 24)

    def get_state(self, job_id: str) -> Optional[JobState]:
        data = self.client.get(f"job:{job_id}")
        if not data:
            return None
        payload = json.loads(data)
        return JobState(status=payload.get("status", "unknown"), detail=payload.get("detail", {}))

    def store_payload(self, job_id: str, payload: dict) -> None:
        self.client.set(f"job_payload:{job_id}", json.dumps(payload), ex=60 * 60)

    def fetch_payload(self, job_id: str) -> Optional[dict]:
        value = self.client.get(f"job_payload:{job_id}")
        if not value:
            return None
        return json.loads(value)

    def store_result(self, job_id: str, result: dict) -> None:
        self.client.set(f"job_result:{job_id}", json.dumps(result), ex=60 * 60 * 24)

    def fetch_result(self, job_id: str) -> Optional[dict]:
        value = self.client.get(f"job_result:{job_id}")
        if not value:
            return None
        return json.loads(value)

    def enqueue_job(self, job_id: str, payload: Optional[dict] = None) -> None:
        message = {"job_id": job_id, "payload": payload or {}}
        self.client.lpush(self.queue_key, json.dumps(message))

    def dequeue_job(self, timeout: int = 5) -> Optional[dict]:
        item = self.client.brpop(self.queue_key, timeout=timeout)
        if not item:
            return None
        _, raw = item
        return json.loads(raw)

    def is_cancelled(self, job_id: str) -> bool:
        try:
            return bool(self.client.exists(self._cancel_flag_key(job_id)))
        except Exception:
            return False

    def remove_job_from_queue(self, job_id: str) -> int:
        """Remove queued entries matching a job_id. Returns number removed."""
        removed = 0
        try:
            queue_items = self.client.lrange(self.queue_key, 0, -1)
        except Exception:
            return removed

        for raw in queue_items:
            try:
                payload = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get("job_id") != job_id:
                continue
            removed += self.client.lrem(self.queue_key, 0, raw)
        return removed

    def cancel_job(self, job_id: str, reason: Optional[str] = None) -> int:
        """Mark job as cancelled and drop it from the queue."""
        state = self.get_state(job_id)
        detail: Dict[str, object] = {}
        if state and state.detail:
            detail = dict(state.detail)
        detail["current_stage"] = "cancelled"
        detail["cancelled_at"] = datetime.now(timezone.utc).isoformat()
        if reason:
            detail["message"] = reason
        self.client.set(self._cancel_flag_key(job_id), "1", ex=60 * 60 * 24)
        self.set_state(job_id, JobState(status="cancelled", detail=detail))
        return self.remove_job_from_queue(job_id)

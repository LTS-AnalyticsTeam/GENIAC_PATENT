from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
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

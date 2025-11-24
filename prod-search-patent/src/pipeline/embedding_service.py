from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from typing import Deque, Iterable, List, Optional, Tuple

from openai import AzureOpenAI

from .config import PipelineConfig
from .exceptions import PipelineStageError

logger = logging.getLogger(__name__)

# AOAI embedding rate limits (per minute)
EMBEDDING_TOKENS_PER_MINUTE = 350_000
EMBEDDING_REQUESTS_PER_MINUTE = 2_100

try:  # Optional for better token estimates; falls back if unavailable
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - best effort only
    _ENCODING = None


def _estimate_tokens(text: str) -> int:
    """Rough token estimator; overestimates to stay under limits."""

    if _ENCODING is not None:
        try:
            return max(1, len(_ENCODING.encode(text)))
        except Exception:  # pragma: no cover - fall back if encoding fails
            pass

    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii_chars = len(text) - ascii_chars
    # Assume ASCII is ~4 chars/token, non-ASCII closer to 1 char/token
    approx = math.ceil(ascii_chars / 4) + math.ceil(non_ascii_chars * 1.3)
    return max(1, approx)


def _estimate_request_tokens(texts: List[str]) -> int:
    return sum(_estimate_tokens(text or "") for text in texts) + max(1, len(texts))


class _RateLimitReservation:
    def __init__(self, limiter: "EmbeddingRateLimiter", reservation_id: int, tokens: int) -> None:
        self.limiter = limiter
        self.reservation_id = reservation_id
        self.tokens = tokens

    async def adjust(self, tokens: int) -> None:
        self.tokens = tokens
        await self.limiter.update_tokens(self.reservation_id, tokens)

    async def cancel(self) -> None:
        await self.limiter.cancel(self.reservation_id)


class EmbeddingRateLimiter:
    """Coarse sliding-window limiter for AOAI embedding endpoints."""

    def __init__(
        self,
        max_tokens_per_minute: int = EMBEDDING_TOKENS_PER_MINUTE,
        max_requests_per_minute: int = EMBEDDING_REQUESTS_PER_MINUTE,
        window_seconds: int = 60,
    ) -> None:
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_minute = max_requests_per_minute
        self.window_seconds = window_seconds
        self._events: Deque[Tuple[int, float, int]] = deque()
        self._lock = asyncio.Lock()
        self._next_id = 0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][1] < cutoff:
            self._events.popleft()

    def _current_usage(self) -> Tuple[int, int]:
        total_tokens = sum(event[2] for event in self._events)
        return total_tokens, len(self._events)

    def _compute_wait(self, tokens: int, now: float, current_tokens: int) -> float:
        token_wait = 0.0
        request_wait = 0.0

        projected_tokens = current_tokens + tokens
        if projected_tokens > self.max_tokens_per_minute:
            excess = projected_tokens - self.max_tokens_per_minute
            cumulative = 0
            for _, ts, tok in self._events:
                cumulative += tok
                if cumulative >= excess:
                    token_wait = max(0.0, ts + self.window_seconds - now)
                    break

        if len(self._events) + 1 > self.max_requests_per_minute and self._events:
            oldest_ts = self._events[0][1]
            request_wait = max(0.0, oldest_ts + self.window_seconds - now)

        return max(token_wait, request_wait, 0.05)

    def _sanitize_tokens(self, tokens: int) -> int:
        return max(1, min(tokens, self.max_tokens_per_minute))

    async def acquire(self, tokens: int) -> _RateLimitReservation:
        tokens = self._sanitize_tokens(tokens)

        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)
                current_tokens, request_count = self._current_usage()

                if (
                    current_tokens + tokens <= self.max_tokens_per_minute
                    and request_count + 1 <= self.max_requests_per_minute
                ):
                    reservation_id = self._next_id
                    self._next_id += 1
                    self._events.append((reservation_id, now, tokens))
                    return _RateLimitReservation(self, reservation_id, tokens)

                wait_for = self._compute_wait(tokens, now, current_tokens)

            if wait_for > 0.0:
                logger.debug(
                    "Embedding rate limit reached; sleeping %.2fs (tokens=%s, requests=%s)",
                    wait_for,
                    tokens,
                    request_count,
                )
                await asyncio.sleep(wait_for)

    async def update_tokens(self, reservation_id: int, tokens: int) -> None:
        tokens = self._sanitize_tokens(tokens)
        async with self._lock:
            for idx, (rid, ts, _) in enumerate(self._events):
                if rid == reservation_id:
                    self._events[idx] = (rid, ts, tokens)
                    break

    async def cancel(self, reservation_id: int) -> None:
        async with self._lock:
            self._events = deque(event for event in self._events if event[0] != reservation_id)


_EMBEDDING_RATE_LIMITER = EmbeddingRateLimiter()


class EmbeddingService:
    """Wrapper to generate embeddings via Azure OpenAI or local vectorizer fallback."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        if not config.azure_openai_key or not config.azure_openai_endpoint:
            raise PipelineStageError("embedding", "Azure OpenAI credentials are not configured.")

        self._client = AzureOpenAI(
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_key,
            api_version=config.azure_openai_api_version,
        )
        self._deployment = config.azure_openai_deployment
        self._max_retries = 5
        self._base_retry_delay = 1.0
        self._max_retry_delay = 20.0

    async def aclose(self) -> None:
        return None

    async def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        payload = list(texts)
        if not payload:
            return []

        return await self._embed_via_azure(payload)

    async def _embed_via_azure(self, texts: List[str]) -> List[List[float]]:
        estimated_tokens = _estimate_request_tokens(texts)
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            reservation = await _EMBEDDING_RATE_LIMITER.acquire(estimated_tokens)
            try:
                response = await asyncio.to_thread(
                    self._client.embeddings.create,
                    model=self._deployment,
                    input=texts,
                )
                prompt_tokens = _extract_prompt_tokens(response) or estimated_tokens
                await reservation.adjust(prompt_tokens)
                embeddings = [item.embedding for item in response.data]
                return embeddings
            except Exception as exc:  # pragma: no cover
                last_error = exc
                retry_after = _retry_after_seconds(exc)
                if attempt >= self._max_retries - 1 or retry_after is None and not _should_retry(exc):
                    raise PipelineStageError("embedding", f"Azure OpenAI failure: {exc}") from exc

                delay = retry_after if retry_after is not None else min(
                    self._max_retry_delay,
                    self._base_retry_delay * (2**attempt),
                )
                logger.warning(
                    "Embedding attempt %s/%s failed (%s). Retrying in %.2fs...",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise PipelineStageError("embedding", f"Azure OpenAI failure: {last_error}")

    async def embed_documents(self, documents: List[dict], vector_field: str) -> List[dict]:
        """Generate unified embeddings (title + summary + claim1)."""
        payloads: List[str] = []

        for doc in documents:
            title = (doc.get("title") or "").strip()
            summary = (doc.get("summary") or "").strip()
            claim1 = (doc.get("claim1") or "").strip()
            claims_text = (doc.get("claims_text") or "").strip()

            combined = "\n\n".join(filter(None, [title, summary, claim1]))
            if not combined:
                combined = "\n\n".join(filter(None, [title, summary, claims_text]))
            if not combined:
                combined = doc.get("patent_id", "")
            payloads.append(combined)

        vectors = await self.embed_texts(payloads)

        enriched_docs: List[dict] = []
        for doc, vector in zip(documents, vectors):
            doc[vector_field] = vector
            enriched_docs.append(doc)
        return enriched_docs


def _extract_prompt_tokens(response: object) -> Optional[int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")

    if isinstance(prompt_tokens, int):
        return prompt_tokens
    return None


def _extract_status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "status", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status", "http_status"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)

    return None


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    headers = headers or getattr(exc, "headers", None)
    if headers:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return None
    return None


def _should_retry(exc: Exception) -> bool:
    status = _extract_status_code(exc)
    if status in {408, 429} or (status is not None and status >= 500):
        return True
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    message = str(exc).lower()
    if "timeout" in message or "timed out" in message:
        return True
    return False

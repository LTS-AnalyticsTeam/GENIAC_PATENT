from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List

from openai import AzureOpenAI

from .config import PipelineConfig
from .exceptions import PipelineStageError

logger = logging.getLogger(__name__)


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

    async def aclose(self) -> None:
        return None

    async def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        payload = list(texts)
        if not payload:
            return []

        return await self._embed_via_azure(payload)

    async def _embed_via_azure(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await asyncio.to_thread(
                self._client.embeddings.create,
                model=self._deployment,
                input=texts,
            )
        except Exception as exc:  # pragma: no cover
            raise PipelineStageError("embedding", f"Azure OpenAI failure: {exc}") from exc

        embeddings = [item.embedding for item in response.data]
        return embeddings

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

from __future__ import annotations

import logging
from typing import Any, Dict, List

import cohere

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _join_parts(parts: List[str]) -> str:
    return "\n".join([part for part in parts if part]).strip()


class CohereReranker:
    """Utility to rerank vector search hits with Cohere rerank API."""

    def __init__(self, config: PipelineConfig) -> None:
        if not config.cohere_api_key:
            raise RuntimeError("Cohere API key missing in configuration.")
        self.model = config.cohere_rerank_model or "rerank-4.0"
        if not hasattr(cohere, "ClientV2"):
            raise RuntimeError(
                "Cohere SDK is outdated. Install/upgrade with `pip install -U cohere` to enable ClientV2."
            )
        self.client = cohere.ClientV2(api_key=config.cohere_api_key)

    @staticmethod
    def _format_document(doc: Dict) -> str:
        parts: List[str] = []
        title = doc.get("title")
        summary = doc.get("summary")
        claim1 = doc.get("claim1")
        if title:
            parts.append(f"Title: {title}")
        if summary:
            parts.append(f"Summary: {summary}")
        if claim1:
            parts.append(f"Claim1: {claim1}")
        formatted = _join_parts(parts)
        if formatted:
            return formatted
        patent_id = doc.get("patent_id")
        return f"Patent {patent_id}" if patent_id else "Patent candidate with no text"

    def rerank(self, query_text: str, documents: List[Dict], top_n: int = 30) -> List[Dict]:
        if not documents:
            return []
        limit = min(top_n, len(documents))
        doc_texts = [self._format_document(doc) for doc in documents]
        try:
            response = self.client.rerank(
                model=self.model,
                query=query_text,
                documents=doc_texts,
                top_n=limit,
            )
            results = response.results
        except Exception:
            logger.exception("Cohere rerank call failed.")
            raise

        ranked: List[Dict] = []
        seen_indices: set[int] = set()
        scored_entries: List[Dict[str, Any]] = []
        for pos, item in enumerate(results):
            idx = getattr(item, "index", None)
            if idx is None or idx >= len(documents):
                continue
            seen_indices.add(idx)
            score = getattr(item, "relevance_score", None)
            scored_entries.append({"idx": idx, "score": score, "pos": pos})

        scored_entries.sort(key=lambda entry: (-(entry["score"] or float("-inf")), entry["pos"]))

        for entry in scored_entries:
            doc = dict(documents[entry["idx"]])
            doc["rerank_score"] = entry["score"]
            ranked.append(doc)

        if len(ranked) < limit:
            for idx, doc in enumerate(documents):
                if len(ranked) >= limit:
                    break
                if idx in seen_indices:
                    continue
                fallback = dict(doc)
                fallback.setdefault("rerank_score", None)
                ranked.append(fallback)

        return ranked[:limit]

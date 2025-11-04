from __future__ import annotations

import asyncio
import json
import logging
from typing import List

from openai import AzureOpenAI

from .config import PipelineConfig
from .exceptions import PipelineStageError

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an expert patent analyst. Given a patent title, summary, and key claim,"
    " craft concise retrieval queries (in Japanese if the inputs are Japanese, otherwise in English) that capture different angles of the invention."
    " Output the queries as a JSON array of plain strings without numbering or additional keys."
)


class QueryGenerator:
    """Generate multiple semantic search queries using Azure OpenAI chat completions."""

    def __init__(self, config: PipelineConfig) -> None:
        if not config.azure_openai_endpoint or not config.azure_openai_key:
            raise PipelineStageError("query_generation", "Azure OpenAI credentials are not configured.")
        if not config.azure_openai_chat_deployment:
            raise PipelineStageError("query_generation", "Azure OpenAI chat deployment name is not configured.")

        self._client = AzureOpenAI(
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_key,
            api_version=config.azure_openai_api_version,
        )
        self._deployment = config.azure_openai_chat_deployment

    async def generate_queries(self, title: str, summary: str, claim1: str, count: int = 10) -> List[str]:
        user_prompt = self._build_prompt(title, summary, claim1, count)
        try:
            response = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=self._deployment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
            )
        except Exception as exc:  # pragma: no cover
            raise PipelineStageError("query_generation", f"Azure OpenAI chat failure: {exc}") from exc

        content = (response.choices[0].message.content or "").strip()
        queries = self._parse_queries(content)
        if not queries:
            logger.warning("Query generation returned no queries. Raw content: %s", content)
        elif len(queries) < count:
            logger.info("Generated %d queries (requested %d)", len(queries), count)
        return queries[:count]

    def _build_prompt(self, title: str, summary: str, claim1: str, count: int) -> str:
        parts = [
            f"Title: {title.strip()}",
            f"Summary: {summary.strip()}",
            f"Claim1: {claim1.strip()}",
            "",
            f"Generate {count} diverse search queries. Return as JSON array of strings.",
        ]
        return "\n".join(parts)

    def _parse_queries(self, content: str) -> List[str]:
        if not content:
            return []

        # Attempt JSON parsing first
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass

        # Fallback: split lines or numbered list
        lines = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove leading numbering or bullets
            line = line.lstrip("-*0123456789.。 ").strip()
            if line:
                lines.append(line)
        return lines

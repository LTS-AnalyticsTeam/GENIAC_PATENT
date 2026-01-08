from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

from neo4j import GraphDatabase

from .config import PipelineConfig


class StopwordCleaner:
    """Utility helper that removes known stopwords and produces token lists."""

    def __init__(self, stopwords: Sequence[str] | None = None) -> None:
        cleaned = [word.strip() for word in (stopwords or []) if word and word.strip()]
        self.stopwords: List[str] = cleaned
        if cleaned:
            # Longer stopwords first to avoid partial replacements.
            escaped = sorted((re.escape(word) for word in cleaned if word), key=len, reverse=True)
            pattern = "|".join(escaped)
        else:
            pattern = ""
        self._pattern = re.compile(pattern) if pattern else None

    def clean_text(self, text: str | None) -> str:
        if not text:
            return ""
        normalized = str(text)
        if self._pattern:
            normalized = self._pattern.sub(" ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def tokenize(self, text: str | None) -> List[str]:
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        tokens = re.findall(r"[A-Za-z0-9]+|[\u3040-\u30ff\u4e00-\u9fff]+", cleaned)
        return [tok.lower() for tok in tokens if tok]


class Stage2Indexer:
    """Handles Stage2 graph enrichment using cleaned Cosmos payloads."""

    def __init__(self, config: PipelineConfig, cleaner: StopwordCleaner | None = None) -> None:
        self.config = config
        self.cleaner = cleaner or StopwordCleaner()
        self.neo4j_driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )

    def close(self) -> None:
        self.neo4j_driver.close()

    def reset_graph(self) -> None:
        """Remove all existing nodes/relationships to start from a clean slate."""

        def work(tx):
            tx.run("MATCH (n) DETACH DELETE n")

        with self.neo4j_driver.session() as session:
            session.execute_write(work)

    def index_documents(self, documents: Iterable[Dict]) -> int:
        """Insert cleaned Patent nodes back into Neo4j."""

        def upsert_patent(tx, patent: Dict) -> None:
            tx.run(
                """
                MERGE (p:Patent {patent_id: $patent_id})
                SET p.title = $title,
                    p.abstract = $abstract,
                    p.technical_field = $technical_field,
                    p.claim1 = $claim1,
                    p.title_tokens = $title_tokens,
                    p.abstract_tokens = $abstract_tokens,
                    p.technical_field_tokens = $technical_tokens,
                    p.claim_tokens = $claim_tokens,
                    p.vector_score = $vector_score
                """,
                **patent,
            )

        count = 0
        with self.neo4j_driver.session() as session:
            for doc in documents:
                patent_id = (doc.get("patent_id") or "").strip()
                if not patent_id:
                    continue

                cleaned_title = self.cleaner.clean_text(doc.get("title"))
                cleaned_abstract = self.cleaner.clean_text(doc.get("abstract"))
                cleaned_technical = self.cleaner.clean_text(doc.get("technical_field"))
                cleaned_claim = self.cleaner.clean_text(doc.get("claim1"))

                payload = {
                    "patent_id": patent_id,
                    "title": cleaned_title,
                    "abstract": cleaned_abstract,
                    "technical_field": cleaned_technical,
                    "claim1": cleaned_claim,
                    "title_tokens": self.cleaner.tokenize(cleaned_title),
                    "abstract_tokens": self.cleaner.tokenize(cleaned_abstract),
                    "technical_tokens": self.cleaner.tokenize(cleaned_technical),
                    "claim_tokens": self.cleaner.tokenize(cleaned_claim),
                    "vector_score": doc.get("vector_score"),
                }
                session.execute_write(upsert_patent, payload)
                count += 1
        return count

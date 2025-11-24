from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

from neo4j import GraphDatabase

from .config import PipelineConfig


class Stage2Indexer:
    """Handles Stage2 graph enrichment."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
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

    @staticmethod
    def _extract_keywords(text: str, limit: int = 8) -> List[str]:
        """Lightweight keyword extractor (ASCII words + JP bi/tri-grams)."""
        if not text:
            return []

        tokens: List[str] = []

        # ASCII-ish tokens (length>=3)
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\\-]{2,}", text.lower()):
            tokens.append(token)

        # Japanese bi/tri-grams (remove whitespace first)
        compact = re.sub(r"\\s+", "", text)
        jp_segments = re.findall(r"[\\u3040-\\u30ff\\u4e00-\\u9fff]{2,}", compact)
        for seg in jp_segments:
            for i in range(len(seg) - 1):
                tokens.append(seg[i : i + 2])
            for i in range(len(seg) - 2):
                tokens.append(seg[i : i + 3])

        # Frequency count
        freq: Dict[str, int] = {}
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1

        sorted_tokens = sorted(freq.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
        unique: List[str] = []
        for tok, _ in sorted_tokens:
            if tok not in unique:
                unique.append(tok)
            if len(unique) >= limit:
                break
        return unique

    @staticmethod
    def _build_section_topics(section_text: str) -> List[str]:
        return Stage2Indexer._extract_keywords(section_text, limit=6)

    @staticmethod
    def _build_claim_keywords(claim_texts: Sequence[str]) -> List[str]:
        merged = " ".join([ct for ct in claim_texts if ct])
        return Stage2Indexer._extract_keywords(merged, limit=12)

    @staticmethod
    def _normalize_citation_ids(entries: Sequence[str]) -> List[str]:
        """Extract patent-like IDs from citation text (best-effort)."""
        ids: List[str] = []
        for raw in entries:
            if not raw:
                continue
            # Pick digit runs of length >=7; prefix JP for consistency
            for match in re.findall(r"\d{7,}", raw):
                norm = f"JP{match}"
                if norm not in ids:
                    ids.append(norm)
        return ids

    def upsert_graph(self, documents: Iterable[dict]) -> None:
        """Ingest Patent/Section/Topic/Claim/CITES into Neo4j."""

        def upsert_patent(tx, patent: Dict) -> None:
            tx.run(
                """
                MERGE (p:Patent {patent_id: $patent_id})
                SET p.title = $title,
                    p.summary = $summary,
                    p.classification_ipc = $classification_ipc,
                    p.ipc_prefixes = $ipc_prefixes,
                    p.claim_keywords = $claim_keywords,
                    p.vector_score = $vector_score
                """,
                patent_id=patent["patent_id"],
                title=patent.get("title"),
                summary=patent.get("summary"),
                classification_ipc=patent.get("classification_ipc"),
                ipc_prefixes=patent.get("ipc_prefixes"),
                claim_keywords=patent.get("claim_keywords"),
                vector_score=patent.get("vector_score"),
            )

        def upsert_section(tx, patent_id: str, section: Dict, idx: int, topics: List[str]) -> None:
            section_key = f"{patent_id}:{section.get('type', 'unknown')}:{idx}"
            tx.run(
                """
                MERGE (p:Patent {patent_id: $patent_id})
                MERGE (s:Section {section_id: $section_id})
                SET s.section_type = $section_type,
                    s.text = $text
                MERGE (p)-[:HAS_SECTION]->(s)
                """,
                patent_id=patent_id,
                section_id=section_key,
                section_type=section.get("type"),
                text=section.get("text"),
            )
            for topic in topics:
                tx.run(
                    """
                    MERGE (t:Topic {text: $text})
                    MERGE (s:Section {section_id: $section_id})
                    MERGE (s)-[:HAS_TOPIC {weight: 1.0}]->(t)
                    """,
                    text=topic,
                    section_id=section_key,
                )

        def upsert_citation(tx, patent_id: str, cited_id: str) -> None:
            tx.run(
                """
                MERGE (p1:Patent {patent_id: $patent_id})
                MERGE (p2:Patent {patent_id: $cited_id})
                MERGE (p1)-[:CITES]->(p2)
                """,
                patent_id=patent_id,
                cited_id=cited_id,
            )

        with self.neo4j_driver.session() as session:
            for doc in documents:
                pid = doc.get("patent_id")
                if not pid:
                    continue

                claims: List[str] = doc.get("claims") or []
                claim_keywords = self._build_claim_keywords(claims) if claims else []
                sections: List[Dict] = doc.get("sections") or []
                ipc_prefixes: List[str] = doc.get("ipc_prefixes") or []
                citations_raw: List[str] = doc.get("citation_texts") or []
                citation_ids = self._normalize_citation_ids(citations_raw)

                patent_payload = {
                    "patent_id": pid,
                    "title": doc.get("title"),
                    "summary": doc.get("summary"),
                    "classification_ipc": doc.get("classification_ipc"),
                    "ipc_prefixes": ipc_prefixes,
                    "claim_keywords": claim_keywords,
                    "vector_score": doc.get("vector_score"),
                }
                session.execute_write(upsert_patent, patent_payload)

                for idx, section in enumerate(sections):
                    text = section.get("text") or ""
                    topics = self._build_section_topics(text)
                    session.execute_write(upsert_section, pid, section, idx, topics)

                for cited_id in citation_ids:
                    session.execute_write(upsert_citation, pid, cited_id)

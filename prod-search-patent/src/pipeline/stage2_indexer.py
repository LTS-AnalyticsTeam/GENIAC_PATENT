from __future__ import annotations

from typing import Iterable

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

    def upsert_graph(self, documents: Iterable[dict]) -> None:
        def work(tx, doc):
            patent_id = doc.get("patent_id")
            tx.run(
                """
                MERGE (p:Patent {patent_id: $patent_id})
                SET p.title = $title,
                    p.summary = $summary,
                    p.classification_ipc = $ipc
                """,
                patent_id=patent_id,
                title=doc.get("title"),
                summary=doc.get("summary"),
                ipc=doc.get("classification_ipc"),
            )

        with self.neo4j_driver.session() as session:
            for doc in documents:
                if not doc.get("patent_id"):
                    continue
                session.execute_write(work, doc)

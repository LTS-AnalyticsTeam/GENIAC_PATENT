from __future__ import annotations

import logging
from typing import Dict, List

from neo4j import GraphDatabase

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class GraphRAGService:
    """Perform Graph-RAG ranking across Neo4j."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def top_k(self, patent_ids: List[str], k: int = 10) -> List[Dict]:
        if not patent_ids:
            return []

        query = """
        UNWIND $ids AS pid
        MATCH (p:Patent {patent_id: pid})
        OPTIONAL MATCH (p)-[r:RELATES_TO]->(q:Patent)
        WITH p, count(r) AS rel_count
        RETURN p.patent_id AS patent_id,
               p.title AS title,
               p.summary AS summary,
               p.classification_ipc AS classification_ipc,
               rel_count
        ORDER BY rel_count DESC, p.patent_id ASC
        LIMIT $k
        """

        with self.driver.session() as session:
            records = session.run(query, ids=patent_ids, k=k)
            results = []
            for rec in records:
                results.append(
                    {
                        "patent_id": rec["patent_id"],
                        "title": rec["title"],
                        "summary": rec["summary"],
                        "classification_ipc": rec["classification_ipc"],
                        "graph_score": rec["rel_count"],
                    }
                )
        return results

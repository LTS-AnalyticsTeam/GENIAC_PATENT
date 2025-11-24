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

    def top_k(self, patent_ids: List[str], alpha_id: str | None, k: int = 10) -> List[Dict]:
        if not patent_ids:
            return []

        query = """
        UNWIND $ids AS pid
        MATCH (p:Patent {patent_id: pid})
        OPTIONAL MATCH (alpha:Patent {patent_id: $alpha_id})
        WITH p, alpha,
             coalesce(p.ipc_prefixes, []) AS p_ipc,
             coalesce(alpha.ipc_prefixes, []) AS a_ipc,
             coalesce(p.claim_keywords, []) AS p_claims,
             coalesce(alpha.claim_keywords, []) AS a_claims
        WITH p,
             size([x IN p_ipc WHERE x IN a_ipc]) AS ipc_overlap,
             size([x IN p_claims WHERE x IN a_claims]) AS claim_overlap,
             alpha
        OPTIONAL MATCH (alpha)-[:HAS_SECTION]->(:Section)-[:HAS_TOPIC]->(t:Topic)
        WITH p, ipc_overlap, claim_overlap, collect(DISTINCT t.text) AS alpha_topics
        OPTIONAL MATCH (p)-[:HAS_SECTION]->(:Section)-[:HAS_TOPIC]->(pt:Topic)
        WITH p, ipc_overlap, claim_overlap, alpha_topics, collect(DISTINCT pt.text) AS cand_topics
        WITH p,
             ipc_overlap,
             claim_overlap,
             size([x IN cand_topics WHERE x IN alpha_topics]) AS topic_overlap
        OPTIONAL MATCH (alpha)-[:CITES]->(p)
        WITH p, ipc_overlap, claim_overlap, topic_overlap, count(alpha) AS cites_from_alpha
        OPTIONAL MATCH (p)-[:CITES]->(alpha)
        WITH p, ipc_overlap, claim_overlap, topic_overlap, cites_from_alpha, count(alpha) AS cites_to_alpha
        WITH p,
             ipc_overlap,
             claim_overlap,
             topic_overlap,
             (cites_from_alpha + cites_to_alpha) AS citation_hits,
             (ipc_overlap * 0.25 +
              topic_overlap * 0.20 +
              claim_overlap * 0.35 +
              (cites_from_alpha + cites_to_alpha) * 0.05) AS graph_score
        RETURN p.patent_id AS patent_id,
               p.title AS title,
               p.summary AS summary,
               p.classification_ipc AS classification_ipc,
               graph_score,
               p.vector_score AS vector_score
        ORDER BY graph_score DESC, vector_score DESC, patent_id ASC
        LIMIT $k
        """

        with self.driver.session() as session:
            records = session.run(query, ids=patent_ids, k=k, alpha_id=alpha_id)
            results = []
            for rec in records:
                results.append(
                    {
                        "patent_id": rec["patent_id"],
                        "title": rec["title"],
                        "summary": rec["summary"],
                        "classification_ipc": rec["classification_ipc"],
                        "graph_score": rec["graph_score"],
                        "vector_score": rec.get("vector_score"),
                    }
                )
        return results

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

    def top_k(self, patent_ids: List[str], alpha_id: str | None, k: int = 30) -> List[Dict]:
        if not patent_ids:
            return []

        query = """
        UNWIND $ids AS pid
        MATCH (p:Patent {patent_id: pid})
        OPTIONAL MATCH (alpha:Patent {patent_id: $alpha_id})
        WITH p,
             coalesce(p.title_tokens, []) AS p_title,
             coalesce(p.abstract_tokens, []) AS p_abstract,
             coalesce(p.technical_field_tokens, []) AS p_technical,
             coalesce(p.claim_tokens, []) AS p_claims,
             CASE WHEN alpha IS NULL THEN [] ELSE coalesce(alpha.title_tokens, []) END AS a_title,
             CASE WHEN alpha IS NULL THEN [] ELSE coalesce(alpha.abstract_tokens, []) END AS a_abstract,
             CASE WHEN alpha IS NULL THEN [] ELSE coalesce(alpha.technical_field_tokens, []) END AS a_technical,
             CASE WHEN alpha IS NULL THEN [] ELSE coalesce(alpha.claim_tokens, []) END AS a_claims
        WITH p,
             size([x IN p_title WHERE x IN a_title]) AS title_overlap,
             size([x IN p_abstract WHERE x IN a_abstract]) AS abstract_overlap,
             size([x IN p_technical WHERE x IN a_technical]) AS technical_overlap,
             size([x IN p_claims WHERE x IN a_claims]) AS claim_overlap
        WITH p,
             title_overlap,
             abstract_overlap,
             technical_overlap,
             claim_overlap,
             (title_overlap * 0.25 +
              abstract_overlap * 0.25 +
              technical_overlap * 0.20 +
              claim_overlap * 0.30) AS graph_score
        RETURN p.patent_id AS patent_id,
               p.title AS title,
               p.abstract AS abstract,
               p.technical_field AS technical_field,
               p.claim1 AS claim1,
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
                        "abstract": rec["abstract"],
                        "technical_field": rec["technical_field"],
                        "claim1": rec["claim1"],
                        "graph_score": rec["graph_score"],
                        "vector_score": rec.get("vector_score"),
                    }
                )
        return results

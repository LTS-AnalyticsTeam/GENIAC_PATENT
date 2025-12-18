from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

from elasticsearch import Elasticsearch, helpers

from .config import PipelineConfig
from .exceptions import PipelineStageError

logger = logging.getLogger(__name__)


class Stage1ElasticsearchIndexer:
    """Handles Stage 1 index creation and bulk upsert."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.client = Elasticsearch(config.es_host, request_timeout=120)

    def ensure_index(self) -> None:
        index = self.config.es_stage1_index
        created = False
        if not self.client.indices.exists(index=index):
            mappings = {
                "mappings": {
                    "properties": {
                        "patent_id": {"type": "keyword"},
                        "job_id": {"type": "keyword"},
                        "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "summary": {"type": "text"},
                        "claim1": {"type": "text"},
                        self.config.es_vector_field: {
                            "type": "dense_vector",
                            "dims": self.config.es_vector_dims,
                            "index": True,
                            "similarity": "cosine",
                        },
                    }
                },
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            }

            self.client.indices.create(index=index, body=mappings)
            created = True
            logger.info("Created Stage1 index %s", index)

        self._ensure_job_id_mapping(index, created)

    def _ensure_job_id_mapping(self, index: str, created: bool) -> None:
        try:
            mapping = self.client.indices.get_mapping(index=index)
            existing_props = mapping.get(index, {}).get("mappings", {}).get("properties", {})
            if "job_id" in (existing_props or {}):
                return
            self.client.indices.put_mapping(index=index, body={"properties": {"job_id": {"type": "keyword"}}})
            if created:
                return
            logger.info("Added job_id field to Stage1 index %s", index)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to ensure job_id mapping on %s: %s", index, exc)

    def bulk_index(self, documents: Iterable[dict]) -> Tuple[int, int]:
        def _actions():
            for doc in documents:
                patent_id = doc.get("patent_id") or doc.get("id")
                if not patent_id:
                    continue
                yield {
                    "_op_type": "index",
                    "_index": self.config.es_stage1_index,
                    "_id": patent_id,
                    "_source": doc,
                }

        success, errors = helpers.bulk(
            self.client,
            _actions(),
            raise_on_error=False,
        )
        if errors:
            logger.warning("Stage1 bulk had %d errors", len(errors))
        return success, len(errors)

    def fetch_existing(self, patent_ids: Iterable[str]) -> Dict[str, dict]:
        ids = [pid for pid in patent_ids if pid]
        if not ids:
            return {}

        index = self.config.es_stage1_index
        if not self.client.indices.exists(index=index):
            return {}

        response = self.client.mget(index=index, ids=ids)
        existing: Dict[str, dict] = {}
        for doc in response.get("docs", []):
            if doc.get("found") and doc.get("_source"):
                existing[str(doc["_id"])] = doc["_source"]
        return existing

    def knn_search(
        self,
        vector: List[float],
        k: int,
        num_candidates: int,
        source_fields: List[str] | None = None,
        job_id: str | None = None,
    ) -> List[dict]:
        index = self.config.es_stage1_index
        if not self.client.indices.exists(index=index):
            return []

        fields = source_fields or [
            "patent_id",
            "title",
            "summary",
            "claim1",
        ]

        knn_query = {
            "field": self.config.es_vector_field,
            "query_vector": vector,
            "k": k,
            "num_candidates": num_candidates,
        }
        if job_id:
            knn_query["filter"] = {"term": {"job_id": job_id}}

        response = self.client.search(
            index=index,
            knn=knn_query,
            size=k,
            _source=fields,
        )
        return response.get("hits", {}).get("hits", [])

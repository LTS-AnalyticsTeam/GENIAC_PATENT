from __future__ import annotations

import logging
from typing import Dict, List

from azure.cosmos import CosmosClient, exceptions  # type: ignore[import]
from azure.core.pipeline.transport import RequestsTransport

from .config import PipelineConfig
from .exceptions import IngestionError, PipelineStageError

logger = logging.getLogger(__name__)


class _CosmosSafeTransport(RequestsTransport):
    """Drop unsupported kwargs the legacy SDK leaks into requests session."""

    def send(self, request, *, proxies=None, **kwargs):  # type: ignore[override]
        kwargs.pop("continuation_token", None)
        return super().send(request, proxies=proxies, **kwargs)


class CosmosPatentClient:
    """Client wrapper for querying Azure Cosmos DB."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        endpoint = (config.cosmos_endpoint or "").strip()
        credential = (config.cosmos_key or "").strip().strip('"')

        if not endpoint or not credential:
            raise PipelineStageError(
                "cosmos-init",
                "Cosmos endpoint or key is not configured. "
                "Set COSMOS_ENDPOINT and COSMOS_KEY in environment variables.",
            )

        database_name = (config.cosmos_database or "").strip().strip('"')
        container_name = (config.cosmos_container or "").strip().strip('"')

        if not database_name or not container_name:
            raise PipelineStageError(
                "cosmos-init",
                "Cosmos database or container name is not configured. "
                "Set COSMOS_DATABASE and COSMOS_CONTAINER in environment variables.",
            )

        try:
            self._client = CosmosClient(
                endpoint,
                credential=credential,
                transport=_CosmosSafeTransport(),
            )
            self._database = self._client.get_database_client(database_name)
            self._container = self._database.get_container_client(container_name)
        except exceptions.CosmosHttpResponseError as exc:
            raise PipelineStageError("cosmos-init", f"Failed to connect to Cosmos DB: {exc}") from exc

    def close(self) -> None:
        try:
            self._client.close()
        except AttributeError:
            # Older SDK versions do not expose close()
            pass

    def query_by_ipc_prefixes(self, ipc_codes: List[str], limit: int) -> List[Dict]:
        if not ipc_codes:
            raise IngestionError("At least one IPC prefix required", status_code=422)

        prefixes: List[str] = []
        for code in ipc_codes:
            if not code:
                continue
            head = code.split("/")[0].strip()
            if head and head not in prefixes:
                prefixes.append(head)

        if not prefixes:
            raise IngestionError("At least one IPC prefix required", status_code=422)

        logger.info("Cosmos query with IPC prefixes: %s (limit=%d)", prefixes[:10], limit)
        parameters = [{"name": f"@prefix{i}", "value": prefix} for i, prefix in enumerate(prefixes)]
        prefix_conditions = [f"STARTSWITH(ipc.text, @prefix{i}, true)" for i in range(len(prefixes))]
        predicate = " OR ".join(prefix_conditions)

        query = f"""
        SELECT VALUE c
        FROM c
        WHERE IS_DEFINED(c.bibliographic.classification.ipc)
          AND ARRAY_LENGTH(c.bibliographic.classification.ipc) > 0
          AND EXISTS (
            SELECT VALUE 1
            FROM ipc IN c.bibliographic.classification.ipc
            WHERE IS_DEFINED(ipc.text)
              AND ({predicate})
          )
        """

        iterator = self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
            max_item_count=limit,
        )

        results: List[Dict] = []
        for item in iterator:
            results.append(item)
            if len(results) >= limit:
                break

        return results

    def fetch_by_patent_ids(self, patent_ids: List[str]) -> Dict[str, Dict]:
        """指定した patent_id ごとの完全JSONをまとめて取得する。"""
        if not patent_ids:
            return {}

        results: Dict[str, Dict] = {}
        query = """
        SELECT VALUE c FROM c
        WHERE (IS_DEFINED(c.patent_id) AND c.patent_id = @lookup)
           OR (IS_DEFINED(c.id) AND c.id = @lookup)
           OR (
                IS_DEFINED(c.bibliographic) AND
                IS_DEFINED(c.bibliographic.publication) AND
                IS_DEFINED(c.bibliographic.publication.doc_number) AND
                c.bibliographic.publication.doc_number = @lookup
              )
        """
        for patent_id in patent_ids:
            if not patent_id:
                continue
            try:
                iterator = self._container.query_items(
                    query=query,
                    parameters=[{"name": "@lookup", "value": patent_id}],
                    enable_cross_partition_query=True,
                    max_item_count=1,
                )
                for doc in iterator:
                    results[patent_id] = doc
                    break
            except exceptions.CosmosHttpResponseError as exc:  # type: ignore[name-defined]
                logger.warning("Failed to fetch Cosmos document %s: %s", patent_id, exc)
        return results

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence, Set

from azure.cosmos import CosmosClient, exceptions  # type: ignore[import]
from azure.core.exceptions import ServiceResponseError
from azure.core.pipeline.transport import RequestsTransport

from .config import PipelineConfig
from .exceptions import IngestionError, PipelineStageError

logger = logging.getLogger(__name__)

_BULK_LOOKUP_CHUNK_SIZE = 50

_SINGLE_LOOKUP_QUERY = """
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

_BULK_LOOKUP_QUERY = """
SELECT VALUE c FROM c
WHERE (
        IS_DEFINED(c.patent_id) AND
        ARRAY_CONTAINS(@ids, c.patent_id, true)
      )
   OR (
        IS_DEFINED(c.id) AND
        ARRAY_CONTAINS(@ids, c.id, true)
      )
   OR (
        IS_DEFINED(c.bibliographic) AND
        IS_DEFINED(c.bibliographic.publication) AND
        IS_DEFINED(c.bibliographic.publication.doc_number) AND
        ARRAY_CONTAINS(@ids, c.bibliographic.publication.doc_number, true)
      )
"""


def _chunked(values: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    for index in range(0, len(values), chunk_size):
        yield list(values[index : index + chunk_size])


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

        deduped: List[str] = []
        seen: Set[str] = set()
        for pid in patent_ids:
            normalized = (pid or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)

        if not deduped:
            return {}

        results: Dict[str, Dict] = {}
        for chunk in _chunked(deduped, _BULK_LOOKUP_CHUNK_SIZE):
            if not chunk:
                continue
            try:
                self._fetch_chunk(chunk, results)
            except (ServiceResponseError, exceptions.CosmosHttpResponseError) as exc:
                logger.warning(
                    "Bulk Cosmos lookup failed for %d ids (fallback to individual queries): %s",
                    len(chunk),
                    exc,
                )
                for pid in chunk:
                    self._fetch_single(pid, results)
        return results

    def _fetch_chunk(self, chunk: Sequence[str], results: Dict[str, Dict]) -> None:
        chunk_set = set(chunk)
        iterator = self._container.query_items(
            query=_BULK_LOOKUP_QUERY,
            parameters=[{"name": "@ids", "value": list(chunk_set)}],
            enable_cross_partition_query=True,
            max_item_count=len(chunk_set),
        )
        for doc in iterator:
            for key in self._candidate_keys(doc):
                if key in chunk_set and key not in results:
                    results[key] = doc

        missing = [pid for pid in chunk if pid not in results]
        if missing:
            logger.debug("Cosmos chunk lookup missing %d ids, retrying individually", len(missing))
            for pid in missing:
                self._fetch_single(pid, results)

    def _fetch_single(self, patent_id: str, results: Dict[str, Dict]) -> None:
        if patent_id in results:
            return
        try:
            iterator = self._container.query_items(
                query=_SINGLE_LOOKUP_QUERY,
                parameters=[{"name": "@lookup", "value": patent_id}],
                enable_cross_partition_query=True,
                max_item_count=1,
            )
            for doc in iterator:
                results[patent_id] = doc
                return
        except exceptions.CosmosHttpResponseError as exc:
            logger.warning("Failed to fetch Cosmos document %s: %s", patent_id, exc)

    @staticmethod
    def _candidate_keys(doc: Dict) -> Set[str]:
        keys: Set[str] = set()
        if not isinstance(doc, dict):
            return keys
        for field in ("patent_id", "id"):
            value = doc.get(field)
            if value:
                text = str(value).strip()
                if text:
                    keys.add(text)
        try:
            biblio = doc.get("bibliographic") or {}
            publication = biblio.get("publication") or {}
            doc_number = publication.get("doc_number")
        except AttributeError:
            doc_number = None
        if doc_number:
            text = str(doc_number).strip()
            if text:
                keys.add(text)
        return keys

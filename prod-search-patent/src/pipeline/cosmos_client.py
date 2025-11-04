from __future__ import annotations

import logging
from typing import Dict, List

from azure.cosmos import CosmosClient, exceptions  # type: ignore[import]
from azure.core.pipeline.transport import RequestsTransport

from .config import PipelineConfig
from .exceptions import IngestionError, PipelineStageError

logger = logging.getLogger(__name__)

_FULLWIDTH_SPACE = "\u3000"


class _CosmosSafeTransport(RequestsTransport):
    """Drop unsupported kwargs the legacy SDK leaks into requests session."""

    def send(self, request, *, proxies=None, **kwargs):  # type: ignore[override]
        kwargs.pop("continuation_token", None)
        return super().send(request, proxies=proxies, **kwargs)


def _sanitize_prefix(value: str) -> str:
    return value.upper().replace(" ", "").replace(_FULLWIDTH_SPACE, "")


def _cosmos_sanitize(field: str) -> str:
    return f"REPLACE(REPLACE(UPPER({field}), ' ', ''), '　', '')"


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
            sanitized = _sanitize_prefix(code)
            if sanitized and sanitized not in prefixes:
                prefixes.append(sanitized)

        if not prefixes:
            raise IngestionError("At least one IPC prefix required", status_code=422)

        prefix_params = [{"name": f"@prefix{i}", "value": prefix} for i, prefix in enumerate(prefixes)]

        def _or_join(expressions: List[str]) -> str:
            return " OR ".join(expr for expr in expressions if expr) or "false"

        # Conditions for top-level classification_ipc array (strings)
        top_array_checks = [
            f"(IS_STRING(ipc) AND STARTSWITH({_cosmos_sanitize('ipc')}, @prefix{i}))"
            for i in range(len(prefixes))
        ]
        top_array_condition = _or_join(top_array_checks)

        # Conditions for bibliographic.classification.ipc array (objects with text)
        biblio_ipc_checks = [
            f"(IS_OBJECT(ipc) AND IS_DEFINED(ipc.text) AND IS_STRING(ipc.text) "
            f"AND STARTSWITH({_cosmos_sanitize('ipc.text')}, @prefix{i}))"
            for i in range(len(prefixes))
        ]
        biblio_ipc_condition = _or_join(biblio_ipc_checks)

        # Conditions for string fields (top-level or national)
        def _string_field_condition(field: str) -> str:
            checks = [
                f"STARTSWITH({_cosmos_sanitize(field)}, @prefix{i})" for i in range(len(prefixes))
            ]
            joined = _or_join(checks)
            return f"(IS_DEFINED({field}) AND IS_STRING({field}) AND ({joined}))"

        string_fields = [
            "c.classification_ipc",
            "c.bibliographic.classification.national.main",
            "c.bibliographic.classification.national.further",
        ]
        string_conditions = [_string_field_condition(field) for field in string_fields]

        where_clauses = [
            f"""(
                IS_DEFINED(c.classification_ipc) AND IS_ARRAY(c.classification_ipc) AND ARRAY_LENGTH(c.classification_ipc) > 0
                AND EXISTS(SELECT VALUE 1 FROM ipc IN c.classification_ipc WHERE {top_array_condition})
            )""",
            f"""(
                IS_DEFINED(c.bibliographic.classification.ipc) AND IS_ARRAY(c.bibliographic.classification.ipc) AND ARRAY_LENGTH(c.bibliographic.classification.ipc) > 0
                AND EXISTS(SELECT VALUE 1 FROM ipc IN c.bibliographic.classification.ipc WHERE {biblio_ipc_condition})
            )""",
            *string_conditions,
        ]

        where_clause = " OR ".join(where_clauses)

        query = f"""
        SELECT VALUE c
        FROM c
        WHERE {where_clause}
        """

        iterator = self._container.query_items(
            query=query,
            parameters=prefix_params,
            enable_cross_partition_query=True,
            max_item_count=limit,
        )

        results: List[Dict] = []
        for item in iterator:
            results.append(item)
            if len(results) >= limit:
                break

        return results

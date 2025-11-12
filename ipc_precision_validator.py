"""
IPC-based patent search precision validator.

Pipeline:
1. Fetch Cosmos DB documents by IPC prefix (STARTSWITH query)
2. Sort by publication date desc and trim to max_docs
3. Generate Azure OpenAI embeddings for claim #1 and abstract separately
4. Store vectors and metadata into a single Elasticsearch index
5. Pick 5 random test cases from test_cases.csv and run claim/abstract hybrid searches
6. Merge results, check whether ground-truth doc_numbers are retrieved, and log counts
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from azure.cosmos import CosmosClient
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
try:
    from openai import AzureOpenAI, OpenAI
except ImportError:  # pragma: no cover
    from openai import AzureOpenAI  # type: ignore
    OpenAI = None  # type: ignore

# Load environment variables early
load_dotenv()

logger = logging.getLogger("ipc_precision_validator")
logging.basicConfig(
    level=os.getenv("IPC_VALIDATOR_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

BULK_BATCH_SIZE = int(os.getenv("ES_BULK_BATCH_SIZE", "100"))


def normalize_patent_number(identifier: str) -> str:
    """Return only digits from a patent identifier like JP2017080220A."""
    if identifier is None:
        return ""
    identifier_str = str(identifier)
    return "".join(ch for ch in identifier_str if ch.isdigit())


def normalize_publication_date(raw_date: Optional[str]) -> Optional[str]:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    if not raw_date:
        return None
    raw = raw_date.strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def format_ipc_query(prefix: str) -> str:
    """
    Cosmos DB stores IPC text as 'A01C 7...' (space after 4th char).
    Convert prefix like 'A01C7' to 'A01C 7'.
    """
    if not prefix:
        return ""
    cleaned = prefix.upper().replace(" ", "")
    if len(cleaned) <= 4:
        return cleaned
    return f"{cleaned[:4]} {cleaned[4:]}"


def truncate_text(value: str, max_chars: int) -> str:
    if not value:
        return ""
    return value[:max_chars]


def clean_ipc_prefix(value: Optional[str]) -> Optional[str]:
    """Normalize IPC prefix strings to the first 6 alphanumeric characters."""
    if not value:
        return None
    normalized = str(value).strip().upper().replace(" ", "")
    if not normalized:
        return None
    normalized = normalized.replace("/", "")
    return normalized[:6] if len(normalized) >= 4 else None


def extract_ipc_prefix_from_document(doc: Dict) -> Optional[str]:
    """Try various classification locations to obtain an IPC prefix."""
    if not isinstance(doc, dict):
        return None

    classification_candidates = [
        doc.get("bibliographic", {}).get("classification"),
        doc.get("classification"),
        doc.get("metadata", {}).get("classification"),
        doc.get("biblio", {}).get("classification"),
    ]

    for classification in classification_candidates:
        if not classification:
            continue
        prefix = _extract_prefix_from_classification_dict(classification)
        if prefix:
            return prefix
    return None


def _extract_prefix_from_classification_dict(classification: Dict) -> Optional[str]:
    for key in ("ipc_prefix", "ipcPrefix", "ipc_prefixes"):
        prefixes = classification.get(key)
        if prefixes:
            if isinstance(prefixes, str):
                candidate = clean_ipc_prefix(prefixes)
                if candidate:
                    return candidate
            elif isinstance(prefixes, (list, tuple, set)):
                for prefix in prefixes:
                    candidate = clean_ipc_prefix(prefix)
                    if candidate:
                        return candidate

    ipc_list = classification.get("ipc") or classification.get("ipcList") or []
    for entry in ipc_list:
        if isinstance(entry, dict):
            text = entry.get("text") or entry.get("value") or entry.get("ipc_text")
            candidate = clean_ipc_prefix(text)
            if candidate:
                return candidate
        else:
            candidate = clean_ipc_prefix(entry)
            if candidate:
                return candidate
    return None


class QueryGenerator:
    """Generate multi-query prompts for hybrid search (vector + keyword)."""

    def __init__(self):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.azure_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv(
            "AZURE_OPENAI_TEXT_DEPLOYMENT"
        )

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        self.provider: Optional[str] = None
        self.client = None

        if self.azure_api_key and self.azure_endpoint and self.azure_chat_deployment:
            self.client = AzureOpenAI(
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
                azure_endpoint=self.azure_endpoint,
            )
            self.provider = "azure"
            logger.info("Azure OpenAI chat client initialized for multi-query generation.")
        elif OpenAI and self.openai_api_key:  # pragma: no cover
            self.client = OpenAI(api_key=self.openai_api_key)
            self.provider = "openai"
            logger.info("OpenAI chat client initialized for multi-query generation.")
        else:
            raise ValueError(
                "Multi-query keyword生成には Azure OpenAI (AZURE_OPENAI_CHAT_DEPLOYMENT)"
                " もしくは OpenAI (OPENAI_API_KEY) の設定が必要です。"
            )

    def _call_chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> str:
        try:
            if self.provider == "azure":
                response = self.client.chat.completions.create(
                    model=self.azure_chat_deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.5,
                    max_tokens=max_tokens,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.openai_chat_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.5,
                    max_tokens=max_tokens,
                )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to generate multi-query prompts: {exc}")

    @staticmethod
    def _parse_json_array(raw_output: str, label: str, expected_len: int, truncate: int) -> List[str]:
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise RuntimeError(f"{label} JSON parsing failed: {exc}; output={raw_output!r}")
        items: List[str] = []
        for entry in data:
            text = str(entry).strip()
            if text:
                items.append(truncate_text(text, truncate))
            if len(items) >= expected_len:
                break
        if len(items) < expected_len:
            raise RuntimeError(f"{label} entries insufficient: {len(items)}/{expected_len}")
        return items

    def generate_queries(self, section_label: str, base_text: str, num_items: int) -> Dict[str, List[str]]:
        content = (base_text or "").strip()
        if not content:
            raise ValueError(f"{section_label} text is empty; cannot generate queries.")

        context = f"{section_label}:\n{truncate_text(content, 2000)}"

        vector_system = (
            "あなたは日本語の特許理解に長けたアシスタントです。"
            "入力の請求項/要約内容に基づき、主題・課題・解決手段・効果・用途など"
            "異なる観点で200文字前後の短文を10本作成してください。"
            "各短文は互いに焦点が重ならないようにしてください。"
            "必ずJSON配列（要素数10の文字列配列）のみを返し、前後に空行・説明文・コードブロックを出力してはいけません。"
        )
        vector_user = f"以下の{section_label}からバリエーションのある検索文を作成してください。\n\n{context}"
        vector_output = self._call_chat(vector_system, vector_user)
        vector_passages = self._parse_json_array(vector_output, f"{section_label} vector queries", num_items, 260)

        keyword_system = (
            "あなたは日本語の特許理解に長けたアシスタントです。"
            "入力テキストに実在する固有語・専門フレーズを抽出し、検索向けのキーワード列を生成してください。"
            "必ずJSON配列（要素数10の文字列配列）のみを返し、前後に余分な文字・説明文・コードブロックを出力してはいけません。"
        )
        keyword_user = (
            f"以下の{section_label}に対して、検索用キーワード列を10本生成してください。\n"
            "# 入力\n"
            f"{context}\n"
            "# 条件\n"
            "- 出力は JSON 配列（文字列10要素）のみ\n"
            "- 各要素は空白区切りで4〜6語程度\n"
            "- 各列に入力テキストに実在する固有語・専門フレーズを2つ以上含める\n"
            "- 助詞や一般語（例: する、できる、方法）は含めない\n"
            "- 説明文・ラベル・コードブロックは禁止\n"
        )
        keyword_output = self._call_chat(keyword_system, keyword_user)
        keyword_queries = self._parse_json_array(keyword_output, f"{section_label} keyword queries", num_items, 200)

        return {
            "vector_passages": vector_passages,
            "keyword_queries": keyword_queries,
        }


def parse_ax_docs(field_value: str) -> List[str]:
    """Split ax_docs field by ';' and trim."""
    if not field_value:
        return []
    return [token.strip() for token in field_value.split(";") if token.strip()]


def timestamp_suffix() -> str:
    """Return mmddhhmmss suffix for run identifier."""
    return datetime.utcnow().strftime("%m%d%H%M%S")


def attach_file_logger(log_path: str) -> logging.Handler:
    """Attach a file handler to the module logger."""
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return handler


def ensure_csv(path: str, headers: List[str]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()


@dataclass
class CaseContext:
    case_id: str
    syutugan: str
    doc_number: str
    ipc_prefix: str
    ipc_query: str
    ax_docs: List[str]
    ax_doc_numbers: List[str]
    claim_text: Optional[str]
    abstract_text: Optional[str]


@dataclass
class CaseResult:
    case_id: str
    syutugan: str
    ipc_prefix: str
    counts: Dict[str, int]
    hit: bool
    matched_ax_docs: List[str] = field(default_factory=list)
    claim_hits: int = 0
    abstract_hits: int = 0
    merged_hits: int = 0
    claim_query_stats: List[str] = field(default_factory=list)
    abstract_query_stats: List[str] = field(default_factory=list)
    ax_in_ipc_filtered: bool = False
    ax_in_sorted: bool = False


class CosmosPatentClient:
    """Thin wrapper around Cosmos DB queries used in this validator."""

    def __init__(self, endpoint: str, key: str, database: str, container: str):
        if not endpoint or not key:
            raise ValueError("COSMOS_ENDPOINT and COSMOS_KEY are required.")
        self.client = CosmosClient(endpoint, key)
        db_client = self.client.get_database_client(database)
        self.container = db_client.get_container_client(container)
        self._doc_query = """
        SELECT *
        FROM c
        WHERE (IS_DEFINED(c.bibliographic.publication.doc_number) AND c.bibliographic.publication.doc_number = @doc)
           OR (IS_DEFINED(c.bibliographic.application.doc_number) AND c.bibliographic.application.doc_number = @doc)
        """

    def fetch_document_by_doc_number(self, doc_number: str, raw_identifier: Optional[str] = None) -> Optional[Dict]:
        candidates: List[str] = []
        if doc_number:
            candidates.append(str(doc_number))
        if raw_identifier is not None:
            raw_str = str(raw_identifier).strip()
            if raw_str:
                candidates.append(raw_str)
                normalized = normalize_patent_number(raw_str)
                if normalized:
                    candidates.append(normalized)
                if raw_str[-1:].upper() in {"A", "B"} and len(raw_str) > 1:
                    candidates.append(raw_str[:-1])

        seen: Set[str] = set()
        fallback: Optional[Dict] = None
        for candidate in candidates:
            value = candidate.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            items = [
                CosmosPatentClient._unwrap_document(item)
                for item in self.container.query_items(
                    query=self._doc_query,
                    parameters=[{"name": "@doc", "value": value}],
                    enable_cross_partition_query=True,
                )
            ]
            if not items:
                continue
            preferred = next((doc for doc in items if extract_ipc_prefix_from_document(doc)), None)
            if preferred:
                return preferred
            if not fallback:
                fallback = items[0]
        return fallback

    def fetch_by_ipc_text(self, ipc_query_prefix: str) -> List[Dict]:
        query = """
        SELECT *
        FROM c
        WHERE EXISTS (
          SELECT VALUE 1
          FROM ipc IN c.bibliographic.classification.ipc
          WHERE IS_DEFINED(ipc.text)
            AND STARTSWITH(ipc.text, @ipc_prefix, true)
        )
        """
        params = [{"name": "@ipc_prefix", "value": ipc_query_prefix}]
        items = [
            CosmosPatentClient._unwrap_document(item)
            for item in self.container.query_items(
                query=query,
                parameters=params,
                enable_cross_partition_query=True,
            )
        ]
        return items

    @staticmethod
    def _unwrap_document(item: Dict) -> Dict:
        """Cosmos queries like 'SELECT c' wrap documents under key 'c'."""
        if isinstance(item, dict) and "c" in item and isinstance(item["c"], dict):
            return item["c"]
        return item


class EmbeddingService:
    """Azure OpenAI embedding helper with simple in-memory cache."""

    def __init__(self):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        if not endpoint or not api_key:
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set.")
        self.deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
        if not self.deployment:
            raise ValueError("Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT.")
        self.client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        self._dimension: Optional[int] = None
        self._cache: Dict[str, List[float]] = {}

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    def ensure_dimension(self) -> int:
        if self._dimension:
            return self._dimension
        probe = self.embed_text("dimension probe for patent embeddings")
        if not probe:
            raise RuntimeError("Failed to infer embedding dimension.")
        return self._dimension or len(probe)

    def embed_text(self, text: Optional[str]) -> Optional[List[float]]:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        if cleaned in self._cache:
            return self._cache[cleaned]
        response = self.client.embeddings.create(input=cleaned, model=self.deployment)
        vector = response.data[0].embedding
        self._cache[cleaned] = vector
        if not self._dimension:
            self._dimension = len(vector)
        return vector


class ElasticsearchIndexManager:
    """Manages the patents_vector_index lifecycle and search operations."""

    def __init__(self, host: str, port: int, index_name: str = "patents_vector_index"):
        scheme = "https" if os.getenv("ELASTICSEARCH_USE_SSL", "false").lower() == "true" else "http"
        username = os.getenv("ELASTICSEARCH_USERNAME")
        password = os.getenv("ELASTICSEARCH_PASSWORD")
        self.index_name = index_name
        self.bulk_chunk_size = int(os.getenv("ES_BULK_CHUNK_SIZE", "25"))
        self.bulk_request_timeout = int(os.getenv("ES_BULK_TIMEOUT", "120"))
        self.client = Elasticsearch(
            hosts=[{"host": host, "port": port, "scheme": scheme}],
            basic_auth=(username, password) if username and password else None,
            verify_certs=False,
            request_timeout=self.bulk_request_timeout,
        )

    def recreate_index(self, dimensions: int):
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        body = {
            "mappings": {
                "properties": {
                    "doc_number": {"type": "keyword"},
                    "publication_date": {"type": "date", "ignore_malformed": True},
                    "ipc_prefix": {"type": "keyword"},
                    "claim_vector": {"type": "dense_vector", "dims": dimensions, "index": False},
                    "abstract_vector": {"type": "dense_vector", "dims": dimensions, "index": False},
                    "claim_text": {"type": "text"},
                    "abstract_text": {"type": "text"},
                }
            }
        }
        self.client.indices.create(index=self.index_name, body=body)
        logger.info("Created index %s with %s dimensions.", self.index_name, dimensions)

    def bulk_index(self, documents: Iterable[Dict]) -> int:
        actions = []
        for doc in documents:
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": doc["doc_number"],
                    "_source": doc,
                }
            )
        if not actions:
            return 0

        chunk_size = max(1, self.bulk_chunk_size)
        success_total = 0
        for start in range(0, len(actions), chunk_size):
            chunk = actions[start : start + chunk_size]
            try:
                success, failures = helpers.bulk(
                    self.client,
                    chunk,
                    chunk_size=len(chunk),
                    request_timeout=self.bulk_request_timeout,
                    raise_on_error=False,
                    raise_on_exception=False,
                )
                success_total += success
                if failures:
                    for failure in failures:
                        op_type = next(iter(failure.keys()))
                        info = failure[op_type]
                        doc_id = info.get("_id")
                        reason = info.get("error", {}).get("reason", "unknown")
                        logger.warning("Bulk index partial failure: doc=%s reason=%s", doc_id, reason)
            except Exception as exc:
                logger.error("Bulk index chunk failed (%s docs): %s", len(chunk), exc)
        self.client.indices.refresh(index=self.index_name)
        return success_total

    def hybrid_search(
        self,
        query_text: Optional[str],
        query_vector: Optional[List[float]],
        vector_field: str,
        text_field: str,
        size: int,
        lexical_weight: float,
        score_field: str,
    ) -> List[Dict]:
        if not query_text or not query_vector:
            return []
        body = {
            "size": size,
            "_source": ["doc_number", "publication_date", "ipc_prefix", "claim_text", "abstract_text"],
            "query": {
                "script_score": {
                    "query": {
                        "match": {
                            text_field: {
                                "query": query_text,
                                "operator": "and",
                            }
                        }
                    },
                    # Combine vector similarity with lexical score. Taking max later during merge per requirements.
                    "script": {
                        "source": f"cosineSimilarity(params.vector, '{vector_field}') + params.weight * _score",
                        "params": {
                            "vector": query_vector,
                            "weight": lexical_weight,
                        },
                    },
                }
            },
        }
        response = self.client.search(index=self.index_name, body=body)
        hits = []
        for hit in response["hits"]["hits"]:
            source = hit.get("_source", {})
            doc_number = source.get("doc_number") or hit.get("_id")
            hits.append(
                {
                    "doc_number": doc_number,
                    "publication_date": source.get("publication_date"),
                    "ipc_prefix": source.get("ipc_prefix", []),
                    "claim_text": source.get("claim_text"),
                    "abstract_text": source.get("abstract_text"),
                    score_field: hit["_score"],
                }
            )
        return hits


class PatentSearchValidator:
    """Coordinates the end-to-end validation pipeline."""

    def __init__(
        self,
        cosmos_client: CosmosPatentClient,
        embedding_service: EmbeddingService,
        es_manager: ElasticsearchIndexManager,
        max_docs: int,
        top_k: int,
        lexical_weight: float,
        num_queries: int,
    ):
        self.cosmos = cosmos_client
        self.embedding = embedding_service
        self.es = es_manager
        self.max_docs = max_docs
        self.top_k = top_k
        self.lexical_weight = lexical_weight
        self.num_queries = num_queries
        self.generator = QueryGenerator()
        self.bulk_batch_size = BULK_BATCH_SIZE

    def load_cases(self, csv_path: str, sample_size: int, seed: Optional[int]) -> List[Dict]:
        df = pd.read_csv(csv_path)
        if sample_size > 0:
            if sample_size >= len(df):
                subset = df
            else:
                subset = df.head(sample_size)
        else:
            subset = df
        return subset.to_dict("records")

    def prepare_cases(self, rows: Sequence[Dict]) -> List[CaseContext]:
        contexts: List[CaseContext] = []
        for row in rows:
            syutugan = row["syutugan"]
            doc_number = normalize_patent_number(syutugan)
            cosmos_doc = self.cosmos.fetch_document_by_doc_number(doc_number, raw_identifier=syutugan)
            if not cosmos_doc:
                logger.warning("Syutugan %s (%s) not found in Cosmos DB.", syutugan, doc_number)
                continue
            ipc_prefix = self._extract_ipc_prefix(cosmos_doc)
            if not ipc_prefix:
                logger.warning("IPC prefix missing for %s.", syutugan)
                continue
            claim_text = self._extract_claim_text(cosmos_doc)
            abstract_text = self._extract_abstract_text(cosmos_doc)
            if not claim_text and not abstract_text:
                logger.warning("Both claim#1 and abstract missing for %s.", syutugan)
                continue
            ax_docs = parse_ax_docs(row.get("ax_docs", ""))
            ax_doc_numbers = [normalize_patent_number(item) for item in ax_docs]
            contexts.append(
                CaseContext(
                    case_id=row["case_id"],
                    syutugan=syutugan,
                    doc_number=doc_number,
                    ipc_prefix=ipc_prefix,
                    ipc_query=format_ipc_query(ipc_prefix),
                    ax_docs=ax_docs,
                    ax_doc_numbers=ax_doc_numbers,
                    claim_text=claim_text,
                    abstract_text=abstract_text,
                )
            )
        return contexts

    def ingest_by_prefixes(self, prefixes: Dict[str, str]) -> Dict[str, Dict[str, int]]:
        stats: Dict[str, Dict[str, int]] = {}
        indexed_doc_ids: set[str] = set()
        for prefix, ipc_query in prefixes.items():
            logger.info("Phase[Ingest] IPC %s: querying Cosmos...", prefix)
            raw_docs = self.cosmos.fetch_by_ipc_text(ipc_query)
            logger.info("IPC %s: fetched %s candidates", prefix, len(raw_docs))
            sorted_docs = sorted(raw_docs, key=self._publication_date_key, reverse=True)
            limited_docs = sorted_docs[: self.max_docs]
            batch: List[Dict] = []
            indexed_count = 0
            filtered_doc_numbers: Set[str] = set()
            limited_doc_numbers: Set[str] = set()
            for doc in raw_docs:
                doc_number = self._extract_doc_number(doc)
                if doc_number:
                    filtered_doc_numbers.add(doc_number)
            for doc in limited_docs:
                doc_number = self._extract_doc_number(doc)
                if not doc_number or doc_number in indexed_doc_ids:
                    continue
                limited_doc_numbers.add(doc_number)
                claim_text = self._extract_claim_text(doc)
                abstract_text = self._extract_abstract_text(doc)
                if not claim_text or not abstract_text:
                    continue
                claim_vector = self.embedding.embed_text(claim_text)
                abstract_vector = self.embedding.embed_text(abstract_text)
                if not claim_vector or not abstract_vector:
                    continue
                batch.append(
                    {
                        "doc_number": doc_number,
                        "publication_date": self._extract_publication_date(doc),
                        "ipc_prefix": doc.get("bibliographic", {}).get("classification", {}).get("ipc_prefix", []),
                        "claim_vector": claim_vector,
                        "abstract_vector": abstract_vector,
                        "claim_text": claim_text,
                        "abstract_text": abstract_text,
                    }
                )
                indexed_doc_ids.add(doc_number)
                if len(batch) >= self.bulk_batch_size:
                    indexed_count += self._flush_index_batch(prefix, batch)
                    batch.clear()
            if batch:
                indexed_count += self._flush_index_batch(prefix, batch)
            stats[prefix] = {
                "ipc_filtered": len(raw_docs),
                "sorted": len(sorted_docs),
                "limited": len(limited_docs),
                "indexed": indexed_count,
                "filtered_doc_numbers": filtered_doc_numbers,
                "limited_doc_numbers": limited_doc_numbers,
            }
            logger.info(
                "IPC %s: %s filtered -> %s sorted -> %s limited -> %s indexed",
                prefix,
                len(raw_docs),
                len(sorted_docs),
                len(limited_docs),
                indexed_count,
            )
        return stats

    def _run_multi_query_field(
        self,
        section_label: str,
        base_text: Optional[str],
        vector_field: str,
        text_field: str,
        score_field: str,
    ) -> Tuple[List[Dict], List[str]]:
        if not base_text:
            return [], [f"{section_label}:0"]

        logger.info(
            "Phase[Search] Generating %s queries for %s (top_k=%s)...",
            self.num_queries,
            section_label,
            self.top_k,
        )
        generated = self.generator.generate_queries(section_label, base_text, self.num_queries)
        vector_passages = generated["vector_passages"]
        keyword_queries = generated["keyword_queries"]

        aggregated: Dict[str, Dict] = {}
        query_stats: List[str] = []

        for idx, (vector_passage, keyword_query) in enumerate(zip(vector_passages, keyword_queries)):
            query_vector = self.embedding.embed_text(vector_passage)
            if not query_vector:
                query_stats.append(f"q{idx}:0")
                continue
            hits = self.es.hybrid_search(
                query_text=keyword_query,
                query_vector=query_vector,
                vector_field=vector_field,
                text_field=text_field,
                size=self.top_k,
                lexical_weight=self.lexical_weight,
                score_field=score_field,
            )
            unique_ids = {hit.get("doc_number") for hit in hits if hit.get("doc_number")}
            query_stats.append(f"q{idx}:{len(unique_ids)}")
            self._update_multi_query_hits(aggregated, hits, score_field)

        aggregated_list = list(aggregated.values())
        aggregated_list.sort(
            key=lambda item: item.get(score_field, float("-inf")),
            reverse=True,
        )
        logger.info(
            "Phase[Search] %s multi-query completed | unique_docs=%s",
            section_label,
            len(aggregated_list),
        )
        return aggregated_list, query_stats

    @staticmethod
    def _update_multi_query_hits(
        aggregated: Dict[str, Dict],
        hits: List[Dict],
        score_field: str,
    ) -> None:
        for hit in hits:
            doc_number = hit.get("doc_number")
            if not doc_number:
                continue
            entry = aggregated.setdefault(
                doc_number,
                {
                    "doc_number": doc_number,
                    "publication_date": hit.get("publication_date"),
                    "ipc_prefix": hit.get("ipc_prefix", []),
                    "claim_text": hit.get("claim_text"),
                    "abstract_text": hit.get("abstract_text"),
                    "times_hit": 0,
                },
            )
            entry["times_hit"] = entry.get("times_hit", 0) + 1
            score = hit.get(score_field)
            if score is None:
                continue
            current_best = entry.get(score_field)
            if current_best is None or score > current_best:
                entry[score_field] = score
                if hit.get("publication_date"):
                    entry["publication_date"] = hit.get("publication_date")
                if hit.get("ipc_prefix"):
                    entry["ipc_prefix"] = hit.get("ipc_prefix")
                if hit.get("claim_text"):
                    entry["claim_text"] = hit.get("claim_text")
                if hit.get("abstract_text"):
                    entry["abstract_text"] = hit.get("abstract_text")

    def _flush_index_batch(self, prefix: str, batch: List[Dict]) -> int:
        if not batch:
            return 0
        logger.info(
            "Phase[Ingest] IPC %s: indexing chunk of %s docs (batch_size=%s)",
            prefix,
            len(batch),
            self.bulk_batch_size,
        )
        count = self.es.bulk_index(batch)
        logger.info("Phase[Ingest] IPC %s: chunk indexed (%s docs).", prefix, count)
        return count

    def evaluate_cases(
        self,
        contexts: Sequence[CaseContext],
        prefix_stats: Dict[str, Dict[str, int]],
        on_case_completed: Optional[Callable[[CaseResult], None]] = None,
    ) -> List[CaseResult]:
        results: List[CaseResult] = []
        for context in contexts:
            counts = {
                "ipc_filtered": prefix_stats.get(context.ipc_prefix, {}).get("ipc_filtered", 0),
                "sorted": prefix_stats.get(context.ipc_prefix, {}).get("sorted", 0),
                "limited": prefix_stats.get(context.ipc_prefix, {}).get("limited", 0),
                "indexed": prefix_stats.get(context.ipc_prefix, {}).get("indexed", 0),
                "claim_hits": 0,
                "abstract_hits": 0,
                "merged_hits": 0,
            }
            filtered_set = prefix_stats.get(context.ipc_prefix, {}).get("filtered_doc_numbers") or set()
            limited_set = prefix_stats.get(context.ipc_prefix, {}).get("limited_doc_numbers") or set()
            ax_in_filtered = any(
                digits and digits in filtered_set for digits in context.ax_doc_numbers if digits
            )
            ax_in_limited = any(
                digits and digits in limited_set for digits in context.ax_doc_numbers if digits
            )
            if not ax_in_filtered:
                logger.info(
                    "Phase[Case] %s skipped: AX not present after IPC filter.",
                    context.case_id,
                )
                skip_result = CaseResult(
                    case_id=context.case_id,
                    syutugan=context.syutugan,
                    ipc_prefix=context.ipc_prefix,
                    counts=counts,
                    hit=False,
                    claim_query_stats=["skipped:ipc_filter"],
                    abstract_query_stats=["skipped:ipc_filter"],
                    ax_in_ipc_filtered=False,
                    ax_in_sorted=False,
                )
                results.append(skip_result)
                if on_case_completed:
                    on_case_completed(skip_result)
                continue
            if not ax_in_limited:
                logger.info(
                    "Phase[Case] %s skipped: AX not present after date sort/limit.",
                    context.case_id,
                )
                skip_result = CaseResult(
                    case_id=context.case_id,
                    syutugan=context.syutugan,
                    ipc_prefix=context.ipc_prefix,
                    counts=counts,
                    hit=False,
                    claim_query_stats=["skipped:date_limit"],
                    abstract_query_stats=["skipped:date_limit"],
                    ax_in_ipc_filtered=True,
                    ax_in_sorted=False,
                )
                results.append(skip_result)
                if on_case_completed:
                    on_case_completed(skip_result)
                continue
            logger.info(
                "Phase[Case] %s | syutugan=%s | IPC=%s -> starting claim/abstract searches.",
                context.case_id,
                context.syutugan,
                context.ipc_prefix,
            )
            try:
                claim_hits, claim_query_stats = self._run_multi_query_field(
                    "請求項1",
                    context.claim_text,
                    "claim_vector",
                    "claim_text",
                    "score_claim",
                )
            except Exception as exc:
                logger.error("Multi-query (claims) failed for %s: %s", context.syutugan, exc)
                claim_hits, claim_query_stats = [], [f"error:{exc}"]

            try:
                abstract_hits, abstract_query_stats = self._run_multi_query_field(
                    "要約",
                    context.abstract_text,
                    "abstract_vector",
                    "abstract_text",
                    "score_abstract",
                )
            except Exception as exc:
                logger.error("Multi-query (abstract) failed for %s: %s", context.syutugan, exc)
                abstract_hits, abstract_query_stats = [], [f"error:{exc}"]

            merged_hits = self._merge_results(claim_hits, abstract_hits)
            merged_doc_numbers = {hit["doc_number"] for hit in merged_hits}
            matched = [
                original
                for original, digits in zip(context.ax_docs, context.ax_doc_numbers)
                if digits and digits in merged_doc_numbers
            ]
            counts.update(
                {
                    "claim_hits": len(claim_hits),
                    "abstract_hits": len(abstract_hits),
                    "merged_hits": len(merged_hits),
                }
            )
            results.append(
                CaseResult(
                    case_id=context.case_id,
                    syutugan=context.syutugan,
                    ipc_prefix=context.ipc_prefix,
                    counts=counts,
                    hit=bool(matched),
                    matched_ax_docs=matched,
                    claim_hits=len(claim_hits),
                    abstract_hits=len(abstract_hits),
                    merged_hits=len(merged_hits),
                    claim_query_stats=claim_query_stats,
                    abstract_query_stats=abstract_query_stats,
                    ax_in_ipc_filtered=ax_in_filtered,
                    ax_in_sorted=ax_in_limited,
                )
            )
            if on_case_completed:
                on_case_completed(results[-1])
            logger.info(
                "Phase[Case] %s completed | claim_hits=%s | abstract_hits=%s | merged=%s | hit=%s",
                context.case_id,
                len(claim_hits),
                len(abstract_hits),
                len(merged_hits),
                bool(matched),
            )
        return results

    @staticmethod
    def _extract_doc_number(doc: Dict) -> Optional[str]:
        bibliographic = doc.get("bibliographic", {})
        publication = bibliographic.get("publication", {})
        doc_number = publication.get("doc_number")
        if doc_number:
            return str(doc_number)
        application = bibliographic.get("application", {})
        if application.get("doc_number"):
            return str(application["doc_number"])
        return None

    @staticmethod
    def _extract_publication_date(doc: Dict) -> Optional[str]:
        bibliographic = doc.get("bibliographic", {})
        publication = bibliographic.get("publication", {})
        return normalize_publication_date(publication.get("date"))

    @staticmethod
    def _extract_claim_text(doc: Dict) -> Optional[str]:
        claims = doc.get("claims") or []
        if not claims:
            return None
        first_claim = claims[0]
        if isinstance(first_claim, str):
            return first_claim.strip()
        if isinstance(first_claim, dict):
            for key in ("text", "claim_text", "body"):
                value = first_claim.get(key)
                if value:
                    return value.strip()
        return None

    @staticmethod
    def _extract_abstract_text(doc: Dict) -> Optional[str]:
        abstract = doc.get("abstract")
        if isinstance(abstract, str):
            return abstract.strip()
        if isinstance(abstract, dict):
            for key in ("text", "abstract", "body"):
                value = abstract.get(key)
                if value:
                    return value.strip()
        if isinstance(abstract, list) and abstract:
            first = abstract[0]
            if isinstance(first, str):
                return first.strip()
            if isinstance(first, dict):
                return (first.get("text") or "").strip()
        return None

    @staticmethod
    def _extract_ipc_prefix(doc: Dict) -> Optional[str]:
        return extract_ipc_prefix_from_document(doc)

    @staticmethod
    def _publication_date_key(doc: Dict) -> str:
        date = PatentSearchValidator._extract_publication_date(doc)
        return date or ""

    @staticmethod
    def _merge_results(claim_hits: List[Dict], abstract_hits: List[Dict]) -> List[Dict]:
        merged: Dict[str, Dict] = {}
        for hit in claim_hits:
            merged[hit["doc_number"]] = dict(hit)
        for hit in abstract_hits:
            entry = merged.setdefault(hit["doc_number"], {})
            entry.setdefault("doc_number", hit["doc_number"])
            entry.setdefault("publication_date", hit.get("publication_date"))
            entry.setdefault("ipc_prefix", hit.get("ipc_prefix", []))
            entry.setdefault("claim_text", hit.get("claim_text"))
            entry.setdefault("abstract_text", hit.get("abstract_text"))
            entry["score_abstract"] = hit.get("score_abstract")
        for doc_number, entry in merged.items():
            claim_score = entry.get("score_claim", float("-inf"))
            abstract_score = entry.get("score_abstract", float("-inf"))
            entry["combined_score"] = max(
                claim_score,
                abstract_score,
            )  # Combining scores via max per requirement
        return sorted(merged.values(), key=lambda item: item.get("combined_score", 0), reverse=True)


def run_validator(args: argparse.Namespace):
    run_suffix = timestamp_suffix()
    run_id = f"validation_{run_suffix}"
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"ipc_validator_{run_suffix}.log")
    file_handler = attach_file_logger(log_path)
    logger.info("Run ID %s started. File log: %s", run_id, log_path)

    result_csv_path = args.result_csv
    result_headers = [
        "run_id",
        "case_id",
        "syutugan",
        "ipc_prefix",
        "ipc_filtered",
        "sorted",
        "limited",
        "indexed",
        "claim_hits",
        "abstract_hits",
        "merged_hits",
        "ax_hit",
        "matched_ax_docs",
        "claim_queries",
        "abstract_queries",
        "ax_in_ipc_filtered",
        "ax_in_sorted",
    ]
    ensure_csv(result_csv_path, result_headers)

    cosmos = CosmosPatentClient(
        endpoint=os.getenv("COSMOS_ENDPOINT"),
        key=os.getenv("COSMOS_KEY"),
        database=os.getenv("COSMOS_DATABASE", "patent_json"),
        container=os.getenv("COSMOS_CONTAINER", "patent_csv1"),
    )
    embedding_service = EmbeddingService()
    es_manager = ElasticsearchIndexManager(
        host=os.getenv("ELASTICSEARCH_HOST", "localhost"),
        port=int(os.getenv("ELASTICSEARCH_PORT", "9200")),
        index_name="patents_vector_index",
    )
    validator = PatentSearchValidator(
        cosmos_client=cosmos,
        embedding_service=embedding_service,
        es_manager=es_manager,
        max_docs=args.max_docs,
        top_k=args.top_k,
        lexical_weight=args.lexical_weight,
        num_queries=args.num_queries,
    )

    logger.info(
        "Loading test cases from %s (sample_size=%s, seed=%s)",
        args.test_cases,
        args.sample_size,
        args.seed,
    )
    cases = validator.load_cases(args.test_cases, args.sample_size, args.seed)
    contexts = validator.prepare_cases(cases)
    if not contexts:
        logger.error("No valid test cases found. Aborting.")
        sys.exit(1)
    logger.info("Prepared %s valid cases (requested sample=%s).", len(contexts), args.sample_size)

    embedding_dim = embedding_service.ensure_dimension()
    logger.info("Embedding dimension determined: %s", embedding_dim)
    es_manager.recreate_index(embedding_dim)

    prefix_map = {}
    for context in contexts:
        prefix_map.setdefault(context.ipc_prefix, context.ipc_query)
    logger.info("Starting ingestion for %s IPC prefixes.", len(prefix_map))
    prefix_stats = validator.ingest_by_prefixes(prefix_map)
    logger.info("Completed ingestion for all IPC prefixes.")

    def append_case_result(case_result: CaseResult):
        row = {
            "run_id": run_id,
            "case_id": case_result.case_id,
            "syutugan": case_result.syutugan,
            "ipc_prefix": case_result.ipc_prefix,
            "ipc_filtered": case_result.counts["ipc_filtered"],
            "sorted": case_result.counts["sorted"],
            "limited": case_result.counts["limited"],
            "indexed": case_result.counts["indexed"],
            "claim_hits": case_result.claim_hits,
            "abstract_hits": case_result.abstract_hits,
            "merged_hits": case_result.merged_hits,
            "ax_hit": case_result.hit,
            "matched_ax_docs": ";".join(case_result.matched_ax_docs) if case_result.matched_ax_docs else "",
            "claim_queries": ";".join(case_result.claim_query_stats),
            "abstract_queries": ";".join(case_result.abstract_query_stats),
            "ax_in_ipc_filtered": case_result.ax_in_ipc_filtered,
            "ax_in_sorted": case_result.ax_in_sorted,
        }
        with open(result_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result_headers)
            writer.writerow(row)

    results = validator.evaluate_cases(contexts, prefix_stats, on_case_completed=append_case_result)
    summary_rows = []
    for result in results:
        row = {
            "run_id": run_id,
            "case_id": result.case_id,
            "syutugan": result.syutugan,
            "ipc_prefix": result.ipc_prefix,
            "ipc_filtered": result.counts["ipc_filtered"],
            "sorted": result.counts["sorted"],
            "limited": result.counts["limited"],
            "indexed": result.counts["indexed"],
            "claim_hits": result.claim_hits,
            "abstract_hits": result.abstract_hits,
            "merged_hits": result.merged_hits,
            "ax_hit": result.hit,
            "matched_ax_docs": ";".join(result.matched_ax_docs) if result.matched_ax_docs else "",
            "claim_queries": ";".join(result.claim_query_stats),
            "abstract_queries": ";".join(result.abstract_query_stats),
            "ax_in_ipc_filtered": result.ax_in_ipc_filtered,
            "ax_in_sorted": result.ax_in_sorted,
        }
        summary_rows.append(row)
        logger.info(
            "[%s] %s IPC=%s | counts=%s | queries(claim=%s, abstract=%s) | hit=%s | matched=%s",
            run_id,
            result.case_id,
            result.ipc_prefix,
            {
                "ipc_filtered": result.counts["ipc_filtered"],
                "sorted": result.counts["sorted"],
                "limited": result.counts["limited"],
                "indexed": result.counts["indexed"],
                "claim_hits": result.claim_hits,
                "abstract_hits": result.abstract_hits,
                "merged_hits": result.merged_hits,
            },
            result.claim_query_stats,
            result.abstract_query_stats,
            result.hit,
            result.matched_ax_docs,
        )

    summary_df = pd.DataFrame(summary_rows)
    pd.set_option("display.max_columns", None)
    print("\n=== Validation Summary ===")
    print(summary_df.to_string(index=False))
    logger.info("Validation run %s completed for %s cases.", run_id, len(results))

    logger.info("Detailed log stored at %s", log_path)
    logger.removeHandler(file_handler)
    file_handler.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IPC-based patent search precision validator.")
    parser.add_argument("--test-cases", default="test_cases.csv", help="Path to test_cases.csv")
    parser.add_argument("--sample-size", type=int, default=0, help="Number of cases to evaluate (0=all)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-docs", type=int, default=30000, help="Maximum documents per IPC prefix")
    parser.add_argument("--top-k", type=int, default=100, help="Top-K per query (claims/abstract multi-query)")
    parser.add_argument("--lexical-weight", type=float, default=0.15, help="Weight for lexical score in hybrid search")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of multi-query prompts per section")
    parser.add_argument("--log-dir", default="logs", help="Directory to store per-run log files")
    parser.add_argument("--result-csv", default="validation_results.csv", help="Path to append per-case results")
    return parser.parse_args()


if __name__ == "__main__":
    run_validator(parse_args())

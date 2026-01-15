from __future__ import annotations

import os
from dataclasses import dataclass, field


def _get_env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "t", "yes", "y"}


@dataclass(slots=True)
class PipelineConfig:
    """Runtime configuration for the ingestion pipeline."""

    es_host: str = field(default_factory=lambda: os.getenv("ES_HOST", "http://localhost:9200"))
    es_stage1_index: str = field(default_factory=lambda: os.getenv("ES_STAGE1_INDEX", "patent_stage1"))
    es_vector_field: str = field(default_factory=lambda: os.getenv("ES_VECTOR_FIELD", "document_vector"))
    es_vector_dims: int = field(default_factory=lambda: int(os.getenv("ES_VECTOR_DIMS", "3072")))
    es_vector_k: int = field(default=1000)
    es_num_candidates: int = field(default_factory=lambda: int(os.getenv("ES_VECTOR_NUM_CANDIDATES", "10000")))
    es_stage1_limit: int = field(default_factory=lambda: int(os.getenv("ES_MAX_STAGE1_RESULTS", "40000")))

    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "neo4j"))

    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    cosmos_endpoint: str = field(default_factory=lambda: os.getenv("COSMOS_ENDPOINT", "http://localhost:7070"))
    cosmos_key: str = field(default_factory=lambda: os.getenv("COSMOS_KEY", "dummy-key"))
    cosmos_database: str = field(
        default_factory=lambda: os.getenv("COSMOS_DATABASE") or os.getenv("DATABASE_NAME", "patent_db")
    )
    cosmos_container: str = field(
        default_factory=lambda: os.getenv("COSMOS_CONTAINER") or os.getenv("CONTAINER_NAME", "patents")
    )

    azure_openai_endpoint: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AOAI_ENDPOINT", "")
    )
    azure_openai_key: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    )
    azure_openai_deployment: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("EMBED_MODEL", "text-embedding-3-large")
    )
    azure_openai_api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"))
    azure_openai_chat_deployment: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("AOAI_CHAT_MODEL", "")
    )

    vectorizer_url: str = field(default_factory=lambda: os.getenv("VECTORIZER_URL", "http://localhost:9000"))
    allow_vectorizer_fallback: bool = field(default_factory=lambda: _get_env_bool("ALLOW_VECTORIZER_FALLBACK", True))

    cohere_api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    cohere_rerank_model: str = field(default_factory=lambda: os.getenv("COHERE_RERANK_MODEL", "rerank-v4.0-pro"))

    def ensure_embedding_credentials(self) -> None:
        if not self.azure_openai_key or not self.azure_openai_endpoint:
            raise RuntimeError("Azure OpenAI credentials are required. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")

    def ensure_rerank_credentials(self) -> None:
        if not self.cohere_api_key:
            raise RuntimeError("Cohere API key is required. Set COHERE_API_KEY for reranking.")

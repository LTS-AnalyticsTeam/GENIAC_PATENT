import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (HnswAlgorithmConfiguration,
                                                   SearchableField,
                                                   SearchField,
                                                   SearchFieldDataType,
                                                   SearchIndex,
                                                   SemanticConfiguration,
                                                   SemanticField,
                                                   SemanticPrioritizedFields,
                                                   SemanticSearch, SimpleField,
                                                   VectorSearch,
                                                   VectorSearchProfile)
from openai import AzureOpenAI

# ==== 環境変数 ====
SERVICE_ENDPOINT = " "
ADMIN_KEY = ""
INDEX_NAME = "fi_classification_index"

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_KEY = ""
EMBED_MODEL = "text-embedding-3-small"

JSONL_DIR = Path("FI/output/JSONL")
FILES = [JSONL_DIR / f"FI_{s}.jsonl" for s in list("ABCDEFGH")]



def ensure_index_exists():
    idx_client = SearchIndexClient(endpoint=SERVICE_ENDPOINT, credential=AzureKeyCredential(ADMIN_KEY))
    try:
        idx_client.get_index(INDEX_NAME)  # あれば何もしない
        print(f"[INFO] index exists: {INDEX_NAME}")
        return
    except ResourceNotFoundError:
        print(f"[INFO] creating index: {INDEX_NAME}")

    # フィールド定義
    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True, facetable=False),
        SearchableField(name="code", type=SearchFieldDataType.String, analyzer="ja.microsoft", filterable=True, sortable=True),
        SearchableField(name="title", type=SearchFieldDataType.String, analyzer="ja.microsoft"),
        SearchableField(name="description", analyzer_name="ja.microsoft"),
        SimpleField(name="level", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="section", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchField(name="breadcrumbs", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True, facetable=False),
        SearchField(name="children_codes", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        SearchField(name="siblings_codes", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        SimpleField(name="parent_code", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_file", type=SearchFieldDataType.String, filterable=False),
        # ベクトル格納先
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=1536,                # text-embedding-3-small
            vector_search_profile_name="vector_profile",
        ),
    ]

    # ベクトル検索設定（HNSW）
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="vector_profile", algorithm_configuration_name="hnsw")],
    )

    # セマンティック検索（任意）
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="sem-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="description")],
                ),
            )
        ]
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )
    idx_client.create_index(index)
    print(f"[OK] created index: {INDEX_NAME}")

# ==== Azure OpenAI client ====
aoai = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def embed_texts(texts: List[str]) -> List[List[float]]:
    # バッチ埋め込み（日本語OK）
    resp = aoai.embeddings.create(
        input=texts,
        model=EMBED_MODEL,
    )
    return [d.embedding for d in resp.data]

def sanitize_key(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "_-=") else "_" for ch in s)

def to_doc(obj: Dict[str, Any]) -> Dict[str, Any]:
    # chunk_id 決定（既にあればそれを使う）
    chunk_id = obj.get("chunk_id") or sanitize_key(obj.get("code") or obj.get("id") or "")
    return {
        "chunk_id": chunk_id,
        "code": obj.get("code"),
        "title": obj.get("title"),
        "description": obj.get("description") or "",
        "level": obj.get("level"),
        "section": obj.get("section"),
        "breadcrumbs": obj.get("breadcrumbs") or [],
        "children_codes": obj.get("children_codes") or [],
        "siblings_codes": obj.get("siblings_codes") or [],
        "parent_code": obj.get("parent_code"),
        "source_file": obj.get("source_file"),
        # "content_vector": 後で埋め込む
    }

def stream_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def upsert_with_embeddings():
    client = SearchClient(SERVICE_ENDPOINT, INDEX_NAME, AzureKeyCredential(ADMIN_KEY))
    BATCH = 64  # 埋め込み＆アップロードのバッチ

    for jf in FILES:
        if not jf.exists():
            print(f"[WARN] not found: {jf}")
            continue

        print(f"[INFO] processing {jf.name}")
        buf_docs, buf_texts = [], []

        for obj in stream_jsonl(jf):
            doc = to_doc(obj)
            buf_docs.append(doc)
            buf_texts.append(doc["description"])

            if len(buf_docs) >= BATCH:
                vecs = embed_texts(buf_texts)
                for d, v in zip(buf_docs, vecs):
                    d["content_vector"] = v
                client.upload_documents(buf_docs)
                buf_docs, buf_texts = [], []

        if buf_docs:
            vecs = embed_texts(buf_texts)
            for d, v in zip(buf_docs, vecs):
                d["content_vector"] = v
            client.upload_documents(buf_docs)

        print(f"[OK] {jf.name} done.")

if __name__ == "__main__":
    ensure_index_exists()
    upsert_with_embeddings()


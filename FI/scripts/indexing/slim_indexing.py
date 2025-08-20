import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (HnswAlgorithmConfiguration,
                                                   SearchableField,
                                                   SearchField,
                                                   SearchFieldDataType,
                                                   SearchIndex, SimpleField,
                                                   VectorSearch,
                                                   VectorSearchProfile)
from openai import AzureOpenAI

# ===================== 環境設定 =====================

# Azure AI Search
SERVICE_ENDPOINT = ""
ADMIN_KEY = ""
INDEX_NAME = "fi_classification_index"

# Azure OpenAI（埋め込み用）
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_KEY = ""
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536

# 入力（統合済みのスリムJSONL：各行 {"chunk_id","code","search_text"}）
INPUT_JSONL = Path("output/JSONL/FI_slim.jsonl")

# バッチサイズ
BATCH = 64

# ===================== クライアント =====================

index_client = SearchIndexClient(endpoint=SERVICE_ENDPOINT, credential=AzureKeyCredential(ADMIN_KEY))
search_client = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(ADMIN_KEY))
aoai = AzureOpenAI(api_key=AZURE_OPENAI_KEY, api_version="2024-02-01", azure_endpoint=AZURE_OPENAI_ENDPOINT)

# ===================== ユーティリティ =====================

def sanitize_key(s: str) -> str:
    if not s:
        return ""
    return "".join(ch if (ch.isalnum() or ch in "_-=") else "_" for ch in s)

def stream_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = aoai.embeddings.create(input=texts, model=EMBED_MODEL)
    return [d.embedding for d in resp.data]

# ===================== インデックス作成 =====================

def ensure_index_exists():
    try:
        index_client.get_index(INDEX_NAME)
        print(f"[INFO] index exists: {INDEX_NAME}")
        return
    except ResourceNotFoundError:
        print(f"[INFO] creating index: {INDEX_NAME}")

    # 最小スキーマ：chunk_id, code, search_text, content_vector
    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=False),
        SimpleField(name="code", type=SearchFieldDataType.String, filterable=True, sortable=False),
        SearchableField(name="search_text", analyzer_name="ja.microsoft"),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=EMBED_DIMS,
            vector_search_profile_name="vector_profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="vector_profile", algorithm_configuration_name="hnsw")],
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        # semantic_search は付けない（スリム化）
    )
    index_client.create_index(index)
    print(f"[OK] created index: {INDEX_NAME}")

# ===================== アップサート =====================

def to_doc(raw: Dict[str, Any]) -> Dict[str, Any]:
    # 必須3フィールドのみを受け取り、Search用のdocに整形
    chunk_id = raw.get("chunk_id") or sanitize_key(raw.get("code", ""))
    return {
        "chunk_id": chunk_id,
        "code": raw.get("code", ""),
        "search_text": raw.get("search_text", "") or "",
        # content_vector は後で付与
    }

def upload_documents_with_retry(docs: List[Dict[str, Any]], retries: int = 2):
    # クォータ超過はリトライしても通りませんが、短い一時的な429/503向けに軽く実装
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            result = search_client.upload_documents(docs)
            # 失敗したレコードがないかだけチェック
            failed = [r for r in result if r.succeeded is False]
            if failed:
                print(f"[WARN] {len(failed)} docs failed (showing first 3 ids): {[r.key for r in failed[:3]]}")
            return
        except HttpResponseError as e:
            print(f"[WARN] upload attempt {attempt+1}/{retries+1} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2.0

def upsert_with_embeddings(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"not found: {input_path}")

    buf_docs: List[Dict[str, Any]] = []
    buf_texts: List[str] = []

    total = 0
    for raw in stream_jsonl(input_path):
        doc = to_doc(raw)
        buf_docs.append(doc)
        buf_texts.append(doc["search_text"])

        if len(buf_docs) >= BATCH:
            vecs = embed_texts(buf_texts)
            for d, v in zip(buf_docs, vecs):
                d["content_vector"] = v
            upload_documents_with_retry(buf_docs)

            total += len(buf_docs)
            print(f"[INFO] uploaded {total} docs...")
            buf_docs, buf_texts = [], []

    if buf_docs:
        vecs = embed_texts(buf_texts)
        for d, v in zip(buf_docs, vecs):
            d["content_vector"] = v
        upload_documents_with_retry(buf_docs)
        total += len(buf_docs)

    print(f"[OK] done. total uploaded: {total}")

# ===================== エントリポイント =====================

if __name__ == "__main__":
    ensure_index_exists()
    upsert_with_embeddings(INPUT_JSONL)

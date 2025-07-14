import os
import time

from elasticsearch import ConnectionError as ESConnectionError
from elasticsearch import Elasticsearch, NotFoundError
from fastapi import FastAPI, HTTPException

ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")


def connect_es_with_retry(retries: int = 12, wait: int = 5) -> Elasticsearch:
    """
    Elasticsearch へ再試行付きで接続
    - retries: 最大試行回数
    - wait:    再試行間隔（秒）
    """
    for i in range(1, retries + 1):
        try:
            es_client = Elasticsearch(ES_HOST, request_timeout=30)
            # 任意の軽い API で疎通確認
            if es_client.ping():
                print("Connected to Elasticsearch")
                return es_client
        except ESConnectionError:
            pass
        print(f"[{i}/{retries}] Elasticsearch 未応答、{wait}s 後に再試行…")
        time.sleep(wait)
    raise RuntimeError("Elasticsearch に接続できませんでした")


es = connect_es_with_retry()

# ───────────────────────────────────────────────────────────────
# 1️⃣ Elasticsearch クライアント初期化
# ───────────────────────────────────────────────────────────────
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
es = Elasticsearch(ES_HOST, request_timeout=30)

INDEX_NAME = "documents"

# ───────────────────────────────────────────────────────────────
# 2️⃣ FastAPI アプリ定義
# ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Elasticsearch + FastAPI サンプル",
    description="Docker Compose で起動する開発用スタック",
    version="0.1.0",
)


# ───────────────────────────────────────────────────────────────
# 3️⃣ アプリ起動時にインデックスを自動作成（存在しない場合のみ）
# ───────────────────────────────────────────────────────────────
@app.on_event("startup")
def create_index_if_needed() -> None:
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(
            index=INDEX_NAME,
            mappings={
                "properties": {
                    "title":   {"type": "text"},
                    "content": {"type": "text"},
                    "tags":    {"type": "keyword"},
                }
            },
        )


# ───────────────────────────────────────────────────────────────
# 4️⃣ CRUD + 検索エンドポイント
# ───────────────────────────────────────────────────────────────
@app.post("/documents/{doc_id}")
def create_document(doc_id: str, body: dict):
    """
    ドキュメント登録（全文置換アップサート）
    body 例:
      {
        "title": "Hello",
        "content": "World",
        "tags": ["greeting"]
      }
    """
    return es.index(index=INDEX_NAME, id=doc_id, document=body)


@app.get("/documents/{doc_id}")
def read_document(doc_id: str):
    """
    単一ドキュメント取得
    """
    try:
        res = es.get(index=INDEX_NAME, id=doc_id)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    return res["_source"]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """
    ドキュメント削除
    """
    try:
        es.delete(index=INDEX_NAME, id=doc_id)
        return {"deleted": doc_id}
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")


@app.get("/search")
def search(q: str, size: int = 10):
    """
    簡易全文検索
    - q:   検索クエリ文字列
    - size: 返却件数
    """
    query = {
        "query": {
            "multi_match": {
                "query":  q,
                "fields": ["title^2", "content", "tags"],
                "type":   "best_fields",
            }
        }
    }
    res = es.search(index=INDEX_NAME, body=query, size=size)
    return [hit["_source"] | {"_score": hit["_score"]} for hit in res["hits"]["hits"]]

import os
import time

from dotenv import load_dotenv
from elasticsearch import ConnectionError as ESConnectionError
from elasticsearch import Elasticsearch, NotFoundError
from fastapi import FastAPI, HTTPException
from vector_search import (
    HybridSearchRequest,
    SimilarDocumentRequest,
    VectorSearchRequest,
    VectorSearchService,
)

load_dotenv()

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
VECTOR_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX", "patent_vectors")

# ───────────────────────────────────────────────────────────────
# 2️⃣ FastAPI アプリ定義
# ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Patent Vector Search API",
    description="Patent document search with vector similarity and hybrid search",
    version="0.2.0",
)

# Initialize vector search service
vector_search_service = VectorSearchService(es)


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


# ───────────────────────────────────────────────────────────────
# 5️⃣ ベクトル検索エンドポイント
# ───────────────────────────────────────────────────────────────

@app.post("/vector-search")
async def vector_search(request: VectorSearchRequest):
    """
    ベクトル類似度検索

    Args:
        request: 検索リクエスト（query, k, min_score, filters）

    Returns:
        類似度の高い特許文書のリスト
    """
    results = await vector_search_service.vector_search(
        query_text=request.query,
        k=request.k,
        min_score=request.min_score,
        filters=request.filters
    )
    return {"results": results, "count": len(results)}


@app.post("/hybrid-search")
async def hybrid_search(request: HybridSearchRequest):
    """
    ハイブリッド検索（RRFによるテキスト + ベクトル融合）

    Args:
        request: 検索リクエスト（query, k, rank_constant, filters）
        - rank_constant: RRFランク定数（デフォルト60、小さいほど上位結果を重視）

    Returns:
        検索結果のリスト
    """
    results = await vector_search_service.hybrid_search(
        query_text=request.query,
        k=request.k,
        rank_constant=request.rank_constant,
        filters=request.filters
    )
    return {"results": results, "count": len(results)}


@app.post("/similar-documents")
async def find_similar_documents(request: SimilarDocumentRequest):
    """
    類似文書検索

    Args:
        request: 類似文書検索リクエスト（patent_id, k, min_score）

    Returns:
        指定された特許に類似する文書のリスト
    """
    results = await vector_search_service.find_similar_documents(
        patent_id=request.patent_id,
        k=request.k,
        min_score=request.min_score
    )
    return {"results": results, "count": len(results)}


@app.get("/search-by-classification/{classification_code}")
async def search_by_classification(
    classification_code: str,
    classification_type: str = "ipc",
    k: int = 20
):
    """
    分類コードによる検索

    Args:
        classification_code: 分類コード
        classification_type: 分類タイプ（ipc, fi, f_term, theme_code）
        k: 返却件数

    Returns:
        該当する特許文書のリスト
    """
    results = await vector_search_service.search_by_classification(
        classification_code=classification_code,
        classification_type=classification_type,
        k=k
    )
    return {"results": results, "count": len(results)}


@app.get("/health")
def health_check():
    """
    ヘルスチェックエンドポイント
    """
    try:
        # Elasticsearch接続確認
        es_health = es.cluster.health()

        # インデックス情報取得
        vector_index_exists = es.indices.exists(index=VECTOR_INDEX_NAME)

        return {
            "status": "healthy",
            "elasticsearch": {
                "status": es_health["status"],
                "cluster_name": es_health["cluster_name"],
                "number_of_nodes": es_health["number_of_nodes"]
            },
            "vector_index": {
                "name": VECTOR_INDEX_NAME,
                "exists": vector_index_exists
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.get("/")
def root():
    """
    ルートエンドポイント
    """
    return {
        "message": "Patent Vector Search API",
        "version": "0.2.0",
        "endpoints": {
            "vector_search": "/vector-search",
            "hybrid_search": "/hybrid-search",
            "similar_documents": "/similar-documents",
            "search_by_classification": "/search-by-classification/{classification_code}",
            "health": "/health",
            "docs": "/docs"
        }
    }

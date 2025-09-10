"""
FI / F-term ハイブリッド検索（各3件）
- 入力: 出願要約テキスト（引数 or 標準入力）
- 処理: Azure OpenAI で埋め込み → Azure AI Search にハイブリッド検索（BM25 + Vector）
- 出力: 各インデックスから上位3件の code 等をTSVで出力
"""

import argparse
import sys
from typing import Any, Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

# ===== 設定 =====
SEARCH_ENDPOINT = ""
SEARCH_KEY = ""
FI_INDEX_NAME = "fi_classification_index"
FTERM_INDEX_NAME = "fterm_classification_index"


# Azure OpenAI
AOAI_ENDPOINT = ""
AOAI_KEY = ""
EMBED_MODEL = "text-embedding-3-small"

# ===== クライアント =====
aoai = AzureOpenAI(api_key=AOAI_KEY, api_version="2024-02-01", azure_endpoint=AOAI_ENDPOINT)

def make_search_client(index_name: str) -> SearchClient:
    return SearchClient(SEARCH_ENDPOINT, index_name, AzureKeyCredential(SEARCH_KEY))

def embed(text: str) -> List[float]:
    resp = aoai.embeddings.create(input=text, model=EMBED_MODEL)
    return resp.data[0].embedding  # 1536次元想定（text-embedding-3-small）

def search_top3(index_name: str, query_text: str, vec: List[float]) -> List[Dict[str, Any]]:
    client = make_search_client(index_name)

    vq = VectorizedQuery(
        vector=vec,
        fields="content_vector",
        k_nearest_neighbors=3
    )

    results_iter = client.search(
        search_text=query_text if query_text.strip() else None,
        search_fields=["search_text"],
        vector_queries=[vq],
        select=["chunk_id", "code", "search_text"],
        top=3,
    )

    rows: List[Dict[str, Any]] = []
    for r in results_iter:
        rows.append({
            "index": index_name,
            "code": r.get("code"),
            "chunk_id": r.get("chunk_id"),
            "score": r.get("@search.score", 0.0),
        })
        if len(rows) >= 3:
            break
    return rows

def main():
    parser = argparse.ArgumentParser(description="FI / F-term ハイブリッド検索（各3件）")
    parser.add_argument("text", nargs="*", help="検索テキスト（出願要約）。未指定なら標準入力から読み込み")
    args = parser.parse_args()

    if args.text:
        text = " ".join(args.text)
    else:
        try:
            text = sys.stdin.read().strip()
        except KeyboardInterrupt:
            print()
            return
        if not text:
            print("[WARN] 空入力のため終了", file=sys.stderr)
            return

    vec = embed(text)

    fi_rows = search_top3(FI_INDEX_NAME, text, vec)
    ft_rows = search_top3(FTERM_INDEX_NAME, text, vec)

    # 出力: TSV（index, code, chunk_id, score）
    for r in fi_rows + ft_rows:
        print(f"{r['index']}\t{r['code']}\t{r['chunk_id']}\t{r['score']:.6f}")

if __name__ == "__main__":
    main()

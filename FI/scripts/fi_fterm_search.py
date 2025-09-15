"""
FI / F-term ハイブリッド検索（各3件）
- 入力: 出願要約テキスト（引数 or 標準入力）
- 処理: Azure OpenAI で埋め込み → Azure AI Search にハイブリッド検索（BM25 + Vector）
- 出力: 各インデックスから上位3件の code 等をTSVで出力

環境変数（.env対応）
- AZURE_SEARCH_ENDPOINT: 例 https://<service>.search.windows.net
- AZURE_SEARCH_KEY: 管理キー or クエリキー
- FI_INDEX_NAME: FI の検索インデックス名（既定 fi_classification_index）
- FTERM_INDEX_NAME: F-term の検索インデックス名（既定 fterm_classification_index）
- AZURE_OPENAI_ENDPOINT: 例 https://<resource>.openai.azure.com
- AZURE_OPENAI_KEY: Azure OpenAI API key
- AZURE_OPENAI_EMBEDDING_MODEL: 埋め込みのデプロイ名（例 text-embedding-3-small のデプロイ名）
"""

import argparse
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI


def _load_config() -> Dict[str, str]:
    load_dotenv()
    cfg = {
        "SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT", ""),
        "SEARCH_KEY": os.getenv("AZURE_SEARCH_KEY", ""),
        "FI_INDEX_NAME": os.getenv("FI_INDEX_NAME", "fi_classification_index"),
        "FTERM_INDEX_NAME": os.getenv("FTERM_INDEX_NAME", "fterm_classification_index"),
        "AOAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "AOAI_KEY": os.getenv("AZURE_OPENAI_KEY", ""),
        "EMBED_MODEL": os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    }
    missing = [k for k in ("SEARCH_ENDPOINT", "SEARCH_KEY", "AOAI_ENDPOINT", "AOAI_KEY") if not cfg[k]]
    if missing:
        # 明示的にエラー終了（呼び出し元は非ゼロ終了で候補無しとして継続）
        print(f"[ERROR] Missing config: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)
    return cfg


def _make_clients(cfg: Dict[str, str]):
    aoai = AzureOpenAI(
        api_key=cfg["AOAI_KEY"],
        api_version="2024-02-01",
        azure_endpoint=cfg["AOAI_ENDPOINT"],
    )
    def make_search_client(index_name: str) -> SearchClient:
        return SearchClient(cfg["SEARCH_ENDPOINT"], index_name, AzureKeyCredential(cfg["SEARCH_KEY"]))
    return aoai, make_search_client


def embed(aoai: AzureOpenAI, model: str, text: str) -> List[float]:
    resp = aoai.embeddings.create(input=text, model=model)
    return resp.data[0].embedding


def search_top3(make_sc, index_name: str, query_text: str, vec: List[float]) -> List[Dict[str, Any]]:
    client = make_sc(index_name)
    vq = VectorizedQuery(vector=vec, fields="content_vector", k_nearest_neighbors=3)
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

    cfg = _load_config()
    aoai, make_sc = _make_clients(cfg)
    vec = embed(aoai, cfg["EMBED_MODEL"], text)

    fi_rows = search_top3(make_sc, cfg["FI_INDEX_NAME"], text, vec)
    ft_rows = search_top3(make_sc, cfg["FTERM_INDEX_NAME"], text, vec)

    # 出力: TSV（index, code, chunk_id, score）
    for r in fi_rows + ft_rows:
        print(f"{r['index']}\t{r['code']}\t{r['chunk_id']}\t{r['score']:.6f}")


if __name__ == "__main__":
    main()

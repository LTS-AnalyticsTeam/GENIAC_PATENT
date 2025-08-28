"""
FI / F-term ハイブリッド検索（各3件）
- 入力: 出願要約テキスト（引数 or 標準入力）
- 処理: Azure OpenAI で埋め込み → Azure AI Search にハイブリッド検索（BM25 + Vector）
- 出力: 各インデックスから上位3件の code 等をTSVで出力
"""

import argparse
import os
import subprocess
import sys
from typing import Any, Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

# ===== 設定 =====


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

    # 1) 取得フィールドに breadcrumbs を追加
    results_iter = client.search(
        search_text=query_text if query_text.strip() else None,
        search_fields=["search_text"],
        vector_queries=[vq],
        select=["chunk_id", "code", "search_text", "breadcrumbs"],  # 追加
        top=1,
    )

    rows: List[Dict[str, Any]] = []
    for r in results_iter:
        rows.append({
            "index": index_name,
            "code": r.get("code"),
            "chunk_id": r.get("chunk_id"),
            "score": r.get("@search.score", 0.0),
            "breadcrumbs": r.get("breadcrumbs"),
        })
        if len(rows) >= 3:
            break
    return rows

# 2) 投影ユーティリティ
def _project_code(row: Dict[str, Any], r: float) -> str:
    bc = row.get("breadcrumbs")
    code = row.get("code")
    if not isinstance(bc, list) or not bc:
        return str(code) if code else ""
    # ルート=深さ0, 葉=深さ(L) として投影
    leaf_depth = len(bc) - 1
    if leaf_depth <= 0:
        return str(code) if code else ""
    target_depth = int(round(leaf_depth * r))
    target_depth = max(0, min(leaf_depth, target_depth))
    # パンくずのその層をコードとして採用
    return str(bc[target_depth]) if bc[target_depth] else (str(code) if code else "")

# 3) 支持数で集約（FI / F-term 別）
from collections import defaultdict


def _aggregate_by_projection(rows: List[Dict[str, Any]], r: float):
    counts = defaultdict(int)
    score_sum = defaultdict(float)
    for row in rows:
        pcode = _project_code(row, r)
        if not pcode:
            continue
        counts[pcode] += 1
        score_sum[pcode] += float(row.get("score", 0.0))
    # ソート: 支持数(desc) → スコア和(desc)
    ranked = sorted(counts.keys(), key=lambda k: (counts[k], score_sum[k]), reverse=True)
    return ranked, counts, score_sum

def _format_fi_code(code: str) -> str:
    # FIコードの末尾表記はそのまま活かし、フィールド指定だけ付与
    return f"{code}/FI"

def _format_fterm_code(code: str) -> str:
    return f"{code}/FT"

def _build_jplatpat_from_results(fi_codes: List[str], ft_codes: List[str], tx_terms: List[str]) -> str:
    # (FI/FI * F/F) を + で繋いで [] に入れる
    n = min(len(fi_codes), len(ft_codes))
    pairs = []
    for i in range(n):
        pairs.append(f"({_format_fi_code(fi_codes[i])} * {_format_fterm_code(ft_codes[i])})")
    head = "[" + " + ".join(pairs) + "]" if pairs else ""

    # /TX 群はユーザー指定語だけ
    tx_part = " * ".join(f"{t}/TX" for t in tx_terms if t and t.strip())

    # 全体連結（空要素は除外）
    return " * ".join([p for p in [head, tx_part] if p])

def main():
    parser = argparse.ArgumentParser(description="FI / F-term ハイブリッド検索（各3件）")
    parser.add_argument("text", nargs="*", help="検索テキスト（出願要約）。未指定なら標準入力から読み込み")
    # 連携オプション
    parser.add_argument(
        "--run-generator",
        action="store_true",
        help="検索結果のcodeをキーワードとしてAzure版ジェネレータを実行"
    )
    parser.add_argument(
        "--jplatpat",
        action="store_true",
        help="ジェネレータ出力をJ-PlatPat形式にする"
    )
    parser.add_argument(
        "--tx-terms",
        nargs="+",
        help="ジェネレータ出力をJ-PlatPat形式にする際に、ユーザー指定語として展開するテキスト。"
    )
    parser.add_argument(
        "--projection-rate", type=float, default=0.6,
        help="相対深さの投影率 r (0.0=根, 1.0=葉)。既定: 0.6"
    )
    parser.add_argument(
        "--aoai-endpoint",
        type=str,
        default=os.getenv("AOAI_ENDPOINT", ""),
        help="Azure OpenAI エンドポイント（省略時は環境変数 AOAI_ENDPOINT）"
    )
    parser.add_argument(
        "--aoai-key",
        type=str,
        default=os.getenv("AOAI_KEY", os.getenv("AOAI_API_KEY", "")),
        help="Azure OpenAI APIキー（省略時は環境変数 AOAI_KEY または AOAI_API_KEY）"
    )
    parser.add_argument(
        "--aoai-model",
        type=str,
        default=os.getenv("AOAI_CHAT_MODEL", "gpt-4o-mini"),
        help="Azure OpenAI チャットモデル名（既定: gpt-4o-mini）"
    )
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

    # ===== 連携（任意） =====
    if args.run_generator:
        # --jplatpat かつ --tx-terms がある場合は、ローカルでJ-PlatPat式を構築
        if args.jplatpat and args.tx_terms:
            # 5) J-PlatPat構築で投影後トップNを採用
            # （--jplatpat and --tx-terms 分岐の直前で、fi_codes/ft_codes を投影ベースに作る）

            # 既存の fi_rows / ft_rows から投影集約
            fi_ranked, _, _ = _aggregate_by_projection(fi_rows, args.projection_rate)
            ft_ranked, _, _ = _aggregate_by_projection(ft_rows, args.projection_rate)

            # 使用する本数（例: 各2本まで）
            fi_codes = fi_ranked[:2] if fi_ranked else []
            ft_codes = ft_ranked[:2] if ft_ranked else []

            # 以降は既存の _build_jplatpat_from_results(fi_codes, ft_codes, args.tx_terms) を使用
            if not fi_codes or not ft_codes:
                print("[WARN] FIまたはFタームのコードが不足しているため、式を構築できません。", file=sys.stderr)
                return

            expr = _build_jplatpat_from_results(fi_codes, ft_codes, args.tx_terms)
            print("\n# ===== 生成された検索式 =====")
            print(expr)
            return

        # 上記以外は従来通り、Azure版ジェネレータをサブプロセスで呼び出す
        # （既存の subprocess.run(...) ブロックをこのまま残す）
        # 上位ヒットの code をキーワードとして利用（重複排除/順序保持）
        seen = set()
        keywords: List[str] = []
        for row in fi_rows + ft_rows:
            code = row.get("code")
            if code and code not in seen:
                keywords.append(str(code))
                seen.add(code)

        if not keywords:
            print("[WARN] ジェネレータに渡すキーワードがありません", file=sys.stderr)
            return

        # azure_patent_search_generator.py のパスを解決
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        gen_path = os.path.join(
            repo_root,
            "create_patent_search",
            "azure_patent_search_generator.py",
        )
        if not os.path.exists(gen_path):
            print(f"[ERROR] ジェネレータが見つかりません: {gen_path}", file=sys.stderr)
            return

        cmd = [
            sys.executable,
            gen_path,
            "--aoai-endpoint",
            args.aoai_endpoint,
            "--aoai-key",
            args.aoai_key,
            "--aoai-model",
            args.aoai_model,
        ]
        if args.jplatpat:
            cmd.append("--jplatpat")
        cmd.extend(keywords)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            generated = result.stdout.strip()
            if generated:
                print("\n# ===== 生成された検索式 =====")
                print(generated)
        except subprocess.CalledProcessError as e:
            print("[ERROR] ジェネレータ実行に失敗しました", file=sys.stderr)
            print(e.stderr, file=sys.stderr)

if __name__ == "__main__":
    main()

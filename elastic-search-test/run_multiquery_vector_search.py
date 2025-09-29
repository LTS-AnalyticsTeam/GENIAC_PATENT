#!/usr/bin/env python3
"""
単一の特許文書からのマルチクエリベクトル検索。

処理フロー:
- Cosmos DBからpatent_idで特許を取得
- 要約/請求項から10個の約200文字の日本語文章を生成（OpenAIが利用可能な場合は使用、それ以外は代替手段）
- Azure OpenAIエンベディングで10個の文章を埋め込み
- 各埋め込みに対してElasticsearchでKNNベクトル検索を実行（上位100件）
- クエリ間で結果をマージし重複を除去

使用方法:
  python run_multiquery_vector_search.py --patent JP2011050607 --k 100 \\
    --num-queries 10 --exclude-source

環境変数（elastic-search-test/.envから）:
- COSMOS_* Cosmos DB用
- ELASTICSEARCH_HOST, ELASTICSEARCH_INDEX
- AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
- AZURE_OPENAI_DEPLOYMENT (エンベディング用)
- テキスト生成用（オプション）: AZURE_OPENAI_CHAT_DEPLOYMENT
"""
import argparse
import asyncio
import json
import logging
import os
import re
# Local imports
import sys as _sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_sys.path.append(str(Path(__file__).parent))

try:
    from openai import AzureOpenAI, OpenAI
except Exception:  # pragma: no cover
    AzureOpenAI = None  # type: ignore
    OpenAI = None  # type: ignore

from data_sync.cosmos_client import CosmosDBClient  # noqa: E402
from data_sync.elasticsearch_indexer import ElasticsearchIndexer  # noqa: E402
from data_sync.embedding_processor import EmbeddingProcessor  # noqa: E402

load_dotenv()

logger = logging.getLogger("multiquery_search")


def normalize_patent_id(raw: str) -> str:
    """特許IDを正規化（数字以外を除去、例: 'JP2011050607A' -> '2011050607'）。"""
    digits = re.sub(r"\D", "", str(raw or ""))
    return digits


def safe_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    elif isinstance(x, list):
        return " ".join(x)
    else:
        return str(x or "")


def truncate_chars(text: str, max_chars: int) -> str:
    if not text:
        return ""
    return text[:max_chars]


class PassageGenerator:
    """Cosmos特許文書から約200文字の日本語文章を生成（OpenAIが利用可能な場合は使用）。"""

    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
        )
        # 生成用の必須チャット/テキストデプロイメント（Azure）
        self.chat_deployment = (
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or
            os.getenv("AZURE_OPENAI_TEXT_DEPLOYMENT")
        )

        # オプション: 生成用の標準OpenAI認証情報
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        self.provider: Optional[str] = None
        self.client = None

        # 完全に設定されている場合はAzureを優先、
        # そうでなければOpenAIが利用可能な場合は使用
        if (AzureOpenAI and self.api_key and self.endpoint and
                self.chat_deployment):
            try:
                self.client = AzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                )
                self.provider = "azure"
                logger.info(
                    "文章生成用のAzure OpenAIチャットクライアントを初期化しました"
                )
            except Exception as e:
                raise RuntimeError(
                    f"生成AIクライアント(Azure)の初期化に失敗しました: {e}"
                )
        elif OpenAI and self.openai_key:
            try:
                self.client = OpenAI(api_key=self.openai_key)
                self.provider = "openai"
                logger.info(
                    "文章生成用のOpenAIチャットクライアントを初期化しました"
                )
            except Exception as e:
                raise RuntimeError(
                    f"生成AIクライアント(OpenAI)の初期化に失敗しました: {e}"
                )
        else:
            # 厳格: ヒューリスティックな代替手段なし; プロバイダーのいずれかが必要
            raise RuntimeError(
                "生成AIが未設定です。Azure を使う場合は "
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_CHAT_DEPLOYMENT を、OpenAI を使う場合は "
                "OPENAI_API_KEY (必要に応じて OPENAI_CHAT_MODEL) を設定してください。"
            )

    async def generate(
        self, doc: Dict[str, Any], num_passages: int = 10
    ) -> List[str]:
        """要約/請求項/タイトルからOpenAIまたはヒューリスティックを使用して文章を生成。"""
        title = safe_text(doc.get("title", ""))
        summary = safe_text(doc.get("summary", ""))
        claims = doc.get("claims", [])
        if isinstance(claims, list):
            claims_text = "\n".join(safe_text(c) for c in claims[:5])
        else:
            claims_text = safe_text(claims)

        # 一部のデータセットでは説明が不足している場合がある;
        # 要約 + 請求項に依存
        context = (
            f"タイトル: {truncate_chars(title, 200)}\n\n"
            f"要約: {truncate_chars(summary, 1500)}\n\n"
            f"請求項(抜粋): {truncate_chars(claims_text, 2000)}\n"
        )

        # OpenAIチャット生成を厳格に要求（代替手段なし）
        try:
            system_msg = (
                """
                # あなたは日本語の特許理解に長けたアシスタントです
                入力の特許の「タイトル・要約・請求項」を読み取り、
                互いに焦点が重ならない検索文を【10本】作成してください

                # 出力形式
                - 出力は JSON 配列のみ（各要素は文字列）
                - 説明文・ラベル・コードブロックは一切禁止
                - 各要素は1文で完結した自然文。句読点と助詞を含む通常の日本語
                - 各文は自立的（「この発明」等の指示語で依存しない）

                # 各クエリの必須条件
                1. 長さ：150~200文字前後
                2. 具体性：入力テキストに実在する「具体的な名詞」を必ず含める
                3. 観点の多様化：各クエリは主たる観点を明確に変える
                候補観点（例）：主題／課題／解決手段／効果／用途／入力手段／表示手法／時間・期間制御／状態遷移など
                4. 言い換え：同義領域は語彙を散らす
                6. アンカー語：各文には少なくとも1語の「希少・固有っぽい語」を入れる
                - 極端に汎用的な語のみは不可
                - 入力に存在しない数値・型番・固有名の捏造は禁止（実在語の再利用のみ）
                8. 重複防止
                - 10文の間で主要名詞・句パターンの重なりを最小化する
                - 可能なら、各文の先頭5〜7語は互いに異なる語で始める

                # 重要
                - 出力は JSON配列（文字列10要素）のみ
                - 各文は固有の観点と語彙で明確に差別化すること
                """
            )
            user_msg = (
                "以下の特許情報からバリエーションのある検索文を作成してください。\n\n" + context
            )
            # 選択されたプロバイダーでchat.completions APIを使用
            if self.provider == "azure":
                resp = self.client.chat.completions.create(
                    model=self.chat_deployment,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.5,
                    max_tokens=1200,
                )
            else:  # openai
                resp = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.5,
                    max_tokens=1200,
                )

            content = resp.choices[0].message.content.strip()
            passages = json.loads(content)
            # クリーンアップして数を強制
            cleaned: List[str] = []
            for p in passages:
                s = safe_text(p).strip()
                if s:
                    cleaned.append(truncate_chars(s, 260))
                if len(cleaned) >= num_passages:
                    break
            if len(cleaned) < num_passages:
                raise RuntimeError(
                    f"生成AIの出力数が不足しています: {len(cleaned)}/{num_passages}"
                )
            return cleaned[:num_passages]
        except Exception as e:
            raise RuntimeError(f"生成AIでの文章生成に失敗しました: {e}")


async def multiquery_vector_search(
    patent_id_raw: str,
    k: int = 100,
    num_queries: int = 10,
    min_score: float = 0.0,
    exclude_source: bool = False,
) -> Dict[str, Any]:
    """指定されたpatent_idに対してマルチクエリベクトル検索パイプラインを実行。"""
    # クライアントを初期化
    cosmos = CosmosDBClient()
    es = ElasticsearchIndexer()
    embedder = EmbeddingProcessor()
    generator = PassageGenerator()

    # 特許IDを正規化して文書を取得
    patent_id = normalize_patent_id(patent_id_raw)
    doc = cosmos.get_document_by_id(patent_id)
    if not doc:
        raise ValueError(
            f"Cosmos DBで特許が見つかりません: {patent_id_raw} -> {patent_id}"
        )

    # 文章を生成
    passages = await generator.generate(doc, num_passages=num_queries)

    # 文章を埋め込み
    embeddings = await embedder.generate_batch_embeddings(passages)

    # ベクトル検索を実行してマージ
    aggregated: Dict[str, Dict[str, Any]] = {}
    per_query_hits: List[int] = []
    for qi, vec in enumerate(embeddings):
        if not vec:
            per_query_hits.append(0)
            continue
        hits = es.search_similar_documents(
            query_vector=vec, k=k, min_score=min_score
        )
        per_query_hits.append(len(hits))
        for rank, h in enumerate(hits, start=1):
            doc_id = h.get("_id") or h.get("patent_id")
            if not doc_id:
                continue
            if exclude_source and str(doc_id) == patent_id:
                continue

            entry = aggregated.get(doc_id)
            score = h.get("_score", 0.0)
            if entry is None:
                aggregated[doc_id] = {
                    "patent_id": doc_id,
                    "max_score": score,
                    "first_query_index": qi,
                    "times_hit": 1,
                    "best_rank": rank,
                    "source_example": passages[qi],
                    "document": h,
                }
            else:
                entry["times_hit"] += 1
                if score > entry["max_score"]:
                    entry["max_score"] = score
                    entry["first_query_index"] = qi
                    entry["best_rank"] = rank
                    entry["source_example"] = passages[qi]
                    entry["document"] = h

    # max_score降順でソート、同点の場合はtimes_hit降順、次にbest_rank昇順
    merged = list(aggregated.values())
    merged.sort(key=lambda x: (
        -float(x.get("max_score", 0.0)),
        -int(x.get("times_hit", 0)),
        int(x.get("best_rank", 1e9))
    ))

    result = {
        "query_patent_id": patent_id,
        "generated_queries": passages,
        "per_query_hits": per_query_hits,
        "unique_results": len(merged),
        "results": merged,
    }

    # クリーンアップ
    try:
        cosmos.close()
        es.close()
    except Exception:
        pass

    return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    parser = argparse.ArgumentParser(
        description="特許文書からマルチクエリベクトル検索を実行"
    )
    parser.add_argument(
        "--patent", required=True,
        help="特許番号（例: JP2011050607A または 2011050607）"
    )
    parser.add_argument("--k", type=int, default=100, help="クエリあたりのTop-K")
    parser.add_argument("--num-queries", type=int, default=10, help="生成する文章数")
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="最小類似度スコアフィルター"
    )
    parser.add_argument(
        "--exclude-source", action="store_true",
        help="結果からソース特許を除外"
    )
    parser.add_argument("--output", default="", help="オプションの出力JSONファイルパス")
    args = parser.parse_args()

    res = asyncio.run(
        multiquery_vector_search(
            patent_id_raw=args.patent,
            k=args.k,
            num_queries=args.num_queries,
            min_score=args.min_score,
            exclude_source=args.exclude_source,
        )
    )

    print("\n==== マルチクエリベクトル検索結果（要約） ====")
    print(f"クエリ特許ID: {res['query_patent_id']}")
    print(f"生成されたクエリ: {len(res['generated_queries'])} -> "
          f"クエリあたりのヒット数: {res['per_query_hits']}")
    print(f"ユニークなマージ結果: {res['unique_results']}")

    # 上位20件を表示
    for i, r in enumerate(res["results"][:20], 1):
        print(
            f"{i:>3}. id={r['patent_id']} score={r['max_score']:.4f} "
            f"hits={r['times_hit']} best_rank={r['best_rank']} "
            f"title={r['document'].get('title', '')}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\n詳細なJSONを保存しました: {args.output}")


if __name__ == "__main__":
    main()

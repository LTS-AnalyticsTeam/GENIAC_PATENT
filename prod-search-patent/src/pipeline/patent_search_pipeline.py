#!/usr/bin/env python3
"""
特許検索パイプライン
一件ずつ実行して、類似特許を検索する

使用方法:
  python scripts/patent_search_pipeline.py JP2021106667A
  python scripts/patent_search_pipeline.py JP2021106667A --filter-years
  python scripts/patent_search_pipeline.py JP2021106667A --filter-years --output results.csv

流れ:
1. 所定のフォルダからsyutugan特許のJSONを取得（なければCosmosDBからダウンロード）
2. キーワード抽出実行
3. STAGE1: CosmosDBに対してIPCフィルタリング
4. STAGE2: キーワードスコアリング
5. (オプション) 10,000件以上の場合、発行日10年以内で絞り込み
6. 上位10,000件の特許番号をCSVで出力
"""
from azure.cosmos import CosmosClient
import os
import json
import time
import csv
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv
import sys

# インポートパスを修正（同じディレクトリ内のモジュール）
from .abc_keyword_extractor_v3 import extract_keywords_with_priority

load_dotenv()

logger = logging.getLogger(__name__)

# 設定
INPUT_FOLDER = 'data/input_patents'
OUTPUT_FOLDER = 'data/output_results'

# グローバル変数（遅延初期化）
_cosmos_client = None
_container = None


def _get_container():
    """CosmosDBコンテナを取得（遅延初期化）"""
    global _cosmos_client, _container
    if _container is None:
        _cosmos_client = CosmosClient(
            os.environ.get('COSMOS_ENDPOINT', ''),
            os.environ.get('COSMOS_KEY', '')
        )
        database = _cosmos_client.get_database_client(
            os.environ.get('COSMOS_DATABASE', os.environ.get('DATABASE_NAME', 'patent_json'))
        )
        _container = database.get_container_client(
            os.environ.get('COSMOS_CONTAINER', os.environ.get('CONTAINER_NAME', 'patent_json_v2'))
        )
    return _container


def run_patent_search_from_json(json_payload: dict, job_manager=None, job_id: str = None) -> dict:
    """
    JSONペイロードから特許検索パイプラインを実行（worker用）

    Args:
        json_payload: Redis/APIから受け取ったJSONデータ
        job_manager: JobManagerインスタンス（進捗更新用）
        job_id: ジョブID

    Returns:
        検索結果（patent_idsリストを含む）
    """
    start_time = time.time()
    container = _get_container()

    logger.info("Starting patent search pipeline from JSON payload")

    # JSONからpatentデータを取得
    if "json" in json_payload:
        patent = json.loads(json_payload["json"]) if isinstance(json_payload["json"], str) else json_payload["json"]
    else:
        patent = json_payload

    # 特許番号取得
    patent_id = ""
    if "bibliographic" in patent:
        doc_number = patent["bibliographic"].get("publication", {}).get("doc_number", "")
        patent_id = f"JP{doc_number}A" if doc_number else ""

    if not patent_id:
        patent_id = patent.get("patent_id", "unknown")

    logger.info(f"Target patent: {patent_id}")

    # 特許情報取得
    title = patent.get('bibliographic', {}).get('invention_title', '')
    if isinstance(title, dict):
        title = title.get('ja', '') or title.get('en', '')
    ipc_list = patent.get('bibliographic', {}).get('classification', {}).get('ipc_prefix', [])

    logger.info(f"Title: {str(title)[:50]}...")
    logger.info(f"IPC: {ipc_list[:3]}")

    # STEP2: キーワード抽出
    logger.info("STEP2: Keyword extraction...")

    abstract = patent.get('abstract', '')
    claims_data = patent.get('claims', [])
    claims_texts = []
    if isinstance(claims_data, list):
        claims_texts = [c.get('claim_text', '') for c in claims_data[:3] if isinstance(c, dict)]
    elif isinstance(claims_data, dict):
        claims_list = claims_data.get('claims', [])
        claims_texts = [c.get('claim_text', '') for c in claims_list[:3] if isinstance(c, dict)]

    keywords = extract_keywords_with_priority(
        str(title) if title else "",
        abstract if isinstance(abstract, str) else str(abstract),
        claims_texts
    )

    total_keywords = sum(len(keywords.get(cat, [])) for cat in ['B_MUST', 'B_SHOULD', 'C_MUST', 'C_SHOULD'])
    logger.info(f"Extracted keywords: {total_keywords}")

    # STEP3: STAGE1 IPCフィルタリング
    logger.info("STEP3: STAGE1 IPC filtering...")
    candidates = stage1_ipc_filter(patent)
    logger.info(f"STAGE1 candidates: {len(candidates):,}")

    # STEP4: STAGE2 キーワードスコアリング
    logger.info("STEP4: STAGE2 keyword scoring...")
    scored = stage2_keyword_scoring_fast(candidates, keywords, patent, limit=10000)
    logger.info(f"STAGE2 results: {len(scored):,}")

    elapsed = time.time() - start_time

    # 結果を構成
    result = {
        "target_patent": patent_id,
        "pipeline_stats": {
            "stage1_IPC_candidates": len(candidates),
            "stage2_keyword_filter_results": len(scored),
            "total_keywords": total_keywords,
            "elapsed_seconds": round(elapsed, 2)
        },
        # 後続処理用のpatent_idリスト
        "keyword_search_results": [doc_num for doc_num, score in scored],
        "search_results": [{"doc_number": doc_num, "score": score} for doc_num, score in scored],
        "searched_at": datetime.now().isoformat()
    }

    # 結果をRedisに保存
    if job_manager and job_id:
        job_manager.store_result(job_id, result)
        logger.info(f"Search result stored for job {job_id}")

    logger.info(f"Search completed in {elapsed:.2f}s, {len(scored)} results")

    return result


# 以下は元のコードを維持（container変数を_get_container()に置き換え）


def download_patent_from_cosmos(patent_id):
    """CosmosDBから特許をダウンロードしてJSONとして保存"""
    container = _get_container()
    doc_number = ''.join(filter(str.isdigit, patent_id))

    query = f"SELECT * FROM c WHERE c.bibliographic.publication.doc_number = '{doc_number}'"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))

    if not items:
        return None

    patent = items[0]

    # フォルダが存在しなければ作成
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    # JSONとして保存
    output_path = os.path.join(INPUT_FOLDER, f"{patent_id}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(patent, f, ensure_ascii=False, indent=2)

    print(f"ダウンロード完了: {output_path}")
    return patent


def load_patent_from_folder(patent_id):
    """所定のフォルダから特許JSONを読み込む"""
    json_path = os.path.join(INPUT_FOLDER, f"{patent_id}.json")

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return None


def stage1_ipc_filter(syutugan_patent):
    """STAGE1: IPCベースのフィルタリング（全候補を取得）"""
    container = _get_container()
    ipc_list = syutugan_patent.get('bibliographic', {}).get('classification', {}).get('ipc_prefix', [])

    if not ipc_list:
        return []

    candidates = set()

    for ipc in ipc_list[:5]:
        ipc_5 = ipc[:5] if len(ipc) >= 5 else ipc

        query = f"""
        SELECT c.bibliographic.publication.doc_number
        FROM c
        WHERE ARRAY_CONTAINS(c.bibliographic.classification.ipc_prefix, '{ipc_5}')
           OR EXISTS(SELECT VALUE p FROM p IN c.bibliographic.classification.ipc_prefix WHERE STARTSWITH(p, '{ipc_5}'))
        """

        items = list(container.query_items(query=query, enable_cross_partition_query=True))

        for item in items:
            doc_num = item.get('doc_number', '')
            if doc_num:
                candidates.add(doc_num)

    return list(candidates)


def stage2_keyword_scoring_fast(candidates, keywords, syutugan_patent, batch_size=None, limit=10000):
    """STAGE2: キーワードベースのスコアリング（バッチクエリで高速化）

    候補数に応じて動的にバッチサイズを調整:
    - ~10,000件: batch_size=200 (約2分)
    - 10,000~50,000件: batch_size=500 (約15分)
    - 50,000~100,000件: batch_size=800 (約40分)
    - 100,000件~: batch_size=1000 (約90分)
    """
    container = _get_container()

    # 候補数に応じた動的バッチサイズ
    if batch_size is None:
        num_candidates = len(candidates)
        if num_candidates <= 10000:
            batch_size = 200
        elif num_candidates <= 50000:
            batch_size = 500
        elif num_candidates <= 100000:
            batch_size = 800
        else:
            batch_size = 1000
        print(f"    動的バッチサイズ: {batch_size}件/バッチ", flush=True)

    # 全キーワードをフラットなリストに変換
    all_keywords = set()
    for category in ['B_MUST', 'B_SHOULD', 'C_MUST', 'C_SHOULD']:
        items = keywords.get(category, [])
        for item in items:
            base_kw = item.get('base_keyword', '')
            if base_kw:
                all_keywords.add(base_kw)
            for syn in item.get('synonyms', []):
                if syn:
                    all_keywords.add(syn)

    # カテゴリ別の重み
    category_weights = {
        'B_MUST': 5.0,
        'B_SHOULD': 3.0,
        'C_MUST': 2.0,
        'C_SHOULD': 1.0
    }

    # キーワードごとの重みを計算
    keyword_weights = {}
    for category, weight in category_weights.items():
        items = keywords.get(category, [])
        for item in items:
            base_kw = item.get('base_keyword', '')
            if base_kw:
                keyword_weights[base_kw] = weight
            for syn in item.get('synonyms', []):
                if syn:
                    keyword_weights[syn] = weight * 0.8

    # F-ターム取得
    syutugan_fterms = syutugan_patent.get('bibliographic', {}).get('f_terms', [])

    scored_candidates = []

    # チェックポイント機能
    checkpoint_file = f'/tmp/checkpoint_{len(candidates)}.json'
    processed_count = 0

    # 前回のチェックポイントがあれば復元
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                scored_candidates = checkpoint_data.get('scored_candidates', [])
                processed_count = checkpoint_data.get('processed_count', 0)
                if processed_count > 0:
                    print(f"    チェックポイント復元: {processed_count}件処理済み", flush=True)
        except:
            pass

    # バッチ処理（IN句で複数取得）
    for batch_idx in range(processed_count, len(candidates), batch_size):
        batch = candidates[batch_idx:batch_idx+batch_size]

        if batch_idx % 5000 == 0:
            print(f"    バッチ処理: {batch_idx}/{len(candidates)}件", flush=True)

        # IN句でバッチ取得
        doc_nums_str = "', '".join(batch)
        query = f"""
        SELECT c.bibliographic.publication.doc_number, c.bibliographic.invention_title, c.abstract, c.description, c.claims, c.bibliographic.f_terms
        FROM c
        WHERE c.bibliographic.publication.doc_number IN ('{doc_nums_str}')
        """

        # リトライロジック付きバッチ取得（強化版）
        max_retries = 5
        items = []
        for retry in range(max_retries):
            try:
                items = list(container.query_items(query=query, enable_cross_partition_query=True))
                break
            except Exception as e:
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 10  # 10秒, 20秒, 30秒, 40秒
                    print(f"    接続エラー、{wait_time}秒後にリトライ... ({retry + 1}/{max_retries})", flush=True)
                    time.sleep(wait_time)

                    # クライアント再接続を試みる（3回目のリトライ以降）
                    if retry >= 2:
                        print(f"    クライアント再接続を試行...", flush=True)
                else:
                    # 最終リトライも失敗した場合は個別に処理
                    print(f"    バッチ取得失敗、個別処理に切り替え...", flush=True)
                    for doc_num in batch:
                        try:
                            single_query = f"SELECT * FROM c WHERE c.bibliographic.publication.doc_number = '{doc_num}'"
                            single_items = list(container.query_items(query=single_query, enable_cross_partition_query=True))
                            if single_items:
                                candidate = single_items[0]
                                score = _score_candidate(candidate, keyword_weights, syutugan_fterms, is_gaming_machine)
                                scored_candidates.append((doc_num, score))
                        except:
                            pass  # 個別エラーはスキップ
                    continue

        for candidate in items:
            doc_num = candidate.get('doc_number', '')
            if not doc_num:
                continue

            score = _score_candidate(candidate, keyword_weights, syutugan_fterms)
            scored_candidates.append((doc_num, score))

        # チェックポイント保存（5000件ごと）
        if (batch_idx + batch_size) % 5000 == 0:
            try:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'scored_candidates': scored_candidates,
                        'processed_count': batch_idx + batch_size
                    }, f)
                print(f"    チェックポイント保存: {batch_idx + batch_size}件", flush=True)
            except:
                pass

    # 処理完了後、チェックポイントファイルを削除
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
        except:
            pass

    # スコアでソート
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    # 上位limit件のみを返す
    return scored_candidates[:limit]


def _score_candidate(candidate, keyword_weights, syutugan_fterms, is_gaming_machine=False):
    """候補のスコアを計算

    Args:
        candidate: 候補特許データ
        keyword_weights: キーワード重み辞書
        syutugan_fterms: 出願特許のF-タームリスト
        is_gaming_machine: 遊技機(A63F)かどうか
    """
    # テキスト準備（dict型対応）
    text = ""

    # タイトルを検索対象に追加
    title = candidate.get('bibliographic', {}).get('invention_title', '') or candidate.get('invention_title', '')
    if title:
        if isinstance(title, str):
            text += title + " "
        elif isinstance(title, dict):
            text += str(title) + " "

    abstract = candidate.get('abstract', '')
    if abstract:
        if isinstance(abstract, str):
            text += abstract + " "
        elif isinstance(abstract, dict):
            text += str(abstract.get('p', '')) + " "
    if candidate.get('description'):
        desc = candidate.get('description', '')
        if isinstance(desc, str):
            text += desc[:3000] + " "
        elif isinstance(desc, dict):
            text += str(desc)[:3000] + " "

    # 請求項1〜5を検索対象に追加
    claims_data = candidate.get('claims', [])
    if isinstance(claims_data, list):
        for i, claim in enumerate(claims_data[:5]):
            if isinstance(claim, dict):
                claim_text = claim.get('claim_text', '') or claim.get('text', '')
                if claim_text:
                    text += claim_text + " "
            elif isinstance(claim, str):
                text += claim + " "
    elif isinstance(claims_data, dict):
        claims_list = claims_data.get('claims', [])
        for i, claim in enumerate(claims_list[:5]):
            if isinstance(claim, dict):
                claim_text = claim.get('claim_text', '') or claim.get('text', '')
                if claim_text:
                    text += claim_text + " "

    # キーワードスコア
    score = 0
    for kw, weight in keyword_weights.items():
        if kw in text:
            score += weight * 10

    # F-タームボーナス（遊技機の場合は10倍）
    candidate_fterms = candidate.get('f_terms', [])
    if not candidate_fterms:
        candidate_fterms = candidate.get('bibliographic', {}).get('f_terms', [])
    common_fterms = set(syutugan_fterms) & set(candidate_fterms)

    # 遊技機(A63F)の場合はF-タームボーナスを大幅増加
    fterm_bonus = 1000 if is_gaming_machine else 100
    score += len(common_fterms) * fterm_bonus

    return score


def filter_by_date(scored_candidates, syutugan_patent, years=10):
    """発行日でフィルタリング（10年以内）"""
    container = _get_container()
    syutugan_date_str = syutugan_patent.get('bibliographic', {}).get('publication', {}).get('date', '')

    if not syutugan_date_str:
        print("  警告: 出願特許の発行日が取得できません")
        return scored_candidates

    try:
        syutugan_year = int(syutugan_date_str[:4])
    except:
        return scored_candidates

    min_year = syutugan_year - years

    print(f"  発行日フィルタ: {min_year}年以降")

    filtered = []
    for doc_num, score in scored_candidates:
        query = f"SELECT c.bibliographic.publication.date FROM c WHERE c.bibliographic.publication.doc_number = '{doc_num}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))

        if items:
            date_str = items[0].get('date', '')
            if date_str:
                try:
                    year = int(date_str[:4])
                    if year >= min_year:
                        filtered.append((doc_num, score))
                except:
                    filtered.append((doc_num, score))
            else:
                filtered.append((doc_num, score))

    return filtered


def save_results_to_csv(patent_id, scored_candidates, output_path=None):
    """結果をCSVとして保存"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_FOLDER, f"{patent_id}_results_{timestamp}.csv")

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'patent_number', 'score'])

        for rank, (doc_num, score) in enumerate(scored_candidates, 1):
            # doc_numberからJP形式に変換
            patent_num = f"JP{doc_num}A" if not doc_num.startswith('JP') else doc_num
            writer.writerow([rank, patent_num, score])

    print(f"結果保存: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='特許検索パイプライン')
    parser.add_argument('patent_id', help='検索対象の特許番号 (例: JP2021106667A)')
    parser.add_argument('--filter-years', action='store_true',
                        help='10,000件以上の場合、発行日10年以内で絞り込む')
    parser.add_argument('--output', '-o', help='出力CSVファイルパス')

    args = parser.parse_args()

    patent_id = args.patent_id

    print("="*80)
    print("特許検索パイプライン")
    print("="*80)
    print(f"対象特許: {patent_id}")
    print()

    start_time = time.time()

    # 1. 特許データ取得
    print("【STEP1】特許データ取得...")
    patent = load_patent_from_folder(patent_id)

    if patent:
        print(f"  ローカルファイルから読み込み: {INPUT_FOLDER}/{patent_id}.json")
    else:
        print(f"  ローカルにないためCosmosDBからダウンロード...")
        patent = download_patent_from_cosmos(patent_id)

        if not patent:
            print(f"  エラー: 特許 {patent_id} が見つかりません")
            return

    # 特許情報表示
    title = patent.get('bibliographic', {}).get('invention_title', '')
    ipc_list = patent.get('bibliographic', {}).get('classification', {}).get('ipc_prefix', [])

    print(f"  タイトル: {title[:50]}...")
    print(f"  IPC: {ipc_list[:3]}")
    print()

    # 2. キーワード抽出
    print("【STEP2】キーワード抽出...")

    abstract = patent.get('abstract', '')
    claims_data = patent.get('claims', [])
    claims_texts = []
    if isinstance(claims_data, list):
        claims_texts = [c.get('claim_text', '') for c in claims_data[:3] if isinstance(c, dict)]
    elif isinstance(claims_data, dict):
        claims_list = claims_data.get('claims', [])
        claims_texts = [c.get('claim_text', '') for c in claims_list[:3] if isinstance(c, dict)]

    keywords = extract_keywords_with_priority(
        title,
        abstract if isinstance(abstract, str) else str(abstract),
        claims_texts
    )

    total_keywords = sum(len(keywords.get(cat, [])) for cat in ['B_MUST', 'B_SHOULD', 'C_MUST', 'C_SHOULD'])
    print(f"  抽出キーワード: {total_keywords}件")
    print()

    # 3. STAGE1: IPCフィルタリング
    print("【STEP3】STAGE1: IPCフィルタリング...")
    candidates = stage1_ipc_filter(patent)
    print(f"  候補数: {len(candidates):,}件")
    print()

    # 4. STAGE2: キーワードスコアリング
    print("【STEP4】STAGE2: キーワードスコアリング...")
    scored = stage2_keyword_scoring_fast(candidates, keywords, patent, limit=10000)
    print(f"  スコアリング完了: {len(scored):,}件")
    print()

    # 5. (オプション) 発行日フィルタ
    if args.filter_years and len(scored) >= 10000:
        print("【STEP5】発行日フィルタリング...")
        scored = filter_by_date(scored, patent, years=10)
        print(f"  フィルタ後: {len(scored):,}件")
        print()

    # 6. 結果をCSVに保存
    print("【STEP6】結果保存...")
    output_path = save_results_to_csv(patent_id, scored, args.output)

    elapsed = time.time() - start_time

    # サマリー表示
    print()
    print("="*80)
    print("【処理完了】")
    print("="*80)
    print(f"対象特許: {patent_id}")
    print(f"STAGE1候補: {len(candidates):,}件")
    print(f"最終結果: {len(scored):,}件")
    print(f"処理時間: {elapsed:.1f}秒")
    print(f"出力ファイル: {output_path}")
    print()

    # 上位10件を表示
    print("上位10件:")
    for rank, (doc_num, score) in enumerate(scored[:10], 1):
        patent_num = f"JP{doc_num}A" if not doc_num.startswith('JP') else doc_num
        print(f"  {rank}位: {patent_num} ({score:.0f}点)")


if __name__ == "__main__":
    main()

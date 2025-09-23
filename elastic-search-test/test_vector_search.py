#!/usr/bin/env python3
"""
ベクトル検索のテストスクリプト
"""
import json
from datetime import datetime

import requests

# API エンドポイント
API_BASE_URL = "http://localhost:8000"

def print_section(title):
    """セクションヘッダーを表示"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def print_result(result, index=1):
    """検索結果を整形して表示"""
    print(f"\n--- 結果 {index} ---")
    print(f"特許ID: {result.get('patent_id', 'N/A')}")
    print(f"スコア: {result.get('score', result.get('similarity_score', 'N/A')):.4f}")
    print(f"タイトル: {result.get('title', 'N/A')[:100]}")
    print(f"要約: {result.get('summary', 'N/A')[:200]}...")
    print(f"出願日: {result.get('filing_date', 'N/A')}")
    print(f"公開日: {result.get('publication_date', 'N/A')}")

    # IPC分類があれば表示
    ipc = result.get('classification_ipc', [])
    if ipc:
        print(f"IPC分類: {', '.join(ipc[:5])}")

    # キーワードがあれば表示
    keywords = result.get('keywords', [])
    if keywords:
        print(f"キーワード: {', '.join(keywords[:10])}")

def test_health_check():
    """ヘルスチェック"""
    print_section("ヘルスチェック")

    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"ステータス: {data['status']}")
            print(f"Elasticsearch: {data['elasticsearch']['status']}")
            print(f"ベクトルインデックス: {data['vector_index']['name']}")
            print(f"インデックス存在: {data['vector_index']['exists']}")
            return True
        else:
            print(f"エラー: ステータスコード {response.status_code}")
            return False
    except Exception as e:
        print(f"エラー: {e}")
        return False

def test_vector_search():
    """ベクトル検索のテスト"""
    print_section("ベクトル検索テスト")

    # テストクエリのリスト
    test_queries = [
        {
            "query": "電池の充電効率を向上させる技術",
            "description": "バッテリー関連の特許を検索"
        },
        {
            "query": "人工知能を用いた画像認識システム",
            "description": "AI・画像処理関連の特許を検索"
        },
        {
            "query": "自動車の自動運転制御装置",
            "description": "自動運転関連の特許を検索"
        }
    ]

    for test_case in test_queries:
        print(f"\n\n🔍 検索クエリ: {test_case['query']}")
        print(f"   ({test_case['description']})")
        print("-" * 50)

        # ベクトル検索リクエスト
        payload = {
            "query": test_case["query"],
            "k": 5,  # 上位5件を取得
            "min_score": 0.5  # 最小スコア
        }

        try:
            response = requests.post(
                f"{API_BASE_URL}/vector-search",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                print(f"\n検索結果: {data['count']}件")

                if data['count'] > 0:
                    for i, result in enumerate(data['results'], 1):
                        print_result(result, i)
                else:
                    print("該当する特許が見つかりませんでした。")
            else:
                print(f"エラー: ステータスコード {response.status_code}")
                print(f"詳細: {response.text}")

        except Exception as e:
            print(f"エラー: {e}")

def test_hybrid_search():
    """ハイブリッド検索のテスト"""
    print_section("ハイブリッド検索テスト（テキスト + ベクトル）")

    query = "リチウムイオン電池の安全性向上"

    print(f"\n🔍 検索クエリ: {query}")
    print("-" * 50)

    # ハイブリッド検索リクエスト
    payload = {
        "query": query,
        "k": 5,
        "text_weight": 0.3,  # テキスト検索の重み
        "vector_weight": 0.7  # ベクトル検索の重み
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/hybrid-search",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n検索結果: {data['count']}件")

            if data['count'] > 0:
                for i, result in enumerate(data['results'], 1):
                    print_result(result, i)
            else:
                print("該当する特許が見つかりませんでした。")
        else:
            print(f"エラー: ステータスコード {response.status_code}")
            print(f"詳細: {response.text}")

    except Exception as e:
        print(f"エラー: {e}")

def test_similar_documents():
    """類似文書検索のテスト"""
    print_section("類似文書検索テスト")

    # まず、ベクトル検索で1件取得して、その特許IDを使用
    print("\n1. まず、サンプル特許を検索...")

    payload = {
        "query": "電池",
        "k": 1,
        "min_score": 0.1
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/vector-search",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()
            if data['count'] > 0:
                patent_id = data['results'][0]['patent_id']
                print(f"サンプル特許ID: {patent_id}")
                print(f"タイトル: {data['results'][0].get('title', 'N/A')}")

                # 類似文書検索
                print(f"\n2. 特許ID {patent_id} に類似する文書を検索...")
                print("-" * 50)

                similar_payload = {
                    "patent_id": patent_id,
                    "k": 5,
                    "min_score": 0.5
                }

                similar_response = requests.post(
                    f"{API_BASE_URL}/similar-documents",
                    json=similar_payload,
                    headers={"Content-Type": "application/json"}
                )

                if similar_response.status_code == 200:
                    similar_data = similar_response.json()
                    print(f"\n類似文書: {similar_data['count']}件")

                    if similar_data['count'] > 0:
                        for i, result in enumerate(similar_data['results'], 1):
                            print_result(result, i)
                    else:
                        print("類似する特許が見つかりませんでした。")
                else:
                    print(f"エラー: ステータスコード {similar_response.status_code}")
                    print(f"詳細: {similar_response.text}")
            else:
                print("サンプル特許が見つかりませんでした。")
        else:
            print(f"エラー: ステータスコード {response.status_code}")
            print(f"詳細: {response.text}")

    except Exception as e:
        print(f"エラー: {e}")

def test_classification_search():
    """分類コード検索のテスト"""
    print_section("分類コード検索テスト")

    # テスト用のIPC分類コード（例）
    test_classifications = [
        ("H01M", "電池関連"),
        ("G06F", "コンピュータ関連"),
        ("H04N", "画像通信関連")
    ]

    for ipc_code, description in test_classifications:
        print(f"\n\n🔍 IPC分類コード: {ipc_code} ({description})")
        print("-" * 50)

        try:
            response = requests.get(
                f"{API_BASE_URL}/search-by-classification/{ipc_code}",
                params={
                    "classification_type": "ipc",
                    "k": 3  # 各分類で3件ずつ取得
                }
            )

            if response.status_code == 200:
                data = response.json()
                print(f"\n検索結果: {data['count']}件")

                if data['count'] > 0:
                    for i, result in enumerate(data['results'], 1):
                        print(f"\n--- 結果 {i} ---")
                        print(f"特許ID: {result.get('patent_id', 'N/A')}")
                        print(f"タイトル: {result.get('title', 'N/A')[:100]}")
                        print(f"公開日: {result.get('publication_date', 'N/A')}")
                else:
                    print("該当する特許が見つかりませんでした。")
            else:
                print(f"エラー: ステータスコード {response.status_code}")
                print(f"詳細: {response.text}")

        except Exception as e:
            print(f"エラー: {e}")

def main():
    """メイン処理"""
    print("=" * 60)
    print(" 特許ベクトル検索システム テスト")
    print(f" 実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. ヘルスチェック
    if not test_health_check():
        print("\n⚠️ システムが正常に動作していません。")
        return

    print("\n✅ システムは正常に動作しています。")

    # 2. ベクトル検索テスト
    test_vector_search()

    # 3. ハイブリッド検索テスト
    test_hybrid_search()

    # 4. 類似文書検索テスト
    test_similar_documents()

    # 5. 分類コード検索テスト
    test_classification_search()

    print("\n" + "=" * 60)
    print(" テスト完了")
    print("=" * 60)

if __name__ == "__main__":
    main()

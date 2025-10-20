#!/usr/bin/env python3
"""
データ整合性チェック専用スクリプト
Cosmos DB と Elasticsearch 間のデータ整合性を詳細に検証します
"""
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import json
import os

# パスを追加
sys.path.append(str(Path(__file__).parent))

from data_sync.cosmos_client import CosmosDBClient
from data_sync.elasticsearch_indexer import ElasticsearchIndexer

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'integrity_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

VECTOR_FIELD = os.getenv("ES_VECTOR_FIELD", "claims_vector")


class IntegrityChecker:
    """データ整合性チェック専用クラス"""

    def __init__(self):
        """初期化"""
        self.cosmos_client = CosmosDBClient()
        self.es_indexer = ElasticsearchIndexer()

    async def run_comprehensive_check(self, sample_size: int = 1000) -> Dict[str, Any]:
        """
        包括的な整合性チェックを実行

        Args:
            sample_size: サンプリング検証するドキュメント数

        Returns:
            検証結果の詳細
        """
        logger.info("🔍 包括的データ整合性チェックを開始します")
        start_time = datetime.now()

        results = {
            "timestamp": start_time.isoformat(),
            "basic_counts": {},
            "sample_validation": {},
            "missing_documents": [],
            "failed_documents_analysis": {},
            "recommendations": []
        }

        try:
            # 1. 基本的な件数チェック
            logger.info("📊 基本件数チェック中...")
            results["basic_counts"] = await self._check_basic_counts()

            # 2. サンプリング検証
            logger.info(f"🔍 サンプリング検証中（{sample_size}件）...")
            results["sample_validation"] = await self._sample_validation(sample_size)

            # 3. 失敗ドキュメントの分析
            logger.info("❌ 失敗ドキュメントの分析中...")
            results["failed_documents_analysis"] = await self._analyze_failed_documents()

            # 4. 推奨事項の生成
            results["recommendations"] = self._generate_recommendations(results)

            # 実行時間を記録
            end_time = datetime.now()
            results["duration_seconds"] = (end_time - start_time).total_seconds()

            logger.info("✅ 整合性チェック完了")
            return results

        except Exception as e:
            logger.error(f"整合性チェック中にエラーが発生: {e}")
            results["error"] = str(e)
            return results

        finally:
            self._cleanup()

    async def _check_basic_counts(self) -> Dict[str, Any]:
        """基本的な件数チェック"""
        try:
            # Cosmos DB の件数取得
            cosmos_count = self.cosmos_client.get_total_document_count()
            logger.info(f"   Cosmos DB総数: {cosmos_count:,}件")

            # Elasticsearch の統計取得
            es_stats = self.es_indexer.get_index_stats()
            es_count = es_stats.get("document_count", 0)
            es_size_mb = es_stats.get("size_in_bytes", 0) / (1024 * 1024)

            logger.info(f"   Elasticsearch総数: {es_count:,}件")
            logger.info(f"   Elasticsearchサイズ: {es_size_mb:.2f} MB")

            # 差分計算
            difference = cosmos_count - es_count
            match_percentage = (es_count / cosmos_count * 100) if cosmos_count > 0 else 0

            return {
                "cosmos_total": cosmos_count,
                "elasticsearch_total": es_count,
                "difference": difference,
                "match_percentage": round(match_percentage, 2),
                "elasticsearch_size_mb": round(es_size_mb, 2),
                "counts_match": difference == 0,
                "connection_status": {
                    "cosmos_connected": True,
                    "elasticsearch_connected": self.es_indexer.is_connected()
                }
            }

        except Exception as e:
            logger.error(f"基本件数チェックエラー: {e}")
            return {"error": str(e)}

    async def _sample_validation(self, sample_size: int) -> Dict[str, Any]:
        """サンプリング検証 - Elasticsearchに存在するドキュメントからサンプリング"""
        try:
            # まずElasticsearchから存在するドキュメントIDをランダムサンプリング
            logger.info(f"   Elasticsearchから{sample_size}件のドキュメントIDを取得中...")
            es_document_ids = self.es_indexer.get_random_document_ids(sample_size)

            if not es_document_ids:
                logger.warning("ElasticsearchからドキュメントIDを取得できませんでした")
                return {
                    "error": "Elasticsearchからサンプルドキュメントを取得できませんでした",
                    "sample_size": 0,
                    "valid_documents": 0,
                    "invalid_documents": 0,
                    "success_rate": 0
                }

            actual_sample_size = len(es_document_ids)
            logger.info(f"   実際のサンプルサイズ: {actual_sample_size}件")

            valid_count = 0
            invalid_count = 0
            missing_in_cosmos = []
            validation_errors = []

            # 各ESドキュメントについて、Cosmos DBとの整合性を検証
            for i, patent_id in enumerate(es_document_ids):
                # Elasticsearchからドキュメントを取得
                es_doc = self.es_indexer.get_document_by_id(patent_id)
                if not es_doc:
                    logger.warning(f"ESドキュメントが見つかりません: {patent_id}")
                    invalid_count += 1
                    continue

                # Cosmos DBから対応するドキュメントを取得
                cosmos_doc = self.cosmos_client.get_document_by_id(patent_id)

                if cosmos_doc:
                    # 両方に存在する場合、フィールドを検証
                    validation_result = self._validate_document_fields(cosmos_doc, es_doc)
                    if validation_result["valid"]:
                        valid_count += 1
                    else:
                        invalid_count += 1
                        validation_errors.append({
                            "patent_id": patent_id,
                            "errors": validation_result["errors"]
                        })
                else:
                    # Cosmos DBに存在しない（これは異常）
                    invalid_count += 1
                    missing_in_cosmos.append(patent_id)
                    logger.warning(f"Cosmos DBにドキュメントが見つかりません: {patent_id}")

                # 進捗表示
                if (i + 1) % 100 == 0:
                    logger.info(f"   サンプル検証進捗: {i + 1}/{actual_sample_size}")

            success_rate = (valid_count / actual_sample_size * 100) if actual_sample_size > 0 else 0

            return {
                "sample_size": actual_sample_size,
                "valid_documents": valid_count,
                "invalid_documents": invalid_count,
                "missing_in_cosmos_count": len(missing_in_cosmos),
                "success_rate": round(success_rate, 2),
                "missing_in_cosmos_sample": missing_in_cosmos[:20],  # 最初の20件のみ
                "validation_errors_sample": validation_errors[:10]   # 最初の10件のみ
            }

        except Exception as e:
            logger.error(f"サンプリング検証エラー: {e}")
            return {"error": str(e)}

    def _validate_document_fields(self, cosmos_doc: Dict, es_doc: Dict) -> Dict[str, Any]:
        """ドキュメントフィールドの検証"""
        errors = []

        # 必須フィールドの検証
        required_fields = ["patent_id", "title", "summary"]
        for field in required_fields:
            cosmos_value = cosmos_doc.get(field, "")
            es_value = es_doc.get(field, "")

            if cosmos_value != es_value:
                errors.append(f"{field}: Cosmos='{cosmos_value}' != ES='{es_value}'")

        # エンベディングの存在確認
        if not es_doc.get(VECTOR_FIELD):
            errors.append(f"{VECTOR_FIELD}: エンベディングが存在しません")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    async def _analyze_failed_documents(self) -> Dict[str, Any]:
        """失敗ドキュメントの分析"""
        try:
            es_stats = self.es_indexer.get_index_stats()
            failed_documents = es_stats.get("failed_documents", [])

            if not failed_documents:
                return {
                    "total_failed": 0,
                    "analysis": "失敗ドキュメントはありません"
                }

            # 失敗理由の分類（簡易版）
            failure_categories = {
                "embedding_failed": 0,
                "indexing_failed": 0,
                "unknown": 0
            }

            # サンプル分析（最初の100件）
            sample_failed = failed_documents[:100]
            for doc_id in sample_failed:
                # Cosmos DB からドキュメントを取得して分析
                cosmos_doc = self.cosmos_client.get_document_by_id(doc_id)
                if cosmos_doc:
                    # 簡易的な分類
                    if not cosmos_doc.get("summary"):
                        failure_categories["embedding_failed"] += 1
                    else:
                        failure_categories["indexing_failed"] += 1
                else:
                    failure_categories["unknown"] += 1

            return {
                "total_failed": len(failed_documents),
                "sample_analyzed": len(sample_failed),
                "failure_categories": failure_categories,
                "failed_documents_sample": failed_documents[:20]
            }

        except Exception as e:
            logger.error(f"失敗ドキュメント分析エラー: {e}")
            return {"error": str(e)}

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """推奨事項の生成"""
        recommendations = []

        # 基本件数チェックの結果に基づく推奨事項
        basic_counts = results.get("basic_counts", {})
        if basic_counts.get("difference", 0) > 0:
            difference = basic_counts["difference"]
            recommendations.append(
                f"🔄 {difference:,}件のドキュメントが不足しています。失敗したドキュメントの再処理を検討してください。"
            )

        # サンプリング検証の結果に基づく推奨事項
        sample_validation = results.get("sample_validation", {})
        success_rate = sample_validation.get("success_rate", 100)
        if success_rate < 95:
            recommendations.append(
                f"⚠️ サンプル検証の成功率が{success_rate}%です。データ品質の改善が必要です。"
            )

        # 失敗ドキュメント分析の結果に基づく推奨事項
        failed_analysis = results.get("failed_documents_analysis", {})
        total_failed = failed_analysis.get("total_failed", 0)
        if total_failed > 0:
            recommendations.append(
                f"❌ {total_failed:,}件の失敗ドキュメントがあります。失敗理由を分析して再処理してください。"
            )

        # 接続状態に基づく推奨事項
        if not basic_counts.get("connection_status", {}).get("elasticsearch_connected", True):
            recommendations.append(
                "🔌 Elasticsearch接続が閉じられています。統計情報が不正確な可能性があります。"
            )

        # 一般的な推奨事項
        if not recommendations:
            recommendations.append("✅ データ整合性に大きな問題は見つかりませんでした。")
        else:
            recommendations.append(
                "🛠️ 問題の修正後、再度整合性チェックを実行することを推奨します。"
            )

        return recommendations

    def _cleanup(self):
        """リソースのクリーンアップ"""
        try:
            self.cosmos_client.close()
            self.es_indexer.close()
            logger.info("リソースをクリーンアップしました")
        except Exception as e:
            logger.warning(f"クリーンアップ中にエラー: {e}")


async def main():
    """メイン処理"""
    print("\n" + "🔍" * 35)
    print("データ整合性チェックツール")
    print("🔍" * 35)
    print("Cosmos DB と Elasticsearch 間のデータ整合性を詳細に検証します")
    print("=" * 70)

    # サンプルサイズの設定
    try:
        sample_size = int(input("サンプリング検証するドキュメント数を入力してください (デフォルト: 1000): ") or "1000")
    except ValueError:
        sample_size = 1000

    print(f"\n📋 検証設定:")
    print(f"   サンプルサイズ: {sample_size:,}件")

    response = input("\n整合性チェックを開始しますか？ (y/N): ")
    if response.lower() != 'y':
        print("処理をキャンセルしました")
        sys.exit(0)

    start_time = datetime.now()

    try:
        # 整合性チェッカーを初期化
        checker = IntegrityChecker()

        # 包括的チェックを実行
        results = await checker.run_comprehensive_check(sample_size)

        # 結果の表示
        print("\n" + "=" * 70)
        print("📊 整合性チェック結果")
        print("=" * 70)

        # 基本件数
        basic_counts = results.get("basic_counts", {})
        if "error" not in basic_counts:
            print(f"\n📈 基本統計:")
            print(f"   Cosmos DB総数: {basic_counts.get('cosmos_total', 0):,}件")
            print(f"   Elasticsearch総数: {basic_counts.get('elasticsearch_total', 0):,}件")
            print(f"   差分: {basic_counts.get('difference', 0):,}件")
            print(f"   一致率: {basic_counts.get('match_percentage', 0)}%")
            print(f"   インデックスサイズ: {basic_counts.get('elasticsearch_size_mb', 0)} MB")

        # サンプリング検証
        sample_validation = results.get("sample_validation", {})
        if "error" not in sample_validation:
            print(f"\n🔍 サンプリング検証:")
            print(f"   検証件数: {sample_validation.get('sample_size', 0):,}件")
            print(f"   有効: {sample_validation.get('valid_documents', 0):,}件")
            print(f"   無効: {sample_validation.get('invalid_documents', 0):,}件")
            print(f"   Cosmos DB未存在: {sample_validation.get('missing_in_cosmos_count', 0):,}件")
            print(f"   成功率: {sample_validation.get('success_rate', 0)}%")

        # 失敗ドキュメント分析
        failed_analysis = results.get("failed_documents_analysis", {})
        if "error" not in failed_analysis:
            print(f"\n❌ 失敗ドキュメント分析:")
            print(f"   総失敗件数: {failed_analysis.get('total_failed', 0):,}件")

        # 推奨事項
        recommendations = results.get("recommendations", [])
        if recommendations:
            print(f"\n💡 推奨事項:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

        # 実行時間
        duration = results.get("duration_seconds", 0)
        print(f"\n⏱️ 実行時間: {duration:.1f}秒")

        # 結果をJSONファイルに保存
        output_file = f"integrity_check_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n💾 詳細結果を保存しました: {output_file}")

        print("\n" + "=" * 70)
        print("🎉 整合性チェックが完了しました！")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️ 処理が中断されました")
        sys.exit(1)

    except Exception as e:
        logger.error(f"整合性チェック中にエラーが発生しました: {e}", exc_info=True)
        print(f"\n❌ エラーが発生しました: {e}")
        print("詳細はログファイルを確認してください")
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "🔍" * 35)
    print("データ整合性チェックツール")
    print("🔍" * 35)

    # 非同期処理を実行
    asyncio.run(main())

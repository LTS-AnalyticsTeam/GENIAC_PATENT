#!/usr/bin/env python3
"""
失敗ドキュメント再処理スクリプト
同期に失敗したドキュメントを特定し、再処理を実行します
"""
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# パスを追加
sys.path.append(str(Path(__file__).parent))

from data_sync.cosmos_client import CosmosDBClient
from data_sync.elasticsearch_indexer import ElasticsearchIndexer
from data_sync.embedding_processor import EmbeddingProcessor

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'failed_retry_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class FailedDocumentRetryProcessor:
    """失敗ドキュメント再処理クラス"""

    def __init__(self):
        """初期化"""
        self.cosmos_client = CosmosDBClient()
        self.es_indexer = ElasticsearchIndexer()
        self.embedding_processor = EmbeddingProcessor()

        # 統計情報
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_failed_found": 0,
            "retry_attempted": 0,
            "retry_successful": 0,
            "retry_failed": 0,
            "skipped": 0,
            "errors": []
        }

    async def identify_failed_documents(self, method: str = "missing") -> List[str]:
        """
        失敗ドキュメントを特定

        Args:
            method: 特定方法 ("missing", "es_failed", "comprehensive")

        Returns:
            失敗ドキュメントIDのリスト
        """
        logger.info(f"🔍 失敗ドキュメントを特定中（方法: {method}）...")
        failed_docs = []

        try:
            if method == "missing":
                # Cosmos DB にあるが Elasticsearch にないドキュメントを特定
                failed_docs = await self._find_missing_documents()

            elif method == "es_failed":
                # Elasticsearch indexer の失敗リストから取得
                es_stats = self.es_indexer.get_index_stats()
                failed_docs = es_stats.get("failed_documents", [])

            elif method == "comprehensive":
                # 両方の方法を組み合わせ
                missing_docs = await self._find_missing_documents()
                es_failed_docs = self.es_indexer.get_index_stats().get("failed_documents", [])
                failed_docs = list(set(missing_docs + es_failed_docs))

            logger.info(f"   特定された失敗ドキュメント: {len(failed_docs)}件")
            return failed_docs

        except Exception as e:
            logger.error(f"失敗ドキュメント特定エラー: {e}")
            return []

    async def _find_missing_documents(self, limit: int = None) -> List[str]:
        """Cosmos DB にあるが Elasticsearch にないドキュメントを効率的に特定"""
        missing_docs = []
        checked_count = 0

        try:
            logger.info("   Cosmos DBから全ドキュメントIDを取得中...")

            # Cosmos DBから全ドキュメントIDを取得
            cosmos_ids = set()
            for cosmos_batch in self.cosmos_client.get_all_documents(batch_size=1000):
                batch_ids = []
                for doc in cosmos_batch:
                    patent_id = doc.get("patent_id")
                    if patent_id:
                        cosmos_ids.add(patent_id)
                        batch_ids.append(patent_id)

                checked_count += len(batch_ids)

                # 進捗表示
                if checked_count % 5000 == 0:
                    logger.info(f"   Cosmos ID取得進捗: {checked_count:,}件")

                # 制限がある場合はチェック
                if limit and checked_count >= limit:
                    break

            logger.info(f"   Cosmos DB総ID数: {len(cosmos_ids):,}件")

            # Elasticsearchから存在するドキュメントIDを取得
            logger.info("   Elasticsearchから存在ドキュメントIDを取得中...")
            es_ids = set()

            # 大きなサンプルサイズでElasticsearchの全ドキュメントIDを取得
            es_document_ids = self.es_indexer.get_random_document_ids(sample_size=50000)

            # より確実に全IDを取得するため、scrollを直接使用
            try:
                query = {
                    "query": {"match_all": {}},
                    "_source": False,
                    "size": 10000
                }

                response = self.es_indexer.es.search(
                    index=self.es_indexer.index_name,
                    body=query,
                    scroll='5m'
                )

                # 最初のバッチ
                for hit in response['hits']['hits']:
                    es_ids.add(hit['_id'])

                # 残りのドキュメントをscrollで取得
                scroll_id = response.get('_scroll_id')
                while scroll_id and len(response['hits']['hits']) > 0:
                    response = self.es_indexer.es.scroll(
                        scroll_id=scroll_id,
                        scroll='5m'
                    )

                    for hit in response['hits']['hits']:
                        es_ids.add(hit['_id'])

                    # 進捗表示
                    if len(es_ids) % 5000 == 0:
                        logger.info(f"   ES ID取得進捗: {len(es_ids):,}件")

                # scrollをクリア
                if scroll_id:
                    try:
                        self.es_indexer.es.clear_scroll(scroll_id=scroll_id)
                    except Exception as e:
                        logger.debug(f"Error clearing scroll: {e}")

            except Exception as e:
                logger.warning(f"Scroll取得エラー、代替方法を使用: {e}")
                # 代替方法として既存のメソッドを使用
                es_ids = set(es_document_ids)

            logger.info(f"   Elasticsearch総ID数: {len(es_ids):,}件")

            # 差分を計算（Cosmos DBにあるがElasticsearchにないもの）
            missing_docs = list(cosmos_ids - es_ids)
            logger.info(f"   不足ドキュメント: {len(missing_docs):,}件")

            return missing_docs

        except Exception as e:
            logger.error(f"不足ドキュメント検索エラー: {e}")
            return missing_docs

    async def retry_failed_documents(
        self,
        failed_doc_ids: List[str],
        batch_size: int = 50,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        失敗ドキュメントの再処理を実行

        Args:
            failed_doc_ids: 再処理するドキュメントIDのリスト
            batch_size: バッチサイズ
            max_retries: 最大リトライ回数

        Returns:
            再処理結果の統計
        """
        logger.info(f"🔄 失敗ドキュメントの再処理を開始（{len(failed_doc_ids)}件）...")
        self.stats["start_time"] = datetime.now()
        self.stats["total_failed_found"] = len(failed_doc_ids)

        try:
            # バッチごとに処理
            for i in range(0, len(failed_doc_ids), batch_size):
                batch_ids = failed_doc_ids[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(failed_doc_ids) + batch_size - 1) // batch_size

                logger.info(f"📦 バッチ {batch_num}/{total_batches} を処理中（{len(batch_ids)}件）...")

                # Cosmos DB からドキュメントを取得
                cosmos_docs = []
                for doc_id in batch_ids:
                    try:
                        doc = self.cosmos_client.get_document_by_id(doc_id)
                        if doc:
                            cosmos_docs.append(doc)
                        else:
                            logger.warning(f"   ドキュメント {doc_id} が Cosmos DB に見つかりません")
                            self.stats["skipped"] += 1
                    except Exception as e:
                        logger.error(f"   ドキュメント {doc_id} の取得エラー: {e}")
                        self.stats["errors"].append(f"Cosmos取得エラー {doc_id}: {str(e)}")
                        self.stats["skipped"] += 1

                if not cosmos_docs:
                    logger.warning(f"   バッチ {batch_num} にはドキュメントがありません")
                    continue

                # エンベディング処理とインデックス処理を実行
                await self._process_batch_with_retry(cosmos_docs, max_retries)

                # 進捗表示
                progress = (i + len(batch_ids)) / len(failed_doc_ids) * 100
                logger.info(f"   進捗: {progress:.1f}% ({i + len(batch_ids)}/{len(failed_doc_ids)})")

            # 統計情報を更新
            self.stats["end_time"] = datetime.now()
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

            logger.info(f"✅ 再処理完了（{duration:.1f}秒）")
            return self.stats

        except Exception as e:
            logger.error(f"再処理中にエラー: {e}")
            self.stats["errors"].append(f"再処理エラー: {str(e)}")
            return self.stats

    async def _process_batch_with_retry(self, cosmos_docs: List[Dict], max_retries: int):
        """バッチ処理（リトライ付き）"""
        for attempt in range(max_retries + 1):
            try:
                # エンベディング処理
                processed_docs = await self.embedding_processor.process_documents_parallel(cosmos_docs)

                # 成功したドキュメントと失敗したドキュメントを分類
                successful_docs = []
                failed_docs = []

                for doc in processed_docs:
                    if doc.get("embedding_generated"):
                        successful_docs.append(doc)
                    else:
                        failed_docs.append(doc)

                # 成功したドキュメントをインデックス
                if successful_docs:
                    index_result = self.es_indexer.bulk_index_documents(successful_docs, skip_existing=False)
                    self.stats["retry_attempted"] += len(successful_docs)
                    self.stats["retry_successful"] += index_result["indexed"]
                    self.stats["retry_failed"] += index_result["failed"]

                    # インデックス失敗の詳細をログ
                    for error in index_result.get("errors", []):
                        self.stats["errors"].append(f"インデックスエラー: {error}")

                # エンベディング失敗のドキュメントを記録
                for doc in failed_docs:
                    patent_id = doc.get("patent_id", "unknown")
                    self.stats["retry_failed"] += 1
                    self.stats["errors"].append(f"エンベディング失敗: {patent_id}")

                # 成功したら終了
                break

            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"   バッチ処理失敗（試行 {attempt + 1}/{max_retries + 1}）: {e}")
                    logger.info(f"   {wait_time}秒後にリトライします...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"   バッチ処理が最大リトライ回数後も失敗: {e}")
                    # 全ドキュメントを失敗として記録
                    for doc in cosmos_docs:
                        patent_id = doc.get("patent_id", "unknown")
                        self.stats["retry_failed"] += 1
                        self.stats["errors"].append(f"バッチ処理失敗: {patent_id} - {str(e)}")

    def generate_retry_report(self, results: Dict[str, Any]) -> str:
        """再処理レポートを生成"""
        report_lines = [
            "=" * 70,
            "📊 失敗ドキュメント再処理レポート",
            "=" * 70,
            "",
            f"🕐 実行時間: {results.get('start_time', 'N/A')} - {results.get('end_time', 'N/A')}",
            f"⏱️ 処理時間: {(results.get('end_time', datetime.now()) - results.get('start_time', datetime.now())).total_seconds():.1f}秒",
            "",
            "📈 処理統計:",
            f"   特定された失敗ドキュメント: {results.get('total_failed_found', 0):,}件",
            f"   再処理試行: {results.get('retry_attempted', 0):,}件",
            f"   再処理成功: {results.get('retry_successful', 0):,}件",
            f"   再処理失敗: {results.get('retry_failed', 0):,}件",
            f"   スキップ: {results.get('skipped', 0):,}件",
            "",
        ]

        # 成功率計算
        total_attempted = results.get('retry_attempted', 0)
        if total_attempted > 0:
            success_rate = (results.get('retry_successful', 0) / total_attempted) * 100
            report_lines.append(f"🎯 成功率: {success_rate:.1f}%")
        else:
            report_lines.append("🎯 成功率: N/A（処理なし）")

        # エラー情報
        errors = results.get('errors', [])
        if errors:
            report_lines.extend([
                "",
                f"❌ エラー詳細（最初の10件）:",
            ])
            for i, error in enumerate(errors[:10], 1):
                report_lines.append(f"   {i}. {error}")
            if len(errors) > 10:
                report_lines.append(f"   ... 他 {len(errors) - 10}件のエラー")

        report_lines.extend([
            "",
            "=" * 70
        ])

        return "\n".join(report_lines)

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
    print("\n" + "🔄" * 35)
    print("失敗ドキュメント再処理ツール")
    print("🔄" * 35)
    print("同期に失敗したドキュメントを特定し、再処理を実行します")
    print("=" * 70)

    # 設定の入力
    print("\n📋 設定:")
    print("1. missing - Cosmos DB にあるが Elasticsearch にないドキュメント")
    print("2. es_failed - Elasticsearch indexer の失敗リスト")
    print("3. comprehensive - 上記両方を組み合わせ")

    method = input("\n失敗ドキュメントの特定方法を選択してください (1-3, デフォルト: 1): ").strip()
    method_map = {"1": "missing", "2": "es_failed", "3": "comprehensive"}
    method = method_map.get(method, "missing")

    try:
        batch_size = int(input("バッチサイズを入力してください (デフォルト: 50): ") or "50")
        max_retries = int(input("最大リトライ回数を入力してください (デフォルト: 3): ") or "3")
    except ValueError:
        batch_size = 50
        max_retries = 3

    print(f"\n📋 実行設定:")
    print(f"   特定方法: {method}")
    print(f"   バッチサイズ: {batch_size}")
    print(f"   最大リトライ回数: {max_retries}")

    response = input("\n再処理を開始しますか？ (y/N): ")
    if response.lower() != 'y':
        print("処理をキャンセルしました")
        sys.exit(0)

    try:
        # 再処理プロセッサーを初期化
        processor = FailedDocumentRetryProcessor()

        # 失敗ドキュメントを特定
        failed_doc_ids = await processor.identify_failed_documents(method)

        if not failed_doc_ids:
            print("\n✅ 再処理が必要なドキュメントは見つかりませんでした")
            return

        print(f"\n🔍 {len(failed_doc_ids)}件の失敗ドキュメントが見つかりました")

        # 確認
        if len(failed_doc_ids) > 1000:
            response = input(f"⚠️ 大量のドキュメント（{len(failed_doc_ids)}件）を処理します。続行しますか？ (y/N): ")
            if response.lower() != 'y':
                print("処理をキャンセルしました")
                return

        # 再処理を実行
        results = await processor.retry_failed_documents(failed_doc_ids, batch_size, max_retries)

        # レポートを生成・表示
        report = processor.generate_retry_report(results)
        print(f"\n{report}")

        # 結果をJSONファイルに保存
        output_file = f"failed_retry_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n💾 詳細結果を保存しました: {output_file}")

        print("\n" + "=" * 70)
        print("🎉 失敗ドキュメント再処理が完了しました！")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️ 処理が中断されました")
        sys.exit(1)

    except Exception as e:
        logger.error(f"再処理中にエラーが発生しました: {e}", exc_info=True)
        print(f"\n❌ エラーが発生しました: {e}")
        print("詳細はログファイルを確認してください")
        sys.exit(1)

    finally:
        # クリーンアップ
        try:
            processor._cleanup()
        except:
            pass


if __name__ == "__main__":
    print("\n" + "🔄" * 35)
    print("失敗ドキュメント再処理ツール")
    print("🔄" * 35)

    # 非同期処理を実行
    asyncio.run(main())

#!/usr/bin/env python3
"""
改良版同期機能のテストスクリプト
小規模なデータセットで機能を検証
"""
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from data_sync.sync_orchestrator import SyncOrchestrator

# パスを追加
sys.path.append(str(Path(__file__).parent))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_improved_sync():
    """改良版同期機能をテスト"""
    print("\n" + "="*60)
    print("🧪 改良版同期機能テスト")
    print("="*60)

    try:
        # オーケストレーターを初期化
        orchestrator = SyncOrchestrator()

        # テスト1: 小規模な日付範囲での同期テスト
        print("\n📅 テスト1: 小規模日付範囲同期 (2010年1月)")
        start_time = datetime.now()

        await orchestrator.sync_by_date_range(
            start_date="20100101",
            end_date="20100131",
            force_recreate_index=False
        )

        end_time = datetime.now()
        duration = end_time - start_time

        # 統計を取得
        stats = orchestrator.get_statistics()
        orchestrator_stats = stats['orchestrator_stats']

        print(f"✅ テスト1完了:")
        print(f"   処理時間: {duration}")
        print(f"   処理済み: {orchestrator_stats['documents_processed']}件")
        print(f"   インデックス済み: {orchestrator_stats['documents_indexed']}件")
        print(f"   失敗: {orchestrator_stats['documents_failed']}件")

        # テスト2: データ整合性チェック
        print(f"\n🔍 テスト2: データ整合性チェック")
        validation_results = await orchestrator.validate_data_integrity()

        if 'error' not in validation_results:
            print(f"✅ 整合性チェック完了:")
            print(f"   Cosmos DB: {validation_results['cosmos_total']:,}件")
            print(f"   Elasticsearch: {validation_results['elasticsearch_total']:,}件")
            print(f"   一致: {'✅' if validation_results['count_match'] else '❌'}")
        else:
            print(f"❌ 整合性チェックエラー: {validation_results['error']}")

        # テスト3: 重複防止テスト（同じ範囲を再実行）
        print(f"\n🔄 テスト3: 重複防止テスト（同じ範囲を再実行）")

        # 統計をリセット
        orchestrator.stats = {
            "start_time": None,
            "end_time": None,
            "total_documents": 0,
            "documents_processed": 0,
            "documents_indexed": 0,
            "documents_failed": 0,
            "embeddings_generated": 0,
            "embeddings_failed": 0,
            "errors": []
        }

        start_time = datetime.now()

        await orchestrator.sync_by_date_range(
            start_date="20100101",
            end_date="20100131",
            force_recreate_index=False
        )

        end_time = datetime.now()
        duration = end_time - start_time

        # 統計を取得
        stats = orchestrator.get_statistics()
        orchestrator_stats = stats['orchestrator_stats']

        print(f"✅ テスト3完了:")
        print(f"   処理時間: {duration}")
        print(f"   処理済み: {orchestrator_stats['documents_processed']}件")
        print(f"   新規インデックス: {orchestrator_stats['documents_indexed']}件")
        print(f"   スキップされた重複: 既存データはスキップされるはず")

        # テスト4: 単一ドキュメント同期テスト
        print(f"\n📄 テスト4: 単一ドキュメント同期テスト")

        # 最初のドキュメントを取得してテスト
        test_patent_id = None
        for cosmos_batch in orchestrator.cosmos_client.get_documents_by_date_range("20100101", "20100105", batch_size=1):
            if cosmos_batch:
                test_patent_id = cosmos_batch[0].get("patent_id")
                break

        if test_patent_id:
            success = await orchestrator.sync_single_document(test_patent_id)
            print(f"✅ 単一ドキュメント同期: {'成功' if success else '失敗'} (ID: {test_patent_id})")
        else:
            print("❌ テスト用ドキュメントが見つかりませんでした")

        print("\n" + "="*60)
        print("🎉 全テスト完了！")
        print("="*60)

        # 最終統計
        final_stats = orchestrator.get_statistics()
        es_stats = final_stats.get('elasticsearch_stats', {})

        print(f"\n📊 最終統計:")
        print(f"   Elasticsearchドキュメント数: {es_stats.get('document_count', 0):,}件")
        if es_stats.get('size_in_bytes'):
            size_mb = es_stats['size_in_bytes'] / (1024 * 1024)
            print(f"   インデックスサイズ: {size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"テスト中にエラーが発生しました: {e}", exc_info=True)
        print(f"\n❌ テストエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "🧪"*30)
    print("改良版同期機能テストツール")
    print("🧪"*30)

    print("\n⚠️  このテストは小規模なデータセットを使用します")
    print("実際の全件同期前に機能を検証できます")

    response = input("\nテストを開始しますか？ (y/N): ")
    if response.lower() != 'y':
        print("テストをキャンセルしました")
        sys.exit(0)

    # 非同期処理を実行
    asyncio.run(test_improved_sync())

#!/usr/bin/env python3
"""
Cosmos DBから全件データを取得してElasticsearchに同期するスクリプト
"""
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# パスを追加
sys.path.append(str(Path(__file__).parent))


from data_sync.sync_orchestrator import SyncOrchestrator

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'sync_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """全件データの同期を実行"""
    print("\n" + "="*60)
    print("🚀 Cosmos DB → Elasticsearch 全件同期開始")
    print("="*60)

    start_time = datetime.now()

    try:
        # オーケストレーターを初期化
        orchestrator = SyncOrchestrator()

        # 全ドキュメントを同期（インデックスを再作成）
        print("\n📊 同期処理を開始します...")
        print("   - 総ドキュメント数: 約39,680件")
        print("   - 推定処理時間: 約20-30分（レート制限による）")
        print("   - バッチサイズ: 20文書/バッチ")
        print("   - 並列ワーカー数: 5")
        print("\n⚠️  処理中はこのスクリプトを中断しないでください")
        print("-"*60)

        # 全ドキュメントを同期
        await orchestrator.sync_all_documents(force_recreate_index=True)

        # 処理時間を計算
        end_time = datetime.now()
        duration = end_time - start_time

        # 統計情報を取得
        stats = orchestrator.get_statistics()

        print("\n" + "="*60)
        print("✅ 同期処理完了！")
        print("="*60)

        print(f"\n📊 処理統計:")
        print(f"   処理時間: {duration}")
        print(f"   処理済みドキュメント: {stats['orchestrator_stats']['documents_processed']:,}件")
        print(f"   インデックス済み: {stats['orchestrator_stats']['documents_indexed']:,}件")
        print(f"   失敗: {stats['orchestrator_stats']['documents_failed']}件")

        if stats['orchestrator_stats']['documents_failed'] > 0:
            print(f"\n⚠️  失敗したドキュメント:")
            failed_docs = stats['orchestrator_stats'].get('failed_documents', [])
            for doc_id in failed_docs[:10]:  # 最初の10件を表示
                print(f"   - {doc_id}")
            if len(failed_docs) > 10:
                print(f"   ... 他 {len(failed_docs) - 10}件")

        # Elasticsearchの統計
        es_stats = stats.get('elasticsearch_stats', {})
        if es_stats:
            print(f"\n📦 Elasticsearch統計:")
            print(f"   インデックス内ドキュメント数: {es_stats.get('document_count', 0):,}件")
            if es_stats.get('index_size_bytes'):
                size_mb = es_stats['index_size_bytes'] / (1024 * 1024)
                print(f"   インデックスサイズ: {size_mb:.2f} MB")

        # 成功率を計算
        total_processed = stats['orchestrator_stats']['documents_processed']
        if total_processed > 0:
            success_rate = (stats['orchestrator_stats']['documents_indexed'] / total_processed) * 100
            print(f"\n🎯 成功率: {success_rate:.1f}%")

        print("\n" + "="*60)
        print("🎉 全件同期が完了しました！")
        print("\n次のステップ:")
        print("1. APIエンドポイントでベクトル検索をテスト:")
        print('   curl -X POST http://localhost:3000/vector-search \\')
        print('     -H "Content-Type: application/json" \\')
        print('     -d \'{"query": "画像処理装置", "k": 10}\'')
        print("\n2. Kibanaで可視化:")
        print("   http://localhost:5601")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\n⚠️  処理が中断されました")
        print("再開するには、このスクリプトを再実行してください")
        sys.exit(1)

    except Exception as e:
        logger.error(f"同期処理中にエラーが発生しました: {e}", exc_info=True)
        print(f"\n❌ エラーが発生しました: {e}")
        print("詳細はログファイルを確認してください")
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "🚀"*30)
    print("Cosmos DB → Elasticsearch 全件同期ツール")
    print("🚀"*30)

    # 確認プロンプト
    print("\n⚠️  注意事項:")
    print("- 約39,680件のドキュメントを処理します")
    print("- Azure OpenAI APIのレート制限により、20-30分程度かかる可能性があります")
    print("- 処理中は大量のAPIリクエストが発生します")

    response = input("\n続行しますか？ (y/N): ")
    if response.lower() != 'y':
        print("処理をキャンセルしました")
        sys.exit(0)

    # 非同期処理を実行
    asyncio.run(main())

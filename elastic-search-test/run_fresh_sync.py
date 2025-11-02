#!/usr/bin/env python3
"""
改良版 Cosmos DB → Elasticsearch 全件同期スクリプト（新規インデックス作成）
チェックポイントを無視して、インデックスを最初から作り直します
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
        logging.FileHandler(f'fresh_sync_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """新規インデックスでの全件データ同期を実行"""
    print("\n" + "="*70)
    print("🆕 新規インデックス作成 - Cosmos DB → Elasticsearch 全件同期")
    print("="*70)
    print("✨ 特徴:")
    print("   - インデックスを完全に再作成")
    print("   - チェックポイントを無視")
    print("   - 重複防止機能")
    print("   - 強化されたエラーハンドリング")
    print("   - データ整合性チェック")
    print("="*70)

    start_time = datetime.now()

    try:
        # オーケストレーターを初期化
        orchestrator = SyncOrchestrator()

        print("\n⚠️  注意事項:")
        print("- 約39,680件のドキュメントを処理します")
        print("- インデックスを完全に再作成します")
        print("- Azure OpenAI APIのレート制限により、20-30分程度かかる可能性があります")
        print("- 処理中は大量のAPIリクエストが発生します")

        response = input("\n新規インデックスで同期を開始しますか？ (y/N): ")
        if response.lower() != 'y':
            print("処理をキャンセルしました")
            sys.exit(0)

        print("\n" + "="*70)
        print("🔄 新規インデックスでの同期処理を開始します...")
        print("="*70)

        # 新規インデックスで全ドキュメント同期を実行
        await orchestrator.sync_all_documents(
            force_recreate_index=True,    # インデックスを強制再作成
            resume_from_checkpoint=False  # チェックポイントを無視
        )

        # 処理時間を計算
        end_time = datetime.now()
        duration = end_time - start_time

        # 統計情報を取得
        stats = orchestrator.get_statistics()

        print("\n" + "="*70)
        print("✅ 新規インデックス同期完了！")
        print("="*70)

        print(f"\n📊 処理統計:")
        orchestrator_stats = stats['orchestrator_stats']
        print(f"   処理時間: {duration}")
        print(f"   処理済みドキュメント: {orchestrator_stats['documents_processed']:,}件")
        print(f"   インデックス済み: {orchestrator_stats['documents_indexed']:,}件")
        print(f"   失敗: {orchestrator_stats['documents_failed']}件")
        print(f"   エンベディング生成成功: {orchestrator_stats['embeddings_generated']:,}件")
        print(f"   エンベディング生成失敗: {orchestrator_stats['embeddings_failed']}件")

        if orchestrator_stats['documents_failed'] > 0:
            print(f"\n⚠️  失敗したドキュメント:")
            failed_docs = orchestrator_stats.get('failed_documents', [])
            for doc_id in failed_docs[:10]:  # 最初の10件を表示
                print(f"   - {doc_id}")
            if len(failed_docs) > 10:
                print(f"   ... 他 {len(failed_docs) - 10}件")

        # Elasticsearchの統計
        es_stats = stats.get('elasticsearch_stats', {})
        if es_stats:
            print(f"\n📦 Elasticsearch統計:")
            print(f"   インデックス内ドキュメント数: {es_stats.get('document_count', 0):,}件")
            if es_stats.get('size_in_bytes'):
                size_mb = es_stats['size_in_bytes'] / (1024 * 1024)
                print(f"   インデックスサイズ: {size_mb:.2f} MB")

        # 成功率を計算
        total_processed = orchestrator_stats['documents_processed']
        if total_processed > 0:
            success_rate = (orchestrator_stats['documents_indexed'] / total_processed) * 100
            print(f"\n🎯 成功率: {success_rate:.1f}%")

        # データ整合性チェックを実行
        print(f"\n🔍 データ整合性チェックを実行中...")
        validation_results = await orchestrator.validate_data_integrity()

        if 'error' not in validation_results:
            print(f"   Cosmos DB総数: {validation_results['cosmos_total']:,}件")
            print(f"   Elasticsearch総数: {validation_results['elasticsearch_total']:,}件")
            print(f"   数値一致: {'✅' if validation_results['count_match'] else '❌'}")

            if not validation_results['count_match']:
                sample_val = validation_results['sample_validation']
                print(f"   サンプル検証: {sample_val['checked']}件中{sample_val['valid']}件が一致")
                if validation_results['missing_documents']:
                    print(f"   不足ドキュメント例: {validation_results['missing_documents'][:5]}")
        else:
            print(f"   ❌ 整合性チェックエラー: {validation_results['error']}")

        print("\n" + "="*70)
        print("🎉 新規インデックス同期が完了しました！")
        print("\n次のステップ:")
        print("1. APIエンドポイントでベクトル検索をテスト:")
        print('   curl -X POST http://localhost:3000/vector-search \\')
        print('     -H "Content-Type: application/json" \\')
        print('     -d \'{"query": "画像処理装置", "k": 10}\'')
        print("\n2. Kibanaで可視化:")
        print("   http://localhost:5601")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n⚠️  処理が中断されました")
        print("新しいチェックポイントが保存されているため、再実行時に続きから処理できます")
        sys.exit(1)

    except Exception as e:
        logger.error(f"同期処理中にエラーが発生しました: {e}", exc_info=True)
        print(f"\n❌ エラーが発生しました: {e}")
        print("詳細はログファイルを確認してください")
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "🆕"*35)
    print("新規インデックス作成 - Cosmos DB → Elasticsearch 全件同期ツール")
    print("🆕"*35)

    # 非同期処理を実行
    asyncio.run(main())

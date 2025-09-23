#!/usr/bin/env python3
"""
ベクトル検索システムのテストスクリプト
Cosmos DB → Embedding生成 → Elasticsearch の各コンポーネントをテスト
"""
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# パスを追加
sys.path.append(str(Path(__file__).parent))

from data_sync.cosmos_client import CosmosDBClient
from data_sync.elasticsearch_indexer import ElasticsearchIndexer
from data_sync.embedding_processor import EmbeddingProcessor
from data_sync.sync_orchestrator import SyncOrchestrator

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_cosmos_connection():
    """Cosmos DB接続テスト"""
    print("\n" + "="*60)
    print("📊 Cosmos DB接続テスト")
    print("="*60)

    try:
        client = CosmosDBClient()

        # 総ドキュメント数を取得
        count = client.get_total_document_count()
        print(f"✅ Cosmos DB接続成功")
        print(f"   総ドキュメント数: {count:,}件")

        # 最初の3件を取得してサンプル表示
        print("\n📄 サンプルドキュメント (最初の3件):")
        sample_count = 0
        for batch in client.get_all_documents(batch_size=3):
            for doc in batch:
                sample_count += 1
                print(f"\n   [{sample_count}] Patent ID: {doc.get('patent_id')}")
                print(f"       Title: {doc.get('title', 'N/A')[:50]}...")
                print(f"       Filing Date: {doc.get('filing_date', 'N/A')}")
                print(f"       Summary Length: {len(doc.get('summary', ''))} 文字")

                # 分類情報も表示
                ipc_codes = doc.get('classification_ipc', [])
                if ipc_codes:
                    print(f"       IPC Codes: {', '.join(ipc_codes[:3])}...")
            break

        client.close()
        return True

    except Exception as e:
        print(f"❌ Cosmos DB接続エラー: {e}")
        logger.error(f"Cosmos DB connection error: {e}", exc_info=True)
        return False


async def test_embedding_generation():
    """Azure OpenAI Embedding生成テスト"""
    print("\n" + "="*60)
    print("🤖 Azure OpenAI Embedding生成テスト")
    print("="*60)

    try:
        processor = EmbeddingProcessor()

        # テスト用テキスト（日本語と英語）
        test_texts = [
            "画像処理装置および画像処理方法に関する発明",
            "バリカン式刈刃装置の改良技術",
            "This is a test text for embedding generation."
        ]

        print("📝 テストテキストでEmbedding生成:")
        for i, text in enumerate(test_texts, 1):
            print(f"\n   [{i}] テキスト: {text[:50]}...")

            # Embedding生成
            embedding = await processor.generate_embedding(text)

            if embedding:
                print(f"   ✅ 生成成功")
                print(f"      - ベクトル次元数: {len(embedding)}")
                print(f"      - 最初の3要素: [{embedding[0]:.6f}, {embedding[1]:.6f}, {embedding[2]:.6f}, ...]")
                print(f"      - ベクトルノルム: {sum(x**2 for x in embedding)**0.5:.6f}")
            else:
                print(f"   ❌ 生成失敗")
                return False

        # 統計情報を表示
        stats = processor.get_statistics()
        print(f"\n📊 処理統計:")
        print(f"   モデル: {stats.get('model')}")
        print(f"   処理成功: {stats.get('total_processed')}件")
        print(f"   処理失敗: {stats.get('total_failed')}件")

        return True

    except Exception as e:
        print(f"❌ Embedding生成エラー: {e}")
        logger.error(f"Embedding generation error: {e}", exc_info=True)
        return False


async def test_elasticsearch_connection():
    """Elasticsearch接続テスト"""
    print("\n" + "="*60)
    print("🔍 Elasticsearch接続テスト")
    print("="*60)

    try:
        indexer = ElasticsearchIndexer()

        # インデックスの作成または確認
        indexer.create_index(force_recreate=False)
        print("✅ Elasticsearch接続成功")

        # インデックス統計を取得
        stats = indexer.get_index_stats()
        print(f"\n📊 インデックス情報:")
        print(f"   インデックス名: {stats.get('index_name')}")
        print(f"   既存ドキュメント数: {stats.get('document_count', 0):,}件")

        if stats.get('size_in_bytes'):
            size_mb = stats['size_in_bytes'] / (1024 * 1024)
            print(f"   インデックスサイズ: {size_mb:.2f} MB")

        indexer.close()
        return True

    except Exception as e:
        print(f"❌ Elasticsearch接続エラー: {e}")
        logger.error(f"Elasticsearch connection error: {e}", exc_info=True)
        return False


async def test_small_sync():
    """小規模な同期テスト（5件のドキュメント）"""
    print("\n" + "="*60)
    print("🔄 小規模同期テスト (5件のドキュメント)")
    print("="*60)

    try:
        orchestrator = SyncOrchestrator()

        # コンポーネントを取得
        cosmos_client = orchestrator.cosmos_client
        embedding_processor = orchestrator.embedding_processor
        es_indexer = orchestrator.es_indexer

        # 最初の5件を取得
        documents = []
        print("\n📥 Cosmos DBからドキュメント取得中...")
        for batch in cosmos_client.get_all_documents(batch_size=5):
            documents = batch[:5]
            break

        if not documents:
            print("❌ ドキュメントが取得できませんでした")
            return False

        print(f"✅ {len(documents)}件のドキュメントを取得")

        # 各ドキュメントの情報を表示
        for i, doc in enumerate(documents, 1):
            print(f"   [{i}] Patent ID: {doc.get('patent_id')}, "
                  f"Summary: {len(doc.get('summary', ''))}文字")

        # Embedding生成
        print("\n🤖 Embedding生成中...")
        start_time = datetime.now()
        processed_docs = await embedding_processor.process_documents(documents)
        processing_time = (datetime.now() - start_time).total_seconds()

        # 結果を確認
        success_count = sum(1 for doc in processed_docs if doc.get("embedding_generated"))
        failed_count = len(processed_docs) - success_count

        print(f"✅ Embedding生成完了 (処理時間: {processing_time:.2f}秒)")
        print(f"   成功: {success_count}件")
        print(f"   失敗: {failed_count}件")

        if success_count == 0:
            print("❌ すべてのEmbedding生成が失敗しました")
            return False

        # Elasticsearchに投入
        print("\n📤 Elasticsearchへデータ投入中...")
        result = es_indexer.bulk_index_documents(processed_docs)

        print(f"✅ Elasticsearch投入完了")
        print(f"   成功: {result['indexed']}件")
        print(f"   失敗: {result['failed']}件")

        if result.get('errors'):
            print(f"   エラー詳細: {result['errors'][:3]}")  # 最初の3件のエラーを表示

        # 最終統計
        print("\n📊 同期テスト統計:")
        print(f"   総処理ドキュメント数: {len(documents)}")
        print(f"   Embedding生成成功率: {(success_count/len(documents)*100):.1f}%")
        print(f"   インデックス成功率: {(result['indexed']/len(documents)*100):.1f}%")

        return result['indexed'] > 0

    except Exception as e:
        print(f"❌ 同期テストエラー: {e}")
        logger.error(f"Sync test error: {e}", exc_info=True)
        return False


async def main():
    """メインテスト実行"""
    print("\n" + "🚀"*30)
    print("🚀 ベクトル検索システム統合テスト開始")
    print("🚀"*30)

    # テスト項目の定義
    tests = [
        ("Cosmos DB接続", test_cosmos_connection),
        ("Azure OpenAI Embedding", test_embedding_generation),
        ("Elasticsearch接続", test_elasticsearch_connection),
        ("小規模同期テスト", test_small_sync)
    ]

    results = []
    all_passed = True

    # 各テストを順番に実行
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))

            if not result:
                all_passed = False
                print(f"\n⚠️  {name}で問題が発生しました")

                # 重要なテストが失敗した場合は中断
                if name in ["Cosmos DB接続", "Azure OpenAI Embedding", "Elasticsearch接続"]:
                    print("❌ 基本的な接続テストが失敗したため、後続のテストをスキップします")
                    break

        except Exception as e:
            print(f"❌ {name}で予期しないエラー: {e}")
            results.append((name, False))
            all_passed = False
            break

    # 結果サマリー
    print("\n" + "="*60)
    print("📊 テスト結果サマリー")
    print("="*60)

    for name, result in results:
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"  {name:25} : {status}")

    print("\n" + "="*60)

    if all_passed:
        print("🎉 すべてのテストが成功しました！")
        print("\n次のステップ:")
        print("1. より大規模なデータセットで同期を実行")
        print("2. APIエンドポイントでベクトル検索をテスト")
        print("\n例: python -c \"from data_sync.sync_orchestrator import SyncOrchestrator; import asyncio; asyncio.run(SyncOrchestrator().sync_by_date_range('20100101', '20100131'))\"")
    else:
        print("⚠️  一部のテストが失敗しました")
        print("\nトラブルシューティング:")
        print("1. .envファイルの設定を確認")
        print("2. 各サービスの接続情報を確認")
        print("3. ログファイルで詳細なエラーを確認")

    return all_passed


if __name__ == "__main__":
    # イベントループを実行
    success = asyncio.run(main())

    # 終了コード
    sys.exit(0 if success else 1)

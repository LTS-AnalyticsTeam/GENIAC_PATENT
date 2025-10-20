# Patent Vector Search System

Cosmos DB から Elasticsearch へのベクトル検索システム

## システム概要

このシステムは、Azure Cosmos DB に格納された特許データを取得し、OpenAI API を使用してベクトル埋め込みを生成し、Elasticsearch にインデックス化してベクトル検索を可能にします。

## アーキテクチャ

```
Cosmos DB → データ取得 → Embedding生成 → Elasticsearch → ベクトル検索API
```

## 主要コンポーネント

### 1. データ同期サービス (`data_sync/`)

- **cosmos_client.py**: Cosmos DB からのデータ取得
- **embedding_processor.py**: OpenAI text-embedding-3-large によるベクトル生成
- **elasticsearch_indexer.py**: Elasticsearch へのデータ投入
- **sync_orchestrator.py**: 同期処理の制御
- **utils/**: レート制限とトークンカウント管理

### 2. 検索 API (`app/`)

- **main.py**: FastAPI アプリケーション
- **vector_search.py**: ベクトル検索サービス

## セットアップ

### 1. 環境変数の設定

`.env`ファイルを作成し、以下の環境変数を設定：

```env
# Cosmos DB
COSMOS_ENDPOINT=https://your-cosmos-account.documents.azure.com:443/
COSMOS_KEY=your-cosmos-key
COSMOS_DATABASE=patent_db
COSMOS_CONTAINER=patents

# OpenAI
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=text-embedding-3-large
OPENAI_MAX_TOKENS_PER_MIN=1000000
OPENAI_BATCH_SIZE=20

# Elasticsearch
ELASTICSEARCH_HOST=http://elasticsearch:9200
ELASTICSEARCH_INDEX=patent_vectors
ELASTICSEARCH_BATCH_SIZE=100

# Processing
CONCURRENT_WORKERS=5
CHECKPOINT_ENABLED=true
ENABLE_INCREMENTAL_SYNC=true
```

### 2. 依存関係のインストール

```bash
# データ同期サービス
cd data_sync
pip install -r requirements.txt

# 検索API
cd ../app
pip install -r requirements.txt
```

### 3. Docker コンテナの起動

```bash
docker-compose up -d
```

## データ同期の実行

### 全データの同期

```python
from data_sync.sync_orchestrator import SyncOrchestrator
import asyncio

async def main():
    orchestrator = SyncOrchestrator()
    await orchestrator.sync_all_documents(force_recreate_index=True)

asyncio.run(main())
```

### 日付範囲指定での同期

```python
async def main():
    orchestrator = SyncOrchestrator()
    await orchestrator.sync_by_date_range("20230101", "20231231")

asyncio.run(main())
```

### インクリメンタル同期

```python
async def main():
    orchestrator = SyncOrchestrator()
    await orchestrator.sync_incremental()  # 前回の同期以降の新規/更新データ

asyncio.run(main())
```

## API エンドポイント

### ベクトル検索

```bash
POST /vector-search
{
  "query": "画像処理装置",
  "k": 10,
  "min_score": 0.7
}
```

### ハイブリッド検索（テキスト + ベクトル）

```bash
POST /hybrid-search
{
  "query": "画像処理装置",
  "k": 10,
  "text_weight": 0.3,
  "vector_weight": 0.7
}
```

### 類似文書検索

```bash
POST /similar-documents
{
  "patent_id": "2010000001",
  "k": 10,
  "min_score": 0.7
}
```

### 分類コード検索

```bash
GET /search-by-classification/A61B8%2F00?classification_type=ipc&k=20
```

### ヘルスチェック

```bash
GET /health
```

## 処理の最適化

### レート制限

- OpenAI API: 100 万トークン/分の制限に対応
- バッチ処理: 最大 20 文書/バッチ
- 並列処理: 5 ワーカーまで同時実行

### トークン管理

- tiktoken を使用した正確なトークンカウント
- 8191 トークン/バッチの上限管理
- 自動的なバッチ分割

### エラーハンドリング

- 指数バックオフによるリトライ
- チェックポイント機能による中断からの再開
- 失敗文書の記録と再処理

## モニタリング

### 統計情報の取得

```python
orchestrator = SyncOrchestrator()
stats = orchestrator.get_statistics()
print(stats)
```

出力例：

```json
{
  "orchestrator_stats": {
    "documents_processed": 1000,
    "documents_indexed": 995,
    "documents_failed": 5,
    "embeddings_generated": 995,
    "embeddings_failed": 5
  },
  "elasticsearch_stats": {
    "document_count": 995,
    "index_size_bytes": 104857600
  }
}
```

## トラブルシューティング

### Elasticsearch 接続エラー

```bash
# Elasticsearchの状態確認
curl http://localhost:9200/_cluster/health
```

### OpenAI API エラー

- API キーの確認
- レート制限の確認
- トークン使用量の監視

### Cosmos DB 接続エラー

- エンドポイントとキーの確認
- ファイアウォール設定の確認
- データベース/コンテナ名の確認

## パフォーマンス

- **処理速度**: 約 100-200 文書/分（OpenAI API レート制限による）
- **メモリ使用**: 約 500MB-1GB（バッチサイズによる）
- **ストレージ**: 約 100KB/文書（ベクトル含む）

## 注意事項

1. **コスト**: OpenAI API の使用料金に注意
2. **データサイズ**: 大量データの場合は段階的な同期を推奨
3. **セキュリティ**: API キーは環境変数で管理
4. **バックアップ**: Elasticsearch のスナップショット機能を活用

## ライセンス

内部使用のみ

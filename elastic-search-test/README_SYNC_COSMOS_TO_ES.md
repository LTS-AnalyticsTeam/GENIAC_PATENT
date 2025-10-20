# Cosmos DB → Elasticsearch 同期 手順書

本ドキュメントは、`elastic-search-test/data_sync` に実装されたコンポーネント（Cosmos クライアント / Embedding 生成 / Elasticsearch への投入）と、実行用スクリプト群（`run_*.py`）の使い方をまとめたものです。

- 取得元: Azure Cosmos DB（特許データ）
- 前処理: Azure OpenAI で要約の埋め込みベクトルを生成（3072次元: text-embedding-3-large 既定）
- 保存先: Elasticsearch（`dense_vector` によるベクトル検索対応）
- 付帯機能: 重複防止、バルク投入、チェックポイント、整合性チェック

---

## 構成と主要ファイル

- データ取得: `data_sync/cosmos_client.py`
  - 環境変数 `COSMOS_*` を用いて Cosmos DB から文書をページング取得
  - 日付範囲、増分同期などのクエリをサポート
- 埋め込み生成: `data_sync/embedding_processor.py`
  - Azure OpenAI Embeddings API（`AZURE_OPENAI_*`）を使用
  - バッチ生成・レート制限・再試行を内包
- ES への投入: `data_sync/elasticsearch_indexer.py`
  - インデックス作成、単体/バルクインデックス、存在チェック、KNN 検索など
  - 既定インデックス名: `patent_vectors`
- オーケストレーション: `data_sync/sync_orchestrator.py`
  - 全件/増分/日付範囲/単一ドキュメント同期、チェックポイント保存、統計出力
- 実行スクリプト群（プロジェクト直下）
  - `run_improved_sync.py`: 既存インデックスを活かして重複スキップしつつ全件同期（推奨）
  - `run_fresh_sync.py`: 新規インデックスを作り直して全件同期（破壊的）
  - `run_full_sync.py`: シンプルな全件同期（初期版）
  - `run_integrity_check.py`: Cosmos と ES の整合性チェック
  - `delete_index.py`: 既存インデックスの削除

---

## 前提条件

- Python 3.10+（推奨）
- Docker（Elasticsearch/Kibana を docker-compose で起動する場合）
- Azure 資格情報
  - Cosmos DB: エンドポイントとキー
  - Azure OpenAI: Embeddings 用 API キーとエンドポイント、デプロイ名（例: `text-embedding-3-large`）

---

## セットアップ

1) 依存関係をインストール

```bash
# 仮想環境は任意
python -m venv venv && source venv/bin/activate

# データ同期で必要な依存関係のみ
pip install -r elastic-search-test/data_sync/requirements.txt
```

2) 環境変数を設定

- ファイル: `elastic-search-test/.env.example` を参考に `elastic-search-test/.env` を作成し、値を設定
- 主要項目:
  - Cosmos: `COSMOS_ENDPOINT`, `COSMOS_KEY`, `COSMOS_DATABASE`, `COSMOS_CONTAINER`
  - Azure OpenAI: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`
  - Elasticsearch: `ELASTICSEARCH_HOST`（例: `http://localhost:9200`）, `ELASTICSEARCH_INDEX`
  - チェックポイント: `CHECKPOINT_ENABLED`, `CHECKPOINT_FILE`
  - レート制御: `OPENAI_BATCH_SIZE`, `CONCURRENT_WORKERS` など

3) Elasticsearch を起動（任意: docker-compose）

```bash
cd elastic-search-test
# ES(9200) と Kibana(5601) を起動。FastAPI サービスは任意
docker compose up -d elasticsearch kibana
```

4) 接続確認（任意）

- `check_elasticsearch_existence.py` などを用いて ES の稼働を確認
- `.env` の値が正しいことを確認（Cosmos/Azure OpenAI にアクセスできること）

---

## 使い方（よく使うフロー）

### 1. 改良版 全件同期（推奨）

既存インデックスを維持しつつ、既に存在するドキュメントをスキップして高速に全件同期します。途中失敗してもチェックポイントから再開可能です。

```bash
python elastic-search-test/run_improved_sync.py
```

- 主な挙動
  - ES インデックスが存在しない場合は自動作成
  - バッチ単位で埋め込み生成 → バルク投入
  - 既存ドキュメントは `mget` で存在確認しスキップ
  - 各バッチ完了ごとにチェックポイント保存
  - 統計・整合性チェックをログ出力

### 2. 新規インデックスで全件同期（破壊的）

インデックスを作り直してクリーンに同期したい場合。

```bash
python elastic-search-test/run_fresh_sync.py
```

- 事前に Kibana 等で必要なダッシュボードをバックアップしてください。

### 3. シンプル全件同期（初期版）

```bash
python elastic-search-test/run_full_sync.py
```

### 4. 増分同期

前回同期以降に追加/更新されたドキュメントのみを同期します。

```python
# 例: ライブラリとして使用（スクリプト外から呼ぶ場合はパスを追加）
from pathlib import Path
import sys, asyncio
sys.path.append(str(Path('elastic-search-test')))
from data_sync.sync_orchestrator import SyncOrchestrator

async def run_inc():
    orchestrator = SyncOrchestrator()
    await orchestrator.sync_incremental()  # 既定は直近24時間

asyncio.run(run_inc())
```

### 5. 日付範囲同期 / 単一ドキュメント同期

```python
# 指定日付範囲
await orchestrator.sync_by_date_range("20230101", "20231231")

# 単一ドキュメント
await orchestrator.sync_single_document("2010000001")
```

---

## インデックス仕様（既定）

- インデックス名: `ELASTICSEARCH_INDEX`（既定: `patent_vectors`）
- マッピング主要項目（抜粋）: `data_sync/elasticsearch_indexer.py`
  - `summary_vector`: `dense_vector`, `dims: 3072`, `similarity: cosine`
  - `title`, `summary`: text
  - `classification_*`, `applicants`, `inventors`: keyword
  - 日付: `filing_date`, `publication_date` など（`yyyyMMdd` などを許容）

注意: `dims` は Embeddings モデルに依存します。`text-embedding-3-large` は 3072 次元です。モデルを変更する場合はマッピングも一致させてください。

---

## チェックポイントと再開

- 有効化: `CHECKPOINT_ENABLED=true`
- 保存先: `CHECKPOINT_FILE=./data/checkpoint.json`
- 仕組み: 各バッチ処理完了後に `batch_num`、統計、タイムスタンプ等を保存。失敗後の再実行で続きから再開します。

---

## 整合性チェック

Cosmos と ES の件数・サンプル一致を確認します。

```bash
python elastic-search-test/run_integrity_check.py
```

- 実施内容
  - 基本件数: Cosmos 総数・ES 文書数
  - サンプリング検証: ES 側に存在する ID をランダム取得し、Cosmos と突合
  - 失敗ドキュメントの簡易分析と推奨事項の提示

---

## トラブルシューティング

- Elasticsearch 接続不可
  - `ELASTICSEARCH_HOST` が正しいか、ES が起動済みか
  - Docker 使用時は `docker compose ps` とヘルスチェックを確認
- Embeddings 生成エラー/レート制限
  - `AZURE_OPENAI_API_KEY` と `AZURE_OPENAI_ENDPOINT` の設定を確認
  - `OPENAI_BATCH_SIZE`, `CONCURRENT_WORKERS`, `OPENAI_MAX_TOKENS_PER_MIN` を調整
- 次元不一致エラー
  - ES マッピング `summary_vector.dims` とモデルの次元が一致しているか
- 既存データの上書き防止
  - バルク投入は `_op_type: create` を使用。既存はスキップされます
- 途中で落ちた
  - `run_improved_sync.py` はチェックポイントから再開可能

---

## よくある質問（FAQ）

- Q. インデックス名を変えたい
  - `.env` の `ELASTICSEARCH_INDEX` を変更し、`run_fresh_sync.py` で新規作成するのが安全です。
- Q. ベクトル検索 API でも使いたい
  - `docker-compose.yml` の `api` サービス（FastAPI）を有効化し、`README_VECTOR_SEARCH.md` を参照してください。

---

## 参考

- 環境例: `elastic-search-test/.env.example`
- 実装参照: `data_sync/cosmos_client.py`, `data_sync/embedding_processor.py`, `data_sync/elasticsearch_indexer.py`, `data_sync/sync_orchestrator.py`

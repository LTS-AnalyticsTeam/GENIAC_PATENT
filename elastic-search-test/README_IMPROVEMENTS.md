# Cosmos DB → Elasticsearch 同期システム改良版

## 概要

前回のデータ移行で発生した問題（途中で処理が停止、重複インデックス）を解決するため、同期システムを大幅に改良しました。

## 主な改良点

### 1. 重複防止機能 ✨

- **問題**: 前回は同じドキュメントが複数回インデックスされ、6,563,477 件という異常な数値になった
- **解決策**:
  - Elasticsearch での存在チェック機能を追加
  - バルク存在確認による効率的な重複検出
  - `create`オペレーションによる重複防止

### 2. 強化されたチェックポイント機能 🔄

- **問題**: 処理中断時の復旧が不完全
- **解決策**:
  - バージョン 2.0 のチェックポイント形式
  - 原子的ファイル書き込み（一時ファイル → リネーム）
  - 詳細な検証メタデータ
  - プロセス ID とタイムスタンプの記録

### 3. 堅牢なエラーハンドリング 🛡️

- **問題**: 個別バッチの失敗が全体の処理を停止させる
- **解決策**:
  - バッチレベルでのエラー分離
  - 指数バックオフによるリトライ機能
  - エラー発生時も処理継続
  - 詳細なエラーログとレポート

### 4. データ整合性チェック 🔍

- **新機能**:
  - Cosmos DB と Elasticsearch の件数比較
  - サンプルドキュメントの存在確認
  - 不足ドキュメントの特定
  - 処理完了後の自動検証

### 5. 最適化されたバッチ処理 ⚡

- **改良点**:
  - 小さなチャンクサイズでエラー処理を改善
  - 並列処理の安定性向上
  - レート制限の最適化
  - メモリ使用量の削減

## ファイル構成

```
elastic-search-test/
├── run_improved_sync.py          # 改良版メイン同期スクリプト
├── test_improved_sync.py         # 機能テスト用スクリプト
├── data_sync/
│   ├── sync_orchestrator.py      # 改良されたオーケストレーター
│   ├── elasticsearch_indexer.py  # 重複防止機能付きインデクサー
│   ├── cosmos_client.py          # Cosmos DBクライアント
│   ├── embedding_processor.py    # エンベディング処理
│   └── utils/
│       ├── rate_limiter.py       # レート制限
│       └── token_counter.py      # トークンカウンター
└── data/
    └── checkpoint.json           # チェックポイントファイル
```

## 使用方法

### 1. テスト実行（推奨）

```bash
cd elastic-search-test
python test_improved_sync.py
```

### 2. 改良版全件同期

```bash
cd elastic-search-test
python run_improved_sync.py
```

### 3. 従来版（参考）

```bash
cd elastic-search-test
python run_full_sync.py
```

## 主要な改良内容

### ElasticsearchIndexer の改良

#### 新機能:

- `document_exists()`: 単一ドキュメントの存在確認
- `bulk_check_existence()`: 複数ドキュメントの効率的存在確認
- `bulk_index_documents()`: 重複防止機能付きバルクインデックス

#### 改良点:

- `skip_existing`パラメータで重複スキップ制御
- `create`オペレーションによる重複防止
- 小さなチャンクでのエラー処理改善
- 詳細な統計情報（indexed, failed, skipped）

### SyncOrchestrator の改良

#### 新機能:

- `validate_data_integrity()`: データ整合性チェック
- `_process_batch_with_retry()`: リトライ機能付きバッチ処理
- チェックポイントからの自動復旧

#### 改良点:

- `resume_from_checkpoint`パラメータ
- バッチレベルでのエラー分離
- 強化されたチェックポイント保存
- 詳細な進捗ログ

## 期待される効果

### 1. データ品質の向上

- ✅ 重複ドキュメントの排除
- ✅ 正確な件数（39,680 件）
- ✅ データ整合性の保証

### 2. 処理の安定性

- ✅ 中断からの自動復旧
- ✅ 個別エラーが全体に影響しない
- ✅ 詳細なエラーレポート

### 3. 運用性の向上

- ✅ 進捗の可視化
- ✅ 処理時間の短縮（重複スキップ）
- ✅ 自動検証機能

## トラブルシューティング

### 処理が中断された場合

1. チェックポイントファイルを確認: `data/checkpoint.json`
2. 改良版スクリプトを再実行: `python run_improved_sync.py`
3. 自動的に続きから処理が開始されます

### データ不整合が検出された場合

1. 整合性チェック結果を確認
2. 不足ドキュメントのリストを確認
3. 必要に応じて個別同期を実行

### エラーが多発する場合

1. ログファイルを確認
2. Azure OpenAI API の制限を確認
3. ネットワーク接続を確認
4. Elasticsearch の状態を確認

## 設定可能な環境変数

```bash
# チェックポイント機能
CHECKPOINT_ENABLED=true
CHECKPOINT_FILE=./data/checkpoint.json

# Elasticsearch設定
ELASTICSEARCH_HOST=http://localhost:9200
ELASTICSEARCH_INDEX=patent_vectors
ELASTICSEARCH_BATCH_SIZE=100

# Azure OpenAI設定
AZURE_OPENAI_ENDPOINT=https://patent-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT=text-embedding-3-large

# レート制限設定
OPENAI_MAX_TOKENS_PER_MIN=1000000
OPENAI_BATCH_SIZE=20
CONCURRENT_WORKERS=5
```

## パフォーマンス指標

### 改良前（問題のあった処理）

- 処理済み: 39,680 件
- インデックス済み: 6,563,477 件（異常値）
- 重複率: 約 16,500%
- 完了状態: 不完全（end_time = null）

### 改良後（期待値）

- 処理済み: 39,680 件
- インデックス済み: 39,680 件（正常値）
- 重複率: 0%
- 完了状態: 完全
- 処理時間: 20-30 分（レート制限による）

## 今後の拡張可能性

1. **増分同期の改良**: より効率的な差分検出
2. **並列処理の最適化**: ワーカー数の動的調整
3. **監視機能**: Prometheus/Grafana との連携
4. **自動復旧**: 失敗したドキュメントの自動再処理
5. **スケジューリング**: 定期実行機能

## まとめ

この改良版により、前回発生した以下の問題が解決されます：

1. ✅ **重複インデックス問題**: 完全に防止
2. ✅ **処理中断問題**: 自動復旧機能で解決
3. ✅ **データ整合性問題**: 自動検証で早期発見
4. ✅ **エラーハンドリング問題**: 堅牢な処理継続
5. ✅ **進捗管理問題**: 詳細な進捗表示

これにより、39,680 件すべてのデータが確実に Elasticsearch に移行され、途中で止まることなく完了できます。

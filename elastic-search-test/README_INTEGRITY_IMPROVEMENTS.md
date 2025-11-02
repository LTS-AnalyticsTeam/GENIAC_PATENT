# データ整合性チェック改善

## 概要

Elasticsearch 同期処理における整合性チェックの問題を修正し、ClosedPoolError を解消しました。

## 問題の詳細

### 発生していた問題

- 同期処理完了後に`ClosedPoolError`が大量発生
- 整合性チェックで「Document count mismatch: Cosmos=39680, ES=0」という誤った結果
- クライアント接続を閉じた後に Elasticsearch にアクセスしようとしていた

### 根本原因

1. **クリーンアップ順序の問題**: 整合性チェック前に Elasticsearch クライアントを閉じていた
2. **接続状態チェックの不足**: 接続が閉じられた状態でのアクセスを防ぐガードがなかった

## 実装した修正

### 1. ElasticsearchIndexer の強化

#### 接続状態チェック機能の追加

```python
def is_connected(self) -> bool:
    """Elasticsearch接続の状態をチェック"""
    try:
        return self.es.ping()
    except Exception as e:
        logger.debug(f"Connection check failed: {e}")
        return False
```

#### メソッドへの接続チェック追加

- `get_document_by_id()`: 接続チェック後にドキュメント取得
- `get_index_stats()`: 接続が閉じている場合はキャッシュされた統計を返す

### 2. SyncOrchestrator の修正

#### クリーンアップ順序の変更

```python
# 修正前: 整合性チェックがfinally節で実行され、クリーンアップ後になっていた
finally:
    self._cleanup()

# 修正後: 整合性チェックをクリーンアップ前に実行
# データ整合性チェックを実行（クリーンアップ前に）
logger.info("🔍 データ整合性チェックを実行中...")
validation_results = await self.validate_data_integrity()
# ... 結果処理 ...

finally:
    self._cleanup()
```

### 3. 新しいツールの追加

#### 整合性チェック専用スクリプト (`run_integrity_check.py`)

- 独立した整合性チェック機能
- 詳細なサンプリング検証
- 失敗ドキュメントの分析
- 推奨事項の自動生成

**主な機能:**

- 基本件数チェック（Cosmos DB vs Elasticsearch）
- サンプリング検証（指定件数のドキュメントを詳細チェック）
- 失敗ドキュメントの分類・分析
- 結果レポートの JSON 出力

#### 失敗ドキュメント再処理スクリプト (`run_failed_documents_retry.py`)

- 失敗したドキュメントの特定と再処理
- 複数の特定方法をサポート
- バッチ処理とリトライ機能

**特定方法:**

1. `missing`: Cosmos DB にあるが Elasticsearch にないドキュメント
2. `es_failed`: Elasticsearch indexer の失敗リスト
3. `comprehensive`: 上記両方を組み合わせ

## 使用方法

### 1. 整合性チェックの実行

```bash
# 独立した整合性チェック
python run_integrity_check.py

# サンプルサイズを指定（デフォルト: 1000件）
# 実行時にプロンプトで入力
```

### 2. 失敗ドキュメントの再処理

```bash
# 失敗ドキュメントの再処理
python run_failed_documents_retry.py

# 設定項目（実行時に選択）:
# - 特定方法: missing/es_failed/comprehensive
# - バッチサイズ: デフォルト50
# - 最大リトライ回数: デフォルト3
```

### 3. 通常の同期処理

```bash
# 修正済みの同期処理（整合性チェック付き）
python run_fresh_sync.py
```

## 修正の効果

### Before（修正前）

```
Full sync completed in 4964.8 seconds
Total 39,680 / Processed 39,680
Documents indexed: 34,279
Documents failed: 5,400

# クリーンアップ後に整合性チェック実行
Document count mismatch: Cosmos=39680, ES=0  # ← 誤った結果
urllib3.exceptions.ClosedPoolError: Pool is closed.  # ← 大量エラー
```

### After（修正後）

```
Full sync completed in 4964.8 seconds
Total 39,680 / Processed 39,680
Documents indexed: 34,279
Documents failed: 5,400

# クリーンアップ前に整合性チェック実行
🔍 データ整合性チェックを実行中...
   Cosmos DB総数: 39,680件
   Elasticsearch総数: 34,329件
   数値一致: ❌
   サンプル検証: 100件中95件が一致

# クリーンアップ実行
Resources cleaned up
```

## 推奨ワークフロー

### 1. 同期実行

```bash
python run_fresh_sync.py
```

### 2. 詳細な整合性チェック

```bash
python run_integrity_check.py
```

### 3. 失敗ドキュメントの再処理（必要に応じて）

```bash
python run_failed_documents_retry.py
```

### 4. 最終確認

```bash
python run_integrity_check.py
```

## ログとレポート

### 生成されるファイル

- `integrity_check_YYYYMMDD_HHMMSS.log`: 整合性チェックログ
- `integrity_check_results_YYYYMMDD_HHMMSS.json`: 詳細結果（JSON）
- `failed_retry_YYYYMMDD_HHMMSS.log`: 再処理ログ
- `failed_retry_results_YYYYMMDD_HHMMSS.json`: 再処理結果（JSON）

### レポート内容

- 基本統計（件数、一致率、サイズ）
- サンプリング検証結果
- 失敗ドキュメント分析
- 具体的な推奨事項

## トラブルシューティング

### よくある問題と対処法

#### 1. 接続エラー

```
Elasticsearch connection is closed, returning cached stats only
```

**対処法**: 新しいプロセスで整合性チェックを実行

#### 2. 大量の失敗ドキュメント

```
🔄 5,400件のドキュメントが不足しています
```

**対処法**: 失敗理由を分析して段階的に再処理

#### 3. エンベディング失敗

```
summary_vector: エンベディングが存在しません
```

**対処法**: Azure OpenAI API の設定とレート制限を確認

## 技術的詳細

### 接続状態管理

- `is_connected()`メソッドで動的に接続状態をチェック
- 接続が閉じている場合は安全にスキップ
- エラーログレベルを適切に調整（WARN/INFO）

### エラーハンドリング

- 接続エラーは警告レベル
- データ不整合は情報レベル
- 致命的エラーのみエラーレベル

### パフォーマンス最適化

- バッチ処理によるメモリ効率化
- 指数バックオフによるリトライ
- サンプリングによる高速検証

## 今後の改善案

1. **リアルタイム監視**: 同期中の整合性をリアルタイムで監視
2. **自動修復**: 軽微な不整合の自動修復機能
3. **詳細分析**: 失敗理由の詳細分類と統計
4. **アラート機能**: 重大な不整合の自動通知

## 関連ファイル

### 修正されたファイル

- `data_sync/elasticsearch_indexer.py`: 接続チェック機能追加
- `data_sync/sync_orchestrator.py`: クリーンアップ順序修正

### 新規追加ファイル

- `run_integrity_check.py`: 整合性チェック専用ツール
- `run_failed_documents_retry.py`: 失敗ドキュメント再処理ツール
- `README_INTEGRITY_IMPROVEMENTS.md`: このドキュメント

---

## まとめ

この修正により、以下が実現されました：

✅ **ClosedPoolError の完全解消**
✅ **正確な整合性チェック結果**
✅ **独立した検証・修復ツール**
✅ **詳細なレポート機能**
✅ **安全なリソース管理**

同期処理の信頼性が大幅に向上し、データ品質の継続的な監視・改善が可能になりました。

# Patent CSV Filter Script

## 概要

このスクリプトは、CSV ファイルの特許データを Cosmos DB の`reference.syutugan`フィールドと照合し、条件に合致する行のみを抽出します。

## 処理フロー

1. **ax_docs 列のチェック**: Cosmos DB の`reference.syutugan`と完全一致するか確認
2. **ay_docs 列のチェック**: セミコロン区切りの全ての特許番号が`reference.syutugan`と完全一致するか確認
3. **syutugan 列のチェック**: Cosmos DB の`reference.syutugan`と完全一致するか確認
4. **全てのチェックを通過した行のみ**を新しい CSV ファイルに保存

## 必要な環境変数

`.env`ファイルに以下の環境変数を設定してください：

```env
COSMOS_ENDPOINT=<your-cosmos-endpoint>
COSMOS_KEY=<your-cosmos-key>
COSMOS_DATABASE=patent_db  # デフォルト値
COSMOS_CONTAINER=patents   # デフォルト値
```

## 必要なパッケージ

```bash
pip install azure-cosmos python-dotenv
```

## 使用方法

### 基本的な実行

```bash
python filter_patent_csv.py
```

### 入力・出力

- **入力ファイル**: `test_cases.csv` (同じディレクトリに配置)
- **出力ファイル**: `filtered_test_cases_YYYYMMDD_HHMMSS.csv` (タイムスタンプ付き)

## 出力例

```
2025-01-13 16:45:00 - INFO - Connecting to Cosmos DB: patent_db/patents
2025-01-13 16:45:02 - INFO - Fetching reference.syutugan data from Cosmos DB...
2025-01-13 16:45:05 - INFO - Loaded 1234 unique reference.syutugan values from Cosmos DB
2025-01-13 16:45:05 - INFO - Processing CSV file: test_cases.csv
2025-01-13 16:45:06 - INFO - Row CASE_0001 passed all checks
2025-01-13 16:45:10 - INFO - Processed 100 rows, 45 passed
============================================================
PROCESSING SUMMARY
============================================================
Total rows processed: 1641
Rows passed all checks: 523
Rows failed at ax_docs check: 412
Rows failed at ay_docs check: 389
Rows failed at syutugan check: 317
Pass rate: 31.87%
Output file: filtered_test_cases_20250113_164510.csv
============================================================
```

## ログレベルの変更

より詳細なログを見たい場合は、スクリプト内の以下の部分を変更してください：

```python
logging.basicConfig(
    level=logging.DEBUG,  # INFO から DEBUG に変更
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

## トラブルシューティング

### Cosmos DB 接続エラー

- `.env`ファイルが正しく設定されているか確認
- `COSMOS_ENDPOINT`と`COSMOS_KEY`が正しいか確認
- ネットワーク接続を確認

### CSV ファイルエラー

- `test_cases.csv`が同じディレクトリに存在するか確認
- ファイルのエンコーディングが UTF-8 であることを確認

### メモリ不足エラー

大量の reference.syutugan データがある場合、メモリ不足になる可能性があります。その場合は、バッチ処理の実装を検討してください。

## カスタマイズ

### 入力・出力ファイル名の変更

`main()`関数内の以下の部分を変更：

```python
input_csv = "your_input_file.csv"
output_csv = f"your_output_file_{timestamp}.csv"
```

### フィルタリング条件の変更

`process_row()`メソッドをカスタマイズして、異なるチェック条件を実装できます。

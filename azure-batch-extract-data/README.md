# Azure Batch Extract Data

Azure Batch を使用して特許データから JP 番号に一致するファイルを抽出するプロジェクトです。

## 概要

このプロジェクトは以下の機能を提供します：

1. **extract_data_processor.py**: tar.gz ファイルから特定の JP 番号に一致する text.txt ファイルを抽出
2. **patent_aci_maneger.py**: Azure Container Instances を使用した並列処理
3. **patent_batch_maneger_prod.py**: Azure Batch を使用した大規模並列処理

## セットアップ

### 1. 環境変数の設定

`.env.example`を`.env`にコピーして、適切な値を設定してください：

```bash
cp .env.example .env
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 3. Docker イメージのビルド（オプション）

```bash
docker build -t your-registry.azurecr.io/patent-extractor:latest .
```

## 使用方法

### 単体実行

```bash
python extract_data_processor.py
```

### Azure Container Instances 使用

```bash
python patent_aci_maneger.py
```

### Azure Batch 使用

```bash
python patent_batch_maneger_prod.py
```

## 環境変数

| 変数名                | 説明                                     | 必須 |
| --------------------- | ---------------------------------------- | ---- |
| AZURE_CLIENT_ID       | Azure Service Principal の Client ID     | ✓    |
| AZURE_CLIENT_SECRET   | Azure Service Principal の Client Secret | ✓    |
| AZURE_TENANT_ID       | Azure Tenant ID                          | ✓    |
| AZURE_SUBSCRIPTION_ID | Azure Subscription ID                    | ✓    |
| AZURE_RESOURCE_GROUP  | Azure Resource Group 名                  | ✓    |
| STORAGE_ACCOUNT_URL   | Azure Storage Account の URL             | ✓    |
| STORAGE_ACCOUNT       | Azure Storage Account 名                 | ✓    |
| CONTAINER_NAME        | Blob コンテナ名                          | ✓    |
| CONTAINER_SAS         | Blob コンテナの SAS トークン             | ✓    |
| BATCH_ACCOUNT_NAME    | Azure Batch Account 名                   | ✓    |
| BATCH_ACCOUNT_URL     | Azure Batch Account の URL               | ✓    |
| BATCH_ACCOUNT_KEY     | Azure Batch Account のキー               | ✓    |
| ACR_SERVER            | Azure Container Registry の URL          | ✓    |
| ACR_USER              | Azure Container Registry のユーザー名    | ✓    |
| ACR_PASSWORD          | Azure Container Registry のパスワード    | ✓    |
| DOCKER_IMAGE          | Docker イメージ名                        | ✓    |

## アーキテクチャ

1. **データ取得**: Azure Blob Storage から CSV ファイルを読み込み、対象 JP 番号を取得
2. **ファイル処理**: tar.gz ファイルをダウンロード・展開し、text.txt ファイルを検索
3. **マッチング**: JP 番号と一致するファイルを特定
4. **アップロード**: マッチしたファイルを新しいコンテナにアップロード

## 注意事項

- `.env`ファイルには機密情報が含まれるため、Git にコミットしないでください
- Azure Batch を使用する場合は、適切なクォータ設定が必要です
- 大量のファイル処理には時間がかかる場合があります

## トラブルシューティング

### 認証エラー

- Azure Service Principal の権限を確認してください
- 環境変数が正しく設定されているか確認してください

### ファイルが見つからない

- Blob Storage のコンテナ名とファイルパスを確認してください
- SAS トークンの有効期限を確認してください

### メモリ不足

- VM サイズを大きくするか、並列度を下げてください

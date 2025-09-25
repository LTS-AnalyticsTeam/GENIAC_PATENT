# run_batch_multiquery_from_csv の使い方

`run_batch_multiquery_from_csv.py` は、CSV の各行（`syutugan`, `ax_docs` など）に対して「マルチクエリ・ベクトル検索」を実行し、結果の要約を出力 CSV にまとめるバッチ実行スクリプトです。

内部で `elastic-search-test/run_multiquery_vector_search.py` のパイプラインを呼び出します:
- Cosmos DB から `syutugan`（出願番号）に対応する特許ドキュメントを取得
- 生成 AI で日本語の短文（既定 10 本）を生成
- 各短文を Azure OpenAI Embeddings でベクトル化
- 各ベクトルで Elasticsearch KNN 検索（Top-K）を実施
- 全クエリの結果をスコア・ヒット回数でマージ＆重複排除

---

## 前提条件

- 上位の構成要件は「elastic-search-test/README_SYNC_COSMOS_TO_ES.md」を参照
  - Cosmos DB: `COSMOS_*`
  - Elasticsearch: `ELASTICSEARCH_HOST`, `ELASTICSEARCH_INDEX`
  - Embeddings: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`
  - 生成（Chat）: 以下のいずれか
    - Azure: `AZURE_OPENAI_CHAT_DEPLOYMENT`（または `AZURE_OPENAI_TEXT_DEPLOYMENT`）
    - OpenAI: `OPENAI_API_KEY`（任意で `OPENAI_CHAT_MODEL`、既定は `gpt-4o-mini`）

- 依存関係
  - `pip install -r elastic-search-test/data_sync/requirements.txt`
  - 生成に OpenAI API を使う場合は `openai` クライアントが必要（上記要件に含まれています）

- 環境変数の読み込み
  - ルートの `.env` → `elastic-search-test/.env` の順で読み込み（後者が優先）

---

## 入力 CSV 形式

必須カラム:
- `case_id`: ケース識別子
- `pattern`: 任意のパターン名
- `syutugan`: 出願番号（例: `JP2011050607A` なども可。非数字は自動で削除し数値部へ正規化）
- `ax_docs`: 照合対象の特許ID（数値部）
- `ay_docs`: 任意（使わないが列があっても可）

サンプル: `syutugan_ax_exist_only.csv`

---

## 出力 CSV 形式（列）

- `case_id`, `pattern`, `syutugan`, `ax_docs`
- `unique_results`: マージ後のユニーク件数
- `contains_ax_docs`: 検索結果の中に `ax_docs` が含まれているか
- `match_index`: マージ済みリスト内の 0 始まりインデックス（なければ `-1`）
- `max_score`: ヒット時の最大スコア
- `times_hit`: 何本の生成クエリでヒットしたか
- `best_rank`: 個別クエリ内での最良順位
- `matched_title`: ヒットした文書のタイトル
- `per_query_hits`: 各生成クエリごとのヒット件数（セミコロン区切り）
- `error`: 行レベルのエラー内容（失敗時）

---

## 実行方法

```bash
# 例: 既定の入力/出力パスを使用
python run_batch_multiquery_from_csv.py \
  --input syutugan_ax_exist_only.csv \
  --output multiquery_summary.csv \
  --k 100 \
  --num-queries 10 \
  --min-score 0.0 \
  --exclude-source
```

オプション:
- `--input`: 入力 CSV パス（既定: `syutugan_ax_exist_only.csv`）
- `--output`: 出力 CSV パス（既定: `multiquery_summary.csv`）
- `--k`: 各クエリの Top-K（既定: 100）
- `--num-queries`: 生成する短文クエリ本数（既定: 10）
- `--min-score`: 類似度スコアの下限（既定: 0.0）
- `--exclude-source`: 元の `syutugan` 文書を結果から除外するフラグ

進行表示:
- 各行ごとに `case_id`, `syutugan`, `ax_docs` を表示
- 失敗時は当該行の `error` 列にメッセージを記録し、他行は継続

---

## 仕組み（詳細）

- `run_batch_multiquery_from_csv.py`
  - 環境を読み込み後、`elastic-search-test/run_multiquery_vector_search.py` の `multiquery_vector_search(...)` を 1 行ずつ実行
  - レート制限を考慮し逐次実行（`asyncio.run` で各行を個別に処理）
  - 生成→埋め込み→KNN 検索→マージの各処理が内部で行われます

- `elastic-search-test/run_multiquery_vector_search.py`
  - Cosmos から `syutugan`（数値部へ正規化済み）で文書取得
  - `PassageGenerator` が 200字前後×`num_queries` の日本語短文を生成
  - `EmbeddingProcessor` が一括でベクトル化
  - `ElasticsearchIndexer.search_similar_documents(...)` で KNN 検索を実施
  - スコア最大値、ヒット回数、最良順位でマージ・ソート

---

## ベストプラクティス / 注意点

- 生成 AI の設定
  - Azure を使う場合は `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_CHAT_DEPLOYMENT` を必須設定
  - OpenAI を使う場合は `OPENAI_API_KEY`（必要なら `OPENAI_CHAT_MODEL`）
- Embeddings 次元
  - ES の `summary_vector.dims` と Embeddings モデル（既定 3072）が一致していること
- レート制限
  - デフォルトは逐次処理。大量行・高い `num_queries` / `k` 設定ではリクエストが増加
  - 途中失敗しても出力 CSV の `error` に残るので再実行/再分析が容易
- ES 側のデータ前提
  - ベクターフィールド `summary_vector` が存在し、クエリ対象の文書が投入済みであること

---

## 例: 少数行でのドライラン

```bash
# 入力を少数行に絞った CSV を用意して検証
python run_batch_multiquery_from_csv.py \
  --input sample_small.csv \
  --output out_small.csv \
  --k 50 --num-queries 5 --exclude-source
```

---

## トラブルシューティング

- 「生成AIが未設定」エラー
  - Azure か OpenAI いずれかの Chat 設定が必須です。環境変数を見直してください
- 「Patent not found in Cosmos DB」
  - 入力の `syutugan` が Cosmos 上に存在しない可能性。非数字除去後の数値部が正しいか確認
- 「Elasticsearch 接続失敗」
  - `ELASTICSEARCH_HOST` と ES の起動状態を確認
- 出力 CSV が空/少ない
  - `min-score` が高すぎる可能性。`--min-score 0.0` で試してみてください

---

## 参考

- 実装参照: `run_batch_multiquery_from_csv.py`, `elastic-search-test/run_multiquery_vector_search.py`
- 同期パイプライン: `elastic-search-test/README_SYNC_COSMOS_TO_ES.md`


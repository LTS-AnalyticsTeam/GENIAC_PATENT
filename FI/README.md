
### FI ディレクトリ概要と fi_fterm_search.py 実行ガイド

## ディレクトリ構成（主要）
- `scripts/`
  - `fi_fterm_search.py`: FI / F-term 向けハイブリッド検索
  - `inddexing/`: インデクシング関連スクリプト置き場
  - `parser/`: FI/F-term等のデータパース用スクリプト
  - `util/`: 補助ユーティリティ
- `output/`
  - `JSONL/`: インデクシング元のJSONL（例: `FI_A.jsonl`, `FI_slim.jsonl`）
  - `fi_summary.txt`, `fi_classifications.txt`: 集計・参照用テキスト
- `requirements.txt`: 依存パッケージ定義
- `Index.html`: FIトップページ（ローカル表示用）
- `.venv/`: 仮想環境（任意）
- `data/`: データ置き場（必要に応じて）

## fi_fterm_search.py とは
- Azure OpenAI で入力テキストの埋め込みを作成（`text-embedding-3-small` / 1536次元）
- Azure AI Search の2つのインデックスにハイブリッド検索
  - FI用: `fi_classification_index`
  - F-term用: `fterm_classification_index`
- 各インデックスから上位3件の`code`/`chunk_id`をTSVで出力

想定スキーマ（両インデックス共通の一例）
- 文字列: `chunk_id`, `code`, `search_text`
- ベクトル: `content_vector`（SingleCollection, 次元=1536）
- 検索時の利用フィールド
  - ベクトル: `content_vector`
  - キーワード: `search_text`

## 事前準備
- Azure AI Search に以下の2インデックスが存在すること
  - `fi_classification_index`
  - `fterm_classification_index`
- それぞれに`content_vector`（1536次元）と`search_text`が定義されていること
- Azure OpenAI の埋め込みモデル: `text-embedding-3-small` を使用（1536次元）

## セットアップ
```bash
cd FI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- スクリプト内にエンドポイントやキーが直書きされています（本番では環境変数やKey Vault推奨）。
  - 置き換えポイント（必要に応じて修正）
    - `SEARCH_ENDPOINT`, `SEARCH_KEY`
    - `FI_INDEX_NAME`, `FTERM_INDEX_NAME`
    - `AOAI_ENDPOINT`, `AOAI_KEY`, `EMBED_MODEL`

## 実行方法
- 引数で渡す:
```bash
python scripts/fi_fterm_search.py "スマートフォンの側面にアイコンを表示するタッチパネル"
```

## 出力（TSV）
- 列: `index`  `code`  `chunk_id`  `score`
- 例:
```
fi_classification_index	A47F1/00	FI_A_000123_0004	12.345678
fterm_classification_index	5L200	FT_C_000987_0001	11.234567
```


## 補足
- ハイブリッド検索で使用するキーワード側のフィールドは`search_text`に設定しています。
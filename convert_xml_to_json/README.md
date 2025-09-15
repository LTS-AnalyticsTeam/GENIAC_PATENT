# 特許 XML→JSON 変換・キーワード抽出パイプライン

## 概要

このプロジェクトは、日本語特許 XML ファイル（J-PlatPat 形式）を解析し、

- メタデータ（特許番号、タイトル、分類情報など）
- 参照情報（ひも付き CSV との照合）
- キーワード・トピック抽出（要約文から技術的キーワードを抽出）
  を含む**検索・分析に最適化された JSON ファイル**へ変換するバッチ処理ツールです。

## 主な処理フロー

1. **XML パース**: ElementTree で特許 XML を解析
2. **メタデータ抽出**: 特許番号、タイトル、出願日、分類（IPC/FI/F-term/テーマコード）などを抽出
3. **参照判定**: himotsuki_csv/CSV1.csv, CSV2.csv の syutugan/himotsuki 列と特許番号を照合し、reference 情報を付与
4. **キーワード抽出**: 要約文から KeyBERT+fastText（cc.ja.300.bin）で技術キーワードを抽出し、ストップワード・不要語を除去
5. **トピック抽出**: IPC 分類や「装置」「方法」などで終わる名詞句からトピックらしい語を抽出
6. **JSON 出力**: 検索・分析に最適な構造で output_files/に保存

## 利用している主な技術・モデル

- **Python 3.8 以上**
- **MeCab**: 日本語形態素解析（名詞・動詞抽出、n-gram 生成）
- **KeyBERT**: 文書から意味的に多様なキーワードを抽出
- **fastText (cc.ja.300.bin)**: 日本語分散表現モデル（KeyBERT のベクトル化に利用）
- **scikit-learn TfidfVectorizer**: TF-IDF スコア計算
- **J-PlatPat XML**: 日本特許庁公開の特許 XML データ
- **himotsuki_csv/CSV1.csv, CSV2.csv**: ひも付き判定用 CSV

## 出力 JSON 形式と各項目の説明

```json
{
  "metadata": {
    "patent_id": "2010000001",         // 公開特許番号（数値部）
    "application_number": "2008152953", // 出願番号
    "title": "バリカン式刈刃装置",         // 発明の名称
    "filing_date": "20080611",          // 出願日
    "publication_date": "20100107",     // 公開日
    "reference": {                       // ひも付き参照情報
      "exist": true,                     // syutugan/himotsukiいずれかに該当すればtrue
      "syutugan": ["JP2010000001A"],    // syutugan列で一致したコード
      "himotsuki": []                    // himotsuki列で一致したコード
    },
    "classification_ipc": [              // IPC分類（主要分類記号のみ、空白なし）
      { "code": "A61B8/00", "raw": "A61B   8/00        20060101AFI20091204BHJP" },
      ...
    ],
    "classification_fi": ["A61B8/00", ...], // FI分類
    "f_term": ["2B382GC15", ...],       // F-term
    "theme_code": ["2B382", ...],       // テーマコード
    "keywords": ["対向領域", "焼き付き不良を解消", ...], // 要約から抽出した技術キーワード
    "topics": ["A61B8/00", "画像処理装置", ...]      // IPCや技術名詞句から抽出したトピック
  },
  "source_file": "input_files/result_1/0/JP2010000001A/text.txt", // 元XMLファイルパス
  "ingest_timestamp": "2025-07-06T09:34:27.679005Z",              // 変換日時
  "parse_version": "v1.0.0",                                      // パーススクリプトのバージョン
  "file_size": 85300,                                              // 元ファイルサイズ（バイト）
  "checksum": "...",                                              // 元ファイルSHA256
  "priority_date": "20080520",                                   // 優先日
  "number_of_claims": "7",                                       // 請求項数
  "claims": [ {"num": "1", "text": "..."}, ... ],            // 請求項リスト
  "summary": "...",                                              // 要約文
  "description": "...",                                          // 明細書全文
  "applicants": ["美津濃株式会社"],                                 // 出願人
  "inventors": ["茶園  清隆", ...],                               // 発明者
  "country": "JP",                                               // 国コード
  "kind_code": "A",                                              // 公報種別
  "language": "ja"                                               // 言語
}
```

### 各項目の意味

- **metadata**: 検索・分析に使う主要なメタ情報
  - **patent_id**: 公開特許番号（数値部のみ）
  - **application_number**: 出願番号
  - **title**: 発明の名称
  - **filing_date/publication_date**: 出願日/公開日
  - **reference**: ひも付き判定情報（syutugan/himotsuki 列との照合結果）
  - **classification_ipc**: IPC 分類（code: 空白なし、raw: 元データ）
  - **classification_fi/f_term/theme_code**: 各種分類コード
  - **keywords**: 要約から抽出した技術キーワード（ストップワード除去済み）
  - **topics**: IPC や「装置」「方法」などで終わる名詞句から抽出したトピック語
- **source_file**: 元 XML ファイルのパス
- **ingest_timestamp**: 変換日時（UTC）
- **parse_version**: スクリプトのバージョン
- **file_size/checksum**: 元ファイルのサイズ・ハッシュ
- **priority_date/number_of_claims/claims**: 優先日・請求項数・請求項リスト
- **summary/description**: 要約文・明細書全文
- **applicants/inventors**: 出願人・発明者
- **country/kind_code/language**: 国コード・公報種別・言語

---

## 事前準備

### 1. Poetry のインストール

Poetry は Python の依存管理・仮想環境ツールです。

#### macOS の場合（Homebrew 推奨）

```sh
brew install poetry
```

#### その他（公式インストーラ）

```sh
# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

インストール後、下記でバージョン確認できます。

```sh
poetry --version
```

Poetry の詳細・トラブルシュートは[Poetry 公式ドキュメント](https://python-poetry.org/docs/)を参照してください。

### 2. Python バージョン

- Python 3.11 または 3.12 を推奨します。
- システムに複数バージョンがある場合は、`pyenv` などで切り替えてください。

### 3. 依存パッケージのインストール

このリポジトリのルート（`convert_xml_to_json` ディレクトリ）で以下を実行してください。

```sh
poetry install
```

- 依存パッケージがすべて自動でインストールされます。
- 仮想環境は `~/.cache/pypoetry/virtualenvs/` 配下に作成されます。

### 4. その他

- MeCab（日本語形態素解析器）はシステムにインストールされている必要があります。
  - macOS: `brew install mecab mecab-ipadic`
  - Ubuntu: `sudo apt install mecab libmecab-dev mecab-ipadic-utf8`
- fastText モデル（`cc.ja.300.bin`）は git には入れていないため、別途`https://fasttext.cc/docs/en/crawl-vectors.html `からインストールし、bin/配下に配置してください。
- 入力ファイル(../data/input_files)・CSV は所定のディレクトリに配置してください。

---

## 実行方法

前提：input_files 配下に特許データ txt ファイルが格納されていること

```sh
# input_files/配下のフォルダ全て実施
poetry run python scripts/convert_xml_to_json.py

# 1ファイルだけ変換（ファイルパス指定）
poetry run python scripts/convert_xml_to_json.py --file input_files/result_1/0/JP2010000001A/text.txt

# 先頭5件だけ変換
poetry run python scripts/convert_xml_to_json.py --count 5
```

- デフォルトでは input_files/ 配下のファイルを全て処理します。
- `--file` でファイルパスまたは特許コードを指定すると、そのファイルだけ処理します。
- `--count` で先頭 N 件だけ処理します。
- 出力は output_files/ 配下に JSON で保存されます。

## その他コマンド

- CosmosDB にファイルをアップロードする

前提：
・.env ファイルに cosmosDB へのアクセス情報が入っていること
・output_files 配下に整形済の特許データ json ファイルが格納されていること

```sh
poetry run python scripts/upload_to_cosmos.py
```

---

## 補足

- キーワード抽出には KeyBERT+fastText（cc.ja.300.bin）を利用しています。
- ストップワードや不要語は日本語特許文書向けにカスタマイズされています。
- topics は IPC 分類や「装置」「方法」などで終わる名詞句を優先的に抽出します。
- 参照判定は himotsuki_csv/CSV1.csv, CSV2.csv の syutugan/himotsuki 列と特許番号のコア部で照合します。

## ディレクトリ構成

```
convert_xml_to_json/
├── bin/                # バイナリやモデルファイル（例: cc.ja.300.bin）
├── scripts/            # Pythonスクリプト群
├── data/               # 入出力データやCSV等
├── requirements/       # 環境構築用ファイル（pyproject.toml, poetry.lock）
├── README.md           # このファイル
```

## 主なファイルの場所

- モデルファイル: `bin/cc.ja.300.bin`
- スクリプト: `scripts/` ディレクトリ内
- データ: `data/` ディレクトリ内
- 環境構築: `requirements/pyproject.toml`, `requirements/poetry.lock`

## 使い方

### 例: スクリプトの実行

```sh
cd convert_xml_to_json
python scripts/convert_xml_to_json.py
```

### 例: データの参照

- 入力 CSV: `data/himotsuki_csv/CSV1.csv` など
- 出力ファイル: `data/output_files/` など

---

## 参考

- **J-PlatPat XML 仕様**

  - https://www.j-platpat.inpit.go.jp/web/all/top/BTmTopPage.html
  - 日本特許庁が公開する特許・実用新案公報 XML の公式仕様

- **キーワード抽出・日本語処理**

  - [KeyBERT: Minimal keyword extraction with BERT](https://github.com/MaartenGr/KeyBERT)（Python パッケージとして利用）
  - [fastText: cc.ja.300.bin 日本語モデル](https://fasttext.cc/docs/en/crawl-vectors.html)（ダウンロード済みバイナリを利用）
  - [MeCab: 日本語形態素解析エンジン](https://taku910.github.io/mecab/)（システムにインストールして利用）
  - [scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

- **Poetry による仮想環境管理**

  - [Poetry 公式ドキュメント](https://python-poetry.org/docs/)
  - 仮想環境は `~/.cache/pypoetry/virtualenvs/` 配下に作成される（プロジェクト直下に venv/は作られない）

- **ストップワードリスト**
  - 特許文書特有の語や一般的な日本語ストップワードを config.py で独自実装

## run_text_to_cosmos_csv: テキスト → キーワード →FI/F-term→Cosmos 検索 →CSV

入力テキストから技術キーワードを抽出し、FI/F-term 候補を推定して Cosmos DB を複数レシピで検索します。どの検索式でヒットしたか（SQL + パラメータ）も結果に含めて出力します。

### 入力と出力

- 入力（コマンド引数）
  - `--text <path>`: 解析する特許テキスト（UTF-8、先頭から最大 4000 文字を使用）
  - `--limit <int>`: 最大件数（既定 200）
  - `--csv <path>`: CSV 出力パス（未指定時は `cosmos_results_<timestamp>.csv`）
  - `--json <path>`: JSON 出力パス（任意）
  - `--topn_keywords <int>`: 抽出キーワード上限（既定 12）
  - `--extra_fi / --extra_fterm`: 追加で固定的に検索へ含めるコード（任意）
  - `--patent_id <id>`: Cosmos DB から該当ドキュメントを取得して入力とする（`--text` より優先）
- 出力
  - CSV 列: `patent_id, score, title, publication_date, search_expressions, search_keywords, fi_candidates, fterm_candidates, doc_fi, doc_fterm`
    - `search_expressions`: ヒットした全レシピの検索式（SQL + params）を “ || ” 区切りで連結
    - `search_keywords`: 検索に使用したキーワード（スペース区切り）
    - `fi_candidates` / `fterm_candidates`: 候補として用いた FI/F-term（スペース区切り）
    - `doc_fi` / `doc_fterm`: 各結果ドキュメント側に登録されている FI/F-term（スペース区切り）
  - JSON（指定時）: 各レコードに `search_expressions`（配列）、`signals`、`fi`、`fterm` などを含む

### 必要な環境変数（.env 推奨）

- Cosmos（必須）
  - `COSMOS_ENDPOINT`, `COSMOS_KEY`, `DATABASE_NAME`, `CONTAINER_NAME`
- Azure OpenAI（FI/F-term 候補のベクトル検索に必須）
  - `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_EMBEDDING_MODEL`
- Azure Cognitive Search（FI/F-term 候補のベクトル検索に必須）
  - `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KEY`, `FI_INDEX_NAME`, `FTERM_INDEX_NAME`
- Gemini API（キーワード抽出に必須・フォールバックなし）
  - `GEMINI_API_KEY`

補足（インポート時の参照回避用）

- 一部補助モジュールがインポート時に次の環境変数を参照する場合があります。必要に応じて設定してください。
  - `OPENAI_API_KEY`（未使用なら任意の非空文字列）
  - `AZURE_OPENAI_SEARCH_KEY=${AZURE_OPENAI_KEY}`
  - `GOOGLE_API_KEY=${GEMINI_API_KEY}`

### 依存（Poetry）

プロジェクトルート（このフォルダ）で以下を導入します。

```sh
poetry add python-dotenv azure-cosmos openai azure-search-documents google-generativeai spacy ginza ja_ginza SudachiDict-core pandas
poetry run python -m spacy download ja_ginza
```

Python は `>=3.11,<3.13` を想定しています（Poetry が自動で適合バージョンを選びます）。

### 実行方法（例）

```sh
cd convert_xml_to_json
poetry run python scripts/run_text_to_cosmos_csv.py \
  --text ../JP2011106326A/text.txt \
  --limit 20 \
  --csv out.csv
```

### 内部ロジック（入力 → 出力）

1. 入力取得（scripts/run_text_to_cosmos_csv.py）

- `--patent_id` 指定時は Cosmos DB から該当ドキュメントを取得し、title/summary/claims を連結して入力テキストを構築。文書側 `metadata.keywords` があればそれを優先的に使用。

2. キーワード抽出（abc_keyword_extractor を厳格利用）

- `abc_keyword_extractor.py` を動的ロードして実行。
- `--patent_id` 指定時は、Cosmos から取得したドキュメントの title/summary/claims を連結し、文書側 `metadata.keywords` の有無に関わらず A/B/C 抽出を実施（常に抽出を行う）。
- フォールバックは行いません。依存やキーが無い場合はエラー終了します。

3. FI/F-term 候補推定（VectorSearchPredictor を厳格利用）

- `expect_patent_codes_by_llm.py` の `VectorSearchPredictor.search_fi_fterm(keywords)` を呼び出し。
- 内部で `FI/scripts/fi_fterm_search.py` を利用し、Azure OpenAI の埋め込み + Azure Search のベクトル/キーワード併用で上位候補（各 3 件）を取得。

4. Cosmos DB 検索（複数レシピの併用と集約）

- 実装: `scripts/cosmos_patent_search.py`
- レシピ概要
  - FI 接頭一致 OR F-term 接頭一致、かつ タイトル/要約/請求項/明細（description）に `@kw` を含む
- 収集・統合
  - 各レシピの結果を `patent_id` キーで統合し、ヒットした全レシピの検索式を `search_expressions` として保持。
- スコアリング
  - FI/F-term（厳密・接頭）、タイトル/要約/請求項ヒット、登録キーワードとの共通語数、発行年による僅かなブーストを合算。
- ランキング
  - スコア降順で並べ、`--limit` 件までを最終結果に採用。

5. 出力

- CSV: `patent_id, score, title, publication_date, search_expressions`
- JSON（指定時）: `search_expressions` は配列、`signals`（一致内訳）、`fi`/`fterm` を含む。

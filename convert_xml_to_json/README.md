# 特許XML→JSON変換・キーワード抽出パイプライン

## 概要
このプロジェクトは、日本語特許XMLファイル（J-PlatPat形式）を解析し、
- メタデータ（特許番号、タイトル、分類情報など）
- 参照情報（ひも付きCSVとの照合）
- キーワード・トピック抽出（要約文から技術的キーワードを抽出）
を含む**検索・分析に最適化されたJSONファイル**へ変換するバッチ処理ツールです。

## 主な処理フロー
1. **XMLパース**: ElementTreeで特許XMLを解析
2. **メタデータ抽出**: 特許番号、タイトル、出願日、分類（IPC/FI/F-term/テーマコード）などを抽出
3. **参照判定**: himotsuki_csv/CSV1.csv, CSV2.csvのsyutugan/himotsuki列と特許番号を照合し、reference情報を付与
4. **キーワード抽出**: 要約文からKeyBERT+fastText（cc.ja.300.bin）で技術キーワードを抽出し、ストップワード・不要語を除去
5. **トピック抽出**: IPC分類や「装置」「方法」などで終わる名詞句からトピックらしい語を抽出
6. **JSON出力**: 検索・分析に最適な構造でoutput_files/に保存

## 利用している主な技術・モデル
- **Python 3.8以上**
- **MeCab**: 日本語形態素解析（名詞・動詞抽出、n-gram生成）
- **KeyBERT**: 文書から意味的に多様なキーワードを抽出
- **fastText (cc.ja.300.bin)**: 日本語分散表現モデル（KeyBERTのベクトル化に利用）
- **scikit-learn TfidfVectorizer**: TF-IDFスコア計算
- **J-PlatPat XML**: 日本特許庁公開の特許XMLデータ
- **himotsuki_csv/CSV1.csv, CSV2.csv**: ひも付き判定用CSV

## 出力JSON形式と各項目の説明

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
  - **reference**: ひも付き判定情報（syutugan/himotsuki列との照合結果）
  - **classification_ipc**: IPC分類（code: 空白なし、raw: 元データ）
  - **classification_fi/f_term/theme_code**: 各種分類コード
  - **keywords**: 要約から抽出した技術キーワード（ストップワード除去済み）
  - **topics**: IPCや「装置」「方法」などで終わる名詞句から抽出したトピック語
- **source_file**: 元XMLファイルのパス
- **ingest_timestamp**: 変換日時（UTC）
- **parse_version**: スクリプトのバージョン
- **file_size/checksum**: 元ファイルのサイズ・ハッシュ
- **priority_date/number_of_claims/claims**: 優先日・請求項数・請求項リスト
- **summary/description**: 要約文・明細書全文
- **applicants/inventors**: 出願人・発明者
- **country/kind_code/language**: 国コード・公報種別・言語

---

## 事前準備

### 1. Poetryのインストール

PoetryはPythonの依存管理・仮想環境ツールです。

#### macOSの場合（Homebrew推奨）

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

Poetryの詳細・トラブルシュートは[Poetry公式ドキュメント](https://python-poetry.org/docs/)を参照してください。

### 2. Pythonバージョン

- Python 3.11 または 3.12 を推奨します。
- システムに複数バージョンがある場合は、`pyenv` などで切り替えてください。

### 3. 依存パッケージのインストール

このリポジトリのルート（`convert_json` ディレクトリ）で以下を実行してください。

```sh
poetry install
```

- 依存パッケージがすべて自動でインストールされます。
- 仮想環境は `~/.cache/pypoetry/virtualenvs/` 配下に作成されます。

### 4. その他

- MeCab（日本語形態素解析器）はシステムにインストールされている必要があります。
  - macOS: `brew install mecab mecab-ipadic`
  - Ubuntu: `sudo apt install mecab libmecab-dev mecab-ipadic-utf8`
- fastTextモデル（`cc.ja.300.bin`）は同梱済みです。
- 入力ファイル・CSVは所定のディレクトリに配置してください。

---

## 実行方法

前提：input_files配下に特許データtxtファイルが格納されていること

```sh
# input_files/配下のフォルダ全て実施
poetry run python convert_xml_to_json.py

# 1ファイルだけ変換（ファイルパス指定）
poetry run python convert_xml_to_json.py --file input_files/result_1/0/JP2010000001A/text.txt

# 先頭5件だけ変換
poetry run python convert_xml_to_json.py --count 5
```

- デフォルトではinput_files/ 配下のファイルを全て処理します。
- `--file` でファイルパスまたは特許コードを指定すると、そのファイルだけ処理します。
- `--count` で先頭N件だけ処理します。
- 出力は output_files/ 配下にJSONで保存されます。

## その他コマンド

- CosmosDBにファイルをアップロードする

前提：output_files配下に整形済の特許データjsonファイルが格納されていること

```sh
poetry run python upload_to_cosmos.py
```



---

## 補足
- キーワード抽出にはKeyBERT+fastText（cc.ja.300.bin）を利用しています。
- ストップワードや不要語は日本語特許文書向けにカスタマイズされています。
- topicsはIPC分類や「装置」「方法」などで終わる名詞句を優先的に抽出します。
- 参照判定はhimotsuki_csv/CSV1.csv, CSV2.csvのsyutugan/himotsuki列と特許番号のコア部で照合します。

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

- 入力CSV: `data/himotsuki_csv/CSV1.csv` など
- 出力ファイル: `data/output_files/` など

---

## 参考

- **J-PlatPat XML仕様**
  - https://www.j-platpat.inpit.go.jp/web/all/top/BTmTopPage.html
  - 日本特許庁が公開する特許・実用新案公報XMLの公式仕様

- **キーワード抽出・日本語処理**
  - [KeyBERT: Minimal keyword extraction with BERT](https://github.com/MaartenGr/KeyBERT)（Pythonパッケージとして利用）
  - [fastText: cc.ja.300.bin 日本語モデル](https://fasttext.cc/docs/en/crawl-vectors.html)（ダウンロード済みバイナリを利用）
  - [MeCab: 日本語形態素解析エンジン](https://taku910.github.io/mecab/)（システムにインストールして利用）
  - [scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

- **Poetryによる仮想環境管理**
  - [Poetry公式ドキュメント](https://python-poetry.org/docs/)
  - 仮想環境は `~/.cache/pypoetry/virtualenvs/` 配下に作成される（プロジェクト直下にvenv/は作られない）

- **ストップワードリスト**
  - 特許文書特有の語や一般的な日本語ストップワードをconfig.pyで独自実装

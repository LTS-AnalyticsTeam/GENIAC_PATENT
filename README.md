# Patent Alpha Analyzer Mock

最小構成の FastAPI + React(TypeScript) モックです。特許αの請求項テキストファイルをアップロードすると、ダミーで生成した A(x) / A(y) 候補と参照スニペット・判断根拠を表示します（投入テキストは現在解析されず、固定レスポンスを返します）。ElasticSearch や DB への接続はありません。

## プロジェクト構成

```
backend/
  app/
    __init__.py      # アプリケーションファクトリ
    main.py          # uvicorn エントリポイント
    api/             # FastAPI ルータ
    config.py        # アプリ設定
    data/            # ダミーデータなどのデータ提供レイヤー
    models.py        # レスポンススキーマ(Pydantic)
    services/        # ビジネスロジック（AnalyzerService 等）
    store.py         # インメモリ Run ストア
    alpha_parser.py  # 将来のテキスト解析ユーティリティ（現状未使用）
frontend/
  src/
    App.tsx
    pages/Analyze.tsx
    components/ResultCard.tsx
    components/Snippet.tsx
    lib/highlight.ts
samples/
  alpha.txt          # デモ用テキスト
```

## セットアップ

### 1. バックエンド (FastAPI)

```bash
cd backend
uvicorn app.main:app --reload
```

既定で `http://127.0.0.1:8000` が起動します。CORS はすべて許可済みです。

### 2. フロントエンド (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

既定で `http://127.0.0.1:5173` が起動します。`.env` が無い場合はバックエンドを `http://127.0.0.1:8000` として呼び出します。

## 主な機能

- 請求項テキストを含む `.txt` ファイルをアップロードするとダミー解析を実行（中身は無視）
- A(x) は常に 1 件（特許番号と要約を表示）、A(y) は web検索と通常資料を含む 3 件のダミーデータを返却（web検索候補は特許番号・要約の代わりに参照URLを表示）
- 各カード内で「参照箇所表示」と「判断根拠」をまとめて表示し、<mark> で一致語句をハイライト
- UI 操作はファイル選択＋実行ボタンのみ（ソートや件数指定は無し）
- エラー時は簡易トースト表示

## API

すべて `http://127.0.0.1:8000` 配下。

### `POST /analyze`

- Content-Type: `multipart/form-data`
- フィールド:
  - `file`: 請求項を含むテキストファイル（`.txt` 必須）

レスポンス（抜粋）:

```jsonc
{
  "run_id": "uuid",
  "alpha": {
    "title": "特許αデモ",
    "pub_number": "DUMMY-0001",
    "claim1": "請求項1: …",
    "claims_rest": ["請求項2: …"]
  },
  "Ax": {
    "doc_id": "ax-dummy",
    "title": "エッジ通信制御装置",
    "score": 0.90,
    "snippets": [
      {
        "section": "内部資料",
        "claim_no": 1,
        "text": "暗号化ハンドシェイクと遅延閾値制御を組み合わせたゲートウェイ制御方式の要約。",
        "offset": 0,
        "len": 6,
        "match_type": "phrase",
        "score": 0.82
      }
    ],
    "explanation": {
      "summary": "ゲートウェイ装置の暗号化手順と遅延制御手法が請求項1の構成と一致する想定例。",
      "why_match": ["暗号化ハンドシェイクの流れが請求項1と一致", "閾値ベースの制御切替が対応"],
      "examiner_hints": ["単独引用で課題の大部分を充足する可能性", "ログ管理の差分を補う引用例を検討"]
    }
  },
  "Ay": [
    {
      "doc_id": "ay-dummy-1",
      "title": "ログ解析パイプライン（web検索）",
      "score": 0.84,
      "source_url": "https://example.com/log-analytics",
      "snippets": [{ "...": "..." }],
      "explanation": { "...": "..." }
    },
    {
      "doc_id": "ay-dummy-2",
      "title": "ダッシュボード連携基盤",
      "score": 0.80,
      "snippets": [{ "...": "..." }],
      "explanation": { "...": "..." }
    },
    {
      "doc_id": "ay-dummy-3",
      "title": "フィードバック制御ログ集約装置",
      "score": 0.78,
      "snippets": [{ "...": "..." }],
      "explanation": { "...": "..." }
    }
  ],
  "limits": { "max_total": 3, "Ay_min": 1 }
}
```

### `GET /runs/{run_id}`

前回 `/analyze` 実行時と同じ JSON を返却します（インメモリ保持）。

### `POST /search`

モックのため未実装。`{"detail": "Not implemented in mock."}` を返します。

### `GET /health`

疎通確認用。`{"status": "ok"}`。

## サンプルテキスト

`samples/alpha.txt` にデモ用テキストを用意しています。フロントのフォームにアップロードすると、A(x) の特許概要と 3 件の A(y) 候補（うち 1 件は web検索）を確認できます。

## 備考

- 現在のモックではテキスト入力内容を解析せず、固定のダミーデータを返します。
- スニペット・スコアはテンプレートで生成しており、実システムでは差し替えが必要です。
- Run ID と結果は FastAPI プロセス内のメモリに保持されるため、再起動でリセットされます。

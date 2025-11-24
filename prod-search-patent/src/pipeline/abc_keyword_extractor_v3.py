import os
import csv
import re
from datetime import datetime
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from openai import AzureOpenAI

"""
ABC キーワード抽出 v3
- Aカテゴリは省略
- B・Cカテゴリに優先順位（MUST/SHOULD）を付与
- キーワードは最小単位に分解
- 類似語も含める
- Azure OpenAI gpt-5-mini を使用（フォールバック: gpt-4o）
"""

load_dotenv()

# Azure OpenAI クライアント (gpt-5-mini用)
client = AzureOpenAI(
    api_version=os.environ.get("API_VERSION"),
    azure_endpoint=os.environ.get("ENDPOINT"),
    api_key=os.environ.get("API_KEY"),
)

# Azure OpenAI クライアント (gpt-4o用 - フォールバック)
client_4o = AzureOpenAI(
    api_version=os.environ.get("API_VERSION_4o"),
    azure_endpoint=os.environ.get("ENDPOINT_4o"),
    api_key=os.environ.get("API_KEY_4o"),
)

def extract_keywords_with_priority(title, abstract, claims_texts, description=None):
    """LLMを使用してB・Cカテゴリキーワードを優先順位付きで抽出

    Args:
        title: 特許タイトル
        abstract: 要約
        claims_texts: 請求項テキストのリスト
        description: 明細書データ（技術分野、背景技術、課題、解決手段を含むdict）
    """

    # descriptionから追加セクションを抽出
    technical_field = ""
    background_art = ""
    problem_to_solve = ""
    means_for_solving = ""

    if description:
        # technical-field を抽出
        tf_items = description.get('technical-field', [])
        if tf_items:
            technical_field = " ".join([item.get('text', '') for item in tf_items if isinstance(item, dict)])

        # background-art を抽出
        ba_items = description.get('background-art', [])
        if ba_items:
            background_art = " ".join([item.get('text', '') for item in ba_items if isinstance(item, dict)])

        # summary-of-invention から課題と解決手段を抽出
        summary = description.get('summary-of-invention', {})
        if isinstance(summary, dict):
            # tech-problem (課題)
            tp_items = summary.get('tech-problem', [])
            if tp_items:
                problem_to_solve = " ".join([item.get('text', '') for item in tp_items if isinstance(item, dict)])

            # tech-solution (解決手段)
            ts_items = summary.get('tech-solution', [])
            if ts_items:
                means_for_solving = " ".join([item.get('text', '') for item in ts_items if isinstance(item, dict)])

    prompt = f"""
# タスク
以下の特許文書から、発明の核心となるキーワードを抽出してください。
AカテゴリーはスキップしてBとCのみ抽出します。

## Bカテゴリ（構成/作用）
発明の中核となる構成要素、材料、または主要な作用・プロセスを示すキーワード。

### 抽出ルール
1. **優先順位**:
   - MUST: 発明に絶対不可欠な要素（この要素がないと発明が成立しない）
   - SHOULD: 重要だが必須ではない要素（発明の特徴を示すが代替可能）

2. **分解**: 複合語は最小単位の名詞に分解すること
   - 例: 「変化表示」→「変化」+「表示」
   - 例: 「乱数取得手段」→「乱数」+「取得」+「手段」
   
3. **類似語**: 各キーワードに対して、特許検索で使える類似語を3-5個程度含める
   - 例: 「表示」→「表示」「ディスプレイ」「画面」「表出」
   - 例: 「制御」→「制御」「コントロール」「管理」「調整」

## Cカテゴリ（特徴/差別化）
発明の新規性や進歩性を示す、従来技術との差別化ポイント。

### 抽出ルール
1. **優先順位**:
   - MUST: 発明の最も重要な差別化要素（課題解決の核心）
   - SHOULD: 補助的な特徴や効果

2. **分解**: 複合語は最小単位に分解
   - 例: 「先読み演出」→「先読み」+「演出」
   
3. **類似語**: 類義語・関連語を含める
   - 例: 「先読み」→「先読み」「予測」「事前判定」「予告」

## 抽出ヒント
- 「〜を含む」「〜からなる」の後のキーワードはBカテゴリ
- 「〜を特徴とする」「〜に優れる」「〜ができる」の前のキーワードはCカテゴリ
- 化学品系なら化合物名・化学物質名はBカテゴリのMUST
- 【課題】【解決手段】に記載の内容は特に重要

# 特許インプット
タイトル: {title}
要約: {abstract}
""" + "\n".join([f"請求項{i+1}: {ct[:500]}" for i, ct in enumerate(claims_texts[:5])]) + f"""
技術分野: {technical_field[:500] if technical_field else '(なし)'}
背景技術: {background_art[:800] if background_art else '(なし)'}
発明が解決しようとする課題: {problem_to_solve[:500] if problem_to_solve else '(なし)'}
課題を解決するための手段: {means_for_solving[:800] if means_for_solving else '(なし)'}
"""

# 出力フォーマット（JSON形式）
```json
{{
  "B_MUST": [
    {{
      "base_keyword": "基本キーワード（分解後）",
      "synonyms": ["類似語1", "類似語2", "類似語3"]
    }}
  ],
  "B_SHOULD": [
    {{
      "base_keyword": "基本キーワード",
      "synonyms": ["類似語1", "類似語2"]
    }}
  ],
  "C_MUST": [
    {{
      "base_keyword": "基本キーワード",
      "synonyms": ["類似語1", "類似語2", "類似語3"]
    }}
  ],
  "C_SHOULD": [
    {{
      "base_keyword": "基本キーワード",
      "synonyms": ["類似語1", "類似語2"]
    }}
  ]
}}
```

重要: JSON形式で出力してください。コードブロックの中にJSONを記述してください。
"""
    
    # Azure OpenAI: gpt-5-mini を優先、失敗時に gpt-4o へフォールバック
    models = [
        ("gpt-5-mini", client),
        ("gpt-4o", client_4o)
    ]

    for model_name, api_client in models:
        try:
            # Azure OpenAI Chat Completions API
            # Note: gpt-5-mini does not support temperature parameter (only default 1)
            # Note: gpt-5-mini requires max_completion_tokens, gpt-4o uses max_tokens
            if model_name == "gpt-5-mini":
                response = api_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts keywords from patent documents. Please output valid JSON."},
                        {"role": "user", "content": prompt}
                    ]
                )
            else:
                response = api_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts keywords from patent documents. Please output valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4096
                )
            text = response.choices[0].message.content

            # JSONを抽出
            import json
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    if any(result.get(key) for key in ["B_MUST", "B_SHOULD", "C_MUST", "C_SHOULD"]):
                        return result
                except json.JSONDecodeError:
                    pass

            # 空の結果の場合、次のモデルにフォールバック
            if model_name == "gpt-5-mini":
                print(f"[INFO] gpt-5-mini returned empty result, falling back to gpt-4o")
                continue

        except Exception as e:
            # エラー時は次のモデルを試行
            if model_name == "gpt-5-mini":
                print(f"[WARN] gpt-5-mini error: {e}, falling back to gpt-4o")
                continue
            else:
                print(f"[ERROR] gpt-4o error: {e}")
                pass

    # 全て失敗した場合は空の構造を返す
    return {
        "B_MUST": [],
        "B_SHOULD": [],
        "C_MUST": [],
        "C_SHOULD": []
    }

def format_keywords_for_csv(keywords_dict):
    """キーワード辞書をCSV出力用にフォーマット"""
    def format_category(items):
        parts = []
        for item in items:
            base = item.get("base_keyword", "")
            synonyms = item.get("synonyms", [])
            if synonyms:
                parts.append(f"{base}({', '.join(synonyms)})")
            else:
                parts.append(base)
        return " | ".join(parts)
    
    return {
        "B_MUST": format_category(keywords_dict.get("B_MUST", [])),
        "B_SHOULD": format_category(keywords_dict.get("B_SHOULD", [])),
        "C_MUST": format_category(keywords_dict.get("C_MUST", [])),
        "C_SHOULD": format_category(keywords_dict.get("C_SHOULD", []))
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ABC v3: B・Cカテゴリキーワード抽出（優先順位・分解・類似語対応）")
    parser.add_argument('--limit', type=int, default=1, help='取得件数')
    parser.add_argument('--doc-numbers', type=str, help='特定の特許番号（カンマ区切り）例: 2017080220,2018143828')
    args = parser.parse_args()
    
    COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT")
    COSMOS_KEY = os.environ.get("COSMOS_KEY")
    DATABASE_NAME = os.environ.get("DATABASE_NAME")
    CONTAINER_NAME = os.environ.get("CONTAINER_NAME")
    
    cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    container = cosmos_client.get_database_client(DATABASE_NAME).get_container_client(CONTAINER_NAME)
    
    # 特許番号リストの取得
    doc_numbers = []
    if args.doc_numbers:
        doc_numbers = [num.strip() for num in args.doc_numbers.split(',')]
    else:
        # syutugan_ax_test_data.csvから取得
        csv_path = os.path.join(os.path.dirname(__file__), "../data/syutugan_ax_test_data.csv")
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= args.limit:
                    break
                syutugan = row.get("syutugan", "").strip()
                if syutugan:
                    m = re.match(r"JP(\d+)[A-Z]?", syutugan)
                    if m:
                        doc_numbers.append(m.group(1))
    
    print(f"処理対象の特許番号: {doc_numbers}")
    
    # 特許データを取得して処理
    results = []
    for doc_number in doc_numbers:
        query = "SELECT * FROM c WHERE c.bibliographic.publication.doc_number = @doc_num"
        params = [{"name": "@doc_num", "value": doc_number}]
        items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
        
        if not items:
            print(f"[警告] 特許 {doc_number} が見つかりませんでした")
            continue
        
        doc = items[0]
        title = doc['bibliographic']['title']
        abstract = doc.get('abstract', '')
        claims = doc.get('claims', [])
        claims_texts = [claim['text'] for claim in claims if isinstance(claim, dict) and 'text' in claim]
        
        print(f"\n処理中: {doc_number} - {title}")
        print(f"  要約文字数: {len(abstract)}, 請求項数: {len(claims_texts)}")
        
        # キーワード抽出
        keywords_dict = extract_keywords_with_priority(title, abstract, claims_texts)
        formatted = format_keywords_for_csv(keywords_dict)
        
        print(f"  B_MUST: {len(keywords_dict.get('B_MUST', []))}件")
        print(f"  B_SHOULD: {len(keywords_dict.get('B_SHOULD', []))}件")
        print(f"  C_MUST: {len(keywords_dict.get('C_MUST', []))}件")
        print(f"  C_SHOULD: {len(keywords_dict.get('C_SHOULD', []))}件")
        
        results.append({
            "patent_id": doc_number,
            "title": title,
            "B_MUST": formatted["B_MUST"],
            "B_SHOULD": formatted["B_SHOULD"],
            "C_MUST": formatted["C_MUST"],
            "C_SHOULD": formatted["C_SHOULD"]
        })
    
    # CSV出力
    outdir = os.path.join(os.path.dirname(__file__), "../data/output_keywords")
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(outdir, f"abc_v3_keywords_{timestamp}.csv")
    
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patent_id", "title", "B_MUST", "B_SHOULD", "C_MUST", "C_SHOULD"
        ])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"\n✓ 結果を保存しました: {outfile}")
    print(f"✓ 処理件数: {len(results)}/{len(doc_numbers)}")

if __name__ == "__main__":
    main()

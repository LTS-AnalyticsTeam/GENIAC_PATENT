"""
A/B/Cカテゴリキーワード抽出スクリプト（Gemini Flash 1.5使用版）

【機能】
- 特許データからA（技術分野）/B（構成・作用）/C（特徴・差別化）カテゴリのキーワードを抽出
- Gemini Flash 1.5によるキーワード選定とLLM分解
- 形態素解析による分解列を追加出力
- 安全ブロック回避機能付き（Gemini 2.5が利用可能になったら'gemini-1.5-flash'を'gemini-2.5-flash'に変更してください）

【必要な環境変数】
- GEMINI_API_KEY: Gemini APIキー
- COSMOS_ENDPOINT: CosmosDBエンドポイント
- COSMOS_KEY: CosmosDBキー
- DATABASE_NAME: データベース名
- CONTAINER_NAME: コンテナ名

【実行方法】
cd convert_xml_to_json
poetry run python scripts/abc_keyword_extractor.py --limit 5

【必要パッケージ】
poetry add google-generativeai spacy azure-cosmos python-dotenv
python -m spacy download ja_ginza
"""

import os
import csv
import re
import spacy

# 形態素解析器
nlp = spacy.load("ja_ginza")

def extract_nouns(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ == "NOUN"]

def extract_a_b_c_candidates_from_record(record):
    a_pattern_candidates = set()
    for text in [record.get("summary", "")]:
        for pat in [r"(.+?)を提供する", r"(.+?)に関する", r"(.+?)に用いる", r"(.+?)用"]:
            m = re.search(pat, text)
            if m:
                a_pattern_candidates.add(m.group(1).strip())
    for claim in record.get("claims", []):
        claim_text = claim.get("text", "") if isinstance(claim, dict) else str(claim)
        for pat in [r"(.+?)を提供する", r"(.+?)に関する", r"(.+?)に用いる", r"(.+?)用"]:
            m = re.search(pat, claim_text)
            if m:
                a_pattern_candidates.add(m.group(1).strip())
    title = record.get("title", "")
    title_nouns = extract_nouns(title)
    b_candidates = set()
    for text in [record.get("summary", "")]:
        for m in re.finditer(r"([一-龥ぁ-んァ-ンA-Za-z0-9]{2,})(部|手段|工程|構成|材料|装置|方法|組成|系|体|剤|層|素子|回路|信号|情報|記憶|処理|記録|検出|制御|生成|変換|伝送|出力|入力|演算|演出|選択|操作|用途|応用)", text):
            b_candidates.add(m.group(0))
    for claim in record.get("claims", []):
        claim_text = claim.get("text", "") if isinstance(claim, dict) else str(claim)
        for m in re.finditer(r"([一-龥ぁ-んァ-ンA-Za-z0-9]{2,})(部|手段|工程|構成|材料|装置|方法|組成|系|体|剤|層|素子|回路|信号|情報|記憶|処理|記録|検出|制御|生成|変換|伝送|出力|入力|演算|演出|選択|操作|用途|応用)", claim_text):
            b_candidates.add(m.group(0))
    c_candidates = set()
    for text in [record.get("summary", "")]:
        for pat in [r"(.+?)という特徴", r"(.+?)という効果", r"(.+?)に優れる", r"(.+?)が高い", r"(.+?)が低い", r"(.+?)の差別化"]:
            m = re.search(pat, text)
            if m:
                c_candidates.add(m.group(1).strip())
    for claim in record.get("claims", []):
        claim_text = claim.get("text", "") if isinstance(claim, dict) else str(claim)
        for pat in [r"(.+?)という特徴", r"(.+?)という効果", r"(.+?)に優れる", r"(.+?)が高い", r"(.+?)が低い", r"(.+?)の差別化"]:
            m = re.search(pat, claim_text)
            if m:
                c_candidates.add(m.group(1).strip())
    return list(a_pattern_candidates), list(b_candidates), list(c_candidates), title_nouns

def select_best_keywords_llm(a_candidates, b_candidates, c_candidates, title_nouns, title, summary, claims_texts, max_retries=3):
    import google.generativeai as genai
    import os
    import time

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
# タスク
発明を「A において B が C という特徴を持つもの」と表現するために、以下のカテゴリに分けてキーワードを選定してください。
合計20語以内で、重複語は禁止とします。数値や記号だけの語は避けてください。

A (技術分野/分類/対象): 発明が属する分野。必ず特許本文（タイトル・要約・請求項）から「その特許の所属分野」を推定し、Aカテゴリキーワードに1つ以上含めてください。分類指標（IPCやFタームなどの外部DB情報）は一切使わず、本文から抽出した分野名のみをAカテゴリに含めてください。装置名・部材名レベルの語句はAカテゴリではなくBカテゴリに含めること。
B (発明の根幹となる構成/作用): 発明の中核となる構成要素、材料、または主要な作用・プロセスを示すキーワード。1〜5つの主要概念を表すように、特許の文面から抽出できる関連する類語や言い換え用語を含めてください。特に、化学品系（化学、医薬、材料などの分野）の特許の場合は、化合物名や化学物質名（例：○○化合物、○○酸、○○塩、○○エステル、○○ポリマーなど）をBカテゴリキーワードに必ず1つ以上含めてください。
C (特徴/差別化): 発明の新規性や進歩性を示す、従来の技術との差別化ポイントとなる特徴を示すキーワード。1〜5つの主要概念を表すように、特許の文面から抽出できる関連する類語や言い換え用語を含めてください。

# 重要ルール
複合語は、理解できる「最小単位の名詞」で分解すること**（例: 「色調補正フィルム」→「色調」「補正」「フィルム」）
文脈を考慮しつつ、各ワードにつき最大10個まで、その特許の文脈に適した類語を出すこと。類語の場合は（）で囲むこと。
（例: 「色調」→「色」「カラー」、「調整」→ 「修正」「補正」）特に、発明の概念をより広くカバーできるような類語を優先してください。
「第一」「第１」「第二」など、順番を表すような数値は含めないこと。

# キーワード抽出のヒント
Aカテゴリは、必ず本文から「分野」を推定して含めること。タイトルや「Aにおいて..」のような文言から推測すると良い。類語には、分類コード（IPC,FI）で利用されている項目名となるような類語を採用すること。
Bのキーワードは、「Bを提供する。」「〜〜であるB。」「〜〜を含むB。」のような文言から抽出すると良い。
Cのキーワードは「Cに優れた」「Cができる」「Cを特徴とする」のような文言から抽出すると良い。
キーワードが「製造方法」など一般的過ぎる場合には別の最適なAを探すこと。
発明の「目的」や「課題」、「解決手段」に記載されているキーワードは特に重要のため、注意深く読むこと。
化学物質名などの場合、略称やアルファベット表記などを類語に含めること。

# 補足
特許文書で一般的に使用される専門用語や表現を優先してください。
複合語を分解する際は、分解後の各語が発明の核心を捉えつつ、概念的に近い関係性を保ち、全体として10語の制限内で効率的に表現できるよう考慮してください。

# 特許インプット
タイトル: {title}
要約：{summary}
""" + "\n".join([f"請求項{i+1}: {ct}" for i, ct in enumerate(claims_texts)]) + """

出力フォーマット:
Aカテゴリキーワード: キーワード1（類語1, 類語2, 類語3）, キーワード2, キーワード3, キーワード4, キーワード5
Bカテゴリキーワード: キーワード1, キーワード2, キーワード3
Cカテゴリキーワード: キーワード1, キーワード2, キーワード3
"""

    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                )
            )

            response = model.generate_content(
                prompt,
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            )

            text = response.text

            import re
            a_match = re.search(r"Aカテゴリキーワード[:：]\s*([^\n\r]*)", text)
            b_match = re.search(r"Bカテゴリキーワード[:：]\s*([^\n\r]*)", text)
            c_match = re.search(r"Cカテゴリキーワード[:：]\s*([^\n\r]*)", text)
            a_keywords = a_match.group(1).strip() if a_match else ""
            b_keywords = b_match.group(1).strip() if b_match else ""
            c_keywords = c_match.group(1).strip() if c_match else ""
            return a_keywords, b_keywords, c_keywords

        except Exception as e:
            print(f"Gemini API呼び出しエラー (試行 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # エクスポネンシャルバックオフ
                continue
            else:
                # 最後の試行が失敗した場合、フォールバック処理
                print("Gemini APIでの処理が失敗しました。規則ベースのキーワード抽出を行います。")
                a_keywords = ", ".join(a_candidates[:5])
                b_keywords = ", ".join(b_candidates[:5])
                c_keywords = ", ".join(c_candidates[:5])
                return a_keywords, b_keywords, c_keywords

def main():
    import argparse
    import os
    from datetime import datetime
    from dotenv import load_dotenv
    from azure.cosmos import CosmosClient

    parser = argparse.ArgumentParser(description="A/B/Cカテゴリキーワード抽出（CosmosDBから指定件数取得＋LLM選択）")
    parser.add_argument('--limit', type=int, default=5, help='取得件数')
    args = parser.parse_args()

    load_dotenv()
    COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT")
    COSMOS_KEY = os.environ.get("COSMOS_KEY")
    DATABASE_NAME = os.environ.get("DATABASE_NAME")
    CONTAINER_NAME = os.environ.get("CONTAINER_NAME")
    client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    container = client.get_database_client(DATABASE_NAME).get_container_client(CONTAINER_NAME)

    csv_path = os.path.join(os.path.dirname(__file__), "../data/syutugan_ax_test_data.csv")
    patent_ids = []
    import re
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= args.limit:
                break
            syutugan = row.get("syutugan", "").strip()
            if syutugan:
                m = re.match(r"JP(\d+)[A-Z]?", syutugan)
                if m:
                    patent_ids.append(m.group(1))

    records = []
    for pid in patent_ids:
        query = "SELECT * FROM c WHERE c.metadata.patent_id = @pid"
        params = [{"name": "@pid", "value": pid}]
        items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
        if items:
            c = items[0]
            records.append({
                "patent_id": pid,
                "title": c.get("metadata", {}).get("title", ""),
                "summary": c.get("summary", ""),
                "claims": c.get("claims", [])
            })
    print(f"CosmosDB取得件数: {len(records)} / {len(patent_ids)}")
    if len(records) < len(patent_ids):
        print("取得できなかったpatent_id:", [pid for pid in patent_ids if pid not in [r['patent_id'] for r in records]])

    # csv_path = os.path.join(os.path.dirname(__file__), "../data/csv1_紐付きリスト.csv")
    # group_patent_map = {}
    # with open(csv_path, encoding="shift_jis") as f:
    #     reader = csv.DictReader(f)
    #     for i, row in enumerate(reader):
    #         if i >= len(patent_ids):
    #             break
    #         ids = [pid.strip() for pid in (row.get("patent_id(公開特許番号)") or "").split(",")]
    #         if ids:
    #             group_patent_map[ids[0]] = ids

    import collections
    results = []

    for rec in records:
        a_candidates, b_candidates, c_candidates, title_nouns = extract_a_b_c_candidates_from_record(rec)
        a_keywords, b_keywords, c_keywords = select_best_keywords_llm(a_candidates, b_candidates, c_candidates, title_nouns, rec["title"], rec["summary"], rec["claims"])
        all_keywords = " ".join([a_keywords, b_keywords, c_keywords]).replace(",", " ")
        keyword_list = [kw.strip() for kw in all_keywords.split() if kw.strip()]
        freq_counter = collections.Counter(keyword_list)
        sorted_keywords = [kw for kw, cnt in freq_counter.most_common()]
        sorted_keywords_str = " ".join(sorted_keywords)
        # group_patents = group_patent_map.get(rec["patent_id"], [])
        # group_others = [pid for pid in group_patents if pid != rec["patent_id"]]
        a_decomp_llm = decompose_keywords_llm("A", a_keywords)
        b_decomp_llm = decompose_keywords_llm("B", b_keywords)
        c_decomp_llm = decompose_keywords_llm("C", c_keywords)

        # 形態素解析による分解を追加
        def morphological_analysis(text):
            doc = nlp(text)
            return "/".join([token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]])

        a_morph = morphological_analysis(a_keywords)
        b_morph = morphological_analysis(b_keywords)
        c_morph = morphological_analysis(c_keywords)

        results.append({
            "patent_id": rec["patent_id"],
            "Aカテゴリキーワード": a_keywords,
            "Bカテゴリキーワード": b_keywords,
            "Cカテゴリキーワード": c_keywords,
            "Aカテゴリ分解キーワード": a_decomp_llm,
            "Bカテゴリ分解キーワード": b_decomp_llm,
            "Cカテゴリ分解キーワード": c_decomp_llm,
            "Aカテゴリ形態素解析": a_morph,
            "Bカテゴリ形態素解析": b_morph,
            "Cカテゴリ形態素解析": c_morph,
            "抽出キーワード全体": all_keywords,
            "重要度順キーワード": sorted_keywords_str
        })

    outdir = os.path.join(os.path.dirname(__file__), "../data/output_keywords")
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(outdir, f"abc_a_keywords_{timestamp}.csv")

    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patent_id",
            "Aカテゴリキーワード", "Aカテゴリ分解キーワード", "Aカテゴリ形態素解析",
            "Bカテゴリキーワード", "Bカテゴリ分解キーワード", "Bカテゴリ形態素解析",
            "Cカテゴリキーワード", "Cカテゴリ分解キーワード", "Cカテゴリ形態素解析",
            "抽出キーワード全体", "重要度順キーワード"
        ])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"出力しました: {outfile}")
    print(f"処理件数: {len(results)}件")

def decompose_keywords_llm(category, keywords, max_retries=3):
    import google.generativeai as genai
    import os
    import time
    import re

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
# タスク
以下の{category}カテゴリキーワード列を、複合語を最小単位の名詞に分解してください。
ただし、一般的すぎて特許の文脈上絞り込みに入れるのに不適切なワードは含めないでください。
出力は「分解キーワード1, 分解キーワード2, ...」のカンマ区切りで、重複語は禁止とします。

# 入力キーワード列
{keywords}

# 出力例
色調, 補正, フィルム, カラー, 修正, TCF, 透明, 電極, 膜
"""

    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                )
            )

            response = model.generate_content(
                prompt,
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            )

            text = response.text

            m = re.search(r"([^\n\r]+)", text)
            if m:
                return "/".join([k.strip() for k in m.group(1).split(",") if k.strip()])
            else:
                return ""

        except Exception as e:
            print(f"Gemini API呼び出しエラー (試行 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                # フォールバック：形態素解析を使用
                return "/".join([token.text for token in nlp(keywords) if token.pos_ == "NOUN"][:10])

if __name__ == "__main__":
    main()

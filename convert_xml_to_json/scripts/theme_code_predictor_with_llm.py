#!/usr/bin/env python3
"""
テーマコード予測システム（LLM判定付き）
1. ベクトル検索で上位10件のテーマコード候補を取得
2. enhanced_search_text（FI定義含む）をLLMに渡して最適なものを選択
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI, OpenAI
import google.generativeai as genai

# ===== Azure Search設定 =====
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://patent-classification.search.windows.net")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")  # 環境変数から取得
ENHANCED_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "enhanced_theme_code_index")

# キーが設定されていない場合の警告
if not SEARCH_KEY:
    logger.warning("AZURE_SEARCH_KEY environment variable is not set. Azure Search functionality will be disabled.")
    SEARCH_KEY = None

# ===== Azure OpenAI設定 =====
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://patent-openai.openai.azure.com/")
AOAI_KEY = os.getenv("AZURE_OPENAI_KEY")  # 環境変数から取得

# キーが設定されていない場合の警告
if not AOAI_KEY:
    logger.warning("AZURE_OPENAI_KEY environment variable is not set. Azure OpenAI functionality will be disabled.")
    AOAI_KEY = None
EMBED_MODEL = "text-embedding-3-small"

# ===== LLM設定 =====
# 環境変数から取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Azure OpenAIクライアント（キーが設定されている場合のみ初期化）
aoai_client = None
if AOAI_KEY:
    aoai_client = AzureOpenAI(api_key=AOAI_KEY, api_version="2024-02-01", azure_endpoint=AOAI_ENDPOINT)
else:
    logger.warning("Azure OpenAI client not initialized (no API key)")

def make_search_client() -> SearchClient:
    """Azure Search クライアントを作成"""
    if not SEARCH_KEY:
        raise ValueError("AZURE_SEARCH_KEY environment variable is not set")
    return SearchClient(SEARCH_ENDPOINT, ENHANCED_INDEX_NAME, AzureKeyCredential(SEARCH_KEY))

def embed_text(text: str) -> List[float]:
    """テキストをベクトルに変換"""
    if not aoai_client:
        logger.warning("Azure OpenAI client not available for embedding")
        return []
    try:
        # 長すぎるテキストは切り詰める
        if len(text) > 8000:
            text = text[:8000]
        
        resp = aoai_client.embeddings.create(input=text, model=EMBED_MODEL)
        return resp.data[0].embedding
    except Exception as e:
        print(f"[ERROR] 埋め込みエラー: {e}", file=sys.stderr)
        return []

def search_theme_candidates(query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    ベクトル検索で上位k件のテーマコード候補を取得
    enhanced_search_textも含めて返す
    """
    client = make_search_client()
    vec = embed_text(query_text)
    
    if not vec:
        return []
    
    vq = VectorizedQuery(
        vector=vec,
        fields="enhanced_vector",
        k_nearest_neighbors=top_k
    )
    
    results_iter = client.search(
        search_text=query_text if query_text.strip() else None,
        search_fields=["enhanced_search_text", "description", "fi_definition_scraped", "fi_definition"],
        vector_queries=[vq],
        select=["theme_code", "description", "enhanced_search_text", "fi_definition_scraped", "fi_definition"],
        top=top_k,
    )
    
    candidates = []
    for r in results_iter:
        candidates.append({
            "theme_code": r.get("theme_code"),
            "description": r.get("description", ""),
            "enhanced_search_text": r.get("enhanced_search_text", ""),
            "fi_definition_scraped": r.get("fi_definition_scraped", ""),
            "fi_definition": r.get("fi_definition", ""),
            "score": r.get("@search.score", 0.0),
        })
        if len(candidates) >= top_k:
            break
    
    return candidates

def select_best_theme_code_with_openai(patent_text: str, candidates: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    OpenAI GPT-4を使って最適なテーマコードを選択
    """
    if not OPENAI_API_KEY:
        print("[WARN] OPENAI_API_KEY not set", file=sys.stderr)
        return candidates[0]["theme_code"], "No LLM selection (API key not set)" if candidates else ("", "")
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # 候補リストを作成（enhanced_search_textを使用）
        candidate_text = ""
        for i, c in enumerate(candidates, 1):
            candidate_text += f"\n{i}. テーマコード: {c['theme_code']}\n"
            # enhanced_search_textを優先使用（FI定義を含む詳細情報）
            if c.get('enhanced_search_text'):
                candidate_text += f"   詳細情報: {c['enhanced_search_text'][:400]}...\n"
            else:
                candidate_text += f"   説明: {c['description']}\n"
            candidate_text += f"   検索スコア: {c['score']:.4f}\n"
        
        prompt = f"""あなたは特許分類の専門家です。以下の特許文書に最も適したテーマコードを選択してください。

【特許の技術要点】
{patent_text[:2000]}

【候補テーマコード（検索スコア順）】
{candidate_text}

【選択基準】
1. 特許の主要技術内容との適合度
2. FI定義との整合性
3. 技術的特徴の一致度

最も適切な1つを選択し、選択理由を技術的根拠とともに説明してください。

【回答形式】
選択: [テーマコード]
理由: [技術的根拠]
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは特許分類の専門家です。技術内容とテーマコードの対応を正確に判断してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        result = response.choices[0].message.content
        
        # 選択されたテーマコードを抽出
        for line in result.split('\n'):
            if '選択:' in line or '選択：' in line:
                selected_code = line.split(':')[-1].strip().replace('[', '').replace(']', '')
                # 候補リストにあるか確認
                for c in candidates:
                    if c['theme_code'] == selected_code:
                        return selected_code, result
        
        # 抽出できなかった場合はスコア最上位を返す
        return candidates[0]["theme_code"], f"LLM selection failed, using top score: {result}" if candidates else ("", "")
        
    except Exception as e:
        print(f"[ERROR] OpenAI API error: {e}", file=sys.stderr)
        return candidates[0]["theme_code"], f"Error: {str(e)}" if candidates else ("", "")

def select_best_theme_code_with_gemini(patent_text: str, candidates: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Google Geminiを使って最適なテーマコードを選択
    """
    if not GOOGLE_API_KEY:
        print("[WARN] GOOGLE_API_KEY not set", file=sys.stderr)
        return candidates[0]["theme_code"], "No LLM selection (API key not set)" if candidates else ("", "")
    
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 候補リストを作成（enhanced_search_textを使用）
        candidate_text = ""
        for i, c in enumerate(candidates, 1):
            candidate_text += f"\n{i}. テーマコード: {c['theme_code']}\n"
            # enhanced_search_textを優先使用（FI定義を含む詳細情報）
            if c.get('enhanced_search_text'):
                candidate_text += f"   詳細情報: {c['enhanced_search_text'][:400]}...\n"
            else:
                candidate_text += f"   説明: {c['description']}\n"
            candidate_text += f"   検索スコア: {c['score']:.4f}\n"
        
        prompt = f"""あなたは特許分類の専門家です。以下の特許文書に最も適したテーマコードを選択してください。

【特許の技術要点】
{patent_text[:2000]}

【候補テーマコード（検索スコア順）】
{candidate_text}

【選択基準】
1. 特許の主要技術内容との適合度
2. FI定義との整合性
3. 技術的特徴の一致度

最も適切な1つを選択し、選択理由を技術的根拠とともに説明してください。

【回答形式】
選択: [テーマコード]
理由: [技術的根拠]
"""
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=200,
            )
        )
        
        result = response.text
        
        # 選択されたテーマコードを抽出
        for line in result.split('\n'):
            if '選択:' in line or '選択：' in line:
                selected_code = line.split(':')[-1].strip().replace('[', '').replace(']', '')
                # 候補リストにあるか確認
                for c in candidates:
                    if c['theme_code'] == selected_code:
                        return selected_code, result
        
        # 抽出できなかった場合はスコア最上位を返す
        return candidates[0]["theme_code"], f"LLM selection failed, using top score: {result}" if candidates else ("", "")
        
    except Exception as e:
        print(f"[ERROR] Gemini API error: {e}", file=sys.stderr)
        return candidates[0]["theme_code"], f"Error: {str(e)}" if candidates else ("", "")

def predict_theme_code(patent_text: str, llm_provider: str = "openai", top_k: int = 10) -> Dict[str, Any]:
    """
    特許文書からテーマコードを予測
    
    Args:
        patent_text: 特許文書のテキスト
        llm_provider: 使用するLLM ("openai" or "gemini")
        top_k: 検索候補数
    
    Returns:
        予測結果の辞書
    """
    # 1. ベクトル検索で候補を取得
    candidates = search_theme_candidates(patent_text, top_k)
    
    if not candidates:
        return {
            "status": "error",
            "message": "No candidates found",
            "predicted_theme_code": "",
            "candidates": []
        }
    
    # 2. LLMで最適なものを選択
    if llm_provider == "gemini":
        selected_code, reasoning = select_best_theme_code_with_gemini(patent_text, candidates)
    else:
        selected_code, reasoning = select_best_theme_code_with_openai(patent_text, candidates)
    
    # 選択されたテーマコードの詳細を取得
    selected_detail = None
    for c in candidates:
        if c['theme_code'] == selected_code:
            selected_detail = c
            break
    
    if not selected_detail:
        selected_detail = candidates[0]  # フォールバック
    
    return {
        "status": "success",
        "predicted_theme_code": selected_code,
        "description": selected_detail["description"],
        "enhanced_search_text": selected_detail["enhanced_search_text"],
        "vector_search_score": selected_detail["score"],
        "llm_reasoning": reasoning,
        "candidates": candidates,
        "llm_provider": llm_provider
    }

def main():
    parser = argparse.ArgumentParser(description="テーマコード予測システム（LLM判定付き）")
    parser.add_argument("text", nargs="*", help="特許文書テキスト。未指定なら標準入力から読み込み")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="検索候補数（デフォルト: 10）")
    parser.add_argument("--llm", choices=["openai", "gemini"], default="openai", help="使用するLLM")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細情報を出力")
    args = parser.parse_args()
    
    # 入力テキストを取得
    if args.text:
        text = " ".join(args.text)
    else:
        try:
            text = sys.stdin.read().strip()
        except KeyboardInterrupt:
            print()
            return
        if not text:
            print("[ERROR] 空入力", file=sys.stderr)
            return
    
    # テーマコードを予測
    result = predict_theme_code(text, args.llm, args.top_k)
    
    # 結果を出力
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["status"] == "success":
            print(f"予測テーマコード: {result['predicted_theme_code']}")
            print(f"説明: {result['description']}")
            print(f"ベクトル検索スコア: {result['vector_search_score']:.4f}")
            print(f"LLMプロバイダ: {result['llm_provider']}")
            print()
            print("LLM判定理由:")
            print(result['llm_reasoning'])
            
            if args.verbose:
                print("\n候補リスト:")
                for i, c in enumerate(result['candidates'], 1):
                    print(f"{i}. {c['theme_code']} - {c['description']} (スコア: {c['score']:.4f})")
        else:
            print(f"[ERROR] {result.get('message', 'Unknown error')}", file=sys.stderr)

if __name__ == "__main__":
    main()
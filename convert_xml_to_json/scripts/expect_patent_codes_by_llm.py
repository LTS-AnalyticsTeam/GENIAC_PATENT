"""
特許データ分類コード推測システム (改良版)

【使用方法】
# デフォルトモード（consensusのみ出力）
python expect_patent_codes_by_llm.py

# デバッグモード（全ての詳細結果をCSV出力）
python expect_patent_codes_by_llm.py --debug

【出力モード】
- デフォルトモード: 複数の予測手法の総意（consensus）のみを出力
  - 予測されたテーマコード、FI、Ftermを含む
  - 各種精度情報も表示
- デバッグモード: 全ての予測手法の個別結果を詳細にCSV出力
  - LLM予測結果
  - ベクトル検索結果（複数手法）
  - テーマコード検索結果
  - 最終的なconsensus結果

【改良点】
1. FI予測の精度向上: application_fieldとtechnical_fieldの両方でFI予測を行い、タイトルとの関連度で選択
2. 特許番号の出力: CSVにpatent_idを追加
3. 安全ブロック対策: Geminiがブロックされた場合にOpenAI APIで再試行
4. ベクトル検索用に端的な１用語を抽出し、FI予測に使用
5. 複数手法の共通FIコードを総意として採用
6. デバッグモードの追加: --debugフラグで詳細出力を制御
7. consensus結果にテーマコードとFtermを含めるよう拡張
8. テーマコードとFtermの整合性チェック: 最初5桁の一致を確認し、不整合を修正
9. 実用的最適化アプローチ: 精度重視でLLM結果を優先し、ベクトル検索で補完する戦略
"""

import asyncio
import csv
import json
import logging
import os
import re
import subprocess

# 改良されたハイブリッドシステムをインポート
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
try:
    from enhanced_theme_predictor import EnhancedThemePredictor
except ImportError:
    # logger may not be initialized yet at import time
    print("enhanced_theme_predictor not found, using fallback")
    EnhancedThemePredictor = None

# 外部ライブラリ
try:
    import google.generativeai as genai
    import pandas as pd
    from azure.cosmos import CosmosClient
    from dotenv import load_dotenv
    from openai import OpenAI

    # LLM強化テーマコード予測システム
    from theme_code_predictor_with_llm import predict_theme_code
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    print("以下のコマンドを実行してください: pip install azure-cosmos pandas google-generativeai openai python-dotenv")
    exit(1)

load_dotenv()

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# テーマコード検索：FI/scripts/fi_fterm_search.py に差し替え
import subprocess


def search_theme_codes_via_fi_script(query_text: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    FI/scripts/fi_fterm_search.py を用いて FI / F-term の上位候補を取得し、
    互換の形式（theme_code相当のフィールドにcodeを入れる）で返す。

    注意: 本来のテーマコード検索とは異なり、ここではFI/F-termコードを返します。
    以降の処理はコード文字列として扱えるため、最低限の置き換えとして機能します。

    Args:
        query_text: 検索クエリテキスト
        top_k: 各インデックスからの上位件数を概ね制御（fi_fterm_search側は固定3件）
    Returns:
        リスト[{'theme_code': <code>, 'description': <index名とchunk_id>, 'full_title': '', 'similarity_score': <score>}]
    """
    if not query_text.strip():
        return []

    try:
        # FI/scripts/fi_fterm_search.py のパス
        fi_script_path = os.path.join(os.path.dirname(__file__), '../../FI/scripts/fi_fterm_search.py')

        # スクリプトを実行（標準出力はTSV: index\tcode\tchunk_id\tscore）
        result = subprocess.run(
            ['python', fi_script_path, query_text],
            capture_output=True,
            text=True,
            timeout=40
        )

        if result.returncode != 0:
            logger.warning(f"fi_fterm_search 実行エラー: {result.stderr}")
            return []

        formatted_results: List[Dict[str, str]] = []
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        for ln in lines:
            parts = ln.split('\t')
            if len(parts) < 4:
                continue
            idx, code, chunk_id, score_str = parts[:4]
            try:
                score = float(score_str)
            except Exception:
                score = 0.0
            formatted_results.append({
                'theme_code': code,  # 互換のためにcodeをtheme_codeフィールドに入れる
                'description': f"{idx}:{chunk_id}",
                'full_title': '',
                'similarity_score': score,
            })

        # 上位top_k相当に制限（両インデックス合算後の簡易カット）
        if top_k is not None and top_k > 0:
            formatted_results = formatted_results[: top_k]
        return formatted_results

    except subprocess.TimeoutExpired:
        logger.warning("fi_fterm_search がタイムアウトしました")
        return []
    except FileNotFoundError:
        logger.warning("fi_fterm_search.py が見つかりません。FI/scripts 配下を確認してください。")
        return []
    except Exception as e:
        logger.warning(f"fi_fterm_search 実行中にエラー: {e}")
        return []

# レガシーのThemeSearchHelperクラス（後方互換性のため）
class LegacyThemeSearchHelper:
    def find_similar_themes(self, query_text: str, top_k: int = 5) -> List[Dict]:
        return search_theme_codes_via_fi_script(query_text, top_k)

# 後方互換性のため
ThemeSearchHelper = LegacyThemeSearchHelper


@dataclass
class PatentData:
    """特許データの構造を定義"""
    patent_id: str  # 特許番号を追加
    title: str
    abstract: str
    claims_main: List[str]
    figure_captions: List[str]
    ipc_seed: List[str]
    entities: Dict[str, List[str]]
    applicant: str
    citations_ipc: List[str]
    correct_theme_code: List[str]
    correct_fi: List[str]
    correct_fterm: List[str]


@dataclass
class PredictionResult:
    """推測結果の構造を定義"""
    predicted_theme_code: List[str]
    predicted_fi: List[str]
    predicted_fterm: List[str]
    analysis: Dict[str, any]
    theme_keywords: List[str]
    fi_keywords: List[str]
    model_name: str
    fi_prediction_method: str = ""  # どちらのキーワードを使用したか記録




class CosmosDBClient:
    """CosmosDB接続・データ取得クラス"""
    def __init__(self):
        self.client = CosmosClient(os.environ['COSMOS_ENDPOINT'], os.environ['COSMOS_KEY'])
        self.database = self.client.get_database_client(os.environ['DATABASE_NAME'])
        self.container = self.database.get_container_client(os.environ['CONTAINER_NAME'])

    def get_patent_data(self, limit: int = 100) -> List[PatentData]:
        """CosmosDBから特許データを取得し、PatentDataオブジェクトに変換"""
        try:
            query = "SELECT TOP @limit * FROM c"
            items = list(self.container.query_items(
                query=query,
                parameters=[{"name": "@limit", "value": limit}],
                enable_cross_partition_query=True
            ))
            logger.info(f"CosmosDBから取得したアイテム数: {len(items)}")
            patent_data_list = []
            for item in items:
                metadata = item.get('metadata', {})
                patent_id = metadata.get('patent_id', 'Unknown')  # 特許番号を取得
                claims_list = [f"【請求項{claim.get('num', '')}】{claim.get('text', '')}" for claim in item.get('claims', [])]
                keywords = metadata.get('keywords', [])
                entities = {'objects': keywords[:10], 'actions': [], 'usecase': []}
                figure_captions = [item.get('description', '').split('【選択図】')[1].strip()] if '【選択図】' in item.get('description', '') else []

                patent_data = PatentData(
                    patent_id=patent_id,  # 特許番号を設定
                    title=metadata.get('title', ''),
                    abstract=item.get('summary', ''),
                    claims_main=claims_list,
                    figure_captions=figure_captions,
                    ipc_seed=[ipc.get('code', '') for ipc in metadata.get('classification_ipc', [])],
                    entities=entities,
                    applicant=', '.join(item.get('applicants', [])),
                    citations_ipc=metadata.get('topics', []),
                    correct_theme_code=self._extract_codes(metadata, 'theme_code'),
                    correct_fi=self._extract_codes(metadata, 'classification_fi'),
                    correct_fterm=self._extract_codes(metadata, 'f_term')
                )
                if patent_data.title or patent_data.abstract or patent_data.claims_main:
                    patent_data_list.append(patent_data)
                else:
                    logger.warning(f"空の特許データをスキップしました: {patent_id}")
            logger.info(f"有効な特許データ数: {len(patent_data_list)}")
            return patent_data_list
        except Exception as e:
            logger.error(f"CosmosDBからのデータ取得エラー: {e}")
            return []

    def _extract_codes(self, metadata: dict, key: str) -> List[str]:
        codes = metadata.get(key, [])
        if isinstance(codes, list):
            return [str(code) for code in codes]
        return [str(codes)] if codes else []


class VectorSearchPredictor:
    """ベクトル検索を使用した分類コード推測クラス"""
    def __init__(self, enable_vector_search: bool = True):
        # fi_fterm_search.pyのパスを設定
        self.fi_fterm_search_path = os.path.join(os.path.dirname(__file__), "../../FI/scripts/fi_fterm_search.py")
        self.enable_vector_search = enable_vector_search

        # テーマコード検索ヘルパーを初期化
        self.theme_searcher = None
        if ThemeSearchHelper:
            try:
                self.theme_searcher = ThemeSearchHelper()
                logger.info(f"テーマコード検索ヘルパー初期化完了: {len(self.theme_searcher.theme_data)}個のテーマコード")
            except Exception as e:
                logger.warning(f"テーマコード検索ヘルパー初期化エラー: {e}")
                self.theme_searcher = None

    def search_fi_fterm(self, keywords: List[str]) -> Tuple[List[str], List[str]]:
        """キーワードリストを使用してFIとFタームを検索（evaluate_fi_fterm_from_csv.pyと同じ方法）"""
        if not keywords or not self.enable_vector_search:
            return [], []

        start_time = time.time()
        # キーワードをスペース区切りの文字列に結合
        query_text = " ".join(keywords)
        logger.debug(f"ベクトル検索開始: キーワード数={len(keywords)}")

        try:
            # evaluate_fi_fterm_from_csv.pyと同じ方法でスクリプトパスを構築
            script_path = os.path.join(os.path.dirname(__file__), "../../FI/scripts/fi_fterm_search.py")

            # Use current Python executable
            venv_python = sys.executable

            subprocess_start = time.time()
            result = subprocess.run(
                [venv_python, script_path, query_text],
                capture_output=True,
                text=True
            )
            logger.debug(f"ベクトル検索API呼び出し時間: {time.time() - subprocess_start:.2f}秒")

            if result.returncode != 0:
                logger.warning(f"ベクトル検索が失敗しました（スキップします）: {result.stderr}")
                return [], []

            # evaluate_fi_fterm_from_csv.pyと同じ解析方法だが、新しいインデックス名に対応
            parse_start = time.time()
            fi_codes = []
            fterm_codes = []

            for line in result.stdout.splitlines():
                cols = line.strip().split("\t")
                if len(cols) >= 2:
                    # 新しいインデックス名に対応（_2が付いている）
                    if cols[0] == "fi_classification_index_2":
                        fi_codes.append(cols[1])
                    elif cols[0] == "fterm_classification_index_2":
                        fterm_codes.append(cols[1])

            total_time = time.time() - start_time
            logger.debug(f"ベクトル検索完了: 解析時間={time.time() - parse_start:.2f}秒, 総時間={total_time:.2f}秒, FI={len(fi_codes)}件, Fterm={len(fterm_codes)}件")
            return fi_codes, fterm_codes  # 上位3件は元のスクリプトで制限されているのでそのまま返す

        except Exception as e:
            total_time = time.time() - start_time
            logger.warning(f"ベクトル検索実行エラー: {e}, 時間: {total_time:.2f}秒")
            return [], []

    def search_fterm_with_keywords(self, keywords: List[str]) -> List[str]:
        """Fターム専用キーワードを使ってFタームのみを検索"""
        if not keywords or not self.enable_vector_search:
            return []

        start_time = time.time()
        query_text = " ".join(keywords)
        logger.debug(f"Fターム専用ベクトル検索開始: キーワード数={len(keywords)}")

        try:
            script_path = os.path.join(os.path.dirname(__file__), "../../FI/scripts/fi_fterm_search.py")
            venv_python = sys.executable

            subprocess_start = time.time()
            result = subprocess.run(
                [venv_python, script_path, query_text],
                capture_output=True,
                text=True
            )
            logger.debug(f"Fターム専用ベクトル検索API呼び出し時間: {time.time() - subprocess_start:.2f}秒")

            if result.returncode != 0:
                logger.warning(f"Fターム専用ベクトル検索が失敗しました: {result.stderr}")
                return []

            parse_start = time.time()
            fterm_codes = []

            for line in result.stdout.splitlines():
                cols = line.strip().split("\t")
                if len(cols) >= 2:
                    if cols[0] == "fterm_classification_index_2":
                        fterm_codes.append(cols[1])

            total_time = time.time() - start_time
            logger.debug(f"Fターム専用ベクトル検索完了: 解析時間={time.time() - parse_start:.2f}秒, 総時間={total_time:.2f}秒, Fterm={len(fterm_codes)}件")
            return fterm_codes

        except Exception as e:
            total_time = time.time() - start_time
            logger.warning(f"Fターム専用ベクトル検索実行エラー: {e}, 時間: {total_time:.2f}秒")
            return []

    def search_theme_codes_enhanced(self, query_text: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        改良されたハイブリッドテーマコード予測システム
        シミュレーション逆引き学習 + Azure AI Searchの組み合わせ
        """
        if not query_text:
            return []

        start_time = time.time()
        logger.debug(f"改良テーマコード検索開始: {query_text}")

        try:
            # 改良されたハイブリッド予測システムを使用
            if EnhancedThemePredictor is not None:
                if not hasattr(self, '_enhanced_predictor'):
                    self._enhanced_predictor = EnhancedThemePredictor()
                    logger.info("ハイブリッド予測システム初期化完了")

                predicted_code, description, debug_info = self._enhanced_predictor.predict_theme_code_enhanced(
                    query_text, top_k
                )

                # 結果を従来の形式に変換
                results = []
                if predicted_code:
                    results.append({
                        'theme_code': predicted_code,
                        'description': description,
                        'similarity_score': str(debug_info.get('hybrid_candidates', [{}])[0].get('hybrid_score', 0.0))
                    })

                    # 追加の候補も含める
                    for candidate in debug_info.get('hybrid_candidates', [])[1:top_k]:
                        results.append({
                            'theme_code': candidate.get('theme_code', ''),
                            'description': candidate.get('description', ''),
                            'similarity_score': str(candidate.get('hybrid_score', 0.0))
                        })

                total_time = time.time() - start_time
                logger.info(f"ハイブリッド予測完了: 予測={predicted_code}, 時間={total_time:.2f}秒")
                return results

            else:
                # フォールバック: 既存システム
                logger.warning("ハイブリッドシステムが利用できません。従来システムを使用します。")
                return self.search_theme_codes_fallback(query_text, top_k)

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"ハイブリッドテーマコード検索エラー: {e}, 時間: {total_time:.2f}秒")
            return self.search_theme_codes_fallback(query_text, top_k)

    def search_theme_codes_fallback(self, query_text: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        従来のテーマコードベースのベクトル検索（Azure AI Search経由）
        """
        if not query_text:
            return []

        start_time = time.time()
        logger.debug(f"フォールバック テーマコード検索開始: {query_text}")

        try:
            # FI/scripts/theme_code_search.pyを使用してAzure AI Searchで検索
            results = search_theme_codes_via_fi_script(query_text, top_k=top_k)

            total_time = time.time() - start_time

            # 類似度スコアがある場合はログに記録
            if results and 'similarity_score' in results[0]:
                scores = [r.get('similarity_score', 0) for r in results]
                logger.debug(f"フォールバック テーマコード検索完了: {len(results)}件ヒット, 最高スコア: {max(scores):.3f}, 時間: {total_time:.2f}秒")
            else:
                logger.debug(f"フォールバック テーマコード検索完了: {len(results)}件ヒット, 時間: {total_time:.2f}秒")

            return results

        except Exception as e:
            total_time = time.time() - start_time
            logger.warning(f"フォールバック テーマコード検索エラー: {e}, 時間: {total_time:.2f}秒")
            return []

    def search_theme_codes(self, query_text: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        テーマコードベースのベクトル検索
        改良版を優先使用し、フォールバックで従来版を使用
        """
        # 改良版を使用
        return self.search_theme_codes_enhanced(query_text, top_k)


class LLMPredictor:
    """LLMを使用した分類コード推測クラス"""
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.gemini_model_25 = genai.GenerativeModel('gemini-2.5-flash')
        self.gemini_model_20 = genai.GenerativeModel('gemini-2.0-flash-exp')  # フォールバック用

    def create_unified_prompt(self, patent_data: PatentData) -> str:
        """
        統合プロンプト: 1回のAPI呼び出しで適用分野・技術分野両方の分析を実行
        """
        all_claims = "\n\n".join(patent_data.claims_main) if patent_data.claims_main else "請求項情報なし"

        return f"""あなたは特許分類の専門家です。以下の特許について、適用分野と技術分野の両方の観点からFI分類コード、テーマコード、Ftermを推測してください。

【特許情報】
タイトル: {patent_data.title}
要約: {patent_data.abstract}
請求項: {all_claims}

【指示】
以下の形式のJSONで回答してください。適用分野重視と技術分野重視の両方の分析を含めてください。

{{
    "application_focus": {{
        "predicted_fi": ["FIコード1", "FIコード2"],
        "predicted_theme_code": ["テーマコード1", "テーマコード2"],
        "predicted_fterm": ["Fterm1", "Fterm2"],
        "application_field_keywords": ["キーワード1", "キーワード2"],
        "confidence": 0.8,
        "reasoning": "適用分野重視の判断理由"
    }},
    "technical_focus": {{
        "predicted_fi": ["FIコード1", "FIコード2"],
        "predicted_theme_code": ["テーマコード1", "テーマコード2"],
        "predicted_fterm": ["Fterm1", "Fterm2"],
        "technical_field_keywords": ["キーワード1", "キーワード2"],
        "confidence": 0.8,
        "reasoning": "技術分野重視の判断理由"
    }},
    "recommended_approach": "application_focus または technical_focus",
    "overall_confidence": 0.85
}}

【重要】
- FIコードは4桁以上で記述（例: A61K, C07D, H01L）
- テーマコードは5文字（例: 5K067, 4C085）
- Ftermは5文字+アルファベット数字（例: 5K067AA02）
- キーワードは特許の核心技術を表す語句
- recommended_approachで最適と思われるアプローチを指定"""

    def create_prompt(self, patent_data: PatentData, focus_on_technical: bool = False) -> str:
        """特許データから分類コード推測用のプロンプトを生成

        Args:
            patent_data: 特許データ
            focus_on_technical: Trueの場合、技術分野により焦点を当てる
        """
        all_claims = "\n\n".join(patent_data.claims_main) if patent_data.claims_main else "請求項情報なし"

        # 焦点の指示を変更
        focus_instruction = ""
        if focus_on_technical:
            focus_instruction = """
            【重要】FI分類の推測では、主に「技術分野のキーワード」（構成要素、機能、技術的特徴）に焦点を当ててください。
            適用分野のキーワードは補助的に使用してください。
            """
        else:
            focus_instruction = """
            【重要】FI分類の推測では、主に「適用分野のキーワード」（使用場所、用途、目的）に焦点を当ててください。
            技術分野のキーワードは補助的に使用してください。
            """

        prompt = f"""
          以下の特許データから、適切な分類コード（テーマコード、FI、Fターム）を推測してください。
          まず「適用分野」と「主要な技術分野（構成要素、機能）」を分析し、キーワードを抽出してください。

          {focus_instruction}

          【分析の視点】
          - **適用分野**: この発明が使われる場所や目的（例: 自動車、医療、建築）。
          - **主要な技術分野**: この発明の核心となる技術や構成要素（例: 光学センサー、データ処理、ロボットアーム）。
          - **適用分野の単一キーワード**: 適用分野を最も的確に表現する1つのキーワード（例: 自動車特許なら「自動車」、医療特許なら「医療」）。
          - **特許の主役**: この特許で扱われる「主役」を1-3個の具体的な用語で抽出。テーマコードで使われる用語を意識。例：清掃器具なら「掃除機、清掃、吸引」、車両なら「自動車、エンジン、制御」、表示装置なら「ディスプレイ、タッチパネル、表示」など。製品名と動作/機能を組み合わせる。
          - **特許の特異点**: この発明をFI/Fターム分類の観点で「AにおいてBがC」の形式で表現してください。
            A: 【製品分野】この技術が属する製品カテゴリ（1単語）
            B: 【技術要素】核心となる技術・部品・手段（1単語）
            C: 【技術効果】達成される技術的効果（1単語）

          【重要】特許の具体的用途や構造ではなく、FI/Fターム分類での位置づけを抽出してください。

          【変換の例】
          - 穴の清掃用ホース → A:掃除機, B:ノズル, C:吸引
          - タッチパネルのセンサー → A:入力装置, B:センサー, C:検出
          - 車のエンジン冷却 → A:エンジン, B:冷却器, C:放熱
          - 通信の暗号化 → A:通信, B:暗号, C:セキュリティ

          【注意事項】
          - A、B、Cは必ず1単語で表現してください
          - 特許の表面的表現ではなく、分類体系での抽象概念を使用
          - どの技術分野・製品に分類されるべきかを重視

          【特許データ】
          発明の名称: {patent_data.title}
          要約: {patent_data.abstract}
          請求項: {all_claims}
          図面の簡単な説明: {' '.join(patent_data.figure_captions) if patent_data.figure_captions else '情報なし'}
          出願人: {patent_data.applicant if patent_data.applicant else '情報なし'}


          【出力形式】
          回答は必ず以下のJSON形式でお願いします。各コードはリスト形式で複数指定可能です。

          {{
              "predicted_theme_code": ["推測したテーマコード"],
              "predicted_fi": ["推測したFI分類"],
              "predicted_fterm": ["推測したFターム"],
              "analysis": {{
                  "application_field_keywords": ["適用分野のキーワード"],
                  "technical_field_keywords": ["主要な技術分野のキーワード"],
                  "application_keyword": "適用分野の単一キーワード（最も適切な1つ）",
                  "core_concept": "この特許の主役（1-3個の具体的用語をカンマ区切り。テーマコード粒度の具体性）",
                  "unique_point": {{
                      "A": "対象・場面",
                      "B": "手段・構成",
                      "C": "効果・機能"
                  }}
              }},
              "theme_keywords": ["テーマコード推測の根拠となったキーワード"],
              "fi_keywords": ["FI分類推測の根拠となったキーワード"],
          }}

          【出力例】
          {{
              "predicted_theme_code": ["5J062"],
              "predicted_fi": ["G06F3/041", "G06F3/0488"],
              "predicted_fterm": ["5B087 FF02", "5E555 AA12"],
              "analysis": {{
                  "application_field_keywords": ["タッチパネル", "スマートフォン", "ディスプレイ"],
                  "technical_field_keywords": ["静電容量センサー", "電極パターン", "ノイズ除去"],
                  "application_keyword": "タッチパネル",
                  "core_concept": "タッチパネル, センサー, 入力装置",
                  "unique_point": {{
                      "A": "入力装置",
                      "B": "センサー",
                      "C": "検出"
                  }}
              }},
              "theme_keywords": ["コンピュータ", "入出力"],
              "fi_keywords": ["タッチパネル", "入力装置"],
          }}
          """
        return prompt

    async def predict_with_openai(self, patent_data: PatentData, focus_on_technical: bool = False) -> PredictionResult:
        """OpenAI GPT-5で分類コードを推測"""
        start_time = time.time()
        logger.info(f"OpenAI推測開始: {patent_data.patent_id}")

        try:
            prompt_start = time.time()
            prompt = self.create_prompt(patent_data, focus_on_technical)
            logger.debug(f"Prompt作成時間: {time.time() - prompt_start:.2f}秒")
            api_start = time.time()
            # response = self.openai_client.chat.completions.create(
            #     model="gpt-4o-mini",  # gpt-5-miniは存在しないため、gpt-4o-miniに修正
            #     messages=[
            #         {"role": "system", "content": "You are a helpful assistant that analyzes patent documents and predicts classification codes in JSON format."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     response_format={"type": "json_object"},
            #     temperature=0.1
            # )
            client = OpenAI()
            response = client.responses.create(
                model="gpt-5-mini",
                input=prompt,
                reasoning={"effort": "minimal"}
            )
            logger.debug(f"OpenAI API呼び出し時間: {time.time() - api_start:.2f}秒")

            parse_start = time.time()
            result_text = ""
            for out in response.output:
                if hasattr(out, "content") and out.content:
                    for c in out.content:
                        if hasattr(c, "text"):
                            result_text = c.text
                            break
            result_json = json.loads(result_text)
            logger.debug(f"JSONパーシング時間: {time.time() - parse_start:.2f}秒")

            # キーワードフィルタリング
            context = patent_data.title + " " + patent_data.abstract
            if 'analysis' in result_json:
                if 'application_field_keywords' in result_json['analysis']:
                    result_json['analysis']['application_field_keywords'] = self.keyword_filter.filter_keywords(
                        result_json['analysis']['application_field_keywords'], context
                    )

            result = PredictionResult(
                predicted_theme_code=result_json.get('predicted_theme_code', []),
                predicted_fi=result_json.get('predicted_fi', []),
                predicted_fterm=result_json.get('predicted_fterm', []),
                analysis=result_json.get('analysis', {}),
                theme_keywords=result_json.get('theme_keywords', []),
                fi_keywords=result_json.get('fi_keywords', []),
                model_name="OpenAI_GPT-4o-mini",
                fi_prediction_method="technical" if focus_on_technical else "application"
            )

            total_time = time.time() - start_time
            logger.info(f"OpenAI推測完了: {patent_data.patent_id}, 総時間: {total_time:.2f}秒")
            return result
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"OpenAI推測エラー: {e}, 時間: {total_time:.2f}秒")
            return self._empty_result("OpenAI_GPT-4o-mini")

    async def predict_with_gemini(self, patent_data: PatentData, focus_on_technical: bool = False) -> PredictionResult:
        """Gemini 2.5 flashで分類コードを推測（ブロックされたら2.0で再試行、それでもダメならOpenAIで再試行）"""

        # 最大限に安全設定を緩和
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]

        prompt = self.create_prompt(patent_data, focus_on_technical)

        # まず2.5で試行
        try:
            response = self.gemini_model_25.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0,  # より安定した結果のために0に
                ),
                safety_settings=safety_settings
            )

            # 2.5でのブロック確認
            if not response.candidates:
                finish_reason = response.prompt_feedback.block_reason if hasattr(response.prompt_feedback, 'block_reason') else 'Unknown'
                logger.warning(f"Gemini 2.5: プロンプトがブロックされました。理由: {finish_reason}")
                logger.info("Gemini 2.0で再試行します...")
                return await self._try_gemini_20(patent_data, focus_on_technical, prompt, safety_settings)

            candidate = response.candidates[0]
            if candidate.finish_reason != 1:  # STOP以外
                logger.warning(f"Gemini 2.5: 応答がブロックされました。理由: {candidate.finish_reason}")
                logger.info("Gemini 2.0で再試行します...")
                return await self._try_gemini_20(patent_data, focus_on_technical, prompt, safety_settings)

            # 応答テキストを安全に取得
            if not response.candidates or not response.candidates[0].content.parts:
                logger.warning("Gemini 2.5: 有効な応答内容が返されませんでした")
                logger.info("Gemini 2.0で再試行します...")
                return await self._try_gemini_20(patent_data, focus_on_technical, prompt, safety_settings)

            parse_start = time.time()
            result_text = response.candidates[0].content.parts[0].text
            result_json = json.loads(result_text)
            logger.debug(f"JSON パーシング時間: {time.time() - parse_start:.2f}秒")

            # キーワードフィルタリング
            context = patent_data.title + " " + patent_data.abstract
            if 'analysis' in result_json:
                if 'application_field_keywords' in result_json['analysis']:
                    result_json['analysis']['application_field_keywords'] = self.keyword_filter.filter_keywords(
                        result_json['analysis']['application_field_keywords'], context
                    )

            return PredictionResult(
                predicted_theme_code=result_json.get('predicted_theme_code', []),
                predicted_fi=result_json.get('predicted_fi', []),
                predicted_fterm=result_json.get('predicted_fterm', []),
                analysis=result_json.get('analysis', {}),
                theme_keywords=result_json.get('theme_keywords', []),
                fi_keywords=result_json.get('fi_keywords', []),
                model_name="Gemini_2.5_Flash",
                fi_prediction_method="technical" if focus_on_technical else "application"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Gemini 2.5 JSON解析エラー: {e}")
            logger.info("Gemini 2.0で再試行します...")
            return await self._try_gemini_20(patent_data, focus_on_technical, prompt, safety_settings)
        except Exception as e:
            logger.error(f"Gemini 2.5推測エラー: {e}")
            logger.info("Gemini 2.0で再試行します...")
            return await self._try_gemini_20(patent_data, focus_on_technical, prompt, safety_settings)

    async def _try_gemini_20(self, patent_data: PatentData, focus_on_technical: bool, prompt: str, safety_settings: list) -> PredictionResult:
        """Gemini 2.0で再試行"""
        try:
            response = self.gemini_model_20.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
                safety_settings=safety_settings
            )

            # 2.0でのブロック確認
            if not response.candidates:
                finish_reason = response.prompt_feedback.block_reason if hasattr(response.prompt_feedback, 'block_reason') else 'Unknown'
                logger.warning(f"Gemini 2.0: プロンプトがブロックされました。理由: {finish_reason}")
                logger.info("OpenAIで再試行します...")
                return await self.predict_with_openai(patent_data, focus_on_technical)

            candidate = response.candidates[0]
            if candidate.finish_reason != 1:  # STOP以外
                logger.warning(f"Gemini 2.0: 応答がブロックされました。理由: {candidate.finish_reason}")
                logger.info("OpenAIで再試行します...")
                return await self.predict_with_openai(patent_data, focus_on_technical)

            # 応答テキストを安全に取得
            if not response.candidates or not response.candidates[0].content.parts:
                logger.warning("Gemini 2.0: 有効な応答内容が返されませんでした")
                logger.info("OpenAIで再試行します...")
                return await self.predict_with_openai(patent_data, focus_on_technical)

            result_text = response.candidates[0].content.parts[0].text
            result_json = json.loads(result_text)

            # キーワードフィルタリング
            context = patent_data.title + " " + patent_data.abstract
            if 'analysis' in result_json:
                if 'application_field_keywords' in result_json['analysis']:
                    result_json['analysis']['application_field_keywords'] = self.keyword_filter.filter_keywords(
                        result_json['analysis']['application_field_keywords'], context
                    )

            return PredictionResult(
                predicted_theme_code=result_json.get('predicted_theme_code', []),
                predicted_fi=result_json.get('predicted_fi', []),
                predicted_fterm=result_json.get('predicted_fterm', []),
                analysis=result_json.get('analysis', {}),
                theme_keywords=result_json.get('theme_keywords', []),
                fi_keywords=result_json.get('fi_keywords', []),
                model_name="Gemini_2.0_Flash_Fallback",  # フォールバックであることを明示
                fi_prediction_method="technical" if focus_on_technical else "application"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Gemini 2.0 JSON解析エラー: {e}")
            logger.info("OpenAIで再試行します...")
            return await self.predict_with_openai(patent_data, focus_on_technical)
        except Exception as e:
            logger.error(f"Gemini 2.0推測エラー: {e}")
            logger.info("OpenAIで再試行します...")
            return await self.predict_with_openai(patent_data, focus_on_technical)

    def _calculate_keyword_relevance(self, keywords: List[str], title: str) -> float:
        """キーワードリストとタイトルの関連度を計算"""
        if not keywords or not title:
            return 0.0

        title_lower = title.lower()
        match_count = 0

        for keyword in keywords:
            if keyword.lower() in title_lower:
                match_count += 1

        return match_count / len(keywords) if keywords else 0.0

    async def predict_with_both_approaches(self, patent_data: PatentData, use_gemini: bool = True) -> PredictionResult:
        """両方のアプローチ（適用分野重視と技術分野重視）でFI予測を行い、より良い結果を返す - API呼び出し最適化版"""

        if use_gemini:
            # 🚀 コスト最適化: 1回のAPI呼び出しで両方の結果を取得
            result = await self.predict_with_gemini_unified(patent_data)
        else:
            # OpenAI版は既存ロジック維持（使用頻度が低いため）
            app_result = await self.predict_with_openai(patent_data, focus_on_technical=False)
            tech_result = await self.predict_with_openai(patent_data, focus_on_technical=True)
            result = self._select_best_approach_from_dual_results(app_result, tech_result, patent_data)

        return result

    async def predict_with_gemini_unified(self, patent_data: PatentData) -> PredictionResult:
        """
        統合Gemini予測: 1回のAPI呼び出しで適用分野・技術分野両方の分析を実行
        コスト削減のため重複API呼び出しを統合
        """
        # 最大限に安全設定を緩和
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]

        # 統合プロンプト作成
        unified_prompt = self.create_unified_prompt(patent_data)

        # まず2.5で試行
        try:
            response = self.gemini_model_25.generate_content(
                unified_prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
                safety_settings=safety_settings
            )

            # ブロック確認
            if not response.candidates or response.candidates[0].finish_reason != 1:
                logger.warning("Gemini 2.5がブロックされました。2.0で再試行...")
                return await self._try_gemini_20_unified(patent_data, unified_prompt, safety_settings)

            # 統合レスポンス解析
            return self._parse_unified_response(response.text, patent_data)

        except Exception as e:
            logger.warning(f"Gemini 2.5エラー: {e}。2.0で再試行...")
            return await self._try_gemini_20_unified(patent_data, unified_prompt, safety_settings)

    def _select_best_approach_from_dual_results(self, app_result: PredictionResult, tech_result: PredictionResult, patent_data: PatentData) -> PredictionResult:
        """2つの個別結果から最適なアプローチを選択（OpenAI用）"""
        if not app_result or not tech_result:
            return app_result or tech_result

        # タイトルとの類似度で判断
        app_keywords = app_result.analysis.get('application_field_keywords', [])
        tech_keywords = tech_result.analysis.get('technical_field_keywords', [])

        app_relevance = self._calculate_keyword_relevance(app_keywords, patent_data.title)
        tech_relevance = self._calculate_keyword_relevance(tech_keywords, patent_data.title)

        logger.debug(f"適用分野キーワードのタイトル関連度: {app_relevance:.2%}, 技術分野キーワードのタイトル関連度: {tech_relevance:.2%}")

        # 選択ロジック
        if tech_relevance > app_relevance and (tech_relevance - app_relevance) > 0.2:
            best_result = tech_result
            best_result.fi_prediction_method = "technical_field_focus"
            logger.info(f"technical_field_focus選択: tech={tech_relevance:.2%} vs app={app_relevance:.2%}")
        else:
            best_result = app_result
            best_result.fi_prediction_method = "application_field_focus"
            logger.info(f"application_field_focus選択（優先）: tech={tech_relevance:.2%} vs app={app_relevance:.2%}")

        best_result._alternative_result = tech_result if tech_relevance > app_relevance else app_result
        return best_result

    def _parse_unified_response(self, response_text: str, patent_data: PatentData) -> PredictionResult:
        """統合レスポンスを解析して最適なアプローチを選択"""
        try:
            data = json.loads(response_text)

            # 推奨アプローチを確認
            recommended = data.get("recommended_approach", "application_focus")

            # 両方の結果を取得
            app_data = data.get("application_focus", {})
            tech_data = data.get("technical_focus", {})

            # 推奨に基づいて主要結果を選択
            if recommended == "technical_focus" and tech_data:
                primary_data = tech_data
                method = "technical_field_focus"
                logger.info("統合レスポンス: technical_field_focus推奨")
            else:
                primary_data = app_data
                method = "application_field_focus"
                logger.info("統合レスポンス: application_field_focus推奨")

            # PredictionResult作成
            result = PredictionResult(
                predicted_theme_code=primary_data.get("predicted_theme_code", []),
                predicted_fi=primary_data.get("predicted_fi", []),
                predicted_fterm=primary_data.get("predicted_fterm", []),
                analysis={
                    "application_field_keywords": app_data.get("application_field_keywords", []),
                    "technical_field_keywords": tech_data.get("technical_field_keywords", []),
                    "confidence": primary_data.get("confidence", 0.8),
                    "reasoning": primary_data.get("reasoning", ""),
                    "overall_confidence": data.get("overall_confidence", 0.8),
                    "unified_response": True  # 統合レスポンスであることを示す
                },
                theme_keywords=[],
                fi_keywords=[],
                model_name="gemini-2.5-flash-unified",
                fi_prediction_method=method
            )

            # 代替結果も保存
            alt_data = tech_data if recommended != "technical_focus" else app_data
            if alt_data:
                alt_result = PredictionResult(
                    predicted_theme_code=alt_data.get("predicted_theme_code", []),
                    predicted_fi=alt_data.get("predicted_fi", []),
                    predicted_fterm=alt_data.get("predicted_fterm", []),
                    analysis={
                        "application_field_keywords": alt_data.get("application_field_keywords", []),
                        "technical_field_keywords": alt_data.get("technical_field_keywords", []),
                        "confidence": alt_data.get("confidence", 0.8)
                    },
                    theme_keywords=[],
                    fi_keywords=[],
                    model_name="gemini-2.5-flash-unified-alt"
                )
                result._alternative_result = alt_result

            return result

        except json.JSONDecodeError as e:
            logger.error(f"統合レスポンス解析エラー: {e}")
            # フォールバック: 従来方式で再試行（非同期でない形式）
            logger.warning("統合解析に失敗したため、application_focusで結果を生成します")
            return PredictionResult(
                predicted_theme_code=[],
                predicted_fi=[],
                predicted_fterm=[],
                analysis={"error": "JSON解析エラー"},
                theme_keywords=[],
                fi_keywords=[],
                model_name="gemini-unified-fallback",
                fi_prediction_method="application_field_focus"
            )

    async def _try_gemini_20_unified(self, patent_data: PatentData, prompt: str, safety_settings: list) -> PredictionResult:
        """Gemini 2.0での統合予測再試行"""
        try:
            response = self.gemini_model_20.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
                safety_settings=safety_settings
            )

            if not response.candidates or response.candidates[0].finish_reason != 1:
                logger.warning("Gemini 2.0も失敗。OpenAIで再試行...")
                return await self.predict_with_openai(patent_data, focus_on_technical=False)

            return self._parse_unified_response(response.text, patent_data)

        except Exception as e:
            logger.error(f"Gemini 2.0統合予測エラー: {e}。OpenAIにフォールバック...")
            return await self.predict_with_openai(patent_data, focus_on_technical=False)

    def _count_fi_matches(self, predicted: List[str], correct: List[str]) -> int:
        """FIコードのマッチ数を数える（前方一致）"""
        if not correct:
            return 0

        predicted_normalized = {self._normalize_fi_code(c) for c in predicted}
        correct_normalized = {self._normalize_fi_code(c) for c in correct}

        matches = 0
        for c_code in correct_normalized:
            if any(c_code.startswith(p_code) for p_code in predicted_normalized):
                matches += 1

        return matches

    def _normalize_fi_code(self, code: str) -> str:
        """FIコードから記号やスペースを除去し、スラッシュ前の部分のみを大文字で返す"""
        code_parts = str(code).split('/')
        main_part = code_parts[0] if code_parts else code
        return re.sub(r'[^A-Z0-9]', '', main_part.upper())


    # extract_patent_core_concept_llm メソッドは削除
    # メインプロンプトに統合されたため不要

    def _empty_result(self, model_name: str) -> PredictionResult:
        """エラー時の空結果を返す"""
        return PredictionResult([], [], [], {}, [], [], [], model_name)


class AccuracyCalculator:
    """精度計算クラス"""

    @staticmethod
    def _normalize_fi_code(code: str) -> str:
        """FIコードから記号やスペースを除去し、スラッシュ前の部分のみを大文字で返す"""
        code_parts = str(code).split('/')
        main_part = code_parts[0] if code_parts else code
        return re.sub(r'[^A-Z0-9]', '', main_part.upper())

    @staticmethod
    def calculate_accuracy(predicted: List[str], correct: List[str], code_type: Optional[str] = None) -> Tuple[int, int, float]:
        """精度を計算（ヒット数, 総数, 正解率）"""
        if not correct:
            return 0, 0, 0.0

        # FIコードの判定ロジック（前方一致）
        if code_type == 'fi':
            predicted_normalized = {AccuracyCalculator._normalize_fi_code(c) for c in predicted}
            correct_normalized = {AccuracyCalculator._normalize_fi_code(c) for c in correct}

            hits = 0
            for c_code in correct_normalized:
                if any(c_code.startswith(p_code) for p_code in predicted_normalized):
                    hits += 1

            total = len(correct_normalized)
            accuracy = hits / total if total > 0 else 0.0
            return hits, total, accuracy

        # FI以外は従来のセットでの完全一致判定
        else:
            predicted_processed = set(predicted)
            correct_processed = set(correct)
            hits = len(predicted_processed & correct_processed)
            total = len(correct_processed)
            accuracy = hits / total if total > 0 else 0.0
            return hits, total, accuracy


class ResultExporter:
    """結果出力クラス"""
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_to_csv(self, results: List[Dict], filename_prefix: str = "patent_classification_results"):
        """結果をCSVファイルに出力"""
        filename = f"{filename_prefix}_{self.timestamp}.csv"
        try:
            df = pd.DataFrame(results)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"結果をCSVファイルに出力しました: {filename}")
            return filename
        except Exception as e:
            logger.error(f"CSV出力エラー: {e}")
            return None


def extract_core_tech_terms_from_title(title: str) -> List[str]:
    """
    特許タイトルから核心となる主役（技術・製品・具体的機能）を抽出
    テーマコード粒度に合う具体性で、製品名と動作/機能を組み合わせて抽出
    """
    if not title:
        return []

    # テーマコード粒度に適した語彙（製品＋動作/機能の組み合わせを意識）
    core_vocabulary = {
        # 基本技術要素
        'システム', '装置', '方法', '技術', '機器', '設備',
        # 制御・処理
        '制御', '処理', '解析', '検出', '測定', '監視', '判定', '生成', '変換',
        # 通信・情報
        '通信', 'ネットワーク', 'データ', '情報', '信号', '画像', '音声',
        # 機械・構造
        '機構', '構造', 'ユニット', '部材', '要素', 'モジュール', 'パーツ',
        # 電子・電気
        '回路', '電源', '電極', 'センサー', 'アクチュエータ', 'ディスプレイ',
        # 材料・化学
        '材料', '化合物', '組成物', '溶液', '薬剤', 'ポリマー',
        # 製品・装置（モノ）
        '掃除機', '自動車', 'エンジン', 'モータ', '冷蔵庫', 'エアコン', 'ロボット', 'ドローン',
        'カメラ', 'プリンタ', '洗濯機', 'テレビ', 'スマートフォン', 'パソコン', 'タブレット',
        # 具体的な用途・機器（広すぎる分野は除外）
        '遊技機', 'パチンコ', 'ゲーム機', '工作機械', '建設機械', '医療機器', '検査装置',
        # 動作・機能（テーマコードでよく使われる）
        '清掃', '吸引', '冷却', '加熱', '洗浄', '乾燥', '切断', '研磨',
        '駆動', '作動', '動作', '機能', '性能', '効果', '改善', '向上'
    }

    # タイトルを解析して主役を抽出
    extracted_terms = []
    words = title.replace('、', ' ').replace('の', ' ').split()

    for word in words:
        # 完全一致する重要語彙
        if word in core_vocabulary:
            extracted_terms.append(word)
        # 重要語彙を含む複合語
        else:
            for core_term in core_vocabulary:
                if core_term in word and len(word) > len(core_term):
                    extracted_terms.append(core_term)
                    break

    # 重複除去と優先順位付け
    unique_terms = []
    seen = set()
    for term in extracted_terms:
        if term not in seen:
            unique_terms.append(term)
            seen.add(term)

    # 製品と動作の組み合わせを検出（テーマコード検索に有効）
    products = {'掃除機', '自動車', 'エンジン', 'モータ', '冷蔵庫', 'エアコン', 'ロボット',
                'カメラ', 'プリンタ', '洗濯機', 'テレビ', 'スマートフォン', '遊技機',
                '工作機械', '建設機械', '医療機器', '検査装置'}

    actions = {'清掃', '吸引', '冷却', '加熱', '洗浄', '乾燥', '切断', '研磨',
               '駆動', '制御', '検出', '測定', '表示', '通信', '処理'}

    found_products = [t for t in unique_terms if t in products]
    found_actions = [t for t in unique_terms if t in actions]
    found_others = [t for t in unique_terms if t not in products and t not in actions]

    # 製品＋動作の組み合わせを優先（テーマコードに最適）
    result = []
    if found_products:
        result.extend(found_products[:1])  # 主要製品1個
    if found_actions:
        result.extend(found_actions[:1])   # 主要動作1個
    if len(result) < 3 and found_others:
        result.extend(found_others[:3-len(result)])  # 他の用語で補完

    # もし何も見つからなかった場合の優先順位リスト
    if not result:
        priority_order = [
            '掃除機', '清掃', '吸引', '自動車', 'エンジン', '制御',
            'システム', '装置', '方法', 'センサー', 'ディスプレイ'
        ]

        prioritized = []
        # 優先順位の高い用語から選択
        for priority_term in priority_order:
            if priority_term in unique_terms:
                prioritized.append(priority_term)
                if len(prioritized) >= 3:
                    break

        # 優先リストにない用語も追加（最大3個まで）
        for term in unique_terms:
            if term not in prioritized and len(prioritized) < 3:
                prioritized.append(term)

        return prioritized[:3]  # 最大3個に制限
    else:
        return result[:3]  # 製品＋動作の組み合わせ結果を返す


def find_common_fi_codes(all_predictions: Dict[str, List[str]], min_methods: int = 2) -> List[str]:
    """複数の予測手法から共通するFIコードを見つける"""
    if len(all_predictions) < min_methods:
        return []

    # 各手法のFIコードを正規化
    normalized_predictions = {}
    for method, codes in all_predictions.items():
        normalized_predictions[method] = [
            re.sub(r'[^A-Z0-9]', '', str(code).split('/')[0].upper())
            for code in codes
        ]

    # 共通するコードを見つける
    common_codes = []
    all_codes = set()
    for codes in normalized_predictions.values():
        all_codes.update(codes)

    for code in all_codes:
        count = sum(1 for method_codes in normalized_predictions.values() if code in method_codes)
        if count >= min_methods:
            common_codes.append(code)

    return common_codes


def filter_fterms_by_theme_codes(fterms: List[str], theme_codes: List[str]) -> List[str]:
    """テーマコードに基づいてFtermをフィルタリング（最初5桁の一致チェック）

    Args:
        fterms: フィルタリング対象のFtermリスト
        theme_codes: 基準となるテーマコードリスト

    Returns:
        テーマコードと一致するFtermのリスト
    """
    if not theme_codes or not fterms:
        return fterms

    # テーマコードから最初5桁を抽出
    theme_prefixes = set()
    for theme_code in theme_codes:
        # テーマコードの正規化（数字と英字のみ）
        normalized_theme = re.sub(r'[^A-Z0-9]', '', str(theme_code).upper())
        if len(normalized_theme) >= 5:
            theme_prefixes.add(normalized_theme[:5])

    # Ftermから一致するものを抽出
    filtered_fterms = []
    for fterm in fterms:
        # Ftermの正規化
        normalized_fterm = re.sub(r'[^A-Z0-9]', '', str(fterm).upper())
        if len(normalized_fterm) >= 5:
            fterm_prefix = normalized_fterm[:5]
            if fterm_prefix in theme_prefixes:
                filtered_fterms.append(fterm)

    # フィルタリングした結果が空の場合は元のリストを返す（厳密すぎる制約を避ける）
    return filtered_fterms if filtered_fterms else fterms


def get_consistent_theme_fterm_pair(theme_codes: List[str], fterms: List[str]) -> Tuple[List[str], List[str]]:
    """テーマコードとFtermの整合性を保った組み合わせを取得

    Args:
        theme_codes: 候補となるテーマコードリスト
        fterms: 候補となるFtermリスト

    Returns:
        (整合性のあるテーマコード, 対応するFterm)のタプル
    """
    if not theme_codes:
        return [], fterms
    if not fterms:
        return theme_codes, []

    # 各テーマコードに対して一致するFtermを探す
    consistent_pairs = []

    for theme_code in theme_codes:
        normalized_theme = re.sub(r'[^A-Z0-9]', '', str(theme_code).upper())
        if len(normalized_theme) >= 5:
            theme_prefix = normalized_theme[:5]

            # このテーマコードに一致するFtermを探す
            matching_fterms = []
            for fterm in fterms:
                normalized_fterm = re.sub(r'[^A-Z0-9]', '', str(fterm).upper())
                if len(normalized_fterm) >= 5 and normalized_fterm[:5] == theme_prefix:
                    matching_fterms.append(fterm)

            if matching_fterms:
                consistent_pairs.append((theme_code, matching_fterms))

    # 結果を構築
    if consistent_pairs:
        # 整合性のあるペアが見つかった場合
        final_themes = [pair[0] for pair in consistent_pairs]
        final_fterms = []
        for _, fterm_list in consistent_pairs:
            final_fterms.extend(fterm_list)
        return final_themes, final_fterms
    else:
        # 整合性のあるペアが見つからない場合は、最初のテーマコードを保持し、Ftermは空にする
        logger.warning(f"テーマコードとFtermの整合性が取れませんでした: themes={theme_codes[:2]}, fterms={fterms[:2]}")
        return theme_codes[:1], []  # 最初のテーマコードのみ保持


def calculate_synergy_confidence(theme_codes: List[str], fterms: List[str],
                                vector_themes: List[str] = None, vector_fterms: List[str] = None) -> float:
    """テーマコードとFtermの相乗効果に基づく信頼度スコアを計算

    Args:
        theme_codes: LLM予測のテーマコード
        fterms: LLM予測のFterm
        vector_themes: ベクトル検索のテーマコード
        vector_fterms: ベクトル検索のFterm

    Returns:
        信頼度スコア (0.0-1.0)
    """
    if not theme_codes and not fterms:
        return 0.0

    confidence = 0.0

    # 1. LLM内でのテーマコードとFtermの整合性 (40%)
    if theme_codes and fterms:
        consistent_themes, consistent_fterms = get_consistent_theme_fterm_pair(theme_codes, fterms)
        if consistent_themes and consistent_fterms:
            consistency_ratio = len(consistent_themes) / len(theme_codes)
            confidence += 0.4 * consistency_ratio

    # 2. ベクトル検索とLLMのテーマコード一致 (30%)
    if vector_themes and theme_codes:
        # 最初5桁での一致チェック
        llm_prefixes = set()
        for theme in theme_codes:
            normalized = re.sub(r'[^A-Z0-9]', '', str(theme).upper())
            if len(normalized) >= 5:
                llm_prefixes.add(normalized[:5])

        vector_prefixes = set()
        for theme in vector_themes:
            normalized = re.sub(r'[^A-Z0-9]', '', str(theme).upper())
            if len(normalized) >= 5:
                vector_prefixes.add(normalized[:5])

        if llm_prefixes and vector_prefixes:
            overlap_ratio = len(llm_prefixes & vector_prefixes) / len(llm_prefixes | vector_prefixes)
            confidence += 0.3 * overlap_ratio

    # 3. ベクトル検索とLLMのFterm一致 (20%)
    if vector_fterms and fterms:
        # 最初5桁での一致チェック
        llm_fterm_prefixes = set()
        for fterm in fterms:
            normalized = re.sub(r'[^A-Z0-9]', '', str(fterm).upper())
            if len(normalized) >= 5:
                llm_fterm_prefixes.add(normalized[:5])

        vector_fterm_prefixes = set()
        for fterm in vector_fterms:
            normalized = re.sub(r'[^A-Z0-9]', '', str(fterm).upper())
            if len(normalized) >= 5:
                vector_fterm_prefixes.add(normalized[:5])

        if llm_fterm_prefixes and vector_fterm_prefixes:
            fterm_overlap_ratio = len(llm_fterm_prefixes & vector_fterm_prefixes) / len(llm_fterm_prefixes | vector_fterm_prefixes)
            confidence += 0.2 * fterm_overlap_ratio

    # 4. ボーナス: ベクトル検索結果同士の整合性 (10%)
    if vector_themes and vector_fterms:
        vector_consistent_themes, vector_consistent_fterms = get_consistent_theme_fterm_pair(vector_themes, vector_fterms)
        if vector_consistent_themes and vector_consistent_fterms:
            vector_consistency = len(vector_consistent_themes) / len(vector_themes) if vector_themes else 0
            confidence += 0.1 * vector_consistency

    return min(1.0, confidence)


def enhance_predictions_with_practical_consensus(all_theme_predictions: Dict[str, List[str]]) -> Tuple[List[str], List[str], float]:
    """実用的なConsensusロジック（正解データ不要）

    手法の事前重み付け + 最頻値ベース:
    1. 各手法に事前定義された重みを適用
    2. 重み付き投票でConsensusを決定
    3. 正解データは一切使用しない

    Args:
        all_theme_predictions: 各手法の予測結果 {'method_name': [theme_codes]}

    Returns:
        (最適化されたテーマコード, 最適化されたFterm, 信頼度スコア)
    """
    start_time = time.time()

    # 1. 各手法の事前定義重み（経験的/統計的に決定）
    method_weights = {
        'Gemini_Unified': 0.6,                    # Gemini統合結果
        'Keyword_Based_Theme_Search': 0.5,        # キーワードベース検索
        'Vector_Search_Theme_Application': 0.4,   # 既存Vector_Search_Theme系
        'Vector_Search_Theme_Technical': 0.4,     # 既存Vector_Search_Theme系
        'Vector_Search_Application': 0.3,         # 新追加：アプリケーション視点
        'Vector_Search_Technical': 0.3,           # 新追加：技術視点
        'Vector_Search_Title': 0.2,               # 新追加：タイトルベース
        'Vector_Search_Theme_Title': 0.2,         # 既存（最低重み）
    }

    # 2. 重み付き投票カウント
    weighted_votes = {}
    total_weight = 0.0

    for method, predictions in all_theme_predictions.items():
        if predictions:
            weight = method_weights.get(method, 0.1)  # 未知手法はデフォルト重み0.1
            total_weight += weight

            for theme_code in predictions:
                if theme_code in weighted_votes:
                    weighted_votes[theme_code] += weight
                else:
                    weighted_votes[theme_code] = weight

    # 3. 重み付きスコア順でConsensus決定
    final_themes = []
    confidence_score = 0.0

    if weighted_votes:
        # スコア順にソート
        sorted_themes = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)

        # 上位3個を採用
        final_themes = [theme for theme, score in sorted_themes[:3]]

        # 信頼度計算（最高スコア / 総重み）
        max_score = sorted_themes[0][1]
        confidence_score = min(1.0, max_score / total_weight) if total_weight > 0 else 0.0

        logger.info(f"重み付きConsensus: {final_themes} (最高スコア: {max_score:.3f})")
        logger.debug(f"全スコア: {dict(sorted_themes)}")

    else:
        # 予測結果がない場合
        logger.warning("全手法で予測結果なし")
        confidence_score = 0.0

    # 4. Ftermは空で返す（テーマコード重視）
    final_fterms = []

    processing_time = time.time() - start_time
    logger.debug(f"実用Consensus処理完了: 時間={processing_time:.2f}秒, 信頼度={confidence_score:.3f}")

    return final_themes, final_fterms, confidence_score


def enhance_predictions_with_practical_approach(llm_theme_codes: List[str], llm_fterms: List[str],
                                               vector_theme_codes: List[str], vector_fterms: List[str]) -> Tuple[List[str], List[str], float]:
    """実用的なアプローチで予測結果を最適化（従来版・フォールバック用）

    ポリシー:
    1. テーマコードはLLM結果を優先し、ベクトルで補完
    2. Ftermはテーマコードとの整合性よりも各手法の結果を優先
    3. 空になるよりは予測結果を保持する

    Args:
        llm_theme_codes: LLM予測のテーマコード
        llm_fterms: LLM予測のFterm
        vector_theme_codes: ベクトル検索のテーマコード
        vector_fterms: ベクトル検索のFterm

    Returns:
        (最適化されたテーマコード, 最適化されたFterm, 信頼度スコア)
    """
    start_time = time.time()

    # 1. テーマコードの決定（LLM優先、ベクトルで補完）
    final_themes = []

    # LLM結果をベースにする
    if llm_theme_codes:
        final_themes.extend(llm_theme_codes)

        # ベクトル結果で補完（重複しないもののみ）
        if vector_theme_codes:
            existing_prefixes = set()
            for theme in final_themes:
                prefix = re.sub(r'[^A-Z0-9]', '', str(theme).upper())[:5]
                if len(prefix) >= 5:
                    existing_prefixes.add(prefix)

            # ベクトル結果の上位2個をチェック
            for vector_theme in vector_theme_codes[:2]:
                vector_prefix = re.sub(r'[^A-Z0-9]', '', str(vector_theme).upper())[:5]
                if len(vector_prefix) >= 5 and vector_prefix not in existing_prefixes:
                    final_themes.append(vector_theme)
                    existing_prefixes.add(vector_prefix)
    else:
        # LLM結果がない場合はベクトル結果を使用
        if vector_theme_codes:
            final_themes.extend(vector_theme_codes[:2])

    # 2. Ftermの決定（異なるアプローチ：各手法の結果を組み合わせ）
    final_fterms = []

    # 2-1. テーマコードと整合性があるFtermを優先的に選択
    theme_prefixes = set()
    for theme in final_themes:
        prefix = re.sub(r'[^A-Z0-9]', '', str(theme).upper())[:5]
        if len(prefix) >= 5:
            theme_prefixes.add(prefix)

    # LLMのFtermから整合性のあるものを選択
    if llm_fterms and theme_prefixes:
        for fterm in llm_fterms:
            fterm_prefix = re.sub(r'[^A-Z0-9]', '', str(fterm).upper())[:5]
            if len(fterm_prefix) >= 5 and fterm_prefix in theme_prefixes:
                final_fterms.append(fterm)

    # ベクトルのFtermから整合性のあるものを追加
    if vector_fterms and theme_prefixes:
        for fterm in vector_fterms:
            fterm_prefix = re.sub(r'[^A-Z0-9]', '', str(fterm).upper())[:5]
            if len(fterm_prefix) >= 5 and fterm_prefix in theme_prefixes:
                if fterm not in final_fterms:  # 重複除去
                    final_fterms.append(fterm)

    # 2-2. 整合性あるFtermが少ない場合は、各手法の上位結果を追加（実用性重視）
    if len(final_fterms) < 3:
        # LLM Ftermの上位3個を追加
        if llm_fterms:
            for fterm in llm_fterms[:3]:
                if fterm and fterm not in final_fterms:
                    final_fterms.append(fterm)

        # まだ不足ならベクトル Ftermを追加
        if len(final_fterms) < 3 and vector_fterms:
            for fterm in vector_fterms[:3]:
                if fterm and fterm not in final_fterms and len(final_fterms) < 5:
                    final_fterms.append(fterm)

    # 3. 信頼度の簡素な計算
    confidence_score = 0.0

    # テーマコードがあるか (50%)
    if final_themes:
        confidence_score += 0.5

    # Ftermがあるか (30%)
    if final_fterms:
        confidence_score += 0.3

    # テーマコードとFtermの整合性 (20%)
    if final_themes and final_fterms:
        consistent_count = 0
        for fterm in final_fterms[:3]:  # 上位3個でチェック
            fterm_prefix = re.sub(r'[^A-Z0-9]', '', str(fterm).upper())[:5]
            if len(fterm_prefix) >= 5 and fterm_prefix in theme_prefixes:
                consistent_count += 1

        if consistent_count > 0:
            consistency_ratio = consistent_count / min(len(final_fterms), 3)
            confidence_score += 0.2 * consistency_ratio

    # 重複除去とクリーンアップ
    final_themes = list(dict.fromkeys([t for t in final_themes if t]))[:3]  # 最大3個
    final_fterms = list(dict.fromkeys([f for f in final_fterms if f]))[:8]   # 最大8個

    processing_time = time.time() - start_time
    logger.debug(f"実用的最適化処理完了: 時間={processing_time:.2f}秒, 信頼度={confidence_score:.3f}")
    logger.debug(f"最適化結果 - テーマ: {final_themes}, Fterm: {final_fterms[:3]}...")

    return final_themes, final_fterms, confidence_score


async def process_patent(patent_data: PatentData, predictor: LLMPredictor, vector_predictor: VectorSearchPredictor, calculator: AccuracyCalculator, debug_mode: bool = False) -> List[Dict]:
    """個別の特許データを処理し、結果を返すコルーチン

    Args:
        patent_data: 処理する特許データ
        predictor: LLM予測器
        vector_predictor: ベクトル検索予測器
        calculator: 精度計算器
        debug_mode: Trueの場合、全ての詳細結果を返す。Falseの場合、consensusのみ返す
    """
    logger.info(f"処理中: {patent_data.title[:50]}... (特許番号: {patent_data.patent_id})")

    # 効率化：複数のLLM呼び出しを並行実行
    # 1. FI予測（既存機能）
    # 2. Fターム用キーワード抽出（新機能）
    gemini_task = predictor.predict_with_both_approaches(patent_data, use_gemini=True)
    # 並行実行して結果を取得
    gemini_result = await gemini_task

    patent_results = []
    # OpenAIの結果はコメントアウト
    # for result in [openai_result, gemini_result]:
    for result in [gemini_result]:
        if not result:
            continue

        theme_hits, theme_total, theme_acc = calculator.calculate_accuracy(
            result.predicted_theme_code, patent_data.correct_theme_code
        )
        fi_hits, fi_total, fi_acc = calculator.calculate_accuracy(
            result.predicted_fi, patent_data.correct_fi, code_type='fi'
        )
        fterm_hits, fterm_total, fterm_acc = calculator.calculate_accuracy(
            result.predicted_fterm, patent_data.correct_fterm
        )

        patent_results.append({
            'patent_id': patent_data.patent_id,  # 特許番号を追加
            'patent_title': patent_data.title,
            'model': result.model_name,
            'fi_prediction_method': result.fi_prediction_method,  # どちらの方法を使ったか
            'predicted_theme_code': '; '.join(result.predicted_theme_code),
            'correct_theme_code': '; '.join(patent_data.correct_theme_code),
            'predicted_fi': '; '.join(result.predicted_fi),
            'correct_fi': '; '.join(patent_data.correct_fi),
            'predicted_fterm': '; '.join(result.predicted_fterm),
            'correct_fterm': '; '.join(patent_data.correct_fterm),
            'theme_accuracy': f"{theme_hits}/{theme_total} ({theme_acc:.2%})",
            'fi_accuracy': f"{fi_hits}/{fi_total} ({fi_acc:.2%})",
            'fterm_accuracy': f"{fterm_hits}/{fterm_total} ({fterm_acc:.2%})",
            'application_field_keywords': '; '.join(result.analysis.get('application_field_keywords', [])),
            'technical_field_keywords': '; '.join(result.analysis.get('technical_field_keywords', [])),
        })

    # ベクトル検索の実行
    if gemini_result:
        # Geminiで抽出されたキーワードを使用してベクトル検索
        app_keywords = gemini_result.analysis.get('application_field_keywords', [])
        tech_keywords = gemini_result.analysis.get('technical_field_keywords', [])

        # LLM強化テーマコード予測を実行（technical_field_keywordsを使用）
        if tech_keywords:
            try:
                # technical_field_keywordsから汎用語を除外して検索クエリを生成
                generic_words = {'装置', '方法', 'システム', '手段', '機構', '構成', '処理', '制御'}
                filtered_keywords = [kw for kw in tech_keywords
                                   if len(kw) >= 3 and kw not in generic_words]
                if not filtered_keywords:  # フィルタ後に空になった場合は元のキーワードを使用
                    filtered_keywords = tech_keywords[:4]
                search_query = ' '.join(filtered_keywords[:4])  # 上位4個の技術的に具体的なキーワードを使用

                # ベクトル検索で上位10件のテーマコード候補を取得（predict_theme_codeの検索部分を利用）
                from theme_code_predictor_with_llm import search_theme_candidates
                theme_candidates = search_theme_candidates(search_query, top_k=10)

                if theme_candidates:
                    # 特許文書の要点（タイトル、要約、請求項1のみ）を準備
                    # claims_mainから最初の請求項を取得
                    first_claim = ""
                    if patent_data.claims_main:
                        first_claim = patent_data.claims_main[0][:500] if patent_data.claims_main[0] else 'なし'
                    else:
                        first_claim = 'なし'

                    patent_summary = f"""タイトル: {patent_data.title}
要約: {patent_data.abstract[:500] if patent_data.abstract else 'なし'}
請求項1: {first_claim}"""

                    # LLMで最適なテーマコードを選択（APIコール最小化）
                    from theme_code_predictor_with_llm import (
                        select_best_theme_code_with_openai,
                    )
                    predicted_theme, llm_reasoning = select_best_theme_code_with_openai(patent_summary, theme_candidates)

                    if predicted_theme:
                        # テーマコード精度計算
                        theme_hits_llm, theme_total_llm, theme_acc_llm = calculator.calculate_accuracy(
                            [predicted_theme], patent_data.correct_theme_code
                        )

                        patent_results.append({
                            'patent_id': patent_data.patent_id,
                            'patent_title': patent_data.title,
                            'model': 'Keyword_Based_Theme_Search',
                            'fi_prediction_method': 'technical_keywords_search',
                            'predicted_theme_code': predicted_theme,
                            'correct_theme_code': '; '.join(patent_data.correct_theme_code),
                            'predicted_fi': '',
                            'correct_fi': '; '.join(patent_data.correct_fi),
                            'predicted_fterm': '',
                            'correct_fterm': '; '.join(patent_data.correct_fterm),
                            'theme_accuracy': f"{theme_hits_llm}/{theme_total_llm} ({theme_acc_llm:.2%})",
                            'fi_accuracy': '0/0 (0.00%)',
                            'fterm_accuracy': '0/0 (0.00%)',
                            'application_field_keywords': '',
                            'technical_field_keywords': search_query,
                            'llm_reasoning': llm_reasoning,
                            'vector_search_score': theme_candidates[0].get('score', 0.0) if theme_candidates else 0.0,
                        })
                        logger.info(f"LLM強化テーマ予測完了: {predicted_theme} (技術キーワード: {search_query})")
                    else:
                        logger.warning("LLM強化テーマ予測: テーマコードの選択に失敗")
                else:
                    logger.warning("LLM強化テーマ予測: 候補テーマコードが見つかりません")
            except Exception as e:
                logger.warning(f"LLM強化テーマ予測でエラー: {e}")

        # Application fieldキーワードでベクトル検索 + テーマコード検索
        if app_keywords:
            try:
                fi_codes_app, fterm_codes_app = vector_predictor.search_fi_fterm(app_keywords)

                # テーマコード検索を追加
                theme_codes_app = []
                theme_hits_app, theme_total_app, theme_acc_app = 0, 0, 0.0

                try:
                    app_query = ' '.join(app_keywords[:4])  # 上位4個のキーワードを使用
                    theme_search_results = vector_predictor.search_theme_codes(app_query, top_k=3)
                    if theme_search_results:
                        theme_codes_app = [result['theme_code'] for result in theme_search_results]
                        theme_hits_app, theme_total_app, theme_acc_app = calculator.calculate_accuracy(
                            theme_codes_app, patent_data.correct_theme_code
                        )
                except Exception as theme_e:
                    logger.warning(f"Application fieldテーマコード検索に失敗: {theme_e}")

                if fi_codes_app or fterm_codes_app or theme_codes_app:
                    # 精度計算
                    fi_hits_app, fi_total_app, fi_acc_app = calculator.calculate_accuracy(
                        fi_codes_app, patent_data.correct_fi, code_type='fi'
                    )
                    fterm_hits_app, fterm_total_app, fterm_acc_app = calculator.calculate_accuracy(
                        fterm_codes_app, patent_data.correct_fterm
                    )

                    patent_results.append({
                        'patent_id': patent_data.patent_id,
                        'patent_title': patent_data.title,
                        'model': 'Vector_Search_Application',
                        'fi_prediction_method': 'vector_search_application_keywords',
                        'predicted_theme_code': '; '.join(theme_codes_app),
                        'correct_theme_code': '; '.join(patent_data.correct_theme_code),
                        'predicted_fi': '; '.join(fi_codes_app),
                        'correct_fi': '; '.join(patent_data.correct_fi),
                        'predicted_fterm': '; '.join(fterm_codes_app),
                        'correct_fterm': '; '.join(patent_data.correct_fterm),
                        'theme_accuracy': f"{theme_hits_app}/{theme_total_app} ({theme_acc_app:.2%})",
                        'fi_accuracy': f"{fi_hits_app}/{fi_total_app} ({fi_acc_app:.2%})",
                        'fterm_accuracy': f"{fterm_hits_app}/{fterm_total_app} ({fterm_acc_app:.2%})",
                        'application_field_keywords': '; '.join(app_keywords),
                        'technical_field_keywords': '',
                    })
            except Exception as e:
                logger.warning(f"Application fieldキーワードのベクトル検索に失敗: {e}")

        # Technical fieldキーワードでベクトル検索 + テーマコード検索
        if tech_keywords:
            try:
                fi_codes_tech, fterm_codes_tech = vector_predictor.search_fi_fterm(tech_keywords)

                # テーマコード検索を追加
                theme_codes_tech = []
                theme_hits_tech, theme_total_tech, theme_acc_tech = 0, 0, 0.0

                try:
                    tech_query = ' '.join(tech_keywords[:4])  # 上位4個のキーワードを使用
                    theme_search_results = vector_predictor.search_theme_codes(tech_query, top_k=3)
                    if theme_search_results:
                        theme_codes_tech = [result['theme_code'] for result in theme_search_results]
                        theme_hits_tech, theme_total_tech, theme_acc_tech = calculator.calculate_accuracy(
                            theme_codes_tech, patent_data.correct_theme_code
                        )
                except Exception as theme_e:
                    logger.warning(f"Technical fieldテーマコード検索に失敗: {theme_e}")

                if fi_codes_tech or fterm_codes_tech or theme_codes_tech:
                    # 精度計算
                    fi_hits_tech, fi_total_tech, fi_acc_tech = calculator.calculate_accuracy(
                        fi_codes_tech, patent_data.correct_fi, code_type='fi'
                    )
                    fterm_hits_tech, fterm_total_tech, fterm_acc_tech = calculator.calculate_accuracy(
                        fterm_codes_tech, patent_data.correct_fterm
                    )

                    patent_results.append({
                        'patent_id': patent_data.patent_id,
                        'patent_title': patent_data.title,
                        'model': 'Vector_Search_Technical',
                        'fi_prediction_method': 'vector_search_technical_keywords',
                        'predicted_theme_code': '; '.join(theme_codes_tech),
                        'correct_theme_code': '; '.join(patent_data.correct_theme_code),
                        'predicted_fi': '; '.join(fi_codes_tech),
                        'correct_fi': '; '.join(patent_data.correct_fi),
                        'predicted_fterm': '; '.join(fterm_codes_tech),
                        'correct_fterm': '; '.join(patent_data.correct_fterm),
                        'theme_accuracy': f"{theme_hits_tech}/{theme_total_tech} ({theme_acc_tech:.2%})",
                        'fi_accuracy': f"{fi_hits_tech}/{fi_total_tech} ({fi_acc_tech:.2%})",
                        'fterm_accuracy': f"{fterm_hits_tech}/{fterm_total_tech} ({fterm_acc_tech:.2%})",
                        'application_field_keywords': '',
                        'technical_field_keywords': '; '.join(tech_keywords),
                    })
            except Exception as e:
                logger.warning(f"Technical fieldキーワードのベクトル検索に失敗: {e}")

        # 特許タイトルでベクトル検索 + テーマコード検索（Single_Applicationの代替）
        try:
            fi_codes_title, fterm_codes_title = vector_predictor.search_fi_fterm([patent_data.title])

            # テーマコード検索を追加
            theme_codes_title = []
            theme_hits_title, theme_total_title, theme_acc_title = 0, 0, 0.0

            try:
                title_query = patent_data.title
                theme_search_results = vector_predictor.search_theme_codes(title_query, top_k=3)
                if theme_search_results:
                    theme_codes_title = [result['theme_code'] for result in theme_search_results]
                    theme_hits_title, theme_total_title, theme_acc_title = calculator.calculate_accuracy(
                        theme_codes_title, patent_data.correct_theme_code
                    )
            except Exception as theme_e:
                logger.warning(f"Titleテーマコード検索に失敗: {theme_e}")

            if fi_codes_title or fterm_codes_title or theme_codes_title:
                # 精度計算
                fi_hits_title, fi_total_title, fi_acc_title = calculator.calculate_accuracy(
                    fi_codes_title, patent_data.correct_fi, code_type='fi'
                )
                fterm_hits_title, fterm_total_title, fterm_acc_title = calculator.calculate_accuracy(
                    fterm_codes_title, patent_data.correct_fterm
                )

                patent_results.append({
                    'patent_id': patent_data.patent_id,
                    'patent_title': patent_data.title,
                    'model': 'Vector_Search_Title',
                    'fi_prediction_method': 'vector_search_patent_title',
                    'predicted_theme_code': '; '.join(theme_codes_title),
                    'correct_theme_code': '; '.join(patent_data.correct_theme_code),
                    'predicted_fi': '; '.join(fi_codes_title),
                    'correct_fi': '; '.join(patent_data.correct_fi),
                    'predicted_fterm': '; '.join(fterm_codes_title),
                    'correct_fterm': '; '.join(patent_data.correct_fterm),
                    'theme_accuracy': f"{theme_hits_title}/{theme_total_title} ({theme_acc_title:.2%})",
                    'fi_accuracy': f"{fi_hits_title}/{fi_total_title} ({fi_acc_title:.2%})",
                    'fterm_accuracy': f"{fterm_hits_title}/{fterm_total_title} ({fterm_acc_title:.2%})",
                    'application_field_keywords': patent_data.title,
                    'technical_field_keywords': '',
                })
        except Exception as e:
            logger.warning(f"特許タイトルベクトル検索に失敗: {e}")


    # テーマコードベース検索（統合されたアプローチ）
    try:
        # アプローチ1: タイトル解析による技術用語抽出
        title_based_terms = extract_core_tech_terms_from_title(patent_data.title)

        # アプローチ2: Gemini結果から主役を取得（統合プロンプトから）
        llm_based_terms = []
        if gemini_result and 'core_concept' in gemini_result.analysis:
            core_concept = gemini_result.analysis.get('core_concept', '')
            if core_concept:
                llm_based_terms = [term.strip() for term in core_concept.split(',') if term.strip()]

        # 両方のアプローチをテスト
        approaches_to_test = [
            ('Title_Analysis', title_based_terms),
            ('LLM_Integration', llm_based_terms),
        ]

        # 各アプローチでテーマコード検索を実行
        for approach_name, search_terms in approaches_to_test:
            if not search_terms:
                continue

            search_query = ' '.join(search_terms)
            logger.debug(f"{approach_name}による検索クエリ: {search_query}")

            try:
                similar_themes = vector_predictor.search_theme_codes(search_query, top_k=3)

                if similar_themes:
                    predicted_theme_codes = [theme['theme_code'] for theme in similar_themes]
                    theme_hits_similarity, theme_total_similarity, theme_acc_similarity = calculator.calculate_accuracy(
                        predicted_theme_codes, patent_data.correct_theme_code
                    )

                    theme_results_str = []
                    for theme in similar_themes:
                        if 'similarity_score' in theme:
                            theme_results_str.append(f"{theme['theme_code']}(スコア:{theme['similarity_score']:.3f})")
                        else:
                            theme_results_str.append(f"{theme['theme_code']}({theme['description'][:20]}...)")

                    patent_results.append({
                        'patent_id': patent_data.patent_id,
                        'patent_title': patent_data.title,
                        'model': f'Vector_Search_Theme_{approach_name}',
                        'fi_prediction_method': f'vector_search_theme_{approach_name.lower()}',
                        'predicted_theme_code': '; '.join(predicted_theme_codes),
                        'correct_theme_code': '; '.join(patent_data.correct_theme_code),
                        'predicted_fi': '',
                        'correct_fi': '; '.join(patent_data.correct_fi),
                        'predicted_fterm': '',
                        'correct_fterm': '; '.join(patent_data.correct_fterm),
                        'theme_accuracy': f"{theme_hits_similarity}/{theme_total_similarity} ({theme_acc_similarity:.2%})",
                        'fi_accuracy': '0/0 (0.00%)',
                        'fterm_accuracy': '0/0 (0.00%)',
                        'application_field_keywords': f"抽出手法: {approach_name}, 検索クエリ: {search_query}",
                        'technical_field_keywords': f"類似テーマ: {'; '.join(theme_results_str)}",
                    })

                    logger.info(f"{approach_name}テーマコード検索結果: {len(similar_themes)}個のテーマコード, 精度: {theme_acc_similarity:.2%}")
                else:
                    logger.warning(f"{approach_name}: 検索クエリ '{search_query}' に対して結果が見つかりませんでした")
            except Exception as e:
                logger.warning(f"{approach_name}テーマコード検索に失敗: {e}")

        # 既に新しい手法で処理済み - フォールバック処理は削除
    except Exception as e:
        logger.error(f"テーマコードベース検索でエラーが発生: {e}")

    # 複数手法の共通FIコードを探す（常に実行）
    # すべての予測結果を収集
    all_fi_predictions = {}

    # キーワードを保存するための辞書
    app_keywords_used = []
    tech_keywords_used = []

    # Gemini結果（選択されたアプローチ + 代替アプローチ）
    if gemini_result:
        # タイトルとの類似度で選ばれた方
        method_name = 'Gemini_' + ('technical' if gemini_result.fi_prediction_method == 'technical_field_focus' else 'application')
        all_fi_predictions[method_name] = gemini_result.predicted_fi

        # もう一方のアプローチも含める
        if hasattr(gemini_result, '_alternative_result') and gemini_result._alternative_result:
            alt_method_name = 'Gemini_' + ('application' if method_name.endswith('technical') else 'technical')
            all_fi_predictions[alt_method_name] = gemini_result._alternative_result.predicted_fi

    # ベクトル検索結果を収集
    if 'fi_codes_app' in locals() and fi_codes_app:
        all_fi_predictions['Vector_Application'] = fi_codes_app
        if app_keywords:
            app_keywords_used = app_keywords
    if 'fi_codes_tech' in locals() and fi_codes_tech:
        all_fi_predictions['Vector_Technical'] = fi_codes_tech
        if tech_keywords:
            tech_keywords_used = tech_keywords

    # Vector_Titleは他の結果と比較して最初のアルファベットが異なる場合は除外
    if 'fi_codes_title' in locals() and fi_codes_title:
        # 他のベクトル検索結果と比較
        other_results = []
        if 'fi_codes_app' in locals() and fi_codes_app:
            other_results.extend(fi_codes_app)
        if 'fi_codes_tech' in locals() and fi_codes_tech:
            other_results.extend(fi_codes_tech)

        # タイトル検索結果が他の結果と近いかチェック
        if other_results:
            # 正規化して最初のアルファベットを比較
            title_first_letters = set()
            for code in fi_codes_title:
                normalized = re.sub(r'[^A-Z0-9]', '', str(code).split('/')[0].upper())
                if normalized:
                    title_first_letters.add(normalized[0] if normalized[0].isalpha() else '')

            other_first_letters = set()
            for code in other_results:
                normalized = re.sub(r'[^A-Z0-9]', '', str(code).split('/')[0].upper())
                if normalized:
                    other_first_letters.add(normalized[0] if normalized[0].isalpha() else '')

            # 最初のアルファベットが一致するものがあれば含める
            if title_first_letters & other_first_letters:
                all_fi_predictions['Vector_Title'] = fi_codes_title
                logger.debug(f"Vector_Title included in consensus (matching first letter with other results)")
            else:
                logger.info(f"Vector_Title excluded from consensus (first letter mismatch: {title_first_letters} vs {other_first_letters})")
        else:
            # 他の結果がない場合は含めない（比較対象がないため）
            logger.debug(f"Vector_Title excluded from consensus (no other results to compare)")

    # 共通するFIコードを見つける（2つ以上の手法で一致するもの）
    common_fi_codes = find_common_fi_codes(all_fi_predictions, min_methods=2)

    # 改良されたConsensus結果を構築
    predicted_theme_codes = []
    predicted_fterms = []
    synergy_confidence = 0.0

    # 各手法のテーマコード予測を収集
    all_theme_predictions = {}

    # 1. Gemini統合結果
    if gemini_result and gemini_result.predicted_theme_code:
        all_theme_predictions['Gemini_Unified'] = gemini_result.predicted_theme_code

    # 2. すべてのテーマコード予測手法結果を収集
    for result in patent_results:
        model_name = result.get('model', '')
        if result.get('predicted_theme_code'):
            theme_codes = result['predicted_theme_code'].split('; ')
            clean_codes = [tc.strip() for tc in theme_codes if tc.strip()]
            if clean_codes:
                # テーマコード予測を行う全手法を含める
                if (model_name.startswith('Vector_Search_Theme') or
                    model_name.startswith('Vector_Search_') or  # Vector_Search_Application, Technical, Title
                    model_name == 'Keyword_Based_Theme_Search'):
                    all_theme_predictions[model_name] = clean_codes

    # 改良されたConsensusロジックを適用
    try:
        # 実用的Consensusを適用（正解データ不要）
        enhanced_themes, enhanced_fterms, synergy_confidence = enhance_predictions_with_practical_consensus(
            all_theme_predictions
        )

        predicted_theme_codes = enhanced_themes
        predicted_fterms = enhanced_fterms

        logger.info(f"改良Consensus適用: {len(all_theme_predictions)}手法 → {predicted_theme_codes}")

    except Exception as e:
        logger.warning(f"改良Consensusでエラー、フォールバック: {e}")

        # フォールバック: 従来の実用的アプローチ
        vector_theme_results = []

        # テーマコードベースの検索結果から抽出
        for result in patent_results:
            if result.get('model', '').startswith('Vector_Search_Theme'):
                if result.get('predicted_theme_code'):
                    theme_codes = result['predicted_theme_code'].split('; ')
                    vector_theme_results.extend([tc.strip() for tc in theme_codes if tc.strip()])

        if gemini_result:
            base_theme_codes = gemini_result.predicted_theme_code if gemini_result.predicted_theme_code else []
            base_fterms = gemini_result.predicted_fterm if gemini_result.predicted_fterm else []

            enhanced_themes, enhanced_fterms, synergy_confidence = enhance_predictions_with_practical_approach(
                base_theme_codes,
                base_fterms,
                vector_theme_results,
                []  # vector_ftermsは空
            )
            predicted_theme_codes = enhanced_themes
            predicted_fterms = enhanced_fterms

    if common_fi_codes:
        logger.info(f"共通FIコード発見: {common_fi_codes}")
        logger.info(f"実用的最適化結果 - テーマ: {predicted_theme_codes}, Fterm: {predicted_fterms[:3]}..., 信頼度: {synergy_confidence:.3f}")

        # 精度計算（正解データがある場合のみ）
        fi_hits_consensus = 0
        fi_total_consensus = 0
        fi_acc_consensus = 0.0
        theme_hits_consensus = 0
        theme_total_consensus = 0
        theme_acc_consensus = 0.0
        fterm_hits_consensus = 0
        fterm_total_consensus = 0
        fterm_acc_consensus = 0.0

        if patent_data.correct_fi and len(patent_data.correct_fi) > 0:
            fi_hits_consensus, fi_total_consensus, fi_acc_consensus = calculator.calculate_accuracy(
                common_fi_codes, patent_data.correct_fi, code_type='fi'
            )

        if patent_data.correct_theme_code and len(patent_data.correct_theme_code) > 0:
            theme_hits_consensus, theme_total_consensus, theme_acc_consensus = calculator.calculate_accuracy(
                predicted_theme_codes, patent_data.correct_theme_code
            )

        if patent_data.correct_fterm and len(patent_data.correct_fterm) > 0:
            fterm_hits_consensus, fterm_total_consensus, fterm_acc_consensus = calculator.calculate_accuracy(
                predicted_fterms, patent_data.correct_fterm
            )

        # consensus結果を構築（複数手法の総意）
        consensus_result = {
            'patent_id': patent_data.patent_id,
            'patent_title': patent_data.title,
            'model': 'Consensus_Method',
            'fi_prediction_method': f'consensus_{len([m for m in all_fi_predictions.values() if any(code in [re.sub(r"[^A-Z0-9]", "", str(c).split("/")[0].upper()) for c in m] for code in common_fi_codes)])}methods',
            'predicted_theme_code': '; '.join(predicted_theme_codes),
            'correct_theme_code': '; '.join(patent_data.correct_theme_code),
            'predicted_fi': '; '.join(common_fi_codes),
            'correct_fi': '; '.join(patent_data.correct_fi),
            'predicted_fterm': '; '.join(predicted_fterms),
            'correct_fterm': '; '.join(patent_data.correct_fterm),
            'theme_accuracy': f"{theme_hits_consensus}/{theme_total_consensus} ({theme_acc_consensus:.2%})" if theme_total_consensus > 0 else '0/0 (0.00%)',
            'fi_accuracy': f"{fi_hits_consensus}/{fi_total_consensus} ({fi_acc_consensus:.2%})" if fi_total_consensus > 0 else '0/0 (0.00%)',
            'fterm_accuracy': f"{fterm_hits_consensus}/{fterm_total_consensus} ({fterm_acc_consensus:.2%})" if fterm_total_consensus > 0 else '0/0 (0.00%)',
            'application_field_keywords': '; '.join(app_keywords_used) if app_keywords_used else '',
            'technical_field_keywords': '; '.join(tech_keywords_used) if tech_keywords_used else '',
            'theme_fterm_consistency': 'practical_optimized',
            'synergy_confidence_score': f'{synergy_confidence:.3f}',
        }

        if debug_mode:
            # デバッグモード: 全ての詳細結果とconsensusを含める
            patent_results.append(consensus_result)
        else:
            # 通常モード: consensusのみ返す（効率的な出力）
            return [consensus_result]
    else:
        logger.warning(f"共通FIコードが見つかりませんでした。各手法の予測: {list(all_fi_predictions.keys())}")

        # 共通FIコードが見つからない場合でもconsensus結果を作成
        if not debug_mode:
            # 通常モード: Gemini単体の結果をconsensusとして返す
            if gemini_result:
                # 精度計算
                fi_hits = 0
                fi_total = 0
                fi_acc = 0.0
                theme_hits = 0
                theme_total = 0
                theme_acc = 0.0
                fterm_hits = 0
                fterm_total = 0
                fterm_acc = 0.0

                if patent_data.correct_fi and len(patent_data.correct_fi) > 0:
                    fi_hits, fi_total, fi_acc = calculator.calculate_accuracy(
                        gemini_result.predicted_fi, patent_data.correct_fi, code_type='fi'
                    )

                if patent_data.correct_theme_code and len(patent_data.correct_theme_code) > 0:
                    theme_hits, theme_total, theme_acc = calculator.calculate_accuracy(
                        gemini_result.predicted_theme_code, patent_data.correct_theme_code
                    )

                if patent_data.correct_fterm and len(patent_data.correct_fterm) > 0:
                    # 整合性を保ったFtermを使用して精度計算
                    consistent_fterms = filter_fterms_by_theme_codes(
                        gemini_result.predicted_fterm, gemini_result.predicted_theme_code
                    )
                    fterm_hits, fterm_total, fterm_acc = calculator.calculate_accuracy(
                        consistent_fterms, patent_data.correct_fterm
                    )

                return [{
                    'patent_id': patent_data.patent_id,
                    'patent_title': patent_data.title,
                    'model': 'Consensus_Method',
                    'fi_prediction_method': 'single_gemini_method',
                    'predicted_theme_code': '; '.join(gemini_result.predicted_theme_code),
                    'correct_theme_code': '; '.join(patent_data.correct_theme_code),
                    'predicted_fi': '; '.join(gemini_result.predicted_fi),
                    'correct_fi': '; '.join(patent_data.correct_fi),
                    'predicted_fterm': '; '.join(consistent_fterms),
                    'correct_fterm': '; '.join(patent_data.correct_fterm),
                    'theme_accuracy': f"{theme_hits}/{theme_total} ({theme_acc:.2%})" if theme_total > 0 else '0/0 (0.00%)',
                    'fi_accuracy': f"{fi_hits}/{fi_total} ({fi_acc:.2%})" if fi_total > 0 else '0/0 (0.00%)',
                    'fterm_accuracy': f"{fterm_hits}/{fterm_total} ({fterm_acc:.2%})" if fterm_total > 0 else '0/0 (0.00%)',
                    'application_field_keywords': '; '.join(gemini_result.analysis.get('application_field_keywords', [])),
                    'technical_field_keywords': '; '.join(gemini_result.analysis.get('technical_field_keywords', [])),
                    'theme_fterm_consistency': 'checked' if consistent_fterms != gemini_result.predicted_fterm else 'original',
                }]
            else:
                # Gemini結果もない場合は空のリストを返す
                return []

    return patent_results


async def main(debug_mode: bool = False):
    """メイン処理

    Args:
        debug_mode: Trueの場合、すべての詳細結果をCSVに出力。Falseの場合、consensusのみ出力
    """
    logger.info("特許データ分類コード推測システムを開始します")
    if debug_mode:
        logger.info("デバッグモード: 全ての詳細結果を出力します")
    else:
        logger.info("通常モード: consensusのみ出力します")

    # 初期化
    cosmos_client = CosmosDBClient()
    predictor = LLMPredictor()
    vector_predictor = VectorSearchPredictor(enable_vector_search=True)  # evaluate_fi_fterm_from_csv.pyと同じ方法で実装
    calculator = AccuracyCalculator()
    exporter = ResultExporter()

    # データ取得
    logger.info("CosmosDBから特許データを取得中...")
    patent_data_list = cosmos_client.get_patent_data(limit=5)  # テスト時は件数を絞る（効率化）

    if not patent_data_list:
        logger.error("特許データが取得できませんでした。CosmosDBの接続とデータ構造を確認してください。")
        return

    # 並行処理でタスクを実行（各特許データを非同期で処理）
    tasks = []
    for i, patent_data in enumerate(patent_data_list):
        logger.info(f"特許データ {i+1}/{len(patent_data_list)} の推測タスクを作成中...")
        tasks.append(process_patent(patent_data, predictor, vector_predictor, calculator, debug_mode=debug_mode))

    results_list = await asyncio.gather(*tasks)

    # 結果を平坦化
    results = [item for sublist in results_list for item in sublist]

    # CSV出力
    if results:
        csv_file = exporter.export_to_csv(results)
        if csv_file:
            logger.info(f"処理完了！結果ファイル: {csv_file}")

            # 統計情報を出力
            df = pd.DataFrame(results)
            logger.info("\n=== 統計情報 ===")

            # モデル別のFI精度を表示
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                fi_accuracies = model_df['fi_accuracy'].str.extract(r'\((\d+\.\d+)%\)')[0].astype(float)
                avg_accuracy = fi_accuracies.mean()
                logger.info(f"{model} - 平均FI精度: {avg_accuracy:.2f}%")

                # 予測方法別の統計
                for method in model_df['fi_prediction_method'].unique():
                    method_df = model_df[model_df['fi_prediction_method'] == method]
                    method_count = len(method_df)
                    logger.info(f"  {method}: {method_count}件使用")
        else:
            logger.error("CSV出力に失敗しました")
    else:
        logger.warning("処理結果がありませんでした。")


if __name__ == "__main__":
    import argparse

    # コマンドライン引数のパーサーを設定
    parser = argparse.ArgumentParser(
        description='特許データ分類コード推測システム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  通常実行（consensusのみ）:     python %(prog)s
  デバッグモード（全詳細出力）:   python %(prog)s --debug
        ''')
    parser.add_argument('--debug', action='store_true',
                       help='デバッグモード: 全ての詳細結果をCSVに出力 (デフォルト: consensusのみ出力)')
    args = parser.parse_args()

    # 必要な環境変数の確認
    required_env = [
        'COSMOS_ENDPOINT', 'COSMOS_KEY', 'DATABASE_NAME', 'CONTAINER_NAME',
        'OPENAI_API_KEY', 'GOOGLE_API_KEY'
    ]
    missing_env = [env for env in required_env if not os.environ.get(env)]
    if missing_env:
        logger.error(f"以下の環境変数が設定されていません: {', '.join(missing_env)}")
        exit(1)

    asyncio.run(main(debug_mode=args.debug))

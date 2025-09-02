"""
特許データ分類コード推測システム (改良版)

改良点:
1. FI予測の精度向上: application_fieldとtechnical_fieldの両方でFI予測を行い、タイトルとの関連度で選択
2. 特許番号の出力: CSVにpatent_idを追加
3. 安全ブロック対策: Geminiがブロックされた場合にOpenAI APIで再試行
4. ベクトル検索用に端的な１用語を抽出し、FI予測に使用
5. 複数手法の共通FIコードを総意として採用
"""

import os
import json
import csv
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
import asyncio
import re
import subprocess
from dataclasses import dataclass

# 外部ライブラリ
try:
    from azure.cosmos import CosmosClient
    import pandas as pd
    import google.generativeai as genai
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    print("以下のコマンドを実行してください: pip install azure-cosmos pandas google-generativeai openai python-dotenv")
    exit(1)

load_dotenv()

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    fterm_keywords: List[str]
    model_name: str
    fi_prediction_method: str = ""  # どちらのキーワードを使用したか記録


class KeywordFilter:
    """キーワードフィルタリングクラス"""

    @classmethod
    def filter_keywords(cls, keywords: List[str], context: str = "") -> List[str]:
        """キーワードをそのまま返す（フィルタリングなし）"""
        return keywords if keywords else []


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

            # FI仮想環境のPythonを使用
            venv_python = "/Users/reina.aratani/geniac/GENIAC_PATENT/FI/.venv/bin/python"

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


class LLMPredictor:
    """LLMを使用した分類コード推測クラス"""
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.gemini_model_25 = genai.GenerativeModel('gemini-2.5-flash')
        self.gemini_model_20 = genai.GenerativeModel('gemini-2.0-flash-exp')  # フォールバック用
        self.keyword_filter = KeywordFilter()

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

          【注意事項】
          - application_keywordは必ず1つの単語またはシンプルな複合語で表現してください。
          - その特許の適用分野を最も端的に表現する用語を選んでください。

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
                  "application_keyword": "適用分野の単一キーワード（最も適切な1つ）"
              }},
              "theme_keywords": ["テーマコード推測の根拠となったキーワード"],
              "fi_keywords": ["FI分類推測の根拠となったキーワード"],
              "fterm_keywords": ["Fターム推測の根拠となったキーワード"]
          }}

          【出力例】
          {{
              "predicted_theme_code": ["5J062"],
              "predicted_fi": ["G06F3/041", "G06F3/0488"],
              "predicted_fterm": ["5B087 FF02", "5E555 AA12"],
              "analysis": {{
                  "application_field_keywords": ["タッチパネル", "スマートフォン", "ディスプレイ"],
                  "technical_field_keywords": ["静電容量センサー", "電極パターン", "ノイズ除去"],
                  "application_keyword": "タッチパネル"
              }},
              "theme_keywords": ["コンピュータ", "入出力"],
              "fi_keywords": ["タッチパネル", "入力装置"],
              "fterm_keywords": ["ノイズ対策", "電極構造"]
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
                            text = c.text
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
                fterm_keywords=result_json.get('fterm_keywords', []),
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
                fterm_keywords=result_json.get('fterm_keywords', []),
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
                fterm_keywords=result_json.get('fterm_keywords', []),
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
        """両方のアプローチ（適用分野重視と技術分野重視）でFI予測を行い、より良い結果を返す"""

        if use_gemini:
            # 両方のアプローチで予測
            app_result = await self.predict_with_gemini(patent_data, focus_on_technical=False)
            tech_result = await self.predict_with_gemini(patent_data, focus_on_technical=True)
        else:
            # OpenAIを使用
            app_result = await self.predict_with_openai(patent_data, focus_on_technical=False)
            tech_result = await self.predict_with_openai(patent_data, focus_on_technical=True)

        # タイトルとの類似度で判断
        # application_field_keywordsとtechnical_field_keywordsを使用
        app_keywords = app_result.analysis.get('application_field_keywords', []) if app_result else []
        tech_keywords = tech_result.analysis.get('technical_field_keywords', []) if tech_result else []

        app_relevance = self._calculate_keyword_relevance(app_keywords, patent_data.title)
        tech_relevance = self._calculate_keyword_relevance(tech_keywords, patent_data.title)

        logger.debug(f"適用分野キーワードのタイトル関連度: {app_relevance:.2%}, 技術分野キーワードのタイトル関連度: {tech_relevance:.2%}")

        # より関連度の高い方を選択
        if tech_relevance >= app_relevance:
            best_result = tech_result
            best_result.fi_prediction_method = "technical_field_focus"
        else:
            best_result = app_result
            best_result.fi_prediction_method = "application_field_focus"

        # 両方の結果を保存しておく（後で複数手法の共通FI検出で使用）
        best_result._alternative_result = app_result if tech_relevance >= app_relevance else tech_result

        return best_result

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


async def process_patent(patent_data: PatentData, predictor: LLMPredictor, vector_predictor: VectorSearchPredictor, calculator: AccuracyCalculator) -> List[Dict]:
    """個別の特許データを処理し、結果を返すコルーチン"""
    logger.info(f"処理中: {patent_data.title[:50]}... (特許番号: {patent_data.patent_id})")

    # Geminiで両方のアプローチでFI予測を実行（OpenAIはコメントアウト）
    # openai_result = await predictor.predict_with_both_approaches(patent_data, use_gemini=False)
    gemini_result = await predictor.predict_with_both_approaches(patent_data, use_gemini=True)

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
            'application_keyword': result.analysis.get('application_keyword', ''),
        })

    # ベクトル検索の実行
    if gemini_result:
        # Geminiで抽出されたキーワードを使用してベクトル検索
        app_keywords = gemini_result.analysis.get('application_field_keywords', [])
        tech_keywords = gemini_result.analysis.get('technical_field_keywords', [])
        single_app_keyword = gemini_result.analysis.get('application_keyword', '')

        # Application fieldキーワードでベクトル検索
        if app_keywords:
            try:
                fi_codes_app, fterm_codes_app = vector_predictor.search_fi_fterm(app_keywords)

                if fi_codes_app or fterm_codes_app:
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
                        'predicted_theme_code': '',
                        'correct_theme_code': '; '.join(patent_data.correct_theme_code),
                        'predicted_fi': '; '.join(fi_codes_app),
                        'correct_fi': '; '.join(patent_data.correct_fi),
                        'predicted_fterm': '; '.join(fterm_codes_app),
                        'correct_fterm': '; '.join(patent_data.correct_fterm),
                        'theme_accuracy': '0/0 (0.00%)',
                        'fi_accuracy': f"{fi_hits_app}/{fi_total_app} ({fi_acc_app:.2%})",
                        'fterm_accuracy': f"{fterm_hits_app}/{fterm_total_app} ({fterm_acc_app:.2%})",
                        'application_field_keywords': '; '.join(app_keywords),
                        'technical_field_keywords': '',
                        'application_keyword': '',
                    })
            except Exception as e:
                logger.warning(f"Application fieldキーワードのベクトル検索に失敗: {e}")

        # Technical fieldキーワードでベクトル検索
        if tech_keywords:
            try:
                fi_codes_tech, fterm_codes_tech = vector_predictor.search_fi_fterm(tech_keywords)

                if fi_codes_tech or fterm_codes_tech:
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
                        'predicted_theme_code': '',
                        'correct_theme_code': '; '.join(patent_data.correct_theme_code),
                        'predicted_fi': '; '.join(fi_codes_tech),
                        'correct_fi': '; '.join(patent_data.correct_fi),
                        'predicted_fterm': '; '.join(fterm_codes_tech),
                        'correct_fterm': '; '.join(patent_data.correct_fterm),
                        'theme_accuracy': '0/0 (0.00%)',
                        'fi_accuracy': f"{fi_hits_tech}/{fi_total_tech} ({fi_acc_tech:.2%})",
                        'fterm_accuracy': f"{fterm_hits_tech}/{fterm_total_tech} ({fterm_acc_tech:.2%})",
                        'application_field_keywords': '',
                        'technical_field_keywords': '; '.join(tech_keywords),
                        'application_keyword': '',
                    })
            except Exception as e:
                logger.warning(f"Technical fieldキーワードのベクトル検索に失敗: {e}")

        # 単一適用分野キーワードでベクトル検索
        if single_app_keyword and single_app_keyword.strip():
            try:
                fi_codes_single, fterm_codes_single = vector_predictor.search_fi_fterm([single_app_keyword])

                if fi_codes_single or fterm_codes_single:
                    # 精度計算
                    fi_hits_single, fi_total_single, fi_acc_single = calculator.calculate_accuracy(
                        fi_codes_single, patent_data.correct_fi, code_type='fi'
                    )
                    fterm_hits_single, fterm_total_single, fterm_acc_single = calculator.calculate_accuracy(
                        fterm_codes_single, patent_data.correct_fterm
                    )

                    patent_results.append({
                        'patent_id': patent_data.patent_id,
                        'patent_title': patent_data.title,
                        'model': 'Vector_Search_Single_Application',
                        'fi_prediction_method': 'vector_search_single_application_keyword',
                        'predicted_theme_code': '',
                        'correct_theme_code': '; '.join(patent_data.correct_theme_code),
                        'predicted_fi': '; '.join(fi_codes_single),
                        'correct_fi': '; '.join(patent_data.correct_fi),
                        'predicted_fterm': '; '.join(fterm_codes_single),
                        'correct_fterm': '; '.join(patent_data.correct_fterm),
                        'theme_accuracy': '0/0 (0.00%)',
                        'fi_accuracy': f"{fi_hits_single}/{fi_total_single} ({fi_acc_single:.2%})",
                        'fterm_accuracy': f"{fterm_hits_single}/{fterm_total_single} ({fterm_acc_single:.2%})",
                        'application_field_keywords': '',
                        'technical_field_keywords': '',
                        'application_keyword': single_app_keyword,
                    })
            except Exception as e:
                logger.warning(f"単一適用分野キーワードのベクトル検索に失敗: {e}")

    # 複数手法の共通FIコードを探す（常に実行）
    # すべての予測結果を収集
    all_fi_predictions = {}

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
    if 'fi_codes_tech' in locals() and fi_codes_tech:
        all_fi_predictions['Vector_Technical'] = fi_codes_tech
    if 'fi_codes_single' in locals() and fi_codes_single:
        all_fi_predictions['Vector_Single'] = fi_codes_single

    # 共通するFIコードを見つける
    common_fi_codes = find_common_fi_codes(all_fi_predictions, min_methods=2)

    if common_fi_codes:
        logger.info(f"共通FIコード発見: {common_fi_codes}")

        # 精度計算（正解データがある場合のみ）
        fi_hits_consensus = 0
        fi_total_consensus = 0
        fi_acc_consensus = 0.0

        if patent_data.correct_fi and len(patent_data.correct_fi) > 0:
            fi_hits_consensus, fi_total_consensus, fi_acc_consensus = calculator.calculate_accuracy(
                common_fi_codes, patent_data.correct_fi, code_type='fi'
            )

        # 共通FIコードを使った結果を追加（総意として1行で表現）
        patent_results.append({
            'patent_id': patent_data.patent_id,
            'patent_title': patent_data.title,
            'model': 'Consensus_Method',
            'fi_prediction_method': f'consensus_{len([m for m in all_fi_predictions.values() if any(code in [re.sub(r"[^A-Z0-9]", "", str(c).split("/")[0].upper()) for c in m] for code in common_fi_codes)])}methods',
            'predicted_theme_code': '',
            'correct_theme_code': '; '.join(patent_data.correct_theme_code),
            'predicted_fi': '; '.join(common_fi_codes),
            'correct_fi': '; '.join(patent_data.correct_fi),
            'predicted_fterm': '',
            'correct_fterm': '; '.join(patent_data.correct_fterm),
            'theme_accuracy': '0/0 (0.00%)',
            'fi_accuracy': f"{fi_hits_consensus}/{fi_total_consensus} ({fi_acc_consensus:.2%})" if fi_total_consensus > 0 else '0/0 (0.00%)',
            'fterm_accuracy': '0/0 (0.00%)',
            'application_field_keywords': f"Methods: {', '.join(all_fi_predictions.keys())}",
            'technical_field_keywords': f"Common FI: {', '.join(common_fi_codes)}",
            'application_keyword': f"{len(all_fi_predictions)}手法中{len([m for m in all_fi_predictions.values() if any(code in [re.sub(r'[^A-Z0-9]', '', str(c).split('/')[0].upper()) for c in m] for code in common_fi_codes)])}手法で一致",
        })
    else:
        logger.warning(f"共通FIコードが見つかりませんでした。各手法の予測: {list(all_fi_predictions.keys())}")

    return patent_results


async def main():
    """メイン処理"""
    logger.info("特許データ分類コード推測システムを開始します")

    # 初期化
    cosmos_client = CosmosDBClient()
    predictor = LLMPredictor()
    vector_predictor = VectorSearchPredictor(enable_vector_search=True)  # evaluate_fi_fterm_from_csv.pyと同じ方法で実装
    calculator = AccuracyCalculator()
    exporter = ResultExporter()

    # データ取得
    logger.info("CosmosDBから特許データを取得中...")
    patent_data_list = cosmos_client.get_patent_data(limit=50)  # テスト時は件数を絞る

    if not patent_data_list:
        logger.error("特許データが取得できませんでした。CosmosDBの接続とデータ構造を確認してください。")
        return

    # 並行処理でタスクを実行
    tasks = []
    for i, patent_data in enumerate(patent_data_list):
        logger.info(f"特許データ {i+1}/{len(patent_data_list)} の推測タスクを作成中...")
        tasks.append(process_patent(patent_data, predictor, vector_predictor, calculator))

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
    # 必要な環境変数の確認
    required_env = [
        'COSMOS_ENDPOINT', 'COSMOS_KEY', 'DATABASE_NAME', 'CONTAINER_NAME',
        'OPENAI_API_KEY', 'GOOGLE_API_KEY'
    ]
    missing_env = [env for env in required_env if not os.environ.get(env)]
    if missing_env:
        logger.error(f"以下の環境変数が設定されていません: {', '.join(missing_env)}")
        exit(1)

    asyncio.run(main())

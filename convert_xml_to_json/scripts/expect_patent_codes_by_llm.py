"""
特許データ分類コード推測システム (改善版)

このプログラムは以下の機能を提供します：
1. CosmosDBから特許データを取得
2. OpenAI gpt-5-miniとGemini 2.5 flashを使用してFI、Fターム、テーマコードを推測
3. 推測結果と正解データの比較・精度評価（FIの比較ロジックを改善）
4. 結果をCSV形式で出力（日付付きファイル名）

改善点：
- Geminiの安全ブロック時のログ出力強化
- FIコードの正解率計算ロジックを改善し、表記揺れに対応
- プロンプトにFew-shotの例を追加し、LLMの出力精度を向上
- OpenAI APIの呼び出しを最新の形式に更新
"""

import os
import json
import csv
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import asyncio
import re
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
    analysis: Dict[str, Dict[str, List[str]]]
    theme_keywords: List[str]
    fi_keywords: List[str]
    fterm_keywords: List[str]
    model_name: str


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
                claims_list = [f"【請求項{claim.get('num', '')}】{claim.get('text', '')}" for claim in item.get('claims', [])]
                keywords = metadata.get('keywords', [])
                entities = {'objects': keywords[:10], 'actions': [], 'usecase': []}
                figure_captions = [item.get('description', '').split('【選択図】')[1].strip()] if '【選択図】' in item.get('description', '') else []
                patent_data = PatentData(
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
                    logger.warning(f"空の特許データをスキップしました: {metadata.get('patent_id', 'Unknown')}")
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


class LLMPredictor:
    """LLMを使用した分類コード推測クラス"""
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')

    def create_prompt(self, patent_data: PatentData) -> str:
        """特許データから分類コード推測用のプロンプトを生成"""
        all_claims = "\n\n".join(patent_data.claims_main) if patent_data.claims_main else "請求項情報なし"
        prompt = f"""
          以下の特許データから、適切な分類コード（テーマコード、FI、Fターム）を推測してください。
          まず「適用分野」と「主要な技術分野（構成要素、機能）」を分析し、キーワードを抽出してください。

          【分析の視点】
          - **適用分野**: この発明が使われる場所や目的（例: 自動車、医療、建築）。
          - **主要な技術分野**: この発明の核心となる技術や構成要素（例: 光学センサー、データ処理、ロボットアーム）。

          【特許データ】
          発明の名称: {patent_data.title}
          要約: {patent_data.abstract}
          請求項: {all_claims}
          図面の簡単な説明: {' '.join(patent_data.figure_captions) if patent_data.figure_captions else '情報なし'}
          出願人: {patent_data.applicant if patent_data.applicant else '情報なし'}

          【考察】
          上記の分析に基づき、この発明の技術的本質を考察してください。

          【出力形式】
          回答は必ず以下のJSON形式でお願いします。各コードはリスト形式で複数指定可能です。

          {{
              "predicted_theme_code": ["推測したテーマコード"],
              "predicted_fi": ["推測したFI分類"],
              "predicted_fterm": ["推測したFターム"],
              "analysis": {{
                  "application_field_keywords": ["適用分野のキーワード"],
                  "technical_field_keywords": ["主要な技術分野のキーワード"],
                  "technical_essence": "考察結果"
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
                  "technical_essence": "タッチパネルの電極構造を改良し、ノイズ耐性を向上させる技術"
              }},
              "theme_keywords": ["コンピュータ", "入出力"],
              "fi_keywords": ["タッチパネル", "入力装置"],
              "fterm_keywords": ["ノイズ対策", "電極構造"]
          }}
          """
        return prompt

    async def predict_with_openai(self, patent_data: PatentData) -> PredictionResult:
        """OpenAI gpt-5-miniで分類コードを推測"""
        try:
            prompt = self.create_prompt(patent_data)
            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes patent documents and predicts classification codes in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
                # 【修正点】モデルが対応していないため、temperatureの行を削除
            )
            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)

            return PredictionResult(
                predicted_theme_code=result_json.get('predicted_theme_code', []),
                predicted_fi=result_json.get('predicted_fi', []),
                predicted_fterm=result_json.get('predicted_fterm', []),
                analysis=result_json.get('analysis', {}),
                theme_keywords=result_json.get('theme_keywords', []),
                fi_keywords=result_json.get('fi_keywords', []),
                fterm_keywords=result_json.get('fterm_keywords', []),
                model_name="OpenAI_GPT-5-mini"
            )
        except Exception as e:
            logger.error(f"OpenAI推測エラー: {e}")
            return self._empty_result("OpenAI_GPT-5-mini")

    async def predict_with_gemini(self, patent_data: PatentData) -> PredictionResult:
        """Gemini 2.5 flashで分類コードを推測"""
        result_text = ""
        try:
            prompt = self.create_prompt(patent_data)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
                safety_settings=safety_settings
            )

            # プロンプト自体がブロックされた場合のチェック
            if not response.candidates:
                finish_reason = response.prompt_feedback.block_reason
                logger.warning(f"Gemini: プロンプトがブロックされました。理由: {finish_reason}")
                if response.prompt_feedback.safety_ratings:
                    for rating in response.prompt_feedback.safety_ratings:
                         logger.warning(f"  安全性評価: {rating.category} - {rating.probability}")
                return self._empty_result("Gemini_2.5_Flash")

            # 【修正点】応答が安全フィルターでブロックされた場合のチェックを追加
            candidate = response.candidates[0]
            # finish_reasonが1(STOP)でない場合はエラーとみなす
            if candidate.finish_reason != 1:
                logger.warning(f"Gemini: 応答の生成が停止しました。理由: {candidate.finish_reason.name} ({candidate.finish_reason.value})")
                if candidate.safety_ratings:
                    for rating in candidate.safety_ratings:
                        logger.warning(f"  安全性評価: {rating.category} - {rating.probability}")
                return self._empty_result("Gemini_2.5_Flash")

            result_text = response.text
            result_json = json.loads(result_text)

            return PredictionResult(
                predicted_theme_code=result_json.get('predicted_theme_code', []),
                predicted_fi=result_json.get('predicted_fi', []),
                predicted_fterm=result_json.get('predicted_fterm', []),
                analysis=result_json.get('analysis', {}),
                theme_keywords=result_json.get('theme_keywords', []),
                fi_keywords=result_json.get('fi_keywords', []),
                fterm_keywords=result_json.get('fterm_keywords', []),
                model_name="Gemini_2.5_Flash"
            )
        except json.JSONDecodeError as e:
            logger.error(f"Gemini JSON解析エラー: {e}")
            logger.error(f"レスポンステキスト: {result_text[:500]}...")
            return self._empty_result("Gemini_2.5_Flash")
        except Exception as e:
            logger.error(f"Gemini推測エラー: {e}")
            return self._empty_result("Gemini_2.5_Flash")

    def _empty_result(self, model_name: str) -> PredictionResult:
        """エラー時の空結果を返す"""
        return PredictionResult([], [], [], {}, [], [], [], model_name)


class AccuracyCalculator:
    """精度計算クラス"""

    @staticmethod
    def _normalize_fi_code(code: str) -> str:
        """FIコードから記号やスペースを除去し、英数字のみを大文字で返す"""
        return re.sub(r'[^A-Z0-9]', '', str(code).upper())

    @staticmethod
    def calculate_accuracy(predicted: List[str], correct: List[str], code_type: Optional[str] = None) -> Tuple[int, int, float]:
        """精度を計算（ヒット数, 総数, 正解率）"""
        if not correct:
            return 0, 0, 0.0

        # 【変更点】FIコードの判定ロジックを「前方一致」に変更
        if code_type == 'fi':
            predicted_normalized = {AccuracyCalculator._normalize_fi_code(c) for c in predicted}
            correct_normalized = {AccuracyCalculator._normalize_fi_code(c) for c in correct}

            hits = 0
            # 各「正解コード」に対してループ
            for c_code in correct_normalized:
                # いずれかの「予測コード」が正解コードの先頭部分と一致するかチェック
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


async def main():
    """メイン処理"""
    logger.info("特許データ分類コード推測システムを開始します")
    cosmos_client = CosmosDBClient()
    predictor = LLMPredictor()
    calculator = AccuracyCalculator()
    exporter = ResultExporter()
    logger.info("CosmosDBから特許データを取得中...")
    patent_data_list = cosmos_client.get_patent_data(limit=10) # テスト時は件数を絞る

    if not patent_data_list:
        logger.error("特許データが取得できませんでした。CosmosDBの接続とデータ構造を確認してください。")
        return

    tasks = []
    for i, patent_data in enumerate(patent_data_list):
        logger.info(f"特許データ {i+1}/{len(patent_data_list)} の推測タスクを作成中...")
        tasks.append(process_patent(patent_data, predictor, calculator))

    results_list = await asyncio.gather(*tasks)

    results = [item for sublist in results_list for item in sublist]

    if results:
        csv_file = exporter.export_to_csv(results)
        if csv_file:
            logger.info(f"処理完了！結果ファイル: {csv_file}")
        else:
            logger.error("CSV出力に失敗しました")
    else:
        logger.warning("処理結果がありませんでした。")


async def process_patent(patent_data: PatentData, predictor: LLMPredictor, calculator: AccuracyCalculator) -> List[Dict]:
    """個別の特許データを処理し、結果を返すコルーチン"""
    logger.info(f"処理中: {patent_data.title[:50]}...")

    openai_result, gemini_result = await asyncio.gather(
        predictor.predict_with_openai(patent_data),
        predictor.predict_with_gemini(patent_data)
    )

    patent_results = []
    for result in [openai_result, gemini_result]:
        if not result: continue

        theme_hits, theme_total, theme_acc = calculator.calculate_accuracy(result.predicted_theme_code, patent_data.correct_theme_code)
        fi_hits, fi_total, fi_acc = calculator.calculate_accuracy(result.predicted_fi, patent_data.correct_fi, code_type='fi')
        fterm_hits, fterm_total, fterm_acc = calculator.calculate_accuracy(result.predicted_fterm, patent_data.correct_fterm)

        patent_results.append({
            'patent_title': patent_data.title,
            'model': result.model_name,
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
            'technical_essence': result.analysis.get('technical_essence', ''),
        })
    return patent_results


if __name__ == "__main__":
    required_env = [
        'COSMOS_ENDPOINT', 'COSMOS_KEY', 'DATABASE_NAME', 'CONTAINER_NAME',
        'OPENAI_API_KEY', 'GOOGLE_API_KEY'
    ]
    missing_env = [env for env in required_env if not os.environ.get(env)]
    if missing_env:
        logger.error(f"以下の環境変数が設定されていません: {', '.join(missing_env)}")
        exit(1)

    asyncio.run(main())

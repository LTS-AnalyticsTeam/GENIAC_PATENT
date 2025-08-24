"""
特許データ分類コード推測システム

このプログラムは以下の機能を提供します：
1. CosmosDBから特許データを取得
2. OpenAI GPT-4とGemini Flash 2.5を使用してFI、Fターム、テーマコードを推測
3. 推測結果と正解データの比較・精度評価
4. 結果をCSV形式で出力（日付付きファイル名）
5. グラフによる結果の可視化

要件：
- 各LLMの推測精度を比較
- 分類コード推測に必要なキーワードを「適用分野」と「主要な技術分野」に分けて抽出
- 正解率の計算と可視化

使用API：
- OpenAI API (GPT-4)
- Google Gemini Flash 2.5
- Azure Cosmos DB

環境変数：
COSMOS_ENDPOINT, COSMOS_KEY, DATABASE_NAME, CONTAINER_NAME
OPENAI_API_KEY, GOOGLE_API_KEY
"""

import os
import json
import csv
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import asyncio
from dataclasses import dataclass

# 外部ライブラリ
try:
    from azure.cosmos import CosmosClient
    import pandas as pd
    import google.generativeai as genai
    from openai import OpenAI
    from dotenv import load_dotenv
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    print("以下のコマンドを実行してください: pip install azure-cosmos pandas google-generativeai openai python-dotenv matplotlib seaborn")
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
    claims_main: List[str]  # 全ての請求項のリスト
    figure_captions: List[str]
    ipc_seed: List[str]
    entities: Dict[str, List[str]]
    applicant: str
    citations_ipc: List[str]
    # 正解データ
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
        self.client = CosmosClient(
            os.environ['COSMOS_ENDPOINT'],
            os.environ['COSMOS_KEY']
        )
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

                # 全ての請求項を取得
                claims_list = []
                claims = item.get('claims', [])
                for claim in claims:
                    claims_list.append(f"【請求項{claim.get('num', '')}】{claim.get('text', '')}")

                # エンティティ情報を構築（keywordsから）
                keywords = metadata.get('keywords', [])
                entities = {
                    'objects': keywords[:10] if len(keywords) > 10 else keywords,
                    'actions': [],
                    'usecase': []
                }

                # 図面の説明を取得（descriptionから抽出）
                figure_captions = []
                description = item.get('description', '')
                if '【選択図】' in description:
                    figure_captions = [description.split('【選択図】')[1].strip()]

                patent_data = PatentData(
                    title=metadata.get('title', ''),
                    abstract=item.get('summary', ''),
                    claims_main=claims_list,
                    figure_captions=figure_captions,
                    ipc_seed=[ipc.get('code', '') for ipc in metadata.get('classification_ipc', [])],
                    entities=entities,
                    applicant=', '.join(item.get('applicants', [])),
                    citations_ipc=metadata.get('topics', []),
                    correct_theme_code=self._extract_theme_codes(item),
                    correct_fi=self._extract_fi_codes(item),
                    correct_fterm=self._extract_fterm_codes(item)
                )

                if patent_data.title or patent_data.abstract or patent_data.claims_main:
                    patent_data_list.append(patent_data)
                    logger.debug(f"追加した特許: {patent_data.title[:50]}...")
                else:
                    logger.warning(f"空の特許データをスキップしました: {metadata.get('patent_id', 'Unknown')}")

            logger.info(f"有効な特許データ数: {len(patent_data_list)}")
            return patent_data_list

        except Exception as e:
            logger.error(f"CosmosDBからのデータ取得エラー: {e}")
            return []

    def _extract_theme_codes(self, item: dict) -> List[str]:
        metadata = item.get('metadata', {})
        theme_codes = metadata.get('theme_code', [])
        return [str(code) for code in theme_codes] if isinstance(theme_codes, list) else [str(theme_codes)] if theme_codes else []

    def _extract_fi_codes(self, item: dict) -> List[str]:
        metadata = item.get('metadata', {})
        fi_codes = metadata.get('classification_fi', [])
        return [str(code) for code in fi_codes] if isinstance(fi_codes, list) else [str(fi_codes)] if fi_codes else []

    def _extract_fterm_codes(self, item: dict) -> List[str]:
        metadata = item.get('metadata', {})
        fterm_codes = metadata.get('f_term', [])
        return [str(code) for code in fterm_codes] if isinstance(fterm_codes, list) else [str(fterm_codes)] if fterm_codes else []


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
          その際、まず「適用分野」と「主要な技術分野（構成要素、機能）」を分けて分析し、それぞれの視点からキーワードを抽出してください。

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
          上記の分析を基に、この発明の技術的本質が「どのような道具を、どのように応用するか」にあるか、あるいは「どのような作業を行うための専用の道具か」にあるかを考察してください。

          回答は必ず以下のJSON形式でお願いします：
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
          """
        return prompt

    async def predict_with_openai(self, patent_data: PatentData) -> PredictionResult:
        """OpenAI GPT-4で分類コードを推測"""
        try:
            prompt = self.create_prompt(patent_data)
            response = self.openai_client.responses.create(
                model="gpt-5-mini", # コストと速度を考慮し、o-miniを使用
                input=prompt,
                reasoning={"effort": "minimal"}
            )
            import re
            text = ""
            for out in response.output:
                if hasattr(out, "content") and out.content:
                    for c in out.content:
                        if hasattr(c, "text"):
                            text = c.text
                            break
            result_text = text

            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()

            result_json = json.loads(result_text)

            return PredictionResult(
                predicted_theme_code=result_json.get('predicted_theme_code', []),
                predicted_fi=result_json.get('predicted_fi', []),
                predicted_fterm=result_json.get('predicted_fterm', []),
                analysis=result_json.get('analysis', {}),
                theme_keywords=result_json.get('theme_keywords', []),
                fi_keywords=result_json.get('fi_keywords', []),
                fterm_keywords=result_json.get('fterm_keywords', []),
                model_name="OpenAI_GPT5o-mini"
            )

        except Exception as e:
            logger.error(f"OpenAI推測エラー: {e}")
            return self._empty_result("OpenAI_GPT-5o-mini")

    async def predict_with_gemini(self, patent_data: PatentData) -> PredictionResult:
        """Gemini Flash 2.5で分類コードを推測"""
        try:
            prompt = self.create_prompt(patent_data)

            # 安全性設定を追加してフィルターを緩和
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                ),
                safety_settings=safety_settings
            )

            # レスポンスの詳細チェック
            if not response.candidates:
                logger.warning("Gemini: レスポンス候補が存在しません")
                return self._empty_result("Gemini_Flash_2.5")

            candidate = response.candidates[0]

            # finish_reasonをチェック
            if candidate.finish_reason == 2:  # SAFETY
                logger.warning("Gemini: 安全性フィルターによりブロックされました")
                # 安全性理由の詳細をログ出力
                if hasattr(candidate, 'safety_ratings'):
                    for rating in candidate.safety_ratings:
                        logger.warning(f"安全性評価: {rating.category} - {rating.probability}")
                return self._empty_result("Gemini_Flash_2.5")
            elif candidate.finish_reason == 3:  # RECITATION
                logger.warning("Gemini: 引用フィルターによりブロックされました")
                return self._empty_result("Gemini_Flash_2.5")
            elif candidate.finish_reason not in [0, 1]:  # STOP, MAX_TOKENS以外
                logger.warning(f"Gemini: 予期しない終了理由: {candidate.finish_reason}")
                return self._empty_result("Gemini_Flash_2.5")

            # テキストが存在するかチェック
            if not candidate.content or not candidate.content.parts:
                logger.warning("Gemini: レスポンスにテキストが含まれていません")
                return self._empty_result("Gemini_Flash_2.5")

            # テキストを安全に取得
            result_text = ""
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    result_text += part.text

            if not result_text.strip():
                logger.warning("Gemini: レスポンステキストが空です")
                return self._empty_result("Gemini_Flash_2.5")

            result_text = result_text.strip()

            # JSONの抽出
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()

            result_json = json.loads(result_text)

            return PredictionResult(
                predicted_theme_code=result_json.get('predicted_theme_code', []),
                predicted_fi=result_json.get('predicted_fi', []),
                predicted_fterm=result_json.get('predicted_fterm', []),
                analysis=result_json.get('analysis', {}),
                theme_keywords=result_json.get('theme_keywords', []),
                fi_keywords=result_json.get('fi_keywords', []),
                fterm_keywords=result_json.get('fterm_keywords', []),
                model_name="Gemini_flash_2.5"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Gemini JSON解析エラー: {e}")
            logger.error(f"レスポンステキスト: {result_text[:500]}...")
            return self._empty_result("Gemini_Flash_2.5")
        except Exception as e:
            logger.error(f"Gemini推測エラー: {e}")
            return self._empty_result("Gemini_Flash_2.5")

    def _empty_result(self, model_name: str) -> PredictionResult:
        """エラー時の空結果を返す"""
        return PredictionResult(
            predicted_theme_code=[],
            predicted_fi=[],
            predicted_fterm=[],
            analysis={},
            theme_keywords=[],
            fi_keywords=[],
            fterm_keywords=[],
            model_name=model_name
        )


class AccuracyCalculator:
    """精度計算クラス"""

    @staticmethod
    def calculate_accuracy(predicted: List[str], correct: List[str], code_type: Optional[str] = None) -> Tuple[int, int, float]:
        """精度を計算（ヒット数, 総数, 正解率）"""
        if not correct:
            return 0, 0, 0.0

        if code_type == 'fi':
            predicted_processed = [code.split('/')[0] for code in predicted]
            correct_processed = [code.split('/')[0] for code in correct]
        else:
            predicted_processed = predicted
            correct_processed = correct

        hits = len(set(predicted_processed) & set(correct_processed))
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
    patent_data_list = cosmos_client.get_patent_data(limit=10)

    if not patent_data_list:
        logger.error("特許データが取得できませんでした。CosmosDBの接続とデータ構造を確認してください。")
        return

    results = []

    for i, patent_data in enumerate(patent_data_list):
        logger.info(f"特許データ {i+1}/{len(patent_data_list)} を処理中...")

        # OpenAIで推測
        openai_result = await predictor.predict_with_openai(patent_data)

        # Geminiで推測
        gemini_result = await predictor.predict_with_gemini(patent_data)

        # 精度計算と結果記録
        for result in [openai_result, gemini_result]:
            theme_hits, theme_total, theme_acc = calculator.calculate_accuracy(
                result.predicted_theme_code, patent_data.correct_theme_code
            )
            fi_hits, fi_total, fi_acc = calculator.calculate_accuracy(
                result.predicted_fi, patent_data.correct_fi, code_type='fi'
            )
            fterm_hits, fterm_total, fterm_acc = calculator.calculate_accuracy(
                result.predicted_fterm, patent_data.correct_fterm
            )

            results.append({
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
                'theme_keywords': '; '.join(result.theme_keywords),
                'fi_keywords': '; '.join(result.fi_keywords),
                'fterm_keywords': '; '.join(result.fterm_keywords)
            })
        await asyncio.sleep(1)

    # 結果をCSVに出力
    csv_file = exporter.export_to_csv(results)

    if csv_file:
        logger.info(f"処理完了！結果ファイル: {csv_file}")

        # 統計情報と可視化
        df = pd.DataFrame(results)
        logger.info("\n=== 全体統計 ===")
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            logger.info(f"\n{model}:")
            logger.info(f"処理件数: {len(model_data)}")

        try:
            logger.info("正解率の可視化を開始します...")
            df['theme_accuracy_rate'] = df['theme_accuracy'].apply(lambda x: float(x.split('(')[1].replace('%)', '')) / 100 if '(' in x else 0.0)
            df['fi_accuracy_rate'] = df['fi_accuracy'].apply(lambda x: float(x.split('(')[1].replace('%)', '')) / 100 if '(' in x else 0.0)
            df['fterm_accuracy_rate'] = df['fterm_accuracy'].apply(lambda x: float(x.split('(')[1].replace('%)', '')) / 100 if '(' in x else 0.0)

            accuracy_df = df.groupby('model')[['theme_accuracy_rate', 'fi_accuracy_rate', 'fterm_accuracy_rate']].mean() * 100
            accuracy_df.columns = ['テーマコード', 'FI', 'Fターム']

            plt.style.use('ggplot')
            accuracy_df.plot(kind='bar', figsize=(10, 6))
            plt.title('LLM別 特許分類コード推測精度比較', fontsize=16)
            plt.xlabel('LLMモデル', fontsize=12)
            plt.ylabel('正解率 (%)', fontsize=12)
            plt.xticks(rotation=0)
            plt.ylim(0, 100)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='分類コード')
            plt.tight_layout()

            image_filename = f"classification_accuracy_{exporter.timestamp}.png"
            plt.savefig(image_filename)
            logger.info(f"正解率のグラフを保存しました: {image_filename}")
            plt.show()

        except Exception as e:
            logger.error(f"グラフ生成エラー: {e}")

    else:
        logger.error("CSV出力に失敗しました")


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

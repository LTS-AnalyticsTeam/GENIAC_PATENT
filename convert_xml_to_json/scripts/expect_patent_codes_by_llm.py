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

# テーマコード検索：FI/scripts/theme_code_search.pyを使用
import subprocess
import json as json_module

def search_theme_codes_via_fi_script(query_text: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    FI/scripts/theme_code_search.pyを使用してテーマコード検索を実行
    
    Args:
        query_text: 検索クエリテキスト
        top_k: 上位何件を返すか
        
    Returns:
        テーマコード検索結果のリスト
    """
    if not query_text.strip():
        return []
    
    try:
        # FI/scripts/theme_code_search.pyのパス
        fi_script_path = os.path.join(os.path.dirname(__file__), '../../FI/scripts/theme_code_search.py')
        
        # スクリプトを実行
        result = subprocess.run(
            ['python', fi_script_path, '--json', '--top-k', str(top_k), query_text],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # JSON形式の結果をパース
            search_results = json_module.loads(result.stdout)
            
            # 結果を統一形式に変換
            formatted_results = []
            for item in search_results:
                formatted_results.append({
                    'theme_code': item.get('theme_code', ''),
                    'description': item.get('description', ''),
                    'full_title': item.get('full_title', ''),
                    'similarity_score': float(item.get('score', 0.0))
                })
            
            return formatted_results
        else:
            logger.warning(f"テーマコード検索スクリプトエラー: {result.stderr}")
            return []
            
    except subprocess.TimeoutExpired:
        logger.warning("テーマコード検索がタイムアウトしました")
        return []
    except Exception as e:
        logger.warning(f"テーマコード検索実行エラー: {e}")
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
    fterm_keywords: List[str]
    model_name: str
    fi_prediction_method: str = ""  # どちらのキーワードを使用したか記録


class KeywordFilter:
    """キーワードフィルタリングクラス"""
    
    # 汎用技術語彙辞書
    TECHNICAL_VOCABULARY = {
        # 材料・構造関連
        '材料': ['金属', 'プラスチック', '樹脂', '繊維', '複合材料', '合金', 'セラミック'],
        '構造': ['フレーム', '筐体', 'ケース', '支持体', '基板', '接続部', '結合部'],
        '機械要素': ['軸', 'ベアリング', 'ギア', 'スプリング', 'ボルト', 'ナット', 'シール'],
        
        # 処理・方法関連
        '加工': ['切削', '研削', '穿孔', '成形', '鋳造', '鍛造', '溶接'],
        '制御': ['制御', '調整', '監視', '検出', '測定', '計測', 'フィードバック'],
        '処理': ['データ処理', '信号処理', '画像処理', '音声処理', '圧縮', '変換'],
        
        # 機能・効果関連
        '性能向上': ['効率化', '高速化', '精度向上', '安定化', '最適化', '省エネ'],
        '品質改善': ['ノイズ低減', '振動抑制', '耐久性向上', '信頼性向上', '強度向上'],
        
        # 用途・分野関連
        '産業分野': ['自動車', '航空宇宙', '建築', '電子', '医療', '通信', '製造業'],
        '応用分野': ['センサー', 'アクチュエータ', 'ディスプレイ', 'エンジン', 'モータ'],
    }
    
    @classmethod
    def generalize_keywords(cls, keywords: List[str]) -> List[str]:
        """個別最適なキーワードを汎用的なキーワードに置換"""
        generalized = []
        
        for keyword in keywords:
            # 汎用語彙辞書から最適なマッチを探す
            best_match = cls._find_best_general_term(keyword)
            if best_match:
                generalized.append(best_match)
            else:
                # マッチしない場合は上位概念を推定
                general_term = cls._extract_general_concept(keyword)
                if general_term:
                    generalized.append(general_term)
        
        # 重複を除去して返す
        return list(dict.fromkeys(generalized))  # 順序を保持しつつ重複除去
    
    @classmethod
    def _find_best_general_term(cls, keyword: str) -> str:
        """キーワードに最も適合する汎用語彙を見つける"""
        # 各カテゴリの語彙と照合
        for category, terms in cls.TECHNICAL_VOCABULARY.items():
            for term in terms:
                if term in keyword or keyword in term:
                    return category  # カテゴリ名を返す
        
        # 部分的なマッチングも試行
        for category, terms in cls.TECHNICAL_VOCABULARY.items():
            for term in terms:
                # 3文字以上の共通部分があればマッチとみなす
                if len(term) >= 3 and (term[:3] in keyword or keyword[:3] in term):
                    return category
        
        return None
    
    @classmethod
    def _extract_general_concept(cls, keyword: str) -> str:
        """キーワードから一般概念を抽出"""
        # よくある技術用語のパターンマッチング
        patterns = {
            'システム': '制御システム',
            '装置': '機械装置', 
            '方法': '処理方法',
            '構造': '機械構造',
            '部品': '機械要素',
            'センサー': 'センサー',
            '制御': '制御システム',
            '検出': '検出システム',
            '処理': 'データ処理',
            '接続': '接続構造',
            '固定': '固定機構',
        }
        
        for pattern, general_term in patterns.items():
            if pattern in keyword:
                return general_term
        
        # デフォルトでより一般的な用語を返す
        if len(keyword) > 5:  # 長いキーワードは「技術要素」に分類
            return '技術要素'
        
        return None

    @classmethod
    def calculate_specificity_score(cls, keyword: str) -> float:
        """キーワードの具体性スコアを計算（0.0-1.0、高いほど具体的）"""
        if not keyword or len(keyword) < 2:
            return 0.0
        
        score = 0.0
        
        # 1. 語長による評価（長いほど具体的な傾向）
        length_score = min(len(keyword) / 10.0, 0.3)  # 最大0.3
        score += length_score
        
        # 2. カタカナ比率（技術用語の特徴）
        katakana_ratio = cls._count_katakana(keyword) / len(keyword)
        if katakana_ratio > 0.5:  # カタカナが半分以上
            score += 0.2
        
        # 3. 英数字を含む（仕様・規格の特徴）
        if any(c.isdigit() for c in keyword):  # 数字を含む
            score += 0.2
        if any(c.isalpha() for c in keyword):  # 英字を含む
            score += 0.1
        
        # 4. 複合語の特徴（具体的な機能や構造を表す）
        compound_indicators = ['ー', '・', '_', '-']
        if any(indicator in keyword for indicator in compound_indicators):
            score += 0.1
        
        # 5. 語尾による判定（抽象度の高い語尾をペナルティ）
        abstract_suffixes = ['性', '化', '的', '用', '系', '式', '法', '業']
        if any(keyword.endswith(suffix) for suffix in abstract_suffixes):
            score -= 0.2
        
        return max(0.0, min(1.0, score))  # 0.0-1.0に正規化

    @classmethod
    def filter_keywords(cls, keywords: List[str], _context: str = "") -> List[str]:
        """具体性スコアに基づいてキーワードをフィルタリング"""
        if not keywords:
            return []
        
        # 各キーワードに具体性スコアを付与
        scored_keywords = []
        for keyword in keywords:
            keyword = keyword.strip()
            if keyword:
                score = cls.calculate_specificity_score(keyword)
                scored_keywords.append((keyword, score))
        
        # スコアでソート（高い順）
        scored_keywords.sort(key=lambda x: x[1], reverse=True)
        
        # 閾値以上のキーワードのみを選択（動的閾値）
        if not scored_keywords:
            return []
        
        # 上位スコアの50%以上を閾値とする（相対的評価）
        max_score = scored_keywords[0][1]
        threshold = max_score * 0.5 if max_score > 0 else 0.3
        
        filtered = [kw for kw, score in scored_keywords if score >= threshold]
        
        return filtered[:10]  # 最大10個に制限
    
    @classmethod
    def _count_katakana(cls, text: str) -> int:
        """カタカナ文字数をカウント"""
        return sum(1 for c in text if '\u30A0' <= c <= '\u30FF')

    @classmethod
    def calculate_title_similarity_score(cls, keyword: str, title: str) -> float:
        """タイトルとキーワードの意味的類似度スコアを計算"""
        if not keyword or not title:
            return 0.0
        
        keyword_lower = keyword.lower()
        title_lower = title.lower()
        
        # 1. 完全一致
        if keyword_lower in title_lower:
            return 1.0
        
        # 2. 部分一致（キーワードの一部がタイトルに含まれる）
        keyword_chars = set(keyword_lower)
        title_chars = set(title_lower)
        char_overlap = len(keyword_chars & title_chars) / len(keyword_chars) if keyword_chars else 0
        
        # 3. 語の境界を考慮した類似度
        keyword_parts = [part for part in keyword_lower if len(part) > 1]
        title_parts = title_lower
        
        partial_matches = 0
        for part in keyword_parts:
            if len(part) > 1 and part in title_parts:
                partial_matches += 1
        
        partial_score = partial_matches / len(keyword_parts) if keyword_parts else 0
        
        # 最終スコア（重み付け平均）
        similarity_score = (char_overlap * 0.3 + partial_score * 0.7)
        
        return min(1.0, similarity_score)

    @classmethod
    def merge_and_prioritize_keywords(cls, app_keywords: List[str], tech_keywords: List[str], title: str = "") -> List[str]:
        """タイトル類似度と具体性を考慮してキーワードをマージ・優先順位付け"""
        # 全キーワードを収集して総合スコア付け
        all_keywords = []
        
        for keyword in tech_keywords:
            if keyword.strip():
                kw = keyword.strip()
                specificity_score = cls.calculate_specificity_score(kw)
                title_similarity = cls.calculate_title_similarity_score(kw, title) if title else 0
                
                # technical_keywordsは具体性を重視（重み: 具体性70%, タイトル類似度30%）
                total_score = specificity_score * 0.7 + title_similarity * 0.3
                all_keywords.append((kw, total_score, 'tech', specificity_score, title_similarity))
        
        for keyword in app_keywords:
            keyword = keyword.strip()
            if keyword and keyword not in [kw for kw, _, _, _, _ in all_keywords]:
                specificity_score = cls.calculate_specificity_score(keyword)
                title_similarity = cls.calculate_title_similarity_score(keyword, title) if title else 0
                
                # application_keywordsはタイトル類似度を重視（重み: タイトル類似度60%, 具体性40%）
                total_score = title_similarity * 0.6 + specificity_score * 0.4
                all_keywords.append((keyword, total_score, 'app', specificity_score, title_similarity))
        
        # 総合スコアでソート（高い順）
        all_keywords.sort(key=lambda x: x[1], reverse=True)
        
        # 上位8個を選択、できればtech/appのバランスを考慮
        selected = []
        tech_count = 0
        app_count = 0
        
        for kw, _score, source, _spec_score, _title_score in all_keywords[:12]:  # 少し多めに候補を確保
            if len(selected) >= 8:
                break
                
            # バランス調整：techが6個超えたらappを優先、appが6個超えたらtechを優先
            if source == 'tech' and tech_count >= 6:
                continue
            if source == 'app' and app_count >= 6:
                continue
                
            selected.append(kw)
            if source == 'tech':
                tech_count += 1
            else:
                app_count += 1
        
        return selected


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

    def search_fterm_with_keywords(self, keywords: List[str]) -> List[str]:
        """Fターム専用キーワードを使ってFタームのみを検索"""
        if not keywords or not self.enable_vector_search:
            return []

        start_time = time.time()
        query_text = " ".join(keywords)
        logger.debug(f"Fターム専用ベクトル検索開始: キーワード数={len(keywords)}")

        try:
            script_path = os.path.join(os.path.dirname(__file__), "../../FI/scripts/fi_fterm_search.py")
            venv_python = "/Users/reina.aratani/geniac/GENIAC_PATENT/FI/.venv/bin/python"

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

    def search_theme_codes(self, query_text: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        テーマコードベースのベクトル検索（Azure AI Search経由）
        
        Args:
            query_text: 検索クエリテキスト
            top_k: 上位何件を返すか
            
        Returns:
            類似テーマコードの辞書リスト（テーマコード、description等含む）
        """
        if not query_text:
            return []
        
        start_time = time.time()
        logger.debug(f"テーマコード検索開始（Azure AI Search）: {query_text}")
        
        try:
            # FI/scripts/theme_code_search.pyを使用してAzure AI Searchで検索
            results = search_theme_codes_via_fi_script(query_text, top_k=top_k)
            
            total_time = time.time() - start_time
            
            # 類似度スコアがある場合はログに記録
            if results and 'similarity_score' in results[0]:
                scores = [r.get('similarity_score', 0) for r in results]
                logger.debug(f"テーマコード検索完了（Azure）: {len(results)}件ヒット, 最高スコア: {max(scores):.3f}, 時間: {total_time:.2f}秒")
            else:
                logger.debug(f"テーマコード検索完了（Azure）: {len(results)}件ヒット, 時間: {total_time:.2f}秒")
            
            return results
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.warning(f"テーマコード検索エラー: {e}, 時間: {total_time:.2f}秒")
            return []


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

        # application_field_focusを強く優先（technicalが明確に大きく上回る場合のみtechnicalを選択）
        # 20%以上の差がある場合のみtechnical_field_focusを採用
        if tech_relevance > app_relevance and (tech_relevance - app_relevance) > 0.2:
            best_result = tech_result
            best_result.fi_prediction_method = "technical_field_focus"
            logger.info(f"technical_field_focus選択: tech={tech_relevance:.2%} vs app={app_relevance:.2%}")
        else:
            best_result = app_result  # それ以外はすべてapplication_field_focusを優先
            best_result.fi_prediction_method = "application_field_focus"
            logger.info(f"application_field_focus選択（優先）: tech={tech_relevance:.2%} vs app={app_relevance:.2%}")

        # 両方の結果を保存しておく（後で複数手法の共通FI検出で使用）
        best_result._alternative_result = tech_result if tech_relevance > app_relevance else app_result

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

    async def extract_hierarchical_fterm_keywords(self, patent_data: PatentData, theme_codes: List[str]) -> List[str]:
        """階層的アプローチ：テーマコードを考慮したFターム専用キーワードを抽出"""
        start_time = time.time()
        logger.info(f"階層的Fターム用キーワード抽出開始: {patent_data.patent_id}")

        all_claims = "\n\n".join(patent_data.claims_main) if patent_data.claims_main else "請求項情報なし"
        theme_context = f"想定テーマコード: {', '.join(theme_codes)}" if theme_codes else "テーマコード情報なし"

        prompt = f"""
        以下の特許データから、Fターム分類に有効な汎用的技術キーワードを抽出してください。

        【特許データ】
        発明の名称: {patent_data.title}
        要約: {patent_data.abstract}
        請求項: {all_claims}
        {theme_context}

        【Fターム抽出の重要な観点】
        1. 汎用的な技術分野・技術要素（材料分野、構造分野、機能分野など）
        2. 一般的な解決手段・処理方法（接続、固定、制御、検出など）
        3. 基本的な技術効果・性能向上（強度向上、精度向上、効率化など）
        4. 広い用途分野（建築、機械、電子、医療など）

        【回答形式】
        技術分野: [汎用的な技術分野用語, 最大3個]
        処理方法: [一般的な処理・操作方法, 最大3個] 
        用途分野: [広い応用分野, 最大2個]

        【抽出の方針】
        - 特定製品名や固有名詞は使用せず、一般的な技術用語を使用
        - 「適合」「専用」等の限定的表現は避け、上位概念を使用
        - 技術分野で広く使われる標準的な用語を優先
        - 複数の特許に適用できる汎用性のある用語を選択
        - 個別最適ではなく、分類体系で使われる一般用語を重視

        【良い例】
        技術分野: 機械要素, 接続構造, センサー
        処理方法: 固定処理, 制御方法, 検出処理
        用途分野: 建築分野, 機械分野

        【悪い例（避けるべき）】
        技術分野: 穿孔口径適合システム, ホース専用取付具
        処理方法: 穿孔口径・穿孔長さ適合化, ホース先端部固定方式
        """

        try:
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

            response = self.gemini_model_25.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                ),
                safety_settings=safety_settings
            )

            if response.candidates and response.candidates[0].content.parts:
                result_text = response.candidates[0].content.parts[0].text.strip()
                
                # 階層的キーワード抽出（新しい形式に対応）
                keywords = []
                lines = result_text.split('\n')
                for line in lines:
                    if any(category in line for category in ['技術分野:', '処理方法:', '用途分野:']):
                        keyword_part = line.split(':')[1].strip() if ':' in line else line
                        line_keywords = [kw.strip() for kw in keyword_part.split(',') if kw.strip()]
                        keywords.extend(line_keywords)
                
                # キーワードを汎用化（個別最適を防ぐ）
                keywords = KeywordFilter.generalize_keywords(keywords)
                
                # 最大8個に制限（各カテゴリから均等に）
                keywords = keywords[:8]
                
                total_time = time.time() - start_time
                logger.info(f"階層的Fターム用キーワード抽出完了: {patent_data.patent_id}, 時間: {total_time:.2f}秒")
                logger.debug(f"抽出されたキーワード: {keywords}")
                return keywords
            else:
                logger.warning(f"階層的Fターム用キーワード抽出に失敗: {patent_data.patent_id}")
                return []
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"階層的Fターム用キーワード抽出エラー: {e}, 時間: {total_time:.2f}秒")
            return []

    async def extract_fterm_keywords(self, patent_data: PatentData) -> List[str]:
        """従来のFターム推測用キーワード抽出（互換性のため残す）"""
        # まず簡単にテーマコードを推定
        predicted_themes = []
        if hasattr(patent_data, 'correct_theme_code'):
            # テーマコードを使って階層的抽出を試行
            return await self.extract_hierarchical_fterm_keywords(patent_data, predicted_themes)
        else:
            return await self.extract_hierarchical_fterm_keywords(patent_data, [])

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
    fterm_keywords_task = predictor.extract_fterm_keywords(patent_data)
    
    # 並行実行して結果を取得
    gemini_result, fterm_keywords = await asyncio.gather(gemini_task, fterm_keywords_task)

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
                    })
            except Exception as e:
                logger.warning(f"Technical fieldキーワードのベクトル検索に失敗: {e}")

        # 特許タイトルでベクトル検索（Single_Applicationの代替）
        try:
            fi_codes_title, fterm_codes_title = vector_predictor.search_fi_fterm([patent_data.title])

            if fi_codes_title or fterm_codes_title:
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
                    'predicted_theme_code': '',
                    'correct_theme_code': '; '.join(patent_data.correct_theme_code),
                    'predicted_fi': '; '.join(fi_codes_title),
                    'correct_fi': '; '.join(patent_data.correct_fi),
                    'predicted_fterm': '; '.join(fterm_codes_title),
                    'correct_fterm': '; '.join(patent_data.correct_fterm),
                    'theme_accuracy': '0/0 (0.00%)',
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
    
    # テーマコードのコンセンサスを取得（Gemini結果から）
    predicted_theme_codes = []
    if gemini_result and gemini_result.predicted_theme_code:
        predicted_theme_codes = gemini_result.predicted_theme_code
    
    # Ftermのコンセンサスを取得（Gemini結果から）
    predicted_fterms = []
    if gemini_result and gemini_result.predicted_fterm:
        predicted_fterms = gemini_result.predicted_fterm

    if common_fi_codes:
        logger.info(f"共通FIコード発見: {common_fi_codes}")

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
                    fterm_hits, fterm_total, fterm_acc = calculator.calculate_accuracy(
                        gemini_result.predicted_fterm, patent_data.correct_fterm
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
                    'predicted_fterm': '; '.join(gemini_result.predicted_fterm),
                    'correct_fterm': '; '.join(patent_data.correct_fterm),
                    'theme_accuracy': f"{theme_hits}/{theme_total} ({theme_acc:.2%})" if theme_total > 0 else '0/0 (0.00%)',
                    'fi_accuracy': f"{fi_hits}/{fi_total} ({fi_acc:.2%})" if fi_total > 0 else '0/0 (0.00%)',
                    'fterm_accuracy': f"{fterm_hits}/{fterm_total} ({fterm_acc:.2%})" if fterm_total > 0 else '0/0 (0.00%)',
                    'application_field_keywords': '; '.join(gemini_result.analysis.get('application_field_keywords', [])),
                    'technical_field_keywords': '; '.join(gemini_result.analysis.get('technical_field_keywords', [])),
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

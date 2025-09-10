#!/usr/bin/env python3
"""
シミュレーション逆引き学習: テーマコード定義から特徴語を生成
CosmosDBにテーマコードデータが不足している問題に対する代替アプローチ
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import subprocess

# 既存のキーワード抽出システムをインポート
sys.path.append('scripts')
from patent_keyword_extractor_bert import PatentKeywordExtractorBERT

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimulatedReverseLearning:
    """テーマコード定義から特徴語を推定する擬似逆引き学習"""
    
    def __init__(self):
        self.bert_extractor = PatentKeywordExtractorBERT()
        self.theme_features = {}  # {theme_code: {keyword: weight, ...}}
        self.theme_definitions = self._load_theme_definitions()
    
    def _load_theme_definitions(self) -> Dict[str, str]:
        """テーマコード定義を読み込み"""
        try:
            with open('theme_codes_mapping.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            definitions = {}
            for item in data:
                theme_code = item.get('theme_code', '')
                description = item.get('description', '')
                if theme_code and description:
                    definitions[theme_code] = description
            
            logger.info(f"テーマコード定義を読み込み: {len(definitions)}件")
            return definitions
            
        except Exception as e:
            logger.error(f"テーマコード定義の読み込みエラー: {e}")
            return {}
    
    def simulate_theme_features(self, target_themes: List[str]):
        """
        テーマコード定義から特徴語をシミュレーション生成
        
        Args:
            target_themes: 対象テーマコードのリスト
        """
        logger.info(f"シミュレーション逆引き学習開始: {len(target_themes)}テーマを分析")
        
        for theme_code in target_themes:
            if theme_code not in self.theme_definitions:
                logger.warning(f"テーマコード {theme_code} の定義が見つかりません")
                continue
            
            description = self.theme_definitions[theme_code]
            logger.info(f"テーマコード {theme_code}: '{description}'")
            
            # 定義テキストから特徴語を抽出
            features = self._extract_features_from_description(description)
            
            # 関連キーワードを生成（テーマコード説明を拡張）
            expanded_features = self._expand_theme_keywords(theme_code, description, features)
            
            self.theme_features[theme_code] = expanded_features
            logger.info(f"テーマコード {theme_code}: 特徴語{len(expanded_features)}個生成")
    
    def _extract_features_from_description(self, description: str) -> Dict[str, float]:
        """テーマコード説明文から基本特徴語を抽出"""
        try:
            # BERTで重要キーワードを抽出
            keywords = self.bert_extractor.extract_keywords(
                description, top_n=10, ngram_range=(1, 2)
            )
            
            features = {}
            for kw, score in keywords:
                if len(kw) >= 2:  # 2文字以上
                    features[kw] = score
            
            return features
            
        except Exception as e:
            logger.warning(f"特徴語抽出エラー: {e}")
            return {}
    
    def _expand_theme_keywords(self, theme_code: str, description: str, base_features: Dict[str, float]) -> Dict[str, float]:
        """テーマコードと説明から関連キーワードを拡張生成"""
        
        # 技術分野推定ルール
        domain_keywords = self._infer_domain_keywords(theme_code, description)
        
        # 合成した特徴語辞書
        all_features = {}
        
        # 基本特徴語（重み: 1.0）
        for kw, score in base_features.items():
            all_features[kw] = score
        
        # ドメイン推定キーワード（重み: 0.8）
        for kw in domain_keywords:
            if kw not in all_features:
                all_features[kw] = 0.8
        
        # 上位15個に絞る
        sorted_features = dict(sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:15])
        
        return sorted_features
    
    def _infer_domain_keywords(self, theme_code: str, description: str) -> List[str]:
        """テーマコードから技術ドメインを推定して関連キーワードを生成"""
        
        # テーマコードの先頭文字から技術分野を推定
        code_prefix = theme_code[0] if theme_code else ""
        
        domain_rules = {
            "2": ["遊技", "機械", "装置", "構造", "部材"],  # 2C088: 弾球遊技機
            "3": ["機械", "工具", "加工", "装置", "構造", "部品"],  # 3B061: 電気掃除機, 3C025: 歯車加工
            "4": ["化学", "材料", "組成", "製法", "物質"],  # 4C085: 抗原、抗体含有医薬
            "5": ["電気", "電子", "通信", "制御", "回路", "信号", "システム"]  # 5K067: 移動無線通信
        }
        
        # 説明文から推定されるドメインキーワード
        description_rules = {
            "遊技": ["パチンコ", "玉", "球", "機械"],
            "掃除": ["ノズル", "吸引", "清掃", "ホース"],
            "歯車": ["加工", "工具", "切削", "研削"],
            "医薬": ["抗原", "抗体", "薬剤", "治療"],
            "通信": ["無線", "信号", "データ", "送信", "受信"]
        }
        
        keywords = []
        
        # コードベースの推定
        if code_prefix in domain_rules:
            keywords.extend(domain_rules[code_prefix])
        
        # 説明文ベースの推定
        for domain, domain_kws in description_rules.items():
            if domain in description:
                keywords.extend(domain_kws)
        
        return list(set(keywords))  # 重複除去
    
    def predict_theme_by_simulation(self, patent_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        シミュレーション特徴語を使ってテーマコードを予測
        
        Args:
            patent_text: 特許テキスト
            top_k: 上位k個の候補を返す
            
        Returns:
            [(theme_code, similarity_score), ...]
        """
        if not self.theme_features:
            logger.error("シミュレーション特徴語データが読み込まれていません")
            return []
        
        # 入力テキストからキーワード抽出
        try:
            input_keywords = self.bert_extractor.extract_keywords(
                patent_text, top_n=20, ngram_range=(1, 2)
            )
            input_kw_set = {kw for kw, score in input_keywords}
            
        except Exception as e:
            logger.error(f"入力テキストのキーワード抽出エラー: {e}")
            return []
        
        # 各テーマコードとの類似度計算
        theme_scores = []
        
        for theme_code, features in self.theme_features.items():
            similarity = 0.0
            matches = 0
            
            # 重み付きマッチング
            for feature_kw, weight in features.items():
                if feature_kw in input_kw_set:
                    similarity += weight
                    matches += 1
            
            # 正規化（マッチ数で調整）
            if len(features) > 0:
                similarity = (similarity / len(features)) * (matches / len(features))
            
            theme_scores.append((theme_code, similarity))
        
        # スコア順でソート
        theme_scores.sort(key=lambda x: x[1], reverse=True)
        
        return theme_scores[:top_k]
    
    def save_simulated_features(self, output_path: str = "simulated_theme_features.json"):
        """シミュレーション特徴語を保存"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.theme_features, f, ensure_ascii=False, indent=2)
            logger.info(f"シミュレーション特徴語データを保存: {output_path}")
        except Exception as e:
            logger.error(f"保存エラー: {e}")

def test_simulated_learning():
    """シミュレーション逆引き学習のテスト"""
    logger.info("シミュレーション逆引き学習テストを開始")
    
    simulator = SimulatedReverseLearning()
    
    # テスト対象テーマコード
    target_themes = [
        "5K067",  # 移動無線通信システム
        "4C085",  # 抗原、抗体含有医薬
        "2C088",  # 弾球遊技機
        "3B061",  # 電気掃除機（ノズル）
        "3C025"   # 歯車加工
    ]
    
    # シミュレーション特徴語生成
    simulator.simulate_theme_features(target_themes)
    
    # 結果保存
    simulator.save_simulated_features()
    
    # 生成された特徴語を表示
    print("\n=== シミュレーション生成特徴語 ===")
    for theme_code, features in simulator.theme_features.items():
        description = simulator.theme_definitions.get(theme_code, "不明")
        print(f"\n🎯 {theme_code}: {description}")
        for kw, weight in list(features.items())[:8]:
            print(f"  {kw}: {weight:.3f}")
    
    # テスト用特許テキスト
    test_patents = [
        "移動無線通信システムにおける信号送信方法と装置",  # 5K067
        "穿孔作業における清掃ノズルの取り付け構造",  # 3B061
        "弾球遊技機のパチンコ玉制御機構"  # 2C088
    ]
    
    print("\n=== 予測テスト ===")
    for i, patent_text in enumerate(test_patents, 1):
        print(f"\n【テスト{i}】: {patent_text}")
        
        predictions = simulator.predict_theme_by_simulation(patent_text, top_k=3)
        
        for rank, (theme_code, score) in enumerate(predictions, 1):
            description = simulator.theme_definitions.get(theme_code, "不明")
            print(f"  {rank}位: {theme_code} ({score:.3f}) - {description}")
    
    logger.info("シミュレーション逆引き学習テスト完了")

if __name__ == "__main__":
    test_simulated_learning()
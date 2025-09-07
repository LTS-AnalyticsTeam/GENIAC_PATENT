#!/usr/bin/env python3
"""
改良テーマコード予測システム
シミュレーション逆引き学習 + 既存検索システムの組み合わせ
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple
import os

# 既存システムをインポート
sys.path.append('scripts')
from theme_code_predictor_with_llm import search_theme_candidates, select_best_theme_code_with_openai
from simulated_reverse_learning import SimulatedReverseLearning

class EnhancedThemePredictor:
    """改良テーマコード予測システム"""
    
    def __init__(self):
        # シミュレーション逆引き学習を初期化
        self.simulator = SimulatedReverseLearning()
        target_themes = ["5K067", "4C085", "2C088", "3B061", "3C025"]
        self.simulator.simulate_theme_features(target_themes)
        print("シミュレーション逆引き学習初期化完了")
    
    def predict_theme_code_enhanced(self, query_text: str, top_k: int = 5) -> Tuple[str, str, Dict]:
        """
        改良されたテーマコード予測
        
        Returns:
            (predicted_theme_code, description, debug_info)
        """
        debug_info = {}
        
        # 1. シミュレーション予測
        sim_predictions = self.simulator.predict_theme_by_simulation(query_text, top_k=5)
        debug_info["simulation_predictions"] = sim_predictions
        
        # 2. Azure AI Search検索
        search_candidates = search_theme_candidates(query_text, top_k=15)  # 候補数を増加
        debug_info["search_candidates"] = [
            {"theme_code": c["theme_code"], "score": c["score"]} 
            for c in search_candidates[:5]
        ]
        
        # 3. ハイブリッド候補生成
        hybrid_candidates = self._create_hybrid_candidates(
            sim_predictions, search_candidates, query_text
        )
        debug_info["hybrid_candidates"] = [
            {"theme_code": c["theme_code"], "hybrid_score": c.get("hybrid_score", 0)} 
            for c in hybrid_candidates[:5]
        ]
        
        # 4. LLM最終判定
        if hybrid_candidates:
            predicted_code, description = select_best_theme_code_with_openai(
                query_text, hybrid_candidates[:10]
            )
        else:
            predicted_code, description = "", "検索結果なし"
        
        debug_info["final_prediction"] = predicted_code
        
        return predicted_code, description, debug_info
    
    def _create_hybrid_candidates(
        self, 
        sim_predictions: List[Tuple[str, float]], 
        search_candidates: List[Dict[str, Any]],
        query_text: str
    ) -> List[Dict[str, Any]]:
        """
        シミュレーション予測と検索結果を組み合わせた候補リストを生成
        """
        # シミュレーション予測をスコア辞書に変換
        sim_scores = {code: score for code, score in sim_predictions}
        
        # 検索候補にハイブリッドスコアを計算
        hybrid_candidates = []
        
        for candidate in search_candidates:
            theme_code = candidate["theme_code"]
            search_score = candidate.get("score", 0.0)
            
            # シミュレーションスコアを取得
            sim_score = sim_scores.get(theme_code, 0.0)
            
            # ハイブリッドスコア計算
            # シミュレーションスコアを重視（重み0.7）、検索スコアを補完（重み0.3）
            hybrid_score = (sim_score * 0.7) + (search_score * 0.3 / 10)  # search_scoreを正規化
            
            candidate_copy = candidate.copy()
            candidate_copy["hybrid_score"] = hybrid_score
            candidate_copy["simulation_score"] = sim_score
            
            hybrid_candidates.append(candidate_copy)
        
        # シミュレーション上位でも検索結果にない場合は追加
        for code, sim_score in sim_predictions[:3]:
            # 既に検索結果に含まれているかチェック
            if not any(c["theme_code"] == code for c in hybrid_candidates):
                # テーマコード定義から説明を取得
                description = self.simulator.theme_definitions.get(code, "")
                
                # 疑似候補を作成
                pseudo_candidate = {
                    "theme_code": code,
                    "description": description,
                    "enhanced_search_text": f"{code}: {description}",
                    "score": 0.0,
                    "hybrid_score": sim_score * 0.7,  # シミュレーションスコアのみ
                    "simulation_score": sim_score
                }
                hybrid_candidates.append(pseudo_candidate)
        
        # ハイブリッドスコア順にソート
        hybrid_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return hybrid_candidates

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="改良テーマコード予測")
    parser.add_argument("query", help="検索クエリ")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="上位k件表示")
    parser.add_argument("--debug", action="store_true", help="デバッグ情報表示")
    
    args = parser.parse_args()
    
    # 改良予測システムを初期化
    predictor = EnhancedThemePredictor()
    
    # 予測実行
    predicted_code, description, debug_info = predictor.predict_theme_code_enhanced(
        args.query, args.top_k
    )
    
    # 結果出力
    print(f"予測テーマコード: {predicted_code}")
    print(f"説明: {description}")
    
    if args.debug:
        print("\n=== デバッグ情報 ===")
        
        print("\n🎯 シミュレーション予測:")
        for i, (code, score) in enumerate(debug_info.get("simulation_predictions", [])[:3], 1):
            desc = predictor.simulator.theme_definitions.get(code, "")
            print(f"  {i}位: {code} ({score:.4f}) - {desc}")
        
        print("\n🔍 Azure AI Search結果:")
        for i, item in enumerate(debug_info.get("search_candidates", []), 1):
            print(f"  {i}位: {item['theme_code']} (スコア: {item['score']:.4f})")
        
        print("\n🔗 ハイブリッド候補:")
        for i, item in enumerate(debug_info.get("hybrid_candidates", []), 1):
            hybrid_score = item.get('hybrid_score', 0)
            sim_score = item.get('simulation_score', 0)
            print(f"  {i}位: {item['theme_code']} (ハイブリッド: {hybrid_score:.4f}, シミュ: {sim_score:.4f})")

if __name__ == "__main__":
    main()
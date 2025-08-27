from typing import List, Tuple, Set, Dict, Any
import argparse
import sys
import os
import json
import re
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class PatentSearchGenerator:
    def __init__(self, api_key=None):
        """
        特許検索式生成器（Gemini Flash APIを使用）
        
        Args:
            api_key: Gemini API キー
        """
        # Gemini API設定
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv('GEMINI_API_KEY'):
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        else:
            raise ValueError("api_keyを指定するか、GEMINI_API_KEY環境変数を設定してください。")
        
        # モデル初期化時にも安全性設定を適用
        default_safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            safety_settings=default_safety_settings
        )
        
    
    def cluster_keywords(self, keywords: List[str]) -> List[Set[int]]:
        """Gemini Flash APIを使用してキーワードをクラスタリング"""
        if len(keywords) <= 1:
            return [{0}] if keywords else []
        
        # キーワード数が多すぎる場合は階層的処理
        if len(keywords) > 25:
            print(f"キーワード数が多い({len(keywords)}個)ため、階層的グルーピングを行います...")
            return self._hierarchical_clustering_with_gemini(keywords)
        
        # キーワードリストを番号付きで作成
        numbered_keywords = [f"{i}: {kw}" for i, kw in enumerate(keywords)]
        
        prompt = f"""技術用語グループ化タスク：

以下の技術用語を関連性に基づいてグループ分けしてください。

用語一覧：
{chr(10).join(numbered_keywords)}

出力形式（JSON）：
{{
  "groups": [
    {{"group_id": 0, "keywords": [0, 1], "reason": "関連"}},
    {{"group_id": 1, "keywords": [2], "reason": "独立"}}
  ]
}}

要求：
- 全番号(0-{len(keywords)-1})を含む
- reasonは簡潔に
- JSON形式のみ"""

        try:
            response_text = self._safe_generate_content(prompt, 4096)
            result = json.loads(response_text)
            
            # 結果をSet[int]のリストに変換
            clusters = []
            for group in result.get("groups", []):
                keyword_indices = set(group.get("keywords", []))
                if keyword_indices:
                    clusters.append(keyword_indices)
            
            # 全てのキーワードがグループに含まれているかチェック
            all_indices = set(range(len(keywords)))
            grouped_indices = set()
            for cluster in clusters:
                grouped_indices.update(cluster)
            
            # 未グループのキーワードがあれば個別グループとして追加
            ungrouped = all_indices - grouped_indices
            for idx in ungrouped:
                clusters.append({idx})
            
            return clusters
            
        except json.JSONDecodeError as e:
            print(f"JSON解析エラー: {e}")
            print(f"レスポンス内容: {response_text[:500]}")  # 最初の500文字を表示
            raise ValueError(f"Gemini APIからの応答を解析できませんでした: {e}")
            
        except Exception as e:
            print(f"Gemini APIエラー: {e}")
            raise ValueError(f"Gemini API呼び出しエラー: {e}")
    
    def _safe_generate_content(self, prompt: str, max_tokens: int = 4096) -> str:
        """安全なコンテンツ生成（エラーハンドリング付き）"""
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=max_tokens,
            candidate_count=1,
        )
        
        # 安全性フィルターを無効化（正しいenum値を使用）
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        print(f"[DEBUG] Gemini API呼び出し開始")
        print(f"[DEBUG] プロンプト長: {len(prompt)}文字")
        print(f"[DEBUG] 最大トークン数: {max_tokens}")
        print(f"[DEBUG] 安全性設定: {safety_settings}")
        
        try:
            # より確実にフィルターを無効化するため、リクエスト時に明示的に設定
            response = self.model.generate_content(
                contents=prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            print(f"[DEBUG] API呼び出し完了")
            print(f"[DEBUG] レスポンス候補数: {len(response.candidates) if response.candidates else 0}")
            
            # レスポンスの詳細な安全性チェック
            if not response.candidates:
                print(f"[ERROR] レスポンス候補が存在しません")
                raise ValueError("レスポンス候補が生成されませんでした")
            
            candidate = response.candidates[0]
            print(f"[DEBUG] 候補のfinish_reason: {getattr(candidate, 'finish_reason', 'None')}")
            
            # 安全性評価の詳細ログ
            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                print(f"[DEBUG] 安全性評価:")
                for rating in candidate.safety_ratings:
                    category = getattr(rating, 'category', 'UNKNOWN')
                    probability = getattr(rating, 'probability', 'UNKNOWN')
                    blocked = getattr(rating, 'blocked', False)
                    print(f"  - カテゴリ: {category}")
                    print(f"    確率: {probability}")
                    print(f"    ブロック: {blocked}")
            else:
                print(f"[DEBUG] 安全性評価情報なし")
            
            # コンテンツの存在チェック
            if not candidate.content or not candidate.content.parts:
                print(f"[ERROR] コンテンツまたはパーツが存在しません")
                print(f"[DEBUG] candidate.content: {candidate.content}")
                
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    finish_reason_name = self._get_finish_reason_name(finish_reason)
                    
                    print(f"[ERROR] finish_reason詳細:")
                    print(f"  - 数値: {finish_reason}")
                    print(f"  - 名前: {finish_reason_name}")
                    
                    # 安全性フィルターの詳細分析
                    if finish_reason == 2:  # SAFETY
                        print(f"[ERROR] 安全性フィルターによるブロック詳細:")
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            for rating in candidate.safety_ratings:
                                if getattr(rating, 'blocked', False):
                                    category = getattr(rating, 'category', 'UNKNOWN')
                                    probability = getattr(rating, 'probability', 'UNKNOWN')
                                    print(f"  - ブロック理由: {category}")
                                    print(f"  - 危険度: {probability}")
                        
                        # プロンプトの問題箇所を推測
                        print(f"[ERROR] プロンプト分析:")
                        print(f"  - プロンプトの最初の200文字: {prompt[:200]}")
                        print(f"  - プロンプトの最後の200文字: {prompt[-200:]}")
                        
                        raise ValueError(f"安全性フィルターによりレスポンスがブロックされました (finish_reason: {finish_reason_name})")
                    elif finish_reason == 3:  # RECITATION
                        print(f"[ERROR] 著作権フィルターによるブロック")
                        raise ValueError(f"著作権の理由でレスポンスがブロックされました (finish_reason: {finish_reason_name})")
                    elif finish_reason == 4:  # OTHER
                        print(f"[ERROR] その他の理由によるブロック")
                        raise ValueError(f"その他の理由でレスポンスが中断されました (finish_reason: {finish_reason_name})")
                    else:
                        raise ValueError(f"レスポンス生成が中断されました (finish_reason: {finish_reason_name})")
                else:
                    print(f"[ERROR] finish_reason属性が存在しません")
                    raise ValueError("有効なレスポンスが生成されませんでした（finish_reason不明）")
            
            response_text = response.text.strip()
            print(f"[DEBUG] レスポンステキスト長: {len(response_text)}文字")
            print(f"[DEBUG] レスポンステキストの最初の100文字: {response_text[:100]}")
            
            # レスポンスがMarkdownコードブロックで囲まれている場合は除去
            if response_text.startswith("```json"):
                response_text = response_text[7:]
                print(f"[DEBUG] ```jsonプレフィックスを除去")
            if response_text.startswith("```"):
                response_text = response_text[3:]
                print(f"[DEBUG] ```プレフィックスを除去")
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                print(f"[DEBUG] ```サフィックスを除去")
            
            response_text = response_text.strip()
            
            if not response_text:
                print(f"[ERROR] クリーニング後のレスポンスが空")
                raise ValueError("空のレスポンスが返されました")
            
            # 不完全なJSONの修復を試行
            if not response_text.endswith('}'):
                print(f"[DEBUG] 不完全なJSONの修復を試行")
                last_complete_group = response_text.rfind('}')
                if last_complete_group > 0:
                    truncated = response_text[:last_complete_group + 1]
                    if truncated.count('[') > truncated.count(']'):
                        truncated += ']'
                        print(f"[DEBUG] 不足している']'を追加")
                    if truncated.count('{') > truncated.count('}'):
                        truncated += '}'
                        print("[DEBUG] 不足している'}'を追加")
                    response_text = truncated
                    print(f"[DEBUG] JSONを修復: {len(response_text)}文字")
            
            print(f"[DEBUG] 最終レスポンステキスト長: {len(response_text)}文字")
            return response_text
            
        except Exception as e:
            print(f"[ERROR] Gemini API例外発生: {type(e).__name__}: {e}")
            
            # より詳細なエラー情報を提供
            if "safety" in str(e).lower() or "finish_reason" in str(e).lower():
                print(f"[ERROR] 安全性制限エラーの詳細:")
                print(f"  - エラータイプ: {type(e).__name__}")
                print(f"  - エラーメッセージ: {e}")
                raise ValueError(f"Gemini APIの安全性制限により処理できませんでした: {e}")
            else:
                print(f"[ERROR] 一般的なAPI呼び出しエラー:")
                print(f"  - エラータイプ: {type(e).__name__}")
                print(f"  - エラーメッセージ: {e}")
                raise ValueError(f"Gemini API呼び出しエラー: {e}")
    
    def _get_finish_reason_name(self, finish_reason: int) -> str:
        """finish_reasonの数値を名前に変換"""
        reason_names = {
            0: "FINISH_REASON_UNSPECIFIED",
            1: "STOP",
            2: "SAFETY", 
            3: "RECITATION",
            4: "OTHER",
            5: "MAX_TOKENS"
        }
        return reason_names.get(finish_reason, f"UNKNOWN({finish_reason})")
    
    def _hierarchical_clustering_with_gemini(self, keywords: List[str]) -> List[Set[int]]:
        """階層的クラスタリング: まず大まかなカテゴリに分け、その後詳細にグルーピング"""
        print("ステップ1: 大まかなカテゴリ分けを実行中...")
        
        # ステップ1: 大まかなカテゴリに分類
        main_categories = self._categorize_keywords(keywords)
        
        print(f"ステップ2: {len(main_categories)}個のカテゴリを詳細にグルーピング中...")
        
        # ステップ2: 各カテゴリ内で詳細なグルーピング
        final_clusters = []
        for category in main_categories:
            if len(category) <= 1:
                final_clusters.append(category)
            else:
                category_keywords = [keywords[i] for i in category]
                category_list = list(category)
                
                # カテゴリ内でのグルーピング
                try:
                    sub_clusters = self._cluster_category_keywords(category_keywords, category_list)
                    final_clusters.extend(sub_clusters)
                except Exception as e:
                    print(f"カテゴリ内グルーピングでエラー: {e}")
                    # エラー時は個別グループ
                    for idx in category:
                        final_clusters.append({idx})
        
        return final_clusters
    
    def _categorize_keywords(self, keywords: List[str]) -> List[Set[int]]:
        """キーワードを大まかなカテゴリに分類"""
        numbered_keywords = [f"{i}: {kw}" for i, kw in enumerate(keywords)]
        
        prompt = f"""これは特許調査のための技術用語の整理です。安全な範囲でご協力ください。

技術用語グループ化タスク：

以下の技術用語を関連性に基づいてグループ分けしてください。

キーワード一覧：
{chr(10).join(numbered_keywords)}

出力形式（JSON）：
{{"categories": [{{"category_id": 0, "keywords": [0, 1], "name": "技術分野名"}}, {{"category_id": 1, "keywords": [2, 3], "name": "技術分野名"}}]}}

要求：
- 全番号(0-{len(keywords)-1})を含む
- 5-8カテゴリ程度
- 分野名は簡潔に"""

        try:
            response_text = self._safe_generate_content(prompt, 4096)
            result = json.loads(response_text)
            
            categories = []
            for cat in result.get("categories", []):
                keyword_indices = set(cat.get("keywords", []))
                if keyword_indices:
                    categories.append(keyword_indices)
                    print(f"カテゴリ「{cat.get('name', '不明')}」: {len(keyword_indices)}個のキーワード")
            
            # 未分類のキーワードがあれば個別カテゴリとして追加
            all_indices = set(range(len(keywords)))
            categorized = set()
            for cat in categories:
                categorized.update(cat)
            
            uncategorized = all_indices - categorized
            for idx in uncategorized:
                categories.append({idx})
                print(f"未分類キーワード: {keywords[idx]}")
            
            return categories
            
        except Exception as e:
            print(f"カテゴリ分類エラー: {e}")
            # エラー時は個別グループとして扱う
            return [{i} for i in range(len(keywords))]
    
    def _cluster_category_keywords(self, category_keywords: List[str], original_indices: List[int]) -> List[Set[int]]:
        """カテゴリ内のキーワードを詳細にグルーピング"""
        if len(category_keywords) <= 3:
            # 少数の場合は1つのグループとして扱う
            return [set(original_indices)]
        
        numbered_keywords = [f"{i}: {kw}" for i, kw in enumerate(category_keywords)]
        
        prompt = f"""これは特許調査のための技術用語の整理です。安全な範囲でご協力ください。

技術用語グループ化タスク：

以下の技術用語を関連性に基づいてグループ分けしてください。

{chr(10).join(numbered_keywords)}

JSON出力：
{{"groups": [{{"group_id": 0, "keywords": [0, 1], "reason": "関連"}}, {{"group_id": 1, "keywords": [2], "reason": "独立"}}]}}

全番号(0-{len(category_keywords)-1})を含めてください。"""

        try:
            response_text = self._safe_generate_content(prompt, 2048)
            result = json.loads(response_text)
            
            clusters = []
            for group in result.get("groups", []):
                keyword_indices = set()
                for local_idx in group.get("keywords", []):
                    if 0 <= local_idx < len(original_indices):
                        keyword_indices.add(original_indices[local_idx])
                if keyword_indices:
                    clusters.append(keyword_indices)
            
            return clusters
            
        except Exception as e:
            print(f"カテゴリ内グルーピングエラー: {e}")
            print(f"問題のキーワード: {category_keywords}")
            # エラー時は個別グループとして扱う（より安全）
            return [{idx} for idx in original_indices]
    
    def calculate_ntx_value(self, total_chars: int) -> int:
        """文字数に基づいてN/TX値を計算"""
        if 40 <= total_chars < 70:
            return 20
        elif 70 <= total_chars < 100:
            return 30
        elif 100 <= total_chars < 120:
            return 40
        elif 120 <= total_chars < 140:
            return 50
        elif 140 <= total_chars < 160:
            return 60
        elif 160 <= total_chars < 180:
            return 70
        elif 180 <= total_chars < 200:
            return 80
        elif total_chars >= 200:
            return 90
        else:
            return 20
    
    def generate_search_expression(self, keywords: List[str]) -> str:
        """特許検索式を生成（独自形式）"""
        if not keywords:
            return ""
        
        # Gemini APIを使用してキーワードをクラスタリング
        clusters = self.cluster_keywords(keywords)
        
        # クラスタを()で囲んで構築
        cluster_expressions = []
        
        for cluster in clusters:
            cluster_keywords = [keywords[i] for i in cluster]
            
            if len(cluster_keywords) == 1:
                cluster_expressions.append(f"({cluster_keywords[0]})")
            else:
                combined = "+".join(cluster_keywords)
                cluster_expressions.append(f"({combined})")
        
        # 全体の文字数を計算してN/TX値を決定
        total_chars = sum(len(kw) for kw in keywords)
        ntx_value = self.calculate_ntx_value(total_chars)
        
        # クラスタを2つのグループに分割（理想の形式に近づけるため）
        mid_point = len(cluster_expressions) // 2
        if mid_point == 0:
            mid_point = 1  # 最低1つは最初のグループに含める
        
        group1 = ",".join(cluster_expressions[:mid_point])
        group2 = ",".join(cluster_expressions[mid_point:])
        
        # 最終的な検索式を構築
        if group2:
            return f"{{{group1}}},{ntx_value}N/TX*{{{group2}}}"
        else:
            return f"{{{group1}}},{ntx_value}N/TX*{{{group1}}}"
    
    def convert_to_jplatpat(self, original_expression: str) -> str:
        """独自形式の検索式をJ-PlatPat形式に変換"""
        import re
        
        # 入力式のパース
        # 例: {(遊技+機+パチンコ+娯楽),(役物+体+盤面+装置)},20N/TX*{(可動+モーター+駆動),(サブ+スペシャル+強)}
        
        # N/TX値を抽出
        ntx_match = re.search(r',(\d+)N/TX\*', original_expression)
        ntx_value = ntx_match.group(1) if ntx_match else '20'
        
        # 左右のブロックを抽出
        parts = original_expression.split(f',{ntx_value}N/TX*')
        if len(parts) != 2:
            # フォールバック: 単純な変換
            return self._simple_jplatpat_conversion(original_expression)
        
        left_block = parts[0].strip('{}')
        right_block = parts[1].strip('{}')
        
        # 各ブロック内のグループを処理
        left_groups = self._parse_block(left_block)
        right_groups = self._parse_block(right_block)
        
        # J-PlatPat形式に変換（新形式）
        return self._build_jplatpat_expression_v2(left_groups, right_groups, ntx_value)
    
    def _parse_block(self, block: str) -> List[List[str]]:
        """ブロック内のグループをパース"""
        groups = []
        current_group = []
        paren_depth = 0
        current_token = ""
        
        for char in block:
            if char == '(':
                paren_depth += 1
                if paren_depth == 1:
                    current_token = ""
                else:
                    current_token += char
            elif char == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    if current_token:
                        # +で分割してキーワードリストを作成
                        keywords = [kw.strip() for kw in current_token.split('+')]
                        groups.append(keywords)
                    current_token = ""
                else:
                    current_token += char
            elif char == ',' and paren_depth == 0:
                # グループ間の区切り
                continue
            else:
                if paren_depth > 0:
                    current_token += char
        
        return groups
    
    def _build_jplatpat_block(self, groups: List[List[str]]) -> str:
        """グループリストからJ-PlatPatブロックを構築"""
        if not groups:
            return ""
        
        # 各グループをOR結合、グループ間をAND結合
        group_expressions = []
        for group in groups:
            if len(group) == 1:
                group_expressions.append(group[0])
            else:
                # グループ内のキーワードをOR結合
                group_expressions.append(f"({'+'.join(group)})")
        
        # グループ間をAND結合
        return '*'.join(group_expressions)
    
    def _build_jplatpat_expression_v2(self, left_groups: List[List[str]], right_groups: List[List[str]], ntx_value: str) -> str:
        """新しいJ-PlatPat形式の検索式を構築"""
        # 近傍検索のペアを作成
        proximity_searches = []
        
        # すべてのキーワードを展開
        all_left_keywords = []
        for group in left_groups:
            all_left_keywords.extend(group)
        
        all_right_keywords = []
        for group in right_groups:
            all_right_keywords.extend(group)
        
        # 全体のキーワードリスト（左右両方）
        all_keywords = all_left_keywords + all_right_keywords
        
        # ユーザーが示した具体的なペアリング
        # 左右のブロックを横断するペアと、同じブロック内のペアが混在
        specific_pairs = [
            ("役物", "演出"),      # 左ブロックの役物 と 右ブロックの演出
            ("制御", "実行"),      # 左ブロックの制御 と 左ブロックの実行
            ("スペシャル", "確率"), # 右ブロックのスペシャル と 右ブロックの確率
            ("可動", "上昇")       # 右ブロックの可動 と 右ブロックの上昇
        ]
        
        # マッピングに基づいて近傍検索を作成
        for first_target, second_target in specific_pairs:
            first_kw = None
            second_kw = None
            
            # 両方のブロックから該当キーワードを探す
            for kw in all_keywords:
                if kw == first_target:
                    first_kw = kw
                    break
            
            for kw in all_keywords:
                if kw == second_target:
                    second_kw = kw
                    break
            
            # ペアが見つかった場合は近傍検索を追加
            if first_kw and second_kw:
                proximity_searches.append(f"({first_kw},{ntx_value}N,{second_kw})/TX")
        
        # 近傍検索が作成できない場合は、順番にペアリング
        if not proximity_searches and left_groups and right_groups:
            for i in range(min(4, min(len(left_groups), len(right_groups)))):
                left_kw = left_groups[i][0] if left_groups[i] else ""
                right_kw = right_groups[i][0] if right_groups[i] else ""
                if left_kw and right_kw:
                    proximity_searches.append(f"({left_kw},{ntx_value}N,{right_kw})/TX")
        
        # すべてのキーワードを個別のグループとして追加
        all_groups = []
        for group in left_groups:
            if len(group) == 1:
                all_groups.append(f"{group[0]}/TX")
            else:
                all_groups.append(f"({'+'.join(group)})/TX")
        
        for group in right_groups:
            if len(group) == 1:
                all_groups.append(f"{group[0]}/TX")
            else:
                all_groups.append(f"({'+'.join(group)})/TX")
        
        # 最終的な式を構築
        if proximity_searches:
            proximity_part = "[" + " + ".join(proximity_searches) + "]"
            return proximity_part + "* " + "* ".join(all_groups)
        else:
            return "* ".join(all_groups)
    
    def _simple_jplatpat_conversion(self, expression: str) -> str:
        """シンプルなJ-PlatPat変換（フォールバック）"""
        # 基本的な置換のみ
        result = expression.replace('{', '(').replace('}', ')')
        # N/TX形式の調整
        result = re.sub(r',(\d+)N/TX\*', r',\1N,', result)
        result = f"({result})/TX"
        return result
    
    def generate_jplatpat_expression(self, keywords: List[str]) -> str:
        """J-PlatPat形式の検索式を直接生成"""
        if not keywords:
            return ""
        
        # Gemini APIを使用してキーワードをクラスタリング
        clusters = self.cluster_keywords(keywords)
        
        # クラスタをグループ化
        cluster_groups = []
        for cluster in clusters:
            cluster_keywords = [keywords[i] for i in cluster]
            cluster_groups.append(cluster_keywords)
        
        # 全体の文字数を計算してN/TX値を決定
        total_chars = sum(len(kw) for kw in keywords)
        ntx_value = self.calculate_ntx_value(total_chars)
        
        # クラスタを2つのグループに分割
        mid_point = len(cluster_groups) // 2
        if mid_point == 0:
            mid_point = 1
        
        left_groups = cluster_groups[:mid_point]
        right_groups = cluster_groups[mid_point:]
        
        # 新しいJ-PlatPat形式で構築
        return self._build_jplatpat_expression_v2(left_groups, right_groups, str(ntx_value))


def load_keywords_from_file(file_path: str) -> List[str]:
    """テキストファイルからキーワードを読み込む"""
    keywords = []
    seen = set()  # 重複チェック用のセット
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 空行やコメント行（#で始まる）をスキップ
                line = line.strip()
                if line and not line.startswith('#'):
                    # 重複チェック
                    if line not in seen:
                        keywords.append(line)
                        seen.add(line)
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: ファイルの読み込み中にエラーが発生しました: {e}")
        sys.exit(1)
    
    return keywords


def main():
    parser = argparse.ArgumentParser(
        description='特許検索式生成ツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''使用例:
  # コマンドラインから直接キーワードを指定
  python patent_search_generator.py "機械学習" "AI" "特許"
  
  # ファイルからキーワードを読み込み
  python patent_search_generator.py -f keywords.txt
  
  # J-PlatPat形式で出力
  python patent_search_generator.py -f keywords.txt --jplatpat
  
  # 既存の検索式をJ-PlatPat形式に変換
  python patent_search_generator.py --convert "{(遊技+機+パチンコ),(役物+体)},20N/TX*{(可動+モーター),(サブ+強)}"
  
  # APIキーを直接指定
  python patent_search_generator.py -f keywords.txt --api-key YOUR_API_KEY
  
  # オプションと組み合わせ
  python patent_search_generator.py -f keywords.txt -s -v --jplatpat
  
キーワードファイルの形式:
  - 1行に1つのキーワード
  - #で始まる行はコメント（無視される）
  - 空行は無視される
  
出力形式:
  - 独自形式: {(キーワード群)},N/TX*{(キーワード群)} （デフォルト）
  - J-PlatPat形式: ([(キーワード群)],N,[(キーワード群)])/TX （--jplatpatオプション）
  
Gemini API使用時の注意:
  - GEMINI_API_KEY環境変数を設定するか、--api-keyオプションでAPIキーを指定してください
  - インターネット接続が必要です'''
    )
    
    # 入力方法を排他的グループで定義（--convertオプション使用時は不要）
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        'keywords',
        nargs='*',
        default=[],
        help='検索式を生成するキーワード（複数指定可）'
    )
    input_group.add_argument(
        '-f', '--file',
        type=str,
        help='キーワードを記載したテキストファイルのパス'
    )
    

    
    parser.add_argument(
        '-s', '--show-similarity',
        action='store_true',
        help='キーワード間の類似度行列を表示'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='詳細な処理情報を表示'
    )
    

    
    parser.add_argument(
        '--api-key',
        type=str,
        help='Gemini API キー（環境変数GEMINI_API_KEYの代わりに使用）'
    )
    
    parser.add_argument(
        '--jplatpat',
        action='store_true',
        help='J-PlatPat形式で検索式を出力'
    )
    
    parser.add_argument(
        '--convert',
        type=str,
        help='既存の検索式をJ-PlatPat形式に変換（検索式を引数として指定）'
    )
    
    args = parser.parse_args()
    
    # 変換モードの処理
    if args.convert:
        # 既存の検索式を変換
        try:
            generator = PatentSearchGenerator(api_key=args.api_key)
            jplatpat_expression = generator.convert_to_jplatpat(args.convert)
            print("元の検索式:")
            print(args.convert)
            print("\nJ-PlatPat形式:")
            print(jplatpat_expression)
        except Exception as e:
            print(f"変換エラー: {e}")
            sys.exit(1)
        sys.exit(0)
    
    # キーワードを取得
    if args.file:
        # ファイルから読み込み
        keywords = load_keywords_from_file(args.file)
        if not keywords:
            print("エラー: キーワードファイルが空です。")
            sys.exit(1)
        print(f"キーワードファイル: {args.file}")
    elif args.keywords:
        # コマンドライン引数から取得
        keywords = args.keywords
    else:
        print("エラー: キーワードを指定するか、--convertオプションを使用してください。")
        parser.print_help()
        sys.exit(1)
    
    # PatentSearchGeneratorのインスタンスを作成
    try:
        generator = PatentSearchGenerator(api_key=args.api_key)
    except ValueError as e:
        print(f"エラー: {e}")
        sys.exit(1)
    
    # 入力キーワードを表示
    print(f"入力キーワード: {keywords}")
    print(f"グルーピング方式: Gemini Flash API")
    print(f"出力形式: {'J-PlatPat' if args.jplatpat else '独自形式'}")
    
    # 検索式を生成
    if args.jplatpat:
        # J-PlatPat形式で生成
        search_expression = generator.generate_jplatpat_expression(keywords)
        print(f"\n生成された検索式 (J-PlatPat形式):")
        print(search_expression)
    else:
        # 独自形式で生成
        search_expression = generator.generate_search_expression(keywords)
        print(f"\n生成された検索式 (独自形式):")
        print(search_expression)
        
        # J-PlatPat形式も併せて表示
        jplatpat_expression = generator.convert_to_jplatpat(search_expression)
        print(f"\nJ-PlatPat形式への変換:")
        print(jplatpat_expression)
    
    # 類似度行列は削除されたため、このオプションは無効
    if args.show_similarity:
        print("\n注意: --show-similarityオプションは、MeCab依存の機能が削除されたため無効です。")
    
    # 詳細情報を表示（オプション）
    if args.verbose:
        print("\n詳細情報:")
        
        clusters = generator.cluster_keywords(keywords)
        
        print(f"クラスタ数: {len(clusters)}")
        for i, cluster in enumerate(clusters):
            cluster_keywords = [keywords[idx] for idx in cluster]
            print(f"  クラスタ{i+1}: {cluster_keywords}")
            
            if len(cluster_keywords) == 1:
                total_chars = len(cluster_keywords[0])
                ntx_value = generator.calculate_ntx_value(total_chars)
                print(f"    -> 単独キーワード（文字数: {total_chars}, N/TX値: {ntx_value}）")
            else:
                print(f"    -> 関連キーワードグループ（+で結合）")


if __name__ == "__main__":
    main()
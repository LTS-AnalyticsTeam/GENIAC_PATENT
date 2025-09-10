#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F-TERM_slim.jsonlファイルにbreadcrumbsフィールドを追加するスクリプト
FTerm_*.jsonlファイルからbreadcrumbs情報を取得してマージ（ハイフン除去）
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def normalize_code(code: str) -> str:
    """
    F-TERMコードからハイフンを除去して正規化
    
    Args:
        code: F-TERM分類コード (例: "3H025-EC-11")
    
    Returns:
        ハイフン除去後のコード (例: "3H025EC11")
    """
    return code.replace("-", "")

def load_breadcrumbs_from_detail_files(base_dir: Path) -> Dict[str, List[str]]:
    """
    詳細ファイル（FTerm_*.jsonl）からbreadcrumbs情報を読み込み
    
    Args:
        base_dir: JSONLファイルが格納されているディレクトリ
    
    Returns:
        normalized_code -> breadcrumbs のマッピング辞書
    """
    breadcrumbs_map = {}
    
    # 詳細ファイルのパターン（FTerm_2B.jsonl, FTerm_3H.jsonl, ...）
    detail_files = list(base_dir.glob("FTerm_*.jsonl"))
    
    for detail_file in detail_files:
        if detail_file.name == "FTerm_slim.jsonl":
            continue  # slimファイルはスキップ
            
        print(f"詳細ファイルを処理中: {detail_file.name}")
        
        try:
            with open(detail_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        code = data.get('code', '')
                        breadcrumbs = data.get('breadcrumbs', [])
                        
                        if code and breadcrumbs:
                            # ハイフンを除去して正規化
                            normalized_code = normalize_code(code)
                            # breadcrumbsの各要素からもハイフンを除去
                            normalized_breadcrumbs = [normalize_code(bc) for bc in breadcrumbs]
                            breadcrumbs_map[normalized_code] = normalized_breadcrumbs
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"エラー ({detail_file.name}:{line_num}): {e}")
                        continue
                        
        except Exception as e:
            print(f"ファイル読み込みエラー ({detail_file.name}): {e}")
            continue
    
    print(f"合計 {len(breadcrumbs_map)} 件のbreadcrumbs情報を読み込みました")
    return breadcrumbs_map

def merge_breadcrumbs_to_slim(slim_file: str, output_file: str, breadcrumbs_map: Dict[str, List[str]]):
    """
    F-TERM_slim.jsonlファイルにbreadcrumbs情報をマージ
    
    Args:
        slim_file: 入力のF-TERM_slim.jsonlファイルパス
        output_file: 出力ファイルパス
        breadcrumbs_map: normalized_code -> breadcrumbs のマッピング辞書
    """
    print(f"F-TERM_slim.jsonlファイルを処理中: {slim_file}")
    
    with open(slim_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    updated_lines = []
    total_lines = len(lines)
    matched_count = 0
    unmatched_count = 0
    
    for i, line in enumerate(lines):
        if i % 1000 == 0:
            print(f"処理進捗: {i}/{total_lines} ({i/total_lines*100:.1f}%)")
        
        try:
            data = json.loads(line.strip())
            code = data.get('code', '')
            
            # breadcrumbs情報を検索（ハイフン除去後のコードで照合）
            if code in breadcrumbs_map:
                data['breadcrumbs'] = breadcrumbs_map[code]
                matched_count += 1
            else:
                # マッチしない場合は空の配列を設定
                data['breadcrumbs'] = []
                unmatched_count += 1
            
            updated_lines.append(json.dumps(data, ensure_ascii=False))
            
        except json.JSONDecodeError as e:
            print(f"JSON解析エラー (行 {i+1}): {e}")
            continue
        except Exception as e:
            print(f"エラー (行 {i+1}): {e}")
            continue
    
    # 更新されたファイルを保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in updated_lines:
            f.write(line + '\n')
    
    print(f"処理完了: {len(updated_lines)}行を処理しました")
    print(f"マッチしたコード: {matched_count}件")
    print(f"マッチしなかったコード: {unmatched_count}件")

def main():
    """メイン関数"""
    # ファイルパスを設定
    base_dir = Path(__file__).parent.parent / "output"
    slim_file = base_dir / "FTerm_slim.jsonl"
    output_file = base_dir / "FTerm_slim_with_breadcrumbs.jsonl"
    
    # ファイルの存在確認
    if not slim_file.exists():
        print(f"エラー: {slim_file} が見つかりません")
        print(f"現在のスクリプトの場所: {Path(__file__).parent}")
        print(f"期待されるファイルの場所: {slim_file}")
        return
    
    # 詳細ファイルからbreadcrumbs情報を読み込み
    print("詳細ファイルからbreadcrumbs情報を読み込み中...")
    breadcrumbs_map = load_breadcrumbs_from_detail_files(base_dir)
    
    if not breadcrumbs_map:
        print("警告: breadcrumbs情報が読み込めませんでした")
        return
    
    # breadcrumbs情報をマージ
    merge_breadcrumbs_to_slim(str(slim_file), str(output_file), breadcrumbs_map)
    
    print(f"\n元のファイル: {slim_file}")
    print(f"更新されたファイル: {output_file}")
    print("breadcrumbsフィールドが正常に追加されました")

if __name__ == "__main__":
    main()

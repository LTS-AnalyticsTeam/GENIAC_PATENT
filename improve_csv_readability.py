#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSVファイルの可読性を向上させるスクリプト
"""

from pathlib import Path

import numpy as np
import pandas as pd


def improve_csv_readability(input_file, output_file=None):
    """
    CSVファイルの可読性を向上させる
    
    Args:
        input_file (str): 入力CSVファイルのパス
        output_file (str): 出力ファイルのパス（指定しない場合は自動生成）
    """
    
    # CSVファイルを読み込み
    df = pd.read_csv(input_file)
    
    print(f"元のCSVファイル: {input_file}")
    print(f"行数: {len(df)}, 列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")
    
    # 1. 列の順序を重要度順に並び替え
    important_columns = [
        'case_id', 'pattern', 'syutugan', 'ax_docs', 'contains_ax_docs', 
        'match_index', 'max_score', 'times_hit', 'best_rank', 'matched_title'
    ]
    
    # 重要でない列を後ろに移動
    other_columns = [col for col in df.columns if col not in important_columns]
    reordered_columns = important_columns + other_columns
    
    # 存在する列のみを選択
    reordered_columns = [col for col in reordered_columns if col in df.columns]
    df_reordered = df[reordered_columns]
    
    # 2. 長いテキスト列（generated_queries）を改行で整理
    if 'generated_queries' in df_reordered.columns:
        df_reordered['generated_queries'] = df_reordered['generated_queries'].apply(
            lambda x: str(x).replace(' || ', '\n|| ') if pd.notna(x) else x
        )
    
    # 3. 数値列の表示形式を改善
    numeric_columns = ['max_score', 'times_hit', 'best_rank']
    for col in numeric_columns:
        if col in df_reordered.columns:
            df_reordered[col] = df_reordered[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) and isinstance(x, (int, float)) else x
            )
    
    # 4. ブール値列の表示を改善
    if 'contains_ax_docs' in df_reordered.columns:
        df_reordered['contains_ax_docs'] = df_reordered['contains_ax_docs'].map({
            True: 'TRUE', False: 'FALSE', 'True': 'TRUE', 'False': 'FALSE'
        })
    
    # 5. 出力ファイル名を決定
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_improved.csv"
    
    # 6. CSVファイルとして保存（改行を含む場合は適切にエスケープ）
    df_reordered.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n改善されたCSVファイル: {output_file}")
    print(f"列の順序: {list(df_reordered.columns)}")
    
    return output_file

def create_excel_version(csv_file, excel_file=None):
    """
    CSVファイルをExcelファイルに変換してより見やすくする
    
    Args:
        csv_file (str): 入力CSVファイルのパス
        excel_file (str): 出力Excelファイルのパス
    """
    
    # CSVファイルを読み込み
    df = pd.read_csv(csv_file)
    
    if excel_file is None:
        csv_path = Path(csv_file)
        excel_file = csv_path.parent / f"{csv_path.stem}_improved.xlsx"
    
    # ExcelWriterを使用して複数のシートを作成
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        
        # メインシート（全データ）
        df.to_excel(writer, sheet_name='全データ', index=False)
        
        # 成功したマッチのみのシート
        if 'contains_ax_docs' in df.columns:
            successful_matches = df[df['contains_ax_docs'] == True]
            if len(successful_matches) > 0:
                successful_matches.to_excel(writer, sheet_name='成功マッチ', index=False)
        
        # 失敗したマッチのみのシート
        if 'contains_ax_docs' in df.columns:
            failed_matches = df[df['contains_ax_docs'] == False]
            if len(failed_matches) > 0:
                failed_matches.to_excel(writer, sheet_name='失敗マッチ', index=False)
        
        # 統計情報シート
        stats_data = []
        
        # 基本統計
        stats_data.append(['項目', '値'])
        stats_data.append(['総ケース数', len(df)])
        
        if 'contains_ax_docs' in df.columns:
            success_count = len(df[df['contains_ax_docs'] == True])
            stats_data.append(['成功マッチ数', success_count])
            stats_data.append(['成功率', f"{success_count/len(df)*100:.1f}%"])
        
        if 'max_score' in df.columns:
            valid_scores = df[df['max_score'].notna()]['max_score']
            if len(valid_scores) > 0:
                stats_data.append(['平均スコア', f"{valid_scores.mean():.4f}"])
                stats_data.append(['最高スコア', f"{valid_scores.max():.4f}"])
                stats_data.append(['最低スコア', f"{valid_scores.min():.4f}"])
        
        if 'times_hit' in df.columns:
            valid_hits = df[df['times_hit'].notna()]['times_hit']
            if len(valid_hits) > 0:
                stats_data.append(['平均ヒット数', f"{valid_hits.mean():.1f}"])
        
        stats_df = pd.DataFrame(stats_data[1:], columns=stats_data[0])
        stats_df.to_excel(writer, sheet_name='統計情報', index=False)
    
    print(f"Excelファイルを作成しました: {excel_file}")
    return excel_file

def main():
    """メイン処理"""
    
    # 入力ファイル
    input_file = "/Users/yuki.kawashima/projects/GENIAC_PATENT/multiquery_summary.csv"
    
    print("=== CSVファイルの可読性改善 ===")
    
    # 1. CSVファイルの改善
    improved_csv = improve_csv_readability(input_file)
    
    # 2. Excelファイルの作成
    print("\n=== Excelファイルの作成 ===")
    excel_file = create_excel_version(improved_csv)
    
    print(f"\n完了！以下のファイルが作成されました：")
    print(f"- 改善されたCSV: {improved_csv}")
    print(f"- Excelファイル: {excel_file}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
himotsuki_csvのCSVファイルに記載されているhimotukiコードが
result_1の中に存在するかチェックするスクリプト
"""

import os
import csv
import glob
from collections import defaultdict

def get_syutugan_codes_from_csv(csv_file):
    """CSVファイルからsyutuganコードを抽出"""
    syutugan_codes = set()
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            syutugan_codes.add(row['syutugan'])
    
    return syutugan_codes

def get_existing_patents_from_result1():
    """result_1から既存の特許コードを抽出"""
    existing_patents = set()
    
    # result_1の全ディレクトリを検索
    pattern = os.path.join('input_files', 'result_1', '*', '*', 'text.txt')
    files = glob.glob(pattern)
    
    for file_path in files:
        # パスから特許コードを抽出
        # input_files/result_1/32/JP2012250133A/text.txt -> JP2012250133A
        parts = file_path.split(os.sep)
        if len(parts) >= 4:
            patent_code = parts[-2]  # ディレクトリ名が特許コード
            existing_patents.add(patent_code)
    
    return existing_patents

def main():
    print("start")
    print("=== himotsuki_csvとresult_1の比較（syutuganで一致） ===")
    
    # CSVファイルからsyutuganコードを取得
    csv1_codes = get_syutugan_codes_from_csv('data/himotsuki_csv/CSV1.csv')
    csv2_codes = get_syutugan_codes_from_csv('data/himotsuki_csv/CSV2.csv')
    
    print(f"CSV1のsyutuganコード数: {len(csv1_codes)}")
    print(f"CSV2のsyutuganコード数: {len(csv2_codes)}")
    
    # result_1から既存の特許コードを取得
    existing_patents = get_existing_patents_from_result1()
    print(f"result_1の特許コード数: {len(existing_patents)}")
    
    # 重複チェック
    csv1_found = csv1_codes.intersection(existing_patents)
    csv2_found = csv2_codes.intersection(existing_patents)
    
    print(f"\n=== 結果 ===")
    print(f"CSV1で見つかったsyutuganコード数: {len(csv1_found)}")
    print(f"CSV2で見つかったsyutuganコード数: {len(csv2_found)}")
    
    if csv1_found:
        print(f"\nCSV1で見つかったsyutuganコード（最初の10件）:")
        for code in sorted(list(csv1_found))[:10]:
            print(f"  {code}")
    
    if csv2_found:
        print(f"\nCSV2で見つかったsyutuganコード（最初の10件）:")
        for code in sorted(list(csv2_found))[:10]:
            print(f"  {code}")
    
    # 見つからないコードの例
    csv1_not_found = csv1_codes - existing_patents
    csv2_not_found = csv2_codes - existing_patents
    
    print(f"\nCSV1で見つからないsyutuganコード数: {len(csv1_not_found)}")
    print(f"CSV2で見つからないsyutuganコード数: {len(csv2_not_found)}")
    
    if csv1_not_found:
        print(f"\nCSV1で見つからないsyutuganコード（最初の10件）:")
        for code in sorted(list(csv1_not_found))[:10]:
            print(f"  {code}")
    
    if csv2_not_found:
        print(f"\nCSV2で見つからないsyutuganコード（最初の10件）:")
        for code in sorted(list(csv2_not_found))[:10]:
            print(f"  {code}")

if __name__ == "__main__":
    main() 
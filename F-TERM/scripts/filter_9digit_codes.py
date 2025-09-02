#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fターム JSONL から9桁のコードのみを残すフィルタリングスクリプト
- 入力: FTerm_slim_with_breadcrumbs.jsonl
- 出力: 9桁のコードのみを含むJSONLファイル
- 5桁や7桁のコードは削除される
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict


def is_9digit_code(code: str) -> bool:
    """
    コードが9桁かどうかを判定
    例: 2B001AA00 (9桁) -> True
        2B001 (5桁) -> False
        2B001AA (7桁) -> False
    """
    if not code:
        return False
    
    # 英数字のみで構成され、長さが9文字のものを9桁コードとする
    return bool(re.fullmatch(r'[A-Z0-9]{9}', code))


def filter_9digit_codes(input_path: Path, output_path: Path):
    """
    入力ファイルから9桁のコードのみを残して出力ファイルに書き込む
    """
    count_total = 0
    count_9digit = 0
    
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
                
            try:
                record = json.loads(line)
                count_total += 1
                
                code = record.get("code", "")
                if is_9digit_code(code):
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count_9digit += 1
                    
            except json.JSONDecodeError:
                print(f"[WARN] JSON解析エラー: {line[:100]}...")
                continue
    
    print(f"[OK] 処理完了:")
    print(f"  総レコード数: {count_total}")
    print(f"  9桁コード数: {count_9digit}")
    print(f"  削除されたレコード数: {count_total - count_9digit}")
    print(f"  出力ファイル: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Fターム JSONL から9桁のコードのみを残すフィルタリング")
    ap.add_argument("--input", required=True, help="入力ファイル（例: FTerm_slim_with_breadcrumbs.jsonl）")
    ap.add_argument("--output", required=True, help="出力ファイル（例: FTerm_9digit_only.jsonl）")
    
    args = ap.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise SystemExit(f"[ERR] 入力ファイルが見つかりません: {input_path}")
    
    filter_9digit_codes(input_path, output_path)


if __name__ == "__main__":
    main()

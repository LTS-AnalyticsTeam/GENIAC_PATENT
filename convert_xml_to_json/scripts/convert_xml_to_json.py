import argparse
import csv
import glob
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from parser import find_xml_file_by_patent_code, parse_patent_xml
from typing import Optional

from config import INPUT_DIR, OUTPUT_DIR


def process_xml_file(xml_path):
    try:
        data = parse_patent_xml(xml_path)
        return (xml_path, data, None)

    except Exception as e:
        return (xml_path, None, str(e))


def main():
    parser = argparse.ArgumentParser(description="Patent XML to JSON converter (parallel)")
    parser.add_argument('--file', type=str, help='input XML file path or patent code (e.g. JP2010000113A)')
    parser.add_argument('--count', type=int, default=None, help='number of files to process (Noneで全件)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_root = "input_files"
    xml_files = []

    if args.file:
        if os.path.isfile(args.file):
            xml_files = [args.file]
        else:
            file_path = find_xml_file_by_patent_code(args.file)
            if file_path:
                xml_files = [file_path]
            else:
                print(f"指定されたファイル/特許コードが見つかりません: {args.file}")
                exit(1)
    else:
        # data/input_files配下の全てのtxt/xmlファイルを取得
        xml_files = glob.glob('data/input_files/**/*.txt', recursive=True)
        xml_files += glob.glob('data/input_files/**/*.xml', recursive=True)
        if args.count is not None:
            xml_files = xml_files[:args.count]

    total = len(xml_files)
    print(f"Total files to process: {total}")
    done = 0
    for xml_path in xml_files:
        xml_path, data, error = process_xml_file(xml_path)
        done += 1
        if error:
            print(f"[{done}/{total}] Error: {xml_path} - {error}")
        else:
            # 出力ファイル名の生成
            patent_id = data.get('metadata', {}).get('patent_id') or ''
            publication_date = data.get('metadata', {}).get('publication_date') or ''
            if not patent_id:
                patent_id = os.path.basename(os.path.dirname(xml_path))
            if not publication_date:
                publication_date = ''
            output_filename = f"{patent_id}_{publication_date}.json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            if os.path.exists(output_path):
                print(f"[{done}/{total}] Skipped (already exists): {output_path}")
                continue
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(data, f_out, ensure_ascii=False, indent=2)
            print(f"[{done}/{total}] Processed: {xml_path} -> {output_filename}")

if __name__ == "__main__":
    main()

import re
import csv

def extract_core_patent_number(code):
    """特許番号から中間の数字部分だけを抽出"""
    m = re.match(r'^[A-Z]+([0-9]+)[A-Z0-9]*$', code)
    return m.group(1) if m else code

def load_reference_sets():
    """CSV1/CSV2からsyutugan/himotukiの全リストをsetで取得"""
    syutugan_set = set()
    himotsuki_set = set()
    for csv_file in ["data/himotsuki_csv/CSV1.csv", "data/himotsuki_csv/CSV2.csv"]:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    syutugan_set.add(row['syutugan'])
                    himotsuki_set.add(row['himotuki'])
        except Exception as e:
            print(f"[WARN] {csv_file} 読み込み失敗: {e}")
    return syutugan_set, himotsuki_set 

# 新関数: 行ごとにdictでリスト化

def load_reference_rows():
    """CSV1/CSV2から全行をdictでリスト化"""
    rows = []
    for csv_file in ["data/himotsuki_csv/CSV1.csv", "data/himotsuki_csv/CSV2.csv"]:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception as e:
            print(f"[WARN] {csv_file} 読み込み失敗: {e}")
    return rows 
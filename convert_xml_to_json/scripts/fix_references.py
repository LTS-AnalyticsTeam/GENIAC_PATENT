import os
import json
import csv
import re

def extract_core_patent_number(code):
    m = re.match(r'^[A-Z]+([0-9]+)[A-Z0-9]*$', code)
    return m.group(1) if m else code

# 1. CSV1からsyutugan/himotsukiの全行を読み込む
csv1_path = 'data/himotsuki_csv/CSV1.csv'
rows = []
with open(csv1_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# 2. output_files_references内の各ファイルを修正
ref_dir = 'output_files_references'
for filename in os.listdir(ref_dir):
    if not filename.endswith('.json'):
        continue
    filepath = os.path.join(ref_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    patent_id = data.get('metadata', {}).get('patent_id')
    if not patent_id:
        continue
    core_patent_id = extract_core_patent_number(str(patent_id))
    syutugan_list = set()
    himotsuki_list = set()
    for row in rows:
        syutugan_core = extract_core_patent_number(row['syutugan'])
        himotsuki_core = extract_core_patent_number(row['himotuki'])
        if syutugan_core == core_patent_id or himotsuki_core == core_patent_id:
            syutugan_list.add(row['syutugan'])
            himotsuki_list.add(row['himotuki'])
    data['metadata']['reference'] = {
        'exist': bool(syutugan_list or himotsuki_list),
        'syutugan': sorted(syutugan_list),
        'himotsuki': sorted(himotsuki_list)
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
print('output_files_references内のreferenceを修正しました') 
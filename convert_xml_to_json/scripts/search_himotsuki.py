import re
import os
from dotenv import load_dotenv
from azure.cosmos import CosmosClient

# .envファイルから環境変数を読み込む
load_dotenv()

COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT")
COSMOS_KEY = os.environ.get("COSMOS_KEY")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
CONTAINER_NAME = os.environ.get("CONTAINER_NAME")

client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
container = client.get_database_client(DATABASE_NAME).get_container_client(CONTAINER_NAME)

# ステップ1: 全レコードのpatent_id, syutugan, himotsukiを取得
query = """
SELECT c.metadata.patent_id AS patent_id, c.metadata.reference.syutugan AS syutugan, c.metadata.reference.himotsuki AS himotsuki
FROM c
"""
records = list(container.query_items(query=query, enable_cross_partition_query=True))

# patent_idごとに参照リストを構築
ref_map = {}
for rec in records:
    pid = rec.get('patent_id')
    syutugan = rec.get('syutugan') or []
    himotsuki = rec.get('himotsuki') or []
    ref_map[pid] = set(syutugan + himotsuki)

# 各patent_idについて「自分以外のレコードで自分が参照されているか」を判定
referenced_by_others = {}
# for my_pid in ref_map:
#     referenced = any(
#         my_pid in refs
#         for pid, refs in ref_map.items()
#         if pid != my_pid
#     )
#     referenced_by_others[my_pid] = referenced
#     print(f"{my_pid} is referenced by others: {referenced}")

# print("\n=== 他レコードから参照されているpatent_id一覧 ===")
# for pid, referenced in referenced_by_others.items():
#     if referenced:
#         print(pid)

# === 追加: csv1のsyutuganにpatent_idが一致し、かつその行のhimotsukiがCosmosDBのpatent_idにあるものを抽出 ===
import csv

# CosmosDBに存在するpatent_id一覧を取得
cosmos_patent_ids = set(ref_map.keys())

# CSV1を1行ずつチェック
csv1_path = 'data/himotsuki_csv/CSV1.csv'
with open(csv1_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        syutugan = row['syutugan']
        himotsuki = row['himotuki']
        # syutuganがCosmosDBのpatent_idに一致
        if syutugan in cosmos_patent_ids:
            # その行のhimotsukiもCosmosDBのpatent_idに存在
            if himotsuki in cosmos_patent_ids:
                print(f"syutugan: {syutugan}, himotsuki: {himotsuki}")

# === 追加: csv1のhimotsukiがpatent_idに一致するものを抽出 ===
with open(csv1_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        himotsuki = row['himotuki']
        if himotsuki in cosmos_patent_ids:
            print(f"himotsuki: {himotsuki}")

# === 追加: CSV1のhimotsukiにもあってsyutuganにもあるレコードを抽出 ===
print("\n=== CSV1でhimotsukiにもsyutuganにも同じ値があるレコード ===")
himotsuki_set = set()
syutugan_set = set()
with open(csv1_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        himotsuki_set.add(row['himotuki'])
        syutugan_set.add(row['syutugan'])
common_codes = himotsuki_set & syutugan_set
if common_codes:
    print(f"共通のコード数: {len(common_codes)}")
    for code in sorted(list(common_codes))[:20]:  # 最初の20件だけ表示
        print(f"共通コード: {code}")
else:
    print("共通するコードはありませんでした")

print("\n=== 共通コードのcore番号でpatent_idにHITしたもの一覧 ===")
hit_codes = []
for code in sorted(list(common_codes)):
    # JPと末尾Aを除去
    if code.startswith('JP') and code.endswith('A'):
        core = code[2:-1]
    else:
        core = code
    if core in cosmos_patent_ids:
        hit_codes.append((code, core))
if not hit_codes:
    print("HITするコードはありませんでした")



# === 追加: HITした共通コード(core)と相方(core)のペアからグループ分け（連結成分分解） ===
print("\n=== HITしたコード群のグループ分け（連結成分） ===")
# まずHITしたペアを集める
hit_pairs = []
common_code_set = set(code for code, _ in hit_codes)
with open(csv1_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        himotsuki = row['himotuki']
        syutugan = row['syutugan']
        for code_val, partner_val in [
            (himotsuki, syutugan),
            (syutugan, himotsuki)
        ]:
            if code_val in common_code_set:
                # core番号化
                if code_val.startswith('JP') and code_val.endswith('A'):
                    code_core = code_val[2:-1]
                else:
                    code_core = code_val
                if partner_val.startswith('JP') and partner_val.endswith('A'):
                    partner_core = partner_val[2:-1]
                else:
                    partner_core = partner_val
                if partner_core in cosmos_patent_ids:
                    hit_pairs.append((code_core, partner_core))
# Union-Findでグループ分け
parent = {}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(x, y):
    x_root = find(x)
    y_root = find(y)
    if x_root != y_root:
        parent[y_root] = x_root
# ノード初期化
nodes = set()
for a, b in hit_pairs:
    nodes.add(a)
    nodes.add(b)
for n in nodes:
    parent[n] = n
for a, b in hit_pairs:
    union(a, b)
# グループ化
groups = {}
for n in nodes:
    root = find(n)
    if root not in groups:
        groups[root] = set()
    groups[root].add(n)
print(f"グループ数: {len(groups)}")
for i, (root, members) in enumerate(groups.items(), 1):
    print(f"Group {i} ({len(members)}件): {sorted(members)}")

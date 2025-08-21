import json

MAX_BYTES = 16_777_216  # 16MB

oversized = []

with open("JSONL/FI_H.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error at line {i}: {e}")
            continue

        byte_size = len(line.encode("utf-8"))
        if byte_size > MAX_BYTES:
            print(f"[OVERSIZED] Line {i} is {byte_size} bytes")
            oversized.append((i, byte_size, obj.get("id", "<no id>")))

if not oversized:
    print("✅ 全てのJSON行は正常でサイズも16MB以内です")
else:
    print(f"\n🚨 {len(oversized)} 行が16MBを超えています。")
    for line_no, size, doc_id in oversized:
        print(f" - 行 {line_no} / {size} bytes / ID: {doc_id}")

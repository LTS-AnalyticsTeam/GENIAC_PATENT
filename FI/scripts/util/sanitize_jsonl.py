import json
import re

input_path = "FI_slim.jsonl"
output_path = "FI_sanitized.jsonl"

def sanitize_chunk_id(chunk_id: str) -> str:
    """
    chunk_idの禁止文字(: / @)をハイフンに変換し、
    許可文字 (A-Z, a-z, 0-9, _, -, =) のみ残す
    """
    # 置換ルール
    chunk_id = chunk_id.replace(":", "-")
    chunk_id = chunk_id.replace("/", "-")
    chunk_id = chunk_id.replace("@", "-")
    
    # 念のため許可文字以外を削除
    chunk_id = re.sub(r"[^A-Za-z0-9_\-=]", "", chunk_id)
    return chunk_id

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        if "chunk_id" in data:
            original = data["chunk_id"]
            safe_id = sanitize_chunk_id(original)
            data["chunk_id"] = safe_id
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"安全なchunk_idに変換完了: {output_path}")

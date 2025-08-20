#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fターム JSONL を統合・スリム化して出力
- 入力: 既存の FタームJSONL（1行=1ノード）
- 出力: 単一JSONL（1行=1ノード）で chunk_id, code, search_text のみ
- ポリシー:
  * 同一codeが複数ある場合: search_textの長い方を採用（次点でtitle/description長）
  * --only-subcodes 指定時は subcode レベルのみ出力（theme/viewpointは除外）
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def j_norm(s: str) -> str:
    if not s: return ""
    return re.sub(r"\s+", " ", s).strip()

def level_of(code: str) -> str:
    # Fターム: 2B314, 2B314-DA, 2B314-DA-07
    if re.fullmatch(r"\d+[A-Z]\d{3}", code): return "theme"
    if re.fullmatch(r"\d+[A-Z]\d{3}-[A-Z]{2}", code): return "viewpoint"
    if re.fullmatch(r"\d+[A-Z]\d{3}-[A-Z]{2}-\d{2}", code): return "subcode"
    return "unknown"

def sanitize_key(s: str) -> str:
    if not s: return ""
    return "".join(ch if (ch.isalnum() or ch in "_-=") else "_" for ch in s)

def stream_jsonl(paths: List[Path]):
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def merge_best(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """同一codeの候補から最良を選ぶ（search_text長→title/desc長）。"""
    best = None
    def score(rec):
        st = j_norm(rec.get("search_text", ""))
        ti = j_norm(rec.get("title", ""))
        de = j_norm(rec.get("description", ""))
        return (len(st), len(ti)+len(de))
    for r in records:
        if best is None or score(r) > score(best):
            best = r
    return best or {}

def run(inputs_glob: List[str], out_path: Path, only_subcodes: bool):
    # 入力解決
    in_files: List[Path] = []
    for pattern in inputs_glob:
        in_files.extend(sorted(Path(".").glob(pattern)))
    if not in_files:
        raise SystemExit(f"[ERR] 入力が見つかりません: {inputs_glob}")

    bucket: Dict[str, List[Dict[str, Any]]] = {}
    for rec in stream_jsonl(in_files):
        code = rec.get("code") or ""
        if not code: continue
        if only_subcodes and level_of(code) != "subcode":
            continue
        bucket.setdefault(code, []).append(rec)

    # 最良選択→スリム化
    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for code in sorted(bucket.keys()):
            best = merge_best(bucket[code])
            # search_text が無い場合は title/description から補完（保険）
            st = j_norm(best.get("search_text", "")) or j_norm(best.get("title", "")) or j_norm(best.get("description", "")) or code
            chunk_id = best.get("chunk_id") or f"FT-{code}"
            chunk_id = sanitize_key(chunk_id)

            slim = {
                "chunk_id": chunk_id,
                "code": code,
                "search_text": st
            }
            fout.write(json.dumps(slim, ensure_ascii=False) + "\n")
            count += 1

    print(f"[OK] wrote {count} items -> {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Fターム JSONL スリム化（chunk_id, code, search_textのみ）")
    ap.add_argument("--inputs", nargs="+", required=True, help="入力glob（例: 'FTerm_*.jsonl'）")
    ap.add_argument("--out", required=True, help="出力ファイル（例: FTerm_slim.jsonl）")
    ap.add_argument("--only-subcodes", action="store_true", help="サブコード行のみ出力（theme/viewpointを除外）")
    args = ap.parse_args()
    run(args.inputs, Path(args.out), args.only_subcodes)

if __name__ == "__main__":
    main()

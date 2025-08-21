#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
既存の FI_*.jsonl を統合・スリム化して出力するスクリプト
- 出力: 単一の JSONL（1行=1ノード）で "chunk_id", "code", "search_text" のみ
- 仕様:
  - 祖先（上位階層）の見出し/説明を継承して search_text を構築（「何の何か」を表現）
  - メイングループ（A61G11）→ /00（A61G11/00）補完
  - @サフィックス（A61G11/00@Z）のベース説明も含める
  - 冗長文の軽い重複除去
  - ※ パンくず "A > A61 > ..." は search_text に含めない
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------
# ユーティリティ
# ---------------------------

def j_normalize(s: str) -> str:
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s, flags=re.UNICODE).strip()

def infer_level(code: str) -> str:
    if re.fullmatch(r'[A-H]', code): return "section"
    if re.fullmatch(r'[A-H]\d{2}', code): return "class"
    if re.fullmatch(r'[A-H]\d{2}[A-Z]', code): return "subclass"
    if re.fullmatch(r'[A-H]\d{2}[A-Z]\d+/\d{2}', code): return "group"
    if re.fullmatch(r'[A-H]\d{2}[A-Z]\d+/\d{2}@[A-Z]', code): return "group-detail"
    if re.fullmatch(r'[A-H]\d{2}[A-Z]\d+', code): return "group"
    return "unknown"

def parent_of(code: str) -> Optional[str]:
    m = re.fullmatch(r'([A-H]\d{2}[A-Z]\d+/\d{2})@[A-Z]', code)
    if m: return m.group(1)
    m = re.fullmatch(r'([A-H]\d{2}[A-Z]\d+)/\d{2}', code)
    if m: return m.group(1)
    m = re.fullmatch(r'([A-H]\d{2}[A-Z])\d+', code)
    if m: return m.group(1)
    m = re.fullmatch(r'([A-H]\d{2})[A-Z]', code)
    if m: return m.group(1)
    m = re.fullmatch(r'([A-H])\d{2}', code)
    if m: return m.group(1)
    return None

def canonicalize_group(code: str) -> str:
    """メイングループ 'A61G11' を 'A61G11/00' へ寄せる"""
    if re.fullmatch(r'[A-H]\d{2}[A-Z]\d+', code):
        return f"{code}/00"
    return code

def dedupe_text_segments(text: str) -> str:
    """文単位でゆるい重複除去"""
    if not text:
        return ""
    segs = re.split(r'[。．\.\n]+', text)
    segs = [j_normalize(s) for s in segs if j_normalize(s)]
    uniq: List[str] = []
    for s in segs:
        if any((s == u) or (s in u) or (u in s) for u in uniq):
            continue
        uniq.append(s)
    return "。".join(uniq)

# ---------------------------
# search_text 構築
# ---------------------------

def ancestors_chain(code: str, nodes: Dict[str, Dict]) -> List[str]:
    """上位→下位の順で祖先コードを返す（自分は含めない）"""
    chain: List[str] = []
    seen = set()
    cur = code
    for _ in range(8):
        p = parent_of(cur)
        if not p:
            # 親が判定できない場合は breadcrumbs を補助に使う
            b = nodes.get(cur, {}).get("breadcrumbs") or []
            if b and b[-1] == cur and len(b) > 1:
                chain = b[:-1]  # すでに上位→下位順
            break
        if p in seen:
            break
        chain.append(p)
        seen.add(p)
        cur = p
    return list(reversed(chain))

def title_or_desc(n: Dict) -> str:
    t = j_normalize(n.get("title") or "")
    d = j_normalize(n.get("description") or "")
    return t or d

def build_search_text(code: str, nodes: Dict[str, Dict]) -> str:
    node = nodes.get(code, {})
    parts: List[str] = []

    # 1) 祖先たちの要旨（見出し優先）
    for ac in ancestors_chain(code, nodes):
        n = nodes.get(ac)
        if not n:
            continue
        parts.append(title_or_desc(n) or ac)

    # 2) @サフィックスの場合、ベースのグループ説明も加える
    if '@' in code:
        base = code.split('@', 1)[0]
        base_node = nodes.get(base) or nodes.get(canonicalize_group(base))
        if base_node:
            parts.append(title_or_desc(base_node) or base)

    # 3) 自分の説明
    parts.append(title_or_desc(node) or code)

    # ※ パンくず（"A > A61 > ..."）は入れない

    text = dedupe_text_segments("。".join([p for p in parts if p]))
    if len(text.encode("utf-8")) > 4000:
        text = text[:3000] + "..."
    return text

# ---------------------------
# メイン処理
# ---------------------------

def load_nodes(inputs: List[Path]) -> Dict[str, Dict]:
    """
    既存の FI_*.jsonl を読み込み、code をキーにノードを統合。
    同一コードが複数行にまたがる場合は、説明が長い方・パンくずが深い方を採用。
    """
    nodes: Dict[str, Dict] = {}
    for p in inputs:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                code = rec.get("code")
                if not code:
                    continue

                cur = nodes.get(code)
                if not cur:
                    nodes[code] = rec
                    continue

                # description の長い方
                desc_new = j_normalize(rec.get("description", ""))
                desc_old = j_normalize(cur.get("description", ""))
                if len(desc_new) > len(desc_old):
                    cur["description"] = rec.get("description", "")

                # title の長い方
                t_new = j_normalize(rec.get("title", ""))
                t_old = j_normalize(cur.get("title", ""))
                if len(t_new) > len(t_old):
                    cur["title"] = rec.get("title", "")

                # breadcrumbs は深い方（要素数の多い方）
                b_new = rec.get("breadcrumbs") or []
                b_old = cur.get("breadcrumbs") or []
                if len(b_new) > len(b_old):
                    cur["breadcrumbs"] = b_new

                # parent_code は存在する方を保持
                if not cur.get("parent_code") and rec.get("parent_code"):
                    cur["parent_code"] = rec.get("parent_code")

                # level は推定できる場合に更新
                lvl = infer_level(code)
                if lvl != "unknown":
                    cur["level"] = lvl

    return nodes

def run(inputs_glob: List[str], output_path: Path, only_groups: bool=False):
    # 入力ファイルを解決
    inputs: List[Path] = []
    for g in inputs_glob:
        inputs.extend(sorted(Path(".").glob(g)))
    if not inputs:
        raise SystemExit(f"[ERR] 入力が見つかりません: {inputs_glob}")

    nodes = load_nodes(inputs)

    # 安定した順序で出力（コード昇順）
    codes_sorted = sorted(nodes.keys())

    out_count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for code in codes_sorted:
            node = nodes[code]

            if only_groups:
                lv = infer_level(code)
                if lv not in ("group", "group-detail"):
                    continue

            search_text = build_search_text(code, nodes)
            rec = {
                "chunk_id": f"FI-{code}",
                "code": code,
                "search_text": search_text
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_count += 1

    print(f"[OK] Wrote {out_count} items -> {output_path}")

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="FI JSONL を統合・スリム化（chunk_id, code, search_text のみ）")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="入力JSONLのglob（例: 'FI_*.jsonl' や 'data/FI_*.jsonl'）")
    ap.add_argument("--out", required=True,
                    help="出力ファイルパス（例: FI_slim.jsonl）")
    ap.add_argument("--only-groups", action="store_true",
                    help="group / group-detail のみ出力したい場合に指定")
    args = ap.parse_args()
    run(inputs_glob=args.inputs, output_path=Path(args.out), only_groups=args.only_groups)

if __name__ == "__main__":
    main()

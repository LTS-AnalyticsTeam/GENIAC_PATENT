#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FI（特許分類）HTML群をパースして JSONL（1行=1分類ノード）を出力
- 文字化け: EUC-JP優先で復号
- コード抽出: <a name="...">優先 → 英字入り表示テキスト → 数値（1/00, 3/42等）
- 数値のみのコードはタイトルや接頭から完全形（A01B1/00, A01B3/42 等）へ正規化
- @A/@B/@Z 等は A01B3/42@A のように付与
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

# ---------------------------
# 低レベルユーティリティ
# ---------------------------

def j_normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def infer_level(code: str) -> str:
    # section | class | subclass | group | group-detail
    if re.fullmatch(r'[A-H]', code):
        return "section"
    if re.fullmatch(r'[A-H]\d{2}', code):
        return "class"
    if re.fullmatch(r'[A-H]\d{2}[A-Z]', code):
        return "subclass"
    if re.fullmatch(r'[A-H]\d{2}[A-Z]\d+/\d{2}', code):
        return "group"
    if re.fullmatch(r'[A-H]\d{2}[A-Z]\d+/\d{2}@[A-Z]', code):
        return "group-detail"
    # メイングループ（/00を省略しているA01B1 等）は group とみなす
    if re.fullmatch(r'[A-H]\d{2}[A-Z]\d+', code):
        return "group"
    return "unknown"

def parent_of(code: str) -> Optional[str]:
    # @X → base
    m = re.fullmatch(r'([A-H]\d{2}[A-Z]\d+/\d{2})@[A-Z]', code)
    if m:
        return m.group(1)
    # group → メイングループ or subclass
    m = re.fullmatch(r'([A-H]\d{2}[A-Z]\d+)/\d{2}', code)
    if m:
        return m.group(1)
    # メイングループ → subclass
    m = re.fullmatch(r'([A-H]\d{2}[A-Z])\d+', code)
    if m:
        return m.group(1)
    # subclass → class
    m = re.fullmatch(r'([A-H]\d{2})[A-Z]', code)
    if m:
        return m.group(1)
    # class → section
    m = re.fullmatch(r'([A-H])\d{2}', code)
    if m:
        return m.group(1)
    return None

def make_breadcrumbs(code: str) -> List[str]:
    # 可能な限り一般則で再構成
    sec = code[0]
    crumbs = [sec]
    m = re.match(r'^([A-H]\d{2})', code)
    if m:
        cls = m.group(1)
        if cls != sec:
            crumbs.append(cls)
    m = re.match(r'^([A-H]\d{2}[A-Z])', code)
    if m:
        subc = m.group(1)
        if subc not in crumbs:
            crumbs.append(subc)
    m = re.match(r'^([A-H]\d{2}[A-Z]\d+)', code)
    if m:
        mg = m.group(1)
        if mg not in crumbs:
            crumbs.append(mg)
    m = re.match(r'^([A-H]\d{2}[A-Z]\d+/\d{2})', code)
    if m:
        grp = m.group(1)
        if grp not in crumbs:
            crumbs.append(grp)
    if '@' in code and code not in crumbs:
        crumbs.append(code)
    return crumbs

def decode_euc_jp(content: bytes) -> str:
    for enc in ("euc-jp", "utf-8", "shift_jis"):
        try:
            return content.decode(enc)
        except Exception:
            continue
    return content.decode("utf-8", errors="ignore")


# ---------------------------
# パーサ本体
# ---------------------------

class FIClassificationParser:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.pages: List[Dict] = []  # 各Index.htmlの抽出結果

    def parse_html_file(self, file_path: Path) -> Optional[Dict]:
        """1つの Index.html を解析して (title, file_path, classifications[]) を返す"""
        try:
            html = decode_euc_jp(file_path.read_bytes())
            soup = BeautifulSoup(html, "html.parser")

            title = soup.find("title").get_text() if soup.find("title") else ""
            m = re.search(r'\(([A-Z0-9/]+)\)', title)
            title_code = m.group(1) if m else None
            # サブクラス（A01B）のときだけ数値を補う接頭辞として使う
            page_prefix = title_code if (title_code and re.match(r'^[A-H]\d{2}[A-Z]$', title_code)) else None

            def clean_text(s: str) -> str:
                return j_normalize(s)

            def expand_code(raw_code: str, link_name: Optional[str]) -> str:
                # 既に完全形（英字開始）
                if re.match(r'^[A-H]', raw_code):
                    return raw_code
                # link name 優先（完全形）
                if link_name and re.match(r'^[A-H]\d{2}[A-Z]\d*(?:/\d{2})?(?:[A-Z])?$', link_name):
                    # 例: name=A01B1 かつ raw=1/00 など
                    if re.match(r'^[A-H]\d{2}[A-Z]\d+$', link_name) and re.match(r'^\d+/\d{2}$', raw_code):
                        return f"{link_name}/{raw_code.split('/')[1]}"
                    return link_name
                # タイトルが完全形で raw が x/yy の場合
                if title_code and re.search(r'\d+/\d{2}$', raw_code):
                    if re.search(r'\d+/\d{2}$', title_code) and re.search(r'\d+/\d{2}$', title_code).group(0) == raw_code:
                        return title_code
                    if page_prefix:
                        major, minor = raw_code.split('/')
                        return f"{page_prefix}{int(major)}/{minor}"
                # 接頭辞ありの場合
                if page_prefix and re.match(r'^\d+/\d{2}$', raw_code):
                    major, minor = raw_code.split('/')
                    return f"{page_prefix}{int(major)}/{minor}"
                if page_prefix and re.match(r'^\d+$', raw_code):
                    return f"{page_prefix}{int(raw_code)}/00"
                return raw_code

            classifications = []
            table = soup.find("table")
            if not table:
                return {"title": title, "file_path": str(file_path), "classifications": classifications}

            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue

                # 先に @行（3列目 <a name="A01B3/42A">A</a>）を拾う
                if len(cells) >= 4:
                    name_tag = cells[2].find('a')
                    if name_tag and name_tag.has_attr('name'):
                        name_attr = name_tag['name'].strip()
                        m_at = re.match(r'^([A-H]\d{2}[A-Z]\d+(?:/\d{2})?)([A-Z])$', name_attr)
                        if m_at:
                            base, suf = m_at.group(1), m_at.group(2)
                            full_code = f"{base}@{suf}"
                            description = clean_text(cells[3].get_text())
                            classifications.append({
                                "code": full_code,
                                "href": "",
                                "description": description
                            })

                # 通常行（1列目にコード/リンク）
                cell0 = cells[0]
                link = cell0.find("a")
                code = None

                # 1) <a name="..."> 最優先
                if link and link.has_attr("name"):
                    name_attr = link["name"].strip()
                    if re.match(r'^[A-H]$', name_attr) \
                       or re.match(r'^[A-H]\d{2}$', name_attr) \
                       or re.match(r'^[A-H]\d{2}[A-Z]$', name_attr) \
                       or re.match(r'^[A-H]\d{2}[A-Z]\d+(?:/\d{2})?$', name_attr):
                        code = name_attr

                # 2) 表示テキストがFIコードっぽければ採用
                if code is None and link:
                    txt = link.get_text(strip=True)
                    if re.match(r'^[A-H]$', txt) \
                       or re.match(r'^[A-H]\d{2}$', txt) \
                       or re.match(r'^[A-H]\d{2}[A-Z]$', txt) \
                       or re.match(r'^[A-H]\d{2}[A-Z]\d+(?:/\d{2})?$', txt):
                        code = txt

                # 3) 数値行 → 完全形に展開
                if code is None:
                    raw = cell0.get_text(strip=True)
                    m_num = re.search(r'\d{1,3}/\d{2}', raw) or re.search(r'^\d{1,3}$', raw)
                    if m_num:
                        code = expand_code(
                            m_num.group(0),
                            link["name"].strip() if (link and link.has_attr("name")) else None
                        )

                if code:
                    description = clean_text(cells[1].get_text())
                    classifications.append({
                        "code": code,
                        "href": link.get("href", "") if link else "",
                        "description": description
                    })

            return {"title": title, "file_path": str(file_path), "classifications": classifications}
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def explore_directory(self, directory: Path, max_depth: int = 12, current_depth: int = 0):
        if not directory.exists():
            print(f"[WARN] Directory not found: {directory}")
            return
        if current_depth > max_depth:
            return
        index_file = directory / "Index.html"
        if index_file.exists():
            result = self.parse_html_file(index_file)
            if result:
                self.pages.append(result)
                try:
                    rel = index_file.relative_to(self.root_dir)
                except Exception:
                    rel = index_file
                print(f"Parsed: {rel}")
        for item in directory.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                self.explore_directory(item, max_depth, current_depth + 1)


# ---------------------------
# 集約→JSONL 出力
# ---------------------------

def build_nodes(pages: List[Dict]) -> Dict[str, Dict]:
    nodes: Dict[str, Dict] = {}
    for page in pages:
        file_path = page["file_path"]
        for item in page["classifications"]:
            code = item["code"]
            title = j_normalize(item.get("title") or "")
            desc = j_normalize(item.get("description") or "")
            level = infer_level(code)
            section = code[0] if code else ""

            if not title:
                # descriptionの先頭句や短いフレーズを見出し代用（簡易）
                title = desc.split("（", 1)[0][:120] if desc else ""

            node = nodes.get(code)
            if not node:
                node = {
                    "id": f"FI:{code}",
                    "code": code,
                    "title": title or desc[:120],
                    "description": desc,
                    "level": level,
                    "section": section,
                    "breadcrumbs": make_breadcrumbs(code),
                    "parent_code": parent_of(code),
                    "children_codes": [],
                    "siblings_codes": [],
                    "source_file": file_path,
                    "normalized_text": j_normalize(desc),
                    # "aliases": [],  # 必要なら後で埋める
                    # "text_leaf": "", "text_context": ""  # RAG用を追加したい場合
                }
                nodes[code] = node
            else:
                # 重複行が来たら、長い説明で上書き
                if len(desc) > len(node.get("description", "")):
                    node["description"] = desc
                    node["normalized_text"] = j_normalize(desc)

    # 親子・兄弟リンク埋め
    for code, node in nodes.items():
        p = node.get("parent_code")
        if p and p in nodes:
            nodes[p]["children_codes"].append(code)
    for code, node in nodes.items():
        p = node.get("parent_code")
        if p and p in nodes:
            sibs = [c for c in nodes[p]["children_codes"] if c != code]
            node["siblings_codes"] = sibs

    return nodes

def write_jsonl(nodes: Dict[str, Dict], out_path: Path, section_filter: Optional[str] = None):
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for n in nodes.values():
            if section_filter and n.get("section") != section_filter:
                continue
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} items → {out_path}")


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description='FI HTML → JSONL 変換')
    parser.add_argument('--root', default='.', help='FIルートディレクトリ（A, B, …, H を含むパス）')
    parser.add_argument('--section', default='ALL', help='対象セクション（A〜H or ALL）')
    parser.add_argument('--jsonl-out', help='出力ファイル名（単一セクション時のみ有効。未指定なら FI_<S>.jsonl）')
    parser.add_argument('--max-depth', type=int, default=12, help='最大探索深度（既定: 12）')
    args = parser.parse_args()

    root = Path(args.root)
    sections = list("ABCDEFGH") if args.section.upper() == "ALL" else [args.section.upper()]

    for sec in sections:
        print(f"=== セクション {sec} を解析中 ===")
        section_dir = root / sec
        fi_parser = FIClassificationParser(root)
        fi_parser.explore_directory(section_dir, max_depth=args.max_depth)

        nodes = build_nodes(fi_parser.pages)

        if len(sections) == 1:
            out_path = Path(args.jsonl_out) if args.jsonl_out else Path(f"FI_{sec}.jsonl")
            write_jsonl(nodes, out_path, section_filter=sec)
        else:
            # ALL の場合はセクションごとにファイルを分けて出力
            out_path = Path(f"FI_{sec}.jsonl")
            write_jsonl(nodes, out_path, section_filter=sec)

        print(f"Pages parsed: {len(fi_parser.pages)}")

if __name__ == "__main__":
    main()

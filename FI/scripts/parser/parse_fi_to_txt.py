#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特許分類FIの階層構造を解析してTXTファイルに出力するスクリプト
EUC-JPエンコーディングでHTMLファイルを読み込み、階層構造を解析します
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Optional

from bs4 import BeautifulSoup, NavigableString


class FIClassificationParser:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.classifications = {}
        self.current_path = []
        
    def decode_euc_jp(self, content: bytes) -> str:
        """EUC-JPエンコーディングでデコード"""
        try:
            return content.decode('euc-jp')
        except UnicodeDecodeError:
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                return content.decode('shift_jis', errors='ignore')

    def clean_text(self, text: str) -> str:
        """改行や空白を除去してクリーンなテキストを返す"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)  # 改行や連続空白をスペース1つに
        return text.strip()

    def parse_html_file(self, file_path: Path) -> Optional[Dict]:
        """HTMLファイルを解析して分類情報を抽出"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            html_content = self.decode_euc_jp(content)
            soup = BeautifulSoup(html_content, 'html.parser')

            # タイトル取得（例: PMGS/FI(A01B3/42) → title_code='A01B3/42'）
            title_tag = soup.find('title')
            title = title_tag.get_text() if title_tag else ""
            m = re.search(r'\(([A-Z0-9/]+)\)', title)
            title_code = m.group(1) if m else None

            # サブクラス接頭辞（例: 'A01B'）
            page_prefix = title_code if (title_code and re.match(r'^[A-H]\d{2}[A-Z]$', title_code)) else None

            # 数値コードを完全形に展開
            def expand_code(raw_code: str, link_name: Optional[str]) -> str:
                # 既に完全形（英字で始まる）ならそのまま
                if re.match(r'^[A-H]', raw_code):
                    return raw_code

                # link の name が完全形なら優先（例: A01B1, A01B3/42, A01B3/42A）
                if link_name and re.match(r'^[A-H]\d{2}[A-Z]\d*(?:/\d{2})?(?:[A-Z])?$', link_name):
                    # A01B1 → 1/00 行なら /00 を補う
                    if re.match(r'^[A-H]\d{2}[A-Z]\d+$', link_name) and re.match(r'^\d+/(?:00|0\d)$', raw_code):
                        return f"{link_name}/{raw_code.split('/')[1]}"
                    # A01B3/42A のような @系は caller 側で処理するのでここは素通しでOK
                    return link_name

                # タイトルが完全形で、末尾の数値ペア一致なら置換（例: 3/42 → A01B3/42）
                if title_code and re.search(r'\d+/\d{2}$', raw_code):
                    # タイトルにも /xx が付いている場合はそれを採用
                    if re.search(r'\d+/\d{2}$', title_code) and re.search(r'\d+/\d{2}$', title_code).group(0) == raw_code:
                        return title_code
                    # 接頭辞があれば A01B + 3/42 → A01B3/42
                    if page_prefix:
                        major, minor = raw_code.split('/')
                        return f"{page_prefix}{int(major)}/{minor}"

                # 接頭辞があれば 1/00 → A01B1/00, 1 → A01B1/00
                if page_prefix and re.match(r'^\d+/\d{2}$', raw_code):
                    major, minor = raw_code.split('/')
                    return f"{page_prefix}{int(major)}/{minor}"
                if page_prefix and re.match(r'^\d+$', raw_code):
                    return f"{page_prefix}{int(raw_code)}/00"

                return raw_code

            classifications = []
            table = soup.find('table')
            if not table:
                return {'title': title, 'file_path': str(file_path), 'classifications': classifications}

            rows = table.find_all('tr')
            for row in rows:
                try:
                    cells = row.find_all('td')
                    if len(cells) < 2:
                        continue

                    # ---- 先に @A/@B/@Z などの行（3列目<a name="A01B3/42A">A</a>）を拾う
                    if len(cells) >= 4:
                        name_tag = cells[2].find('a')
                        if name_tag and name_tag.has_attr('name'):
                            name_attr = name_tag['name'].strip()  # 例: A01B3/42A
                            m_at = re.match(r'^([A-H]\d{2}[A-Z]\d+(?:/\d{2})?)([A-Z])$', name_attr)
                            if m_at:
                                base = m_at.group(1)
                                suf  = m_at.group(2)
                                full_code = f"{base}@{suf}"
                                description = self.clean_text(cells[3].get_text())
                                classifications.append({
                                    'code': full_code,
                                    'href': '',
                                    'description': description
                                })
                                # @行はここで完結。引き続き通常行も評価したい場合は continue しない

                    # ---- 通常行（1列目にコード/リンクがある）
                    cell0 = cells[0]
                    link = cell0.find('a')
                    code = None

                    # 1) <a name="..."> 最優先（A01, A01B, A01B3, A01B3/42 など）
                    if link and link.has_attr('name'):
                        name_attr = link['name'].strip()
                        if re.match(r'^[A-H]$', name_attr) \
                        or re.match(r'^[A-H]\d{2}$', name_attr) \
                        or re.match(r'^[A-H]\d{2}[A-Z]$', name_attr) \
                        or re.match(r'^[A-H]\d{2}[A-Z]\d+(?:/\d{2})?$', name_attr):
                            code = name_attr

                    # 2) <a> の表示テキストが FIコードっぽければ採用
                    if code is None and link:
                        txt = link.get_text(strip=True)
                        if re.match(r'^[A-H]$', txt) \
                        or re.match(r'^[A-H]\d{2}$', txt) \
                        or re.match(r'^[A-H]\d{2}[A-Z]$', txt) \
                        or re.match(r'^[A-H]\d{2}[A-Z]\d+(?:/\d{2})?$', txt):
                            code = txt

                    # 3) 数値行（1/00, 3/42, 1 など）→ 完全形に展開
                    if code is None:
                        code_text = cell0.get_text().strip()
                        m_num = re.search(r'\d{1,3}/\d{2}', code_text) or re.search(r'^\d{1,3}$', code_text)
                        if m_num:
                            raw = m_num.group(0)
                            code = expand_code(raw, link['name'].strip() if (link and link.has_attr('name')) else None)

                    if code:
                        description = self.clean_text(cells[1].get_text())
                        classifications.append({
                            'code': code,
                            'href': link.get('href', '') if link else '',
                            'description': description
                        })

                except Exception:
                    # 1行不正でも他行の処理は続ける
                    continue

            return {
                'title': title,
                'file_path': str(file_path),
                'classifications': classifications
            }

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def get_classification_path(self, file_path: Path) -> str:
        """ファイルパスから分類パスを生成"""
        relative_path = file_path.relative_to(self.root_dir)
        parts = relative_path.parts
        
        # Index.htmlを除去してパスを構築
        path_parts = []
        for part in parts:
            if part != 'Index.html':
                path_parts.append(part)
        
        return '/'.join(path_parts) if path_parts else 'ROOT'
    
    def explore_directory(self, directory: Path, max_depth: int = 10,
                         current_depth: int = 0):
        """ディレクトリを再帰的に探索してHTMLファイルを解析"""
        if current_depth > max_depth:
            return
        
        try:
            # Index.htmlファイルを探す
            index_file = directory / 'Index.html'
            if index_file.exists():
                result = self.parse_html_file(index_file)
                if result:
                    classification_path = self.get_classification_path(index_file)
                    self.classifications[classification_path] = result
                    print(f"Parsed: {classification_path}")
            
            # サブディレクトリを探索
            for item in directory.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    self.explore_directory(item, max_depth, current_depth + 1)
                    
        except Exception as e:
            print(f"Error exploring {directory}: {e}")
    
    def generate_txt_output(self, output_file: str = "fi_classifications.txt"):
        """解析結果をTXTファイルに出力"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("特許分類FI 階層構造解析結果\n")
            f.write("=" * 50 + "\n\n")
            
            # パスでソート
            sorted_paths = sorted(self.classifications.keys())
            
            for path in sorted_paths:
                data = self.classifications[path]
                f.write(f"【分類パス】: {path}\n")
                f.write(f"【ファイル】: {data['file_path']}\n")
                f.write(f"【タイトル】: {data['title']}\n")
                f.write("【サブ分類】:\n")
                
                for classification in data['classifications']:
                    f.write(f"  {classification['code']}: "
                           f"{classification['description']}\n")
                    if classification['href']:
                        f.write(f"    → {classification['href']}\n")
                
                f.write("\n" + "-" * 40 + "\n\n")
            
            # 統計情報
            f.write("\n統計情報:\n")
            f.write(f"総分類数: {len(self.classifications)}\n")
            total_sub = sum(len(data['classifications']) 
                           for data in self.classifications.values())
            f.write(f"総サブ分類数: {total_sub}\n")
    
    def generate_summary_txt(self, output_file: str = "fi_summary.txt"):
        """要約版のTXTファイルを生成"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("特許分類FI 要約\n")
            f.write("=" * 30 + "\n\n")
            
            # メインセクション（A, B, C, D, E, F, G, H）の情報を抽出
            main_sections = {}
            
            for path, data in self.classifications.items():
                if '/' not in path and path in ['A', 'B', 'C', 'D', 'E', 'F', 
                                              'G', 'H']:
                    main_sections[path] = data
            
            # メインセクションを出力
            for section in sorted(main_sections.keys()):
                data = main_sections[section]
                f.write(f"{section}セクション:\n")
                
                for classification in data['classifications']:
                    f.write(f"  {classification['code']}: "
                           f"{classification['description']}\n")
                
                f.write("\n")
            
            # 階層構造の概要
            f.write("階層構造概要:\n")
            f.write("-" * 20 + "\n")
            
            for path in sorted(self.classifications.keys()):
                if '/' in path:  # サブディレクトリ
                    depth = path.count('/')
                    indent = "  " * depth
                    f.write(f"{indent}{path}\n")


def main():
    parser = argparse.ArgumentParser(description='特許分類FIの階層構造を解析します')
    parser.add_argument('--root', default='.', help='ルートディレクトリのパス')
    parser.add_argument('--max-depth', type=int, default=10, 
                       help='最大探索深度')
    parser.add_argument('--output', default='fi_classifications.txt', 
                       help='出力ファイル名')
    parser.add_argument('--summary', default='fi_summary.txt', 
                       help='要約ファイル名')
    
    args = parser.parse_args()
    
    print("特許分類FI解析を開始します...")
    
    # パーサーを作成
    fi_parser = FIClassificationParser(args.root)
    
    # ディレクトリを探索
    print(f"ディレクトリを探索中: {args.root}")
    fi_parser.explore_directory(Path(args.root), args.max_depth)
    
    # 結果を出力
    print(f"解析結果を出力中: {args.output}")
    fi_parser.generate_txt_output(args.output)
    
    print(f"要約を出力中: {args.summary}")
    fi_parser.generate_summary_txt(args.summary)
    
    print("解析完了!")
    print(f"解析した分類数: {len(fi_parser.classifications)}")


if __name__ == "__main__":
    main() 
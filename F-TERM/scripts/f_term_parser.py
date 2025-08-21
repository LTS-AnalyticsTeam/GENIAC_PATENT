"""
FタームHTML群をパースして JSONL（1行=1分類ノード）を出力
- 文字化け: EUC-JP優先で復号
- コード抽出: テーマコード + 観点 + サブコードの形式
- 階層構造: テーマコード → 観点 → サブコードの階層を構築
- 親子関係: 各レベルでの親子関係を特定
- 説明文: 各階層での適切な説明文を取得（txtファイル優先）
- 検索最適化: 上位階層の説明を継承した検索用テキストを生成
- 詳細情報: Index2.htmlから詳細説明を取得してsearch_textに反映
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
    # テーマコード | 観点 | サブコード
    if re.fullmatch(r'\d+[A-Z]\d{3}', code):
        return "theme"
    if re.fullmatch(r'\d+[A-Z]\d{3}-[A-Z]{2}', code):
        return "viewpoint"
    if re.fullmatch(r'\d+[A-Z]\d{3}-[A-Z]{2}-\d{2}', code):
        return "subcode"
    return "unknown"


def parent_of(code: str) -> Optional[str]:
    # サブコード → 観点
    m = re.fullmatch(r'(\d+[A-Z]\d{3}-[A-Z]{2})-\d{2}', code)
    if m:
        return m.group(1)
    # 観点 → テーマコード
    m = re.fullmatch(r'(\d+[A-Z]\d{3})-[A-Z]{2}', code)
    if m:
        return m.group(1)
    return None


def make_breadcrumbs(code: str) -> List[str]:
    # 階層構造を構築
    crumbs = []
    
    # テーマコード部分
    m = re.match(r'^(\d+[A-Z]\d{3})', code)
    if m:
        theme = m.group(1)
        crumbs.append(theme)
    
    # 観点部分
    m = re.match(r'^(\d+[A-Z]\d{3}-[A-Z]{2})', code)
    if m:
        viewpoint = m.group(1)
        if viewpoint not in crumbs:
            crumbs.append(viewpoint)
    
    # サブコード部分
    if code not in crumbs:
        crumbs.append(code)
    
    return crumbs


def decode_euc_jp(content: bytes) -> str:
    for enc in ("euc-jp", "utf-8", "shift_jis"):
        try:
            return content.decode(enc)
        except Exception:
            continue
    return content.decode("utf-8", errors="ignore")


def extract_theme_code_from_path(file_path: Path) -> Optional[str]:
    """ファイルパスからテーマコードを抽出"""
    # 例: 3B/3B120/AD/Index001.html → 3B120
    parts = file_path.parts
    for part in parts:
        if re.match(r'\d+[A-Z]\d{3}', part):
            return part
    return None


def extract_viewpoint_from_path(file_path: Path) -> Optional[str]:
    """ファイルパスから観点を抽出"""
    # 例: 3B/3B120/AD/Index001.html → AD
    parts = file_path.parts
    for part in parts:
        if re.match(r'^[A-Z]{2}$', part):
            return part
    return None


def get_section_from_theme_code(theme_code: str) -> str:
    """テーマコードからセクション番号を抽出"""
    # 例: 3B120 → 3
    m = re.match(r'^(\d+)', theme_code)
    return m.group(1) if m else ""


def clean_html_text(s: str) -> str:
    """HTMLタグを除去してテキストをクリーンアップ"""
    if not s:
        return ""
    # HTMLタグを除去
    s = re.sub(r'<[^>]+>', '', s)
    # 改行文字をスペースに変換
    s = s.replace('\n', ' ').replace('\r', ' ')
    # 複数のスペースを1つに
    s = re.sub(r'\s+', ' ', s)
    return j_normalize(s)


def remove_duplicate_text(text: str) -> str:
    """重複するテキストを除去（より厳密な重複除去）"""
    if not text:
        return ""
    
    # 文単位で分割
    sentences = re.split(r'[。！？、]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 重複除去（完全一致、部分一致、類似表現）
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        # 完全一致チェック
        if sentence in seen:
            continue
        
        # 部分一致チェック
        is_duplicate = False
        for existing in unique_sentences:
            if sentence in existing or existing in sentence:
                is_duplicate = True
                break
            # 類似表現のチェック（例：「テーマコード 2F031」と「2F031」）
            if re.sub(r'テーマコード\s*', '', sentence) == existing or \
               re.sub(r'テーマコード\s*', '', existing) == sentence:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_sentences.append(sentence)
            seen.add(sentence)
    
    return '、'.join(unique_sentences)


def create_hierarchical_description(node: Dict, all_nodes: Dict[str, Dict], 
                                 parser: 'FTermParser') -> str:
    """階層的な説明文を作成（上位→下位の順）"""
    descriptions = []
    
    # テーマコードの説明を取得
    theme_code = node.get('theme_code')
    if theme_code:
        theme_desc = parser.theme_descriptions.get(theme_code, "")
        if theme_desc and theme_desc != f"テーマコード {theme_code}" and theme_desc != theme_code:
            descriptions.append(theme_desc)
    
    # 観点の説明を取得
    if node.get('level') in ['viewpoint', 'subcode']:
        viewpoint_code = node.get('viewpoint_code') or f"{theme_code}-{node.get('viewpoint')}"
        if viewpoint_code:
            viewpoint_desc = parser.viewpoint_descriptions.get(viewpoint_code, "")
            if viewpoint_desc and viewpoint_desc != f"観点 {node.get('viewpoint', '')}" and viewpoint_desc != viewpoint_code:
                descriptions.append(viewpoint_desc)
    
    # 自分の説明を追加
    own_desc = node.get('description', '')
    if own_desc and own_desc != node.get('code', ''):
        descriptions.append(own_desc)
    
    # 説明文が全く取得できていない場合は、適切なプレースホルダーを設定
    if not descriptions:
        if node.get('level') == 'theme':
            descriptions.append(f"Fタームテーマコード {theme_code}")
        elif node.get('level') == 'viewpoint':
            descriptions.append(f"Fターム観点 {node.get('viewpoint', '')}")
        elif node.get('level') == 'subcode':
            descriptions.append(f"Fタームサブコード {node.get('subcode', '')}")
    
    # 重複除去とクリーニング
    combined = ' '.join(descriptions)
    cleaned = remove_duplicate_text(combined)
    
    # 長さ制限（3-4KB程度）
    if len(cleaned.encode('utf-8')) > 4000:
        cleaned = cleaned[:3000] + '...'
    
    return cleaned


def create_search_text(node: Dict, desc_hier: str,
                      parser: 'FTermParser') -> str:
    """検索用テキストを作成（本質的な説明のみ）"""
    parts = []
    
    # テーマコードの説明（セクションIndex.htmlから取得したもの）
    theme_code = node.get('theme_code')
    if theme_code:
        theme_desc = parser.get_theme_description(theme_code)
        if theme_desc and theme_desc != f"Fタームテーマコード {theme_code}":
            parts.append(theme_desc)
    
    # 観点の説明
    if node.get('level') in ['viewpoint', 'subcode']:
        viewpoint = node.get('viewpoint', '')
        if viewpoint:
            viewpoint_code = f"{theme_code}-{viewpoint}"
            viewpoint_desc = parser.viewpoint_descriptions.get(viewpoint_code, "")
            if viewpoint_desc and viewpoint_desc != f"観点 {viewpoint}":
                parts.append(viewpoint_desc)
    
    # サブコードの説明
    if node.get('level') == 'subcode':
        own_desc = node.get('description', '')
        if own_desc and own_desc != node.get('code', ''):
            parts.append(own_desc)
    
    # 結合してクリーニング
    search_text = ' '.join(parts)
    search_text = re.sub(r'\s+', ' ', search_text).strip()
    
    return search_text


def get_ancestors(node: Dict, all_nodes: Dict[str, Dict]) -> List[str]:
    """祖先のコードを取得（テーマコードまで）"""
    ancestors = []
    current = node
    
    # 最大2ステップ（サブコード → 観点 → テーマコード）
    for _ in range(2):
        parent_code = current.get("parent_code")
        if not parent_code:
            break
        ancestors.append(parent_code)
        current = all_nodes.get(parent_code, {})
        if not current:
            break
    
    return list(reversed(ancestors))  # テーマ→観点の順


def clean_code_for_display(code: str) -> str:
    """表示用のコードからハイフンを除去"""
    return code.replace('-', '')


# ---------------------------
# パーサ本体
# ---------------------------


class FTermParser:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.pages: List[Dict] = []
        self.theme_descriptions: Dict[str, str] = {}  # テーマコード → 説明
        self.viewpoint_descriptions: Dict[str, str] = {}  # 観点コード → 説明
        self.theme_detail_descriptions: Dict[str, str] = {}  # テーマコード → 詳細説明
        self.viewpoint_detail_descriptions: Dict[str, str] = {}  # 観点コード → 詳細説明
        self.section_theme_descriptions: Dict[str, str] = {} # セクション番号 → テーマコード説明

    def parse_theme_index(self, theme_dir: Path) -> Optional[Dict]:
        """テーマコードのIndex1.htmlを解析して説明を取得"""
        index_file = theme_dir / "Index1.html"
        if not index_file.exists():
            return None
            
        try:
            html = decode_euc_jp(index_file.read_bytes())
            soup = BeautifulSoup(html, "html.parser")
            
            theme_code = extract_theme_code_from_path(index_file)
            if not theme_code:
                return None
                
            # テーマコードの説明を取得
            theme_description = ""
            table = soup.find("table")
            if table:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        first_cell = cells[0].get_text(strip=True)
                        if "テーマコード" in first_cell or "テーマ" in first_cell:
                            theme_description = clean_html_text(cells[1].get_text())
                            break
                
                # 観点の説明を取得
                viewpoint_descriptions = {}
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        first_cell = cells[0].get_text(strip=True)
                        if re.match(r'^[A-Z]{2}$', first_cell):  # AA, BA, CAなど
                            viewpoint = first_cell
                            description = clean_html_text(cells[1].get_text())
                            viewpoint_descriptions[viewpoint] = description
                
                return {
                    "theme_code": theme_code,
                    "theme_description": theme_description,
                    "viewpoint_descriptions": viewpoint_descriptions
                }
        except Exception as e:
            print(f"Error parsing theme index {index_file}: {e}")
            return None
        return None

    def parse_theme_index2(self, theme_dir: Path) -> Optional[Dict]:
        """テーマコードのIndex2.htmlを解析して詳細説明を取得"""
        index2_file = theme_dir / "Index2.html"
        if not index2_file.exists():
            return None
            
        try:
            html = decode_euc_jp(index2_file.read_bytes())
            soup = BeautifulSoup(html, "html.parser")
            
            theme_code = extract_theme_code_from_path(index2_file)
            if not theme_code:
                return None
                
            # テーマの詳細説明を取得
            theme_detail_description = ""
            table = soup.find("table")
            if table:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        first_cell = cells[0].get_text(strip=True)
                        if "テーマの詳細説明" in first_cell or "テーマの説明" in first_cell:
                            theme_detail_description = clean_html_text(cells[1].get_text())
                            break
                
                # 観点の詳細説明を取得
                viewpoint_detail_descriptions = {}
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        first_cell = cells[0].get_text(strip=True)
                        if "観点の詳細説明" in first_cell or "観点の説明" in first_cell:
                            description = clean_html_text(cells[1].get_text())
                            # 観点コードを特定するために、前後の行を確認
                            # 通常、観点の詳細説明の前後に観点コード（AA, BA等）がある
                            for prev_row in rows:
                                prev_cells = prev_row.find_all("td")
                                if len(prev_cells) >= 1:
                                    prev_text = prev_cells[0].get_text(strip=True)
                                    if re.match(r'^[A-Z]{2}$', prev_text):
                                        viewpoint = prev_text
                                        viewpoint_detail_descriptions[viewpoint] = description
                                        break
                            break
                
                return {
                    "theme_code": theme_code,
                    "theme_detail_description": theme_detail_description,
                    "viewpoint_detail_descriptions": viewpoint_detail_descriptions
                }
        except Exception as e:
            print(f"Error parsing theme index2 {index2_file}: {e}")
            return None
        return None

    def find_txt_files(self, theme_dir: Path) -> Dict[str, str]:
        """テーマディレクトリ内のtxtファイルから説明を取得"""
        txt_descriptions = {}
        
        # テーマコードの説明を取得（例: 3B110+txt）
        theme_code = extract_theme_code_from_path(theme_dir)
        if theme_code:
            # セクションディレクトリ内のtxtファイルを探す
            section_dir = theme_dir.parent
            for txt_file in section_dir.glob(f"{theme_code}+txt"):
                try:
                    content = decode_euc_jp(txt_file.read_bytes())
                    if ':' in content:
                        description = content.split(':', 1)[1]
                        txt_descriptions['theme'] = clean_html_text(description)
                        print(f"DEBUG: Found theme description in {txt_file}: {description[:100]}...")
                except Exception as e:
                    print(f"Error parsing theme txt file {txt_file}: {e}")
        
        # 観点別のtxtファイルを探す（例: 06-3B110-AA+txt）
        for txt_file in theme_dir.glob("06-*-*+txt"):
            try:
                content = decode_euc_jp(txt_file.read_bytes())
                # 形式: "06-3B110-AA:説明文<BR>..."
                if ':' in content:
                    parts = content.split(':', 1)
                    if len(parts) == 2:
                        code_part = parts[0]
                        description = parts[1]
                        
                        # 観点コードを抽出（例: 06-3B110-AA → AA）
                        m = re.search(r'-([A-Z]{2})$', code_part)
                        if m:
                            viewpoint = m.group(1)
                            txt_descriptions[viewpoint] = clean_html_text(description)
                            print(f"DEBUG: Found viewpoint description for {viewpoint}: {description[:100]}...")
            except Exception as e:
                print(f"Error parsing viewpoint txt file {txt_file}: {e}")
        
        return txt_descriptions

    def parse_section_index(self, section_dir: Path) -> Dict[str, str]:
        """セクションのIndex.htmlを解析してテーマコードの説明を取得"""
        section_index = section_dir / "Index.html"
        theme_descriptions = {}
        
        if section_index.exists():
            try:
                html = decode_euc_jp(section_index.read_bytes())
                soup = BeautifulSoup(html, "html.parser")
                
                table = soup.find("table")
                if table:
                    rows = table.find_all("tr")
                    for row in rows:
                        cells = row.find_all("td")
                        if len(cells) >= 2:
                            # テーマコード（例: 2F001）
                            code_cell = cells[0]
                            code_link = code_cell.find("a")
                            if code_link:
                                theme_code = code_link.get_text(strip=True)
                                # 説明（例: アナログクオーツ）
                                description = clean_html_text(cells[1].get_text())
                                if theme_code and description:
                                    theme_descriptions[theme_code] = description
                                    print(f"DEBUG: Found theme description {theme_code}: {description}")
            except Exception as e:
                print(f"Error parsing section index {section_index}: {e}")
        
        return theme_descriptions

    def get_theme_description(self, theme_code: str) -> str:
        """テーマコードの説明を取得（優先順位付き）"""
        # 1. セクションIndex.htmlから取得した説明（最優先）
        section_desc = self.section_theme_descriptions.get(theme_code, "")
        if section_desc:
            return section_desc
        
        # 2. Index1.htmlから取得した説明
        theme_desc = self.theme_descriptions.get(theme_code, "")
        if theme_desc and theme_desc != f"テーマコード {theme_code}":
            return theme_desc
        
        # 3. デフォルト
        return f"Fタームテーマコード {theme_code}"

    def explore_directory(self, directory: Path, max_depth: int = 12, current_depth: int = 0):
        if not directory.exists():
            print(f"[WARN] Directory not found: {directory}")
            return
        if current_depth > max_depth:
            return
            
        # セクションレベル（depth=0）でIndex.htmlを解析
        if current_depth == 0:
            section_descriptions = self.parse_section_index(directory)
            # セクション情報を保存
            self.section_theme_descriptions.update(section_descriptions)
        
        # テーマコードのIndex1.htmlとIndex2.htmlを解析
        if current_depth == 1:  # セクション直下のディレクトリ
            theme_info = self.parse_theme_index(directory)
            theme_info2 = self.parse_theme_index2(directory)
            
            if theme_info:
                theme_code = theme_info["theme_code"]
                
                # txtファイルから説明を取得（優先）
                txt_descriptions = self.find_txt_files(directory)
                
                # テーマコードの説明
                theme_description = txt_descriptions.get('theme') or theme_info["theme_description"]
                if theme_description:
                    self.theme_descriptions[theme_code] = theme_description
                    print(f"DEBUG: Set theme description for {theme_code}: {theme_description[:100]}...")
                else:
                    print(f"WARNING: No description found for theme {theme_code}")
                
                # 観点の説明
                for vp, desc in theme_info["viewpoint_descriptions"].items():
                    viewpoint_code = f"{theme_code}-{vp}"
                    # txtファイルの説明を優先
                    txt_desc = txt_descriptions.get(vp)
                    if txt_desc:
                        self.viewpoint_descriptions[viewpoint_code] = txt_desc
                        print(f"DEBUG: Set viewpoint description for {viewpoint_code}: {txt_desc[:100]}...")
                    else:
                        self.viewpoint_descriptions[viewpoint_code] = desc
            
            # Index2.htmlから詳細説明を取得
            if theme_info2:
                theme_code = theme_info2["theme_code"]
                
                # テーマの詳細説明
                if theme_info2["theme_detail_description"]:
                    self.theme_detail_descriptions[theme_code] = theme_info2["theme_detail_description"]
                    print(f"DEBUG: Set theme detail description for {theme_code}: {theme_info2['theme_detail_description'][:100]}...")
                
                # 観点の詳細説明
                for vp, desc in theme_info2["viewpoint_detail_descriptions"].items():
                    viewpoint_code = f"{theme_code}-{vp}"
                    self.viewpoint_detail_descriptions[viewpoint_code] = desc
                    print(f"DEBUG: Set viewpoint detail description for {viewpoint_code}: {desc[:100]}...")
            
        # Index.htmlファイルを探す
        for item in directory.iterdir():
            if (item.is_file() and 
                    item.name.startswith("Index") and 
                    item.suffix == ".html"):
                result = self.parse_html_file(item)
                if result:
                    self.pages.append(result)
                    try:
                        rel = item.relative_to(self.root_dir)
                    except Exception:
                        rel = item
                    item_count = len(result['classifications'])
                    print(f"Parsed: {rel} -> {item_count} items")
        
        # サブディレクトリを探索
        for item in directory.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                self.explore_directory(item, max_depth, current_depth + 1)

    def parse_html_file(self, file_path: Path) -> Optional[Dict]:
        """1つの Index.html を解析して (title, file_path, classifications[]) を返す"""
        try:
            html = decode_euc_jp(file_path.read_bytes())
            soup = BeautifulSoup(html, "html.parser")

            title = soup.find("title").get_text() if soup.find("title") else ""
            
            # テーマコードと観点を抽出
            theme_code = extract_theme_code_from_path(file_path)
            viewpoint = extract_viewpoint_from_path(file_path)
            
            if not theme_code or not viewpoint:
                return None

            classifications = []
            table = soup.find("table")
            if not table:
                return {
                    "title": title, 
                    "file_path": str(file_path), 
                    "classifications": classifications
                }

            rows = table.find_all("tr")
            
            for i, row in enumerate(rows):
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue

                # サブコード（1列目）- NOWRAP属性があっても取得
                subcode_cell = cells[0]
                subcode_text = subcode_cell.get_text(strip=True)
                
                # サブコードが数値の場合のみ処理（00, 01, 02...）
                if re.match(r'^\d{2}$', subcode_text):
                    full_code = f"{theme_code}-{viewpoint}-{subcode_text}"
                    description = clean_html_text(cells[1].get_text())
                    
                    # タイトルを抽出（説明の最初の部分）
                    title_part = description.split('（')[0] if '（' in description else description
                    title_part = title_part[:100]  # 長すぎる場合は切り詰め
                    
                    classifications.append({
                        "code": full_code,
                        "title": clean_html_text(title_part),
                        "description": description,
                        "subcode": subcode_text
                    })

            return {
                "title": title, 
                "file_path": str(file_path), 
                "classifications": classifications,
                "theme_code": theme_code,
                "viewpoint": viewpoint
            }
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None


# ---------------------------
# 集約→JSONL 出力
# ---------------------------


def build_nodes(pages: List[Dict], parser: FTermParser) -> Dict[str, Dict]:
    nodes: Dict[str, Dict] = {}
    
    # テーマコードと観点のノードも作成
    theme_nodes = {}
    viewpoint_nodes = {}
    
    for page in pages:
        file_path = page["file_path"]
        theme_code = page["theme_code"]
        viewpoint = page["viewpoint"]
        
        # テーマコードノード
        if theme_code not in theme_nodes:
            theme_description = parser.theme_descriptions.get(theme_code, f"Fタームテーマコード {theme_code}")
            theme_nodes[theme_code] = {
                "chunk_id": f"FT-{theme_code}",
                "code": theme_code,  # テーマコードはハイフンなし
                "title": f"テーマコード {theme_code}",
                "description": theme_description,
                "level": "theme",
                "section": get_section_from_theme_code(theme_code),
                "breadcrumbs": [theme_code],
                "parent_code": None,
                "children_codes": [],
                "siblings_codes": [],
                "source_file": file_path,
                "content_type": "theme",
                "theme_code": theme_code
            }
        
        # 観点ノード
        viewpoint_code = f"{theme_code}-{viewpoint}"
        if viewpoint_code not in viewpoint_nodes:
            viewpoint_description = parser.viewpoint_descriptions.get(viewpoint_code, f"Fターム観点 {viewpoint}")
            
            viewpoint_nodes[viewpoint_code] = {
                "chunk_id": f"FT-{viewpoint_code}",
                "code": clean_code_for_display(viewpoint_code),  # ハイフン除去
                "title": f"観点 {viewpoint}",
                "description": viewpoint_description,
                "level": "viewpoint",
                "section": get_section_from_theme_code(theme_code),
                "breadcrumbs": [theme_code, viewpoint_code],
                "parent_code": theme_code,
                "children_codes": [],
                "siblings_codes": [],
                "source_file": file_path,
                "content_type": "viewpoint",
                "theme_code": theme_code,
                "viewpoint": viewpoint,
                "viewpoint_code": viewpoint_code
            }
        
        # サブコードノード
        for item in page["classifications"]:
            code = item["code"]
            title = item.get("title", "")
            desc = item.get("description", "")
            subcode = item.get("subcode", "")
            level = infer_level(code)
            section = get_section_from_theme_code(theme_code)

            node = {
                "chunk_id": f"FT-{code}",
                "code": clean_code_for_display(code),  # ハイフン除去
                "title": title or desc[:120],
                "description": desc,
                "level": level,
                "section": section,
                "breadcrumbs": make_breadcrumbs(code),
                "parent_code": parent_of(code),
                "children_codes": [],
                "siblings_codes": [],
                "source_file": file_path,
                "content_type": "subcode",
                "theme_code": theme_code,
                "viewpoint": viewpoint,
                "subcode": subcode,
                "viewpoint_code": f"{theme_code}-{viewpoint}"
            }
            nodes[code] = node

    print(f"DEBUG: Created {len(theme_nodes)} theme nodes, {len(viewpoint_nodes)} viewpoint nodes, {len(nodes)} subcode nodes")

    # 親子・兄弟リンクを埋める
    all_nodes = {**theme_nodes, **viewpoint_nodes, **nodes}
    
    for code, node in all_nodes.items():
        p = node.get("parent_code")
        if p and p in all_nodes:
            all_nodes[p]["children_codes"].append(code)
    
    for code, node in all_nodes.items():
        p = node.get("parent_code")
        if p and p in all_nodes:
            sibs = [c for c in all_nodes[p]["children_codes"] if c != code]
            node["siblings_codes"] = sibs

    # 階層的説明と検索用テキストを生成
    print("Generating hierarchical descriptions and search text...")
    for code, node in all_nodes.items():
        # 階層的説明を生成
        desc_hier = create_hierarchical_description(node, all_nodes, parser)
        node["desc_hier"] = desc_hier
        
        # search_textを詳細情報を含む形で生成
        search_text = create_search_text(node, desc_hier, parser)
        node["search_text"] = search_text
        
        # 祖先のコードを取得
        ancestors = get_ancestors(node, all_nodes)
        node["ancestors"] = ancestors

    return all_nodes


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
    parser = argparse.ArgumentParser(description='Fターム HTML → JSONL 変換')
    parser.add_argument('--root', default='.', help='Fタームルートディレクトリ')
    parser.add_argument('--section', default='ALL', help='対象セクション（2B, 3B, 5L等 or ALL）')
    parser.add_argument('--jsonl-out', help='出力ファイル名')
    parser.add_argument('--max-depth', type=int, default=12, help='最大探索深度（既定: 12）')
    args = parser.parse_args()

    root = Path(args.root)
    
    # セクションを特定
    if args.section.upper() == "ALL":
        # ルートディレクトリからセクションを自動検出
        sections = [
            d.name for d in root.iterdir() 
            if d.is_dir() and re.match(r'^\d+[A-Z]$', d.name)
        ]
    else:
        sections = [args.section.upper()]

    for sec in sections:
        print(f"=== セクション {sec} を解析中 ===")
        section_dir = root / sec
        
        # セクション番号を抽出（3B → 3）
        section_number = re.match(r'^(\d+)', sec)
        if section_number:
            section_number = section_number.group(1)
        else:
            section_number = section_number.group(1)
            
        fterm_parser = FTermParser(root)
        fterm_parser.explore_directory(section_dir, max_depth=args.max_depth)

        nodes = build_nodes(fterm_parser.pages, fterm_parser)

        if len(sections) == 1:
            if args.jsonl_out:
                out_path = Path(args.jsonl_out)
            else:
                out_path = Path(f"FTerm_{sec}.jsonl")
            write_jsonl(nodes, out_path, section_filter=section_number)
        else:
            out_path = Path(f"FTerm_{sec}.jsonl")
            write_jsonl(nodes, out_path, section_filter=section_number)

        print(f"Pages parsed: {len(fterm_parser.pages)}")


if __name__ == "__main__":
    main()

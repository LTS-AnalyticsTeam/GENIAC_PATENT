import os
import sys
import csv
import json
import argparse
import subprocess
from typing import List, Tuple, Dict, Any

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return None

# Local modules
from keyword_extractor import extract_keywords_keybert_fasttext
from cosmos_patent_search import search_cosmos
from azure.cosmos import CosmosClient


def read_text(path: str, max_chars: int = 4000) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    # Keep it bounded for upstream services (embedding, etc.)
    return txt[:max_chars]


def fetch_cosmos_doc_by_patent_id(patent_id: str) -> Dict[str, Any]:
    """Fetch a single document from Cosmos by metadata.patent_id.
    Requires COSMOS_ENDPOINT, COSMOS_KEY, DATABASE_NAME, CONTAINER_NAME.
    """
    load_dotenv()
    endpoint = os.environ.get("COSMOS_ENDPOINT")
    key = os.environ.get("COSMOS_KEY")
    database = os.environ.get("DATABASE_NAME")
    container = os.environ.get("CONTAINER_NAME")
    if not all([endpoint, key, database, container]):
        raise RuntimeError("COSMOS_ENDPOINT, COSMOS_KEY, DATABASE_NAME, CONTAINER_NAME を設定してください")
    client = CosmosClient(endpoint, key)
    cont = client.get_database_client(database).get_container_client(container)
    query = "SELECT TOP 1 c FROM c WHERE c.metadata.patent_id = @pid"
    params = [{"name": "@pid", "value": patent_id}]
    items = cont.query_items(query=query, parameters=params, enable_cross_partition_query=True)
    for it in items:
        # Return inner c if wrapped, else item itself
        return it.get("c", it)
    return {}


def build_text_from_doc(doc: Dict[str, Any], max_chars: int = 4000) -> str:
    meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
    title = meta.get("title", "") or doc.get("title", "") or ""
    summary = doc.get("summary", "") or ""
    claims = doc.get("claims", []) or []
    claims_text = []
    for cl in claims:
        if isinstance(cl, dict):
            t = cl.get("text")
            if t:
                claims_text.append(str(t))
        else:
            claims_text.append(str(cl))
    joined = "\n".join([title, summary, "\n".join(claims_text)])
    return joined[:max_chars]


def run_fi_fterm_search(query_text: str) -> Tuple[List[str], List[str]]:
    """Invoke FI/scripts/fi_fterm_search.py and parse TSV output.
    Returns (fi_codes, fterm_codes).
    """
    fi_codes: List[str] = []
    fterm_codes: List[str] = []

    if not query_text.strip():
        return fi_codes, fterm_codes

    script_path = os.path.join(os.path.dirname(__file__), '../../FI/scripts/fi_fterm_search.py')
    if not os.path.exists(script_path):
        # Script not found; return empty candidates gracefully
        return fi_codes, fterm_codes

    try:
        # NOTE: fi_fterm_search prints TSV lines: index\tcode\tchunk_id\tscore
        proc = subprocess.run(
            [sys.executable, script_path, query_text],
            capture_output=True,
            text=True,
            timeout=60
        )
        if proc.returncode != 0:
            # Azure keys or endpoint may be missing; ignore and continue
            return fi_codes, fterm_codes

        for line in proc.stdout.splitlines():
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            index_name = parts[0]
            code = parts[1].strip()
            if not code:
                continue
            if 'fi_' in index_name:
                fi_codes.append(code)
            elif 'fterm_' in index_name:
                fterm_codes.append(code)
            else:
                # Fallback: if code looks like F-term pattern (e.g., 3G091...), treat as F-term
                if code and code[0].isdigit():
                    fterm_codes.append(code)
                else:
                    fi_codes.append(code)
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        # Any failure here should not abort the whole pipeline
        pass

    # Deduplicate preserving order
    fi_seen = set()
    fi_unique = []
    for c in fi_codes:
        if c not in fi_seen:
            fi_seen.add(c)
            fi_unique.append(c)
    ft_seen = set()
    ft_unique = []
    for c in fterm_codes:
        if c not in ft_seen:
            ft_seen.add(c)
            ft_unique.append(c)

    return fi_unique, ft_unique


def extract_keywords_using_abc(text: str, topn: int = 12) -> List[str]:
    """Use abc_keyword_extractor's A/B/C keyword logic strictly (no fallback)."""
    import importlib.util
    here = os.path.dirname(__file__)
    mod_path = os.path.join(here, 'abc_keyword_extractor.py')
    spec = importlib.util.spec_from_file_location('abc_keyword_extractor', mod_path)
    if spec is None or spec.loader is None:
        raise ImportError('abc_keyword_extractor module not found')
    abc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abc)
    rec = {"title": "", "summary": text, "claims": []}
    a_cands, b_cands, c_cands, title_nouns = abc.extract_a_b_c_candidates_from_record(rec)
    a_kw, b_kw, c_kw = abc.select_best_keywords_llm(
        a_cands, b_cands, c_cands, title_nouns, rec["title"], rec["summary"], rec["claims"]
    )
    merged = " ".join([a_kw or "", b_kw or "", c_kw or ""]).replace(",", " ")
    kws = [k.strip() for k in merged.split() if k.strip()]
    # Deduplicate and cap
    out: List[str] = []
    seen = set()
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
        if len(out) >= topn:
            break
    return out


def fi_fterm_candidates_using_llm_keywords(keywords: List[str]) -> Tuple[List[str], List[str]]:
    """Use expect_patent_codes_by_llm.VectorSearchPredictor strictly (no fallback)."""
    import importlib.util
    here = os.path.dirname(__file__)
    mod_path = os.path.join(here, 'expect_patent_codes_by_llm.py')
    spec = importlib.util.spec_from_file_location('expect_patent_codes_by_llm', mod_path)
    if spec is None or spec.loader is None:
        raise ImportError('expect_patent_codes_by_llm module not found')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    predictor = mod.VectorSearchPredictor(enable_vector_search=True)
    fi, ft = predictor.search_fi_fterm(keywords or [])
    fi = list(dict.fromkeys([c for c in fi if c]))
    ft = list(dict.fromkeys([c for c in ft if c]))
    return fi, ft


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    # Required minimal columns
    fieldnames = [
        'patent_id', 'score', 'title', 'publication_date',
        'search_expressions', 'search_keywords', 'fi_candidates', 'fterm_candidates',
        'doc_fi', 'doc_fterm'
    ]
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            exprs = r.get('search_expressions', r.get('search_expression', ''))
            if isinstance(exprs, list):
                exprs_str = ' || '.join(exprs)
            else:
                exprs_str = exprs or ''
            # keywords / candidates (list -> joined string)
            kws = r.get('search_keywords', [])
            if isinstance(kws, list):
                kws_str = ' '.join(map(str, kws))
            else:
                kws_str = str(kws or '')
            fi_cands = r.get('fi_candidates', [])
            if isinstance(fi_cands, list):
                fi_cands_str = ' '.join(map(str, fi_cands))
            else:
                fi_cands_str = str(fi_cands or '')
            ft_cands = r.get('fterm_candidates', [])
            if isinstance(ft_cands, list):
                ft_cands_str = ' '.join(map(str, ft_cands))
            else:
                ft_cands_str = str(ft_cands or '')
            # doc-side FI / F-term present in search results
            doc_fi = r.get('fi', [])
            doc_ft = r.get('fterm', [])
            if isinstance(doc_fi, list):
                doc_fi_str = ' '.join(map(str, doc_fi))
            else:
                doc_fi_str = str(doc_fi or '')
            if isinstance(doc_ft, list):
                doc_ft_str = ' '.join(map(str, doc_ft))
            else:
                doc_ft_str = str(doc_ft or '')
            writer.writerow({
                'patent_id': r.get('patent_id', ''),
                'score': r.get('score', 0),
                'title': r.get('title', ''),
                'publication_date': r.get('publication_date', ''),
                'search_expressions': exprs_str,
                'search_keywords': kws_str,
                'fi_candidates': fi_cands_str,
                'fterm_candidates': ft_cands_str,
                'doc_fi': doc_fi_str,
                'doc_fterm': doc_ft_str,
            })


def main():
    parser = argparse.ArgumentParser(description='Run: text -> keywords + FI/F-term -> Cosmos DB search -> CSV')
    parser.add_argument('--text', required=False, help='Path to patent text (e.g., text.txt). Ignored if --patent_id is set')
    parser.add_argument('--patent_id', required=False, help='Patent ID to fetch from Cosmos (uses document content as input)')
    parser.add_argument('--csv', required=False, default=None, help='Output CSV path (default: cosmos_results_<timestamp>.csv)')
    parser.add_argument('--json', required=False, default=None, help='Optional JSON output path with full results')
    parser.add_argument('--topn_keywords', type=int, default=12, help='Top-N keywords to use')
    parser.add_argument('--extra_fi', nargs='*', default=[], help='Additional FI codes to include')
    parser.add_argument('--extra_fterm', nargs='*', default=[], help='Additional F-term codes to include')
    parser.add_argument('--limit', type=int, default=200, help='Max results')
    args = parser.parse_args()

    load_dotenv()

    # 0) Input acquisition: from Cosmos by patent_id, or from text file
    cosmos_doc: Dict[str, Any] = {}
    if args.patent_id:
        try:
            cosmos_doc = fetch_cosmos_doc_by_patent_id(args.patent_id)
            if not cosmos_doc:
                print(f"[ERROR] Cosmos に patent_id={args.patent_id} が見つかりませんでした")
                sys.exit(1)
            text = build_text_from_doc(cosmos_doc)
        except Exception as e:
            print(f"[ERROR] Cosmos からの取得に失敗しました: {e}")
            sys.exit(1)
    else:
        if not args.text:
            print("[ERROR] --text か --patent_id のどちらかを指定してください")
            sys.exit(1)
        text = read_text(args.text)

    # 1) Keywords (ABC extractor; no fallback). Always run ABC even if doc has keywords.
    try:
        kws = extract_keywords_using_abc(text, topn=args.topn_keywords)
    except Exception as e:
        print(f"[ERROR] キーワード抽出(abc_keyword_extractor)に失敗しました: {e}")
        sys.exit(1)

    # 2) FI / F-term candidates via LLM/Vector predictor (no fallback)
    #    Always use expect_patent_codes_by_llm (Azure Search) regardless of doc codes.
    try:
        fi_cands, ft_cands = fi_fterm_candidates_using_llm_keywords(kws)
    except Exception as e:
        print(f"[ERROR] FI/F-term候補推定(expect_patent_codes_by_llm)に失敗しました: {e}")
        sys.exit(1)
    if args.extra_fi:
        fi_cands.extend(args.extra_fi)
    if args.extra_fterm:
        ft_cands.extend(args.extra_fterm)

    # De-duplicate
    fi_cands = list(dict.fromkeys(fi_cands))
    ft_cands = list(dict.fromkeys(ft_cands))

    # 3) Cosmos search (multi-recipe)
    results = search_cosmos(kws, fi_cands, ft_cands, limit_total=args.limit)
    # Attach run-level context to each row for CSV visibility
    for r in results:
        r['search_keywords'] = list(kws)
        r['fi_candidates'] = list(fi_cands)
        r['fterm_candidates'] = list(ft_cands)

    # 4) Output
    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({"count": len(results), "results": results}, f, ensure_ascii=False, indent=2)

    out_csv = args.csv
    if not out_csv:
        import datetime as _dt
        ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_csv = f'cosmos_results_{ts}.csv'
    write_csv(results, out_csv)

    print(f"Keywords: {kws}")
    print(f"FI candidates: {fi_cands}")
    print(f"F-term candidates: {ft_cands}")
    print(f"Results: {len(results)} rows -> {out_csv}")


if __name__ == '__main__':
    main()

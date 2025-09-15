import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple, Set

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return None
from azure.cosmos import CosmosClient


def normalize_fi(code: str) -> str:
    if not code:
        return ""
    s = re.sub(r"\s+", "", code).upper()
    return s


def fi_prefix(code: str, min_len: int = 4) -> str:
    s = normalize_fi(code)
    # 例: A61B5/00 -> A61B5 / A61B
    # 斜線前を優先的に短縮しつつ、最低長を確保
    if "/" in s:
        left, right = s.split("/", 1)
        if len(left) >= min_len:
            return left
        return s[:min_len]
    return s[:max(min_len, min(len(s), 6))]


def normalize_fterm(code: str) -> str:
    return (code or "").strip().upper()


def fterm_prefixes(code: str) -> List[str]:
    """
    F-term 例: 3G091AA10 -> 上位桁候補: 3G091, 3G091AA
    """
    s = normalize_fterm(code)
    res: List[str] = []
    if len(s) >= 5:
        res.append(s[:5])  # テーマ
    if len(s) >= 7:
        res.append(s[:7])  # サブテーマ
    return res


def build_kw_pairs(keywords: List[str], limit_pairs: int = 6) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    n = min(len(keywords), 8)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((keywords[i], keywords[j]))
            if len(pairs) >= limit_pairs:
                return pairs
    return pairs


class CosmosPatentSearcher:
    def __init__(self, endpoint: str, key: str, database: str, container: str):
        client = CosmosClient(endpoint, key)
        self.container = client.get_database_client(database).get_container_client(container)

    def _run_query(self, query: str, params: Dict[str, Any], limit: int = 200, recipe: str = "") -> List[Dict[str, Any]]:
        parameters = [{"name": f"@{k}", "value": v} for k, v in params.items()]
        results = []
        items = self.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        )
        # Compose human-readable search expression
        try:
            expr = f"{query} | params: " + json.dumps(params, ensure_ascii=False)
        except Exception:
            expr = f"{query} | params: {params}"
        for item in items:
            # Attach search meta for downstream consumers
            try:
                item["_search_expr"] = expr
                if recipe:
                    item["_search_recipe"] = recipe
            except Exception:
                pass
            results.append(item)
            if len(results) >= limit:
                break
        return results

    def _select_clause(self) -> str:
        return (
            "SELECT c.metadata.patent_id AS patent_id, "
            "c.metadata.title AS title, c.metadata.classification_fi AS classification_fi, "
            "c.metadata.f_term AS f_term, c.summary AS summary, c.claims AS claims, c.description AS description, "
            "c.metadata.keywords AS doc_keywords, c.metadata.classification_ipc AS classification_ipc, "
            "c.metadata.publication_date AS publication_date"
        )

    def recipe_high_precision(self, fi_exact: str, fterm_prefix: str, kw_list: List[str], min_kw_hits: int = 2, limit: int = 100) -> List[Dict[str, Any]]:
        q = self._select_clause() + " FROM c WHERE " \
            "EXISTS (SELECT VALUE 1 FROM fi IN c.metadata.classification_fi WHERE fi = @fi_exact) " \
            "AND EXISTS (SELECT VALUE 1 FROM ft IN c.metadata.f_term WHERE STARTSWITH(ft, @ft_prefix, true)) " \
            "AND (SELECT VALUE COUNT(1) FROM kw IN c.metadata.keywords WHERE ARRAY_CONTAINS(@kw_list, kw, true)) >= @min_hits"
        return self._run_query(q, {
            "fi_exact": fi_exact,
            "ft_prefix": fterm_prefix,
            "kw_list": kw_list,
            "min_hits": min_kw_hits,
        }, limit, recipe=f"high_precision fi={fi_exact} ft_pref={fterm_prefix} min_hits={min_kw_hits} kws={kw_list[:5]}")

    def recipe_balanced(self, fi_pref: str, ft_pref: str, kw: str, limit: int = 200) -> List[Dict[str, Any]]:
        q = self._select_clause() + " FROM c WHERE (" \
            "EXISTS (SELECT VALUE 1 FROM fi IN c.metadata.classification_fi WHERE STARTSWITH(fi, @fi_pref, true)) " \
            "OR EXISTS (SELECT VALUE 1 FROM ft IN c.metadata.f_term WHERE STARTSWITH(ft, @ft_pref, true)) ) " \
            "AND (CONTAINS(c.metadata.title, @kw, true) OR CONTAINS(c.summary, @kw, true) OR CONTAINS(c.description, @kw, true) " \
            "OR EXISTS (SELECT VALUE 1 FROM cl IN c.claims WHERE CONTAINS(cl.text, @kw, true)))"
        return self._run_query(q, {
            "fi_pref": fi_pref,
            "ft_pref": ft_pref,
            "kw": kw,
        }, limit, recipe=f"balanced fi_pref={fi_pref} ft_pref={ft_pref} kw={kw}")

    def recipe_broad(self, fi_pref: str, ft_pref: str, kw: str, limit: int = 300) -> List[Dict[str, Any]]:
        q = self._select_clause() + " FROM c WHERE " \
            "EXISTS (SELECT VALUE 1 FROM fi IN c.metadata.classification_fi WHERE STARTSWITH(fi, @fi_pref, true)) " \
            "OR EXISTS (SELECT VALUE 1 FROM ft IN c.metadata.f_term WHERE STARTSWITH(ft, @ft_pref, true)) " \
            "OR CONTAINS(c.metadata.title, @kw, true)"
        return self._run_query(q, {
            "fi_pref": fi_pref,
            "ft_pref": ft_pref,
            "kw": kw,
        }, limit, recipe=f"broad fi_pref={fi_pref} ft_pref={ft_pref} kw={kw}")

    def recipe_claims_pair(self, kw1: str, kw2: str, limit: int = 100) -> List[Dict[str, Any]]:
        q = self._select_clause() + " FROM c WHERE " \
            "EXISTS (SELECT VALUE 1 FROM cl IN c.claims WHERE CONTAINS(cl.text, @kw1, true) AND CONTAINS(cl.text, @kw2, true))"
        return self._run_query(q, {"kw1": kw1, "kw2": kw2}, limit, recipe=f"claims_pair kw1={kw1} kw2={kw2}")


def count_hits(text: str, terms: List[str]) -> int:
    if not text:
        return 0
    count = 0
    for t in terms:
        if t and t in text:
            count += 1
    return count


def score_document(doc: Dict[str, Any], fi_exact_set: Set[str], fi_pref_set: Set[str], fterm_exact_set: Set[str], fterm_pref_set: Set[str], keywords: List[str]) -> Tuple[int, Dict[str, Any]]:
    score = 0
    signals: Dict[str, Any] = {}

    fi_list = [normalize_fi(x) for x in (doc.get("classification_fi") or [])]
    ft_list = [normalize_fterm(x) for x in (doc.get("f_term") or [])]

    # FI signals
    matched_fi_exact = any(fi in fi_exact_set for fi in fi_list)
    matched_fi_pref = any(any(fi.startswith(pref) for pref in fi_pref_set) for fi in fi_list)
    if matched_fi_exact:
        score += 5
    if matched_fi_pref:
        score += 3
    signals["matched_fi_exact"] = matched_fi_exact
    signals["matched_fi_pref"] = matched_fi_pref

    # F-term signals
    matched_ft_exact = any(ft in fterm_exact_set for ft in ft_list)
    matched_ft_pref = any(any(ft.startswith(pref) for pref in fterm_pref_set) for ft in ft_list)
    if matched_ft_exact:
        score += 5
    if matched_ft_pref:
        score += 3
    signals["matched_ft_exact"] = matched_ft_exact
    signals["matched_ft_pref"] = matched_ft_pref

    # Text hits
    title = doc.get("title") or ""
    summary = doc.get("summary") or ""
    claims = doc.get("claims") or []
    title_hits = count_hits(title, keywords)
    summary_hits = count_hits(summary, keywords)
    claim_hits = 0
    for cl in claims:
        txt = cl.get("text") if isinstance(cl, dict) else ""
        claim_hits += count_hits(txt or "", keywords)
    score += min(5, title_hits * 3 + summary_hits * 2 + min(3, claim_hits))
    signals["title_hits"] = title_hits
    signals["summary_hits"] = summary_hits
    signals["claim_hits"] = claim_hits

    # Keyword intersection with doc keywords
    doc_kws = doc.get("doc_keywords") or []
    kw_intersection = len(set(doc_kws) & set(keywords))
    score += min(3, kw_intersection)  # cap
    signals["kw_intersection"] = kw_intersection

    # Recency slight boost
    pubdate = (doc.get("publication_date") or "")
    try:
        y = int(pubdate[:4]) if len(pubdate) >= 4 else 0
        score += max(0, (y - 2000) // 10)  # +1 per decade since 2000
    except Exception:
        pass

    return score, signals


def search_cosmos(keywords: List[str], fi_codes: List[str], fterm_codes: List[str], limit_total: int = 300) -> List[Dict[str, Any]]:
    load_dotenv()
    endpoint = os.environ.get("COSMOS_ENDPOINT")
    key = os.environ.get("COSMOS_KEY")
    database = os.environ.get("DATABASE_NAME")
    container = os.environ.get("CONTAINER_NAME")
    if not all([endpoint, key, database, container]):
        raise RuntimeError("COSMOS_ENDPOINT, COSMOS_KEY, DATABASE_NAME, CONTAINER_NAME を設定してください")

    searcher = CosmosPatentSearcher(endpoint, key, database, container)

    # Normalization
    fi_exact = list({normalize_fi(x) for x in fi_codes if x})
    fi_prefs = list({fi_prefix(x) for x in fi_codes if x})
    ft_exact = list({normalize_fterm(x) for x in fterm_codes if x})
    ft_prefs = list({p for x in fterm_codes for p in fterm_prefixes(x)})

    # Collect candidates from multiple recipes
    # candidates[patent_id] = {"doc": doc, "exprs": ["[recipe] SQL | params: {...}", ...]}
    candidates: Dict[str, Dict[str, Any]] = {}
    recipes_count = 0

    # High precision recipe disabled by request
    kw_list = keywords[:12]

    # Balanced: FI or F-term prefix with one keyword (try top 3 keywords)
    for fi_p in fi_prefs[:3]:
        for ft_p in ft_prefs[:3]:
            for kw in kw_list[:3]:
                rows = searcher.recipe_balanced(fi_p, ft_p, kw, limit=80)
                recipes_count += 1
                for d in rows:
                    pid = d.get("patent_id")
                    if not pid:
                        continue
                    label = d.get("_search_expr") or ""
                    if d.get("_search_recipe"):
                        label = f"[{d.get('_search_recipe')}] {label}"
                    ag = candidates.setdefault(pid, {"doc": d, "exprs": []})
                    if label and label not in ag["exprs"]:
                        ag["exprs"].append(label)

    # Broad and claims-pair recipes are disabled

    # Scoring
    fi_exact_set = set(fi_exact)
    fi_pref_set = set(fi_prefs)
    ft_exact_set = set(ft_exact)
    ft_pref_set = set(ft_prefs)

    ranked: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for pid, ag in candidates.items():
        doc = ag["doc"]
        s, sig = score_document(doc, fi_exact_set, fi_pref_set, ft_exact_set, ft_pref_set, kw_list)
        ranked.append((s, doc, sig, ag["exprs"]))

    ranked.sort(key=lambda x: x[0], reverse=True)
    ranked = ranked[:limit_total]

    # Build output rows
    out: List[Dict[str, Any]] = []
    for s, d, sig, exprs in ranked:
        out.append({
            "patent_id": d.get("patent_id"),
            "title": d.get("title"),
            "publication_date": d.get("publication_date"),
            "score": s,
            "signals": sig,
            "fi": d.get("classification_fi"),
            "fterm": d.get("f_term"),
            "search_expressions": exprs,
        })
    return out


def load_inputs_from_json(json_path: str) -> Tuple[List[str], List[str], List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    keywords = meta.get("keywords", []) or []
    fi = meta.get("classification_fi", []) or []
    fterm = meta.get("f_term", []) or []
    # classification_fi may be mixed with FI strings; ensure strings only
    fi_strs: List[str] = []
    for x in fi:
        if isinstance(x, str):
            fi_strs.append(x)
        elif isinstance(x, dict):
            v = x.get("code") or x.get("raw")
            if v:
                fi_strs.append(v)
    return keywords, fi_strs, [str(x) for x in fterm if x]


def main():
    parser = argparse.ArgumentParser(description="Cosmos DB Patent Search (multi-recipe)")
    parser.add_argument("--json", type=str, help="input JSON path (converted patent data)")
    parser.add_argument("--keywords", nargs="*", default=[], help="keywords (override or supplement)")
    parser.add_argument("--fi", nargs="*", default=[], help="FI codes")
    parser.add_argument("--fterm", nargs="*", default=[], help="F-term codes")
    parser.add_argument("--limit", type=int, default=200, help="max results")
    args = parser.parse_args()

    kw: List[str] = []
    fi: List[str] = []
    ft: List[str] = []
    if args.json:
        kw, fi, ft = load_inputs_from_json(args.json)
    if args.keywords:
        kw = args.keywords
    if args.fi:
        fi = args.fi
    if args.fterm:
        ft = args.fterm

    if not (kw or fi or ft):
        print("At least one of keywords, fi, fterm must be provided (or via --json)")
        return

    results = search_cosmos(kw, fi, ft, limit_total=args.limit)
    print(json.dumps({
        "count": len(results),
        "results": results
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

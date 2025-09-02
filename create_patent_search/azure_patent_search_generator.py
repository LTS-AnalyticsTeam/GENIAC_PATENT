import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Set

from openai import AzureOpenAI


def calculate_ntx_value(total_chars: int) -> int:
    """文字数に基づいてN/TX値を計算"""
    if 40 <= total_chars < 70:
        return 20
    if 70 <= total_chars < 100:
        return 30
    if 100 <= total_chars < 120:
        return 40
    if 120 <= total_chars < 140:
        return 50
    if 140 <= total_chars < 160:
        return 60
    if 160 <= total_chars < 180:
        return 70
    if 180 <= total_chars < 200:
        return 80
    if total_chars >= 200:
        return 90
    return 20


def group_with_aoai(
    client: AzureOpenAI,
    model: str,
    keywords: List[str]
):
    """Azure OpenAI でキーワードを関連グループに分割する。

    失敗時は各キーワードを単独グループとして返す。
    出力フォーマット期待値:
      {
        "groups": [
          {"keywords": [0, 1]},
          {"keywords": [2]}
        ]
      }
    """
    if not keywords:
        return []
    if len(keywords) == 1:
        return [{0}]

    indexed = [f"{i}: {kw}" for i, kw in enumerate(keywords)]
    sys_prompt = (
        "あなたは特許調査の補助エージェントです。技術用語の関連性に基づき、"
        "番号付きキーワードをグルーピングしてください。出力は厳密にJSONのみ。"
    )
    user_prompt = (
        "キーワード一覧:\n" + "\n".join(indexed) +
        "\n\n出力形式:\n{\n  \"groups\": [\n    {\"keywords\": [0,1]},\n"
        "    {\"keywords\": [2]}\n  ]\n}\n"
        "全ての番号(0-" + str(len(keywords) - 1) + ")を必ず1回ずつ含めてください。"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        text = resp.choices[0].message.content.strip()

        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text)
        groups = []
        for g in data.get("groups", []):
            indices = set()
            for idx in g.get("keywords", []):
                if isinstance(idx, int) and 0 <= idx < len(keywords):
                    indices.add(idx)
            if indices:
                groups.append(indices)

        # 取りこぼしがあれば単独で補完
        covered = set()
        for s in groups:
            covered.update(s)
        for i in range(len(keywords)):
            if i not in covered:
                groups.append({i})

        return groups
    except Exception:
        return [{i} for i in range(len(keywords))]


def build_expression_from_groups(keywords: List[str], clusters: List[Set[int]]) -> str:
    if not keywords:
        return ""
    cluster_exprs = []
    for cluster in clusters:
        terms = [keywords[i] for i in cluster]
        if len(terms) == 1:
            cluster_exprs.append(f"({terms[0]})")
        else:
            cluster_exprs.append("(" + "+".join(terms) + ")")
    total_chars = sum(len(k) for k in keywords)
    ntx = calculate_ntx_value(total_chars)
    mid = max(1, len(cluster_exprs) // 2)
    left = ",".join(cluster_exprs[:mid])
    right = ",".join(cluster_exprs[mid:])
    if right:
        return f"{{{left}}},{ntx}N/TX*{{{right}}}"
    return f"{{{left}}},{ntx}N/TX*{{{left}}}"


def convert_to_jplatpat(original_expression: str) -> str:
    import re
    ntx_match = re.search(r',(\d+)N/TX\*', original_expression)
    ntx_value = ntx_match.group(1) if ntx_match else '20'
    parts = original_expression.split(f',{ntx_value}N/TX*')
    if len(parts) != 2:
        res = original_expression.replace('{', '(').replace('}', ')')
        res = re.sub(r',(\d+)N/TX\*', r',\1N,', res)
        return f"({res})/TX"

    def parse_block(block: str) -> List[List[str]]:
        groups = []
        token = ""
        depth = 0
        for ch in block:
            if ch == '(':
                depth += 1
                if depth == 1:
                    token = ""
                else:
                    token += ch
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    if token:
                        groups.append([t.strip() for t in token.split('+')])
                    token = ""
                else:
                    token += ch
            elif ch == ',' and depth == 0:
                continue
            else:
                if depth > 0:
                    token += ch
        return groups

    left = parts[0].strip('{}')
    right = parts[1].strip('{}')
    left_groups = parse_block(left)
    right_groups = parse_block(right)

    def block_expr(gs: List[List[str]]) -> str:
        segs = []
        for g in gs:
            if len(g) == 1:
                segs.append(f"{g[0]}/TX")
            else:
                segs.append("(" + "+".join(g) + ")/TX")
        return "* ".join(segs)

    pairs = []
    if left_groups and right_groups:
        limit = min(4, min(len(left_groups), len(right_groups)))
        for i in range(limit):
            lk = left_groups[i][0] if left_groups[i] else ""
            rk = right_groups[i][0] if right_groups[i] else ""
            if lk and rk:
                pairs.append(
                    f"({lk},{ntx_value}N,{rk})/TX"
                )

    parts_out = []
    if pairs:
        parts_out.append("[" + " + ".join(pairs) + "]")
    parts_out.append(block_expr(left_groups))
    if right_groups:
        parts_out.append(block_expr(right_groups))
    return "* ".join(parts_out)


def _aggregate_leaf_code(rows: List[Dict[str, Any]]):
    counts = defaultdict(int)
    score_sum = defaultdict(float)
    for row in rows:
        code = row.get("code")
        if not code:
            continue
        code = str(code)
        counts[code] += 1
        score_sum[code] += float(row.get("score", 0.0))
    ranked = sorted(counts.keys(), key=lambda k: (counts[k], score_sum[k]), reverse=True)
    return ranked, counts, score_sum


def main():
    parser = argparse.ArgumentParser(
        description='Azure OpenAI 版 特許検索式生成ツール'
    )
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('keywords', nargs='*', default=[], help='キーワード群')
    input_group.add_argument('-f', '--file', type=str, help='キーワードファイルパス')
    parser.add_argument('--jplatpat', action='store_true', help='J-PlatPat形式で出力')
    parser.add_argument(
        '--aoai-endpoint',
        type=str,
        default=os.getenv('AOAI_ENDPOINT', '')
    )
    parser.add_argument(
        '--aoai-key',
        type=str,
        default=os.getenv('AOAI_KEY', os.getenv('AOAI_API_KEY', ''))
    )
    parser.add_argument(
        '--aoai-model',
        type=str,
        default=os.getenv('AOAI_CHAT_MODEL', 'gpt-4o-mini')
    )
    args = parser.parse_args()

    if args.file:
        kws = []
        seen = set()
        with open(args.file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and line not in seen:
                    kws.append(line)
                    seen.add(line)
    else:
        kws = args.keywords

    if not kws:
        print("", end="")
        return

    client = AzureOpenAI(
        api_key=args.aoai_key,
        api_version="2024-02-01",
        azure_endpoint=args.aoai_endpoint,
    )

    clusters = group_with_aoai(client, args.aoai_model, kws)
    expr = build_expression_from_groups(kws, clusters)
    if args.jplatpat:
        expr = convert_to_jplatpat(expr)
    print(expr)


if __name__ == "__main__":
    main()


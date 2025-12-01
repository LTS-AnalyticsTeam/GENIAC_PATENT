"""
Web検索サービス
arXiv, Crossref, OpenAlexから関連文献を検索し、LLMで要約する
"""
import asyncio
import logging
import os
import re
from typing import Any, Dict, List

import httpx
import feedparser
from bs4 import BeautifulSoup
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

# 設定
ARXIV_MAX_RESULTS = 10
CROSSREF_MAX_RESULTS = 10
OPENALEX_MAX_RESULTS = 10


def normalize_record(title: str, abstract: str, source_url: str, source: str) -> Dict[str, Any]:
    """検索結果を正規化"""
    return {
        "title": title or "",
        "abstract": abstract or "",
        "source_url": source_url or "",
        "source": source,
    }


def deduplicate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """タイトルでレコードを重複排除"""
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in records:
        key = r["title"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


async def fetch_arxiv(client: httpx.AsyncClient, query: str) -> List[Dict[str, Any]]:
    """arXivから検索"""
    try:
        url = "https://export.arxiv.org/api/query"
        params = {"search_query": f"all:{query}", "start": 0, "max_results": ARXIV_MAX_RESULTS}
        resp = await client.get(url, params=params, timeout=30)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        out: List[Dict[str, Any]] = []
        for e in feed.entries:
            out.append(
                normalize_record(
                    e.get("title", ""),
                    e.get("summary", ""),
                    e.get("id", ""),
                    "arxiv",
                )
            )
        return out
    except Exception as e:
        logger.warning(f"arXiv search failed: {e}")
        return []


async def fetch_crossref(client: httpx.AsyncClient, query: str) -> List[Dict[str, Any]]:
    """Crossrefから検索"""
    try:
        url = "https://api.crossref.org/works"
        params = {"query": query, "rows": CROSSREF_MAX_RESULTS}
        resp = await client.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        out: List[Dict[str, Any]] = []
        for it in data.get("message", {}).get("items", []):
            title = (it.get("title") or [""])[0]
            abstract = it.get("abstract", "")
            link = it.get("URL", "")
            out.append(normalize_record(title, abstract, link, "crossref"))
        return out
    except Exception as e:
        logger.warning(f"Crossref search failed: {e}")
        return []


async def fetch_openalex(client: httpx.AsyncClient, query: str) -> List[Dict[str, Any]]:
    """OpenAlexから検索"""
    try:
        url = "https://api.openalex.org/works"
        params = {"search": query, "per-page": OPENALEX_MAX_RESULTS}
        resp = await client.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        out: List[Dict[str, Any]] = []
        for w in data.get("results", []):
            title = (w.get("display_name") or "").strip()
            inv = w.get("abstract_inverted_index") or {}
            if inv:
                max_pos = max(pos for ps in inv.values() for pos in ps)
                words = [""] * (max_pos + 1)
                for w2, ps in inv.items():
                    for pos in ps:
                        if 0 <= pos < len(words):
                            words[pos] = w2
                abstract = " ".join(words)
            else:
                abstract = ""
            link = (w.get("id") or "").replace("https://api.openalex.org/", "https://openalex.org/")
            out.append(normalize_record(title, abstract, link, "openalex"))
        return out
    except Exception as e:
        logger.warning(f"OpenAlex search failed: {e}")
        return []


async def retrieve_candidates(claim_text: str) -> List[Dict[str, Any]]:
    """3つのAPIから候補をまとめて取得"""
    q = claim_text.replace("\n", " ")[:200]  # 先頭200文字
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        arxiv, cr, oa = await asyncio.gather(
            fetch_arxiv(client, q),
            fetch_crossref(client, q),
            fetch_openalex(client, q),
        )
    return arxiv + cr + oa


def select_top10(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """タイトルあり・abstractあり・abstract長い順でソートしてTop10"""
    uniq = deduplicate(records)
    uniq.sort(
        key=lambda x: (bool(x["title"]), bool(x["abstract"]), len(x["abstract"])),
        reverse=True,
    )
    return uniq[:10]


def scrape_page_info(url: str) -> Dict[str, str]:
    """URLからページ情報を取得"""
    info = {
        "page_title": "",
        "page_abstract": "",
        "page_text_excerpt": "",
    }
    if not url:
        return info

    try:
        resp = httpx.get(url, follow_redirects=True, timeout=20)
        resp.raise_for_status()
    except Exception:
        return info

    soup = BeautifulSoup(resp.text, "html.parser")

    def good_text(text: str, min_len: int = 10) -> bool:
        if not text:
            return False
        t = text.strip()
        if len(t) < min_len:
            return False
        if t.lower() in {"j-stage", "jstage", "j stage"}:
            return False
        return True

    # タイトル候補
    title = ""
    meta_ct = soup.find("meta", attrs={"name": "citation_title"})
    if meta_ct and good_text(meta_ct.get("content", "")):
        title = meta_ct["content"].strip()

    if not title:
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and good_text(og_title.get("content", "")):
            title = og_title["content"].strip()

    if not title:
        dc_title = soup.find("meta", attrs={"name": "dc.title"})
        if dc_title and good_text(dc_title.get("content", "")):
            title = dc_title["content"].strip()

    if not title and soup.title and good_text(soup.title.get_text()):
        title = soup.title.get_text().strip()

    info["page_title"] = title

    # abstract / description
    abstract = ""
    for name in ["citation_abstract", "dc.description", "description"]:
        tag = soup.find("meta", attrs={"name": name})
        if tag and good_text(tag.get("content", ""), min_len=40):
            abstract = tag["content"].strip()
            break

    if not abstract:
        og_desc = soup.find("meta", attrs={"property": "og:description"})
        if og_desc and good_text(og_desc.get("content", ""), min_len=40):
            abstract = og_desc["content"].strip()

    if not abstract:
        abs_like = soup.find_all(
            lambda tag: tag.has_attr("class") and any("abstract" in c.lower() for c in tag["class"])
        )
        for tag in abs_like:
            txt = tag.get_text(" ", strip=True)
            if good_text(txt, min_len=40):
                abstract = txt
                break

    if not abstract:
        tag = soup.find(id=lambda v: isinstance(v, str) and "abstract" in v.lower())
        if tag:
            txt = tag.get_text(" ", strip=True)
            if good_text(txt, min_len=40):
                abstract = txt

    info["page_abstract"] = abstract

    # 本文テキスト(p要素から抜粋)
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    body_text = "\n\n".join(t for t in paras if good_text(t, min_len=30))
    info["page_text_excerpt"] = body_text[:4000]

    return info


def extract_json_block(text: str) -> str:
    """LLM出力からJSON部分を抽出"""
    if not text:
        raise ValueError("empty content")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON-looking block found")
    return text[start : end + 1]


def clean_abstract_text(text: str) -> str:
    """abstractから不要なメタデータを除去"""
    if not text:
        return ""

    remove_patterns = [
        r"JSTプロジェクトデータベース.*",
        r"掲載開始日.*",
        r"最終更新日.*",
        r"\d{4}-\d{2}-\d{2}",
        r"産学が連携した研究開発成果の展開.*",
        r"研究成果展開事業.*",
        r"マッチングプランナープログラム.*",
        r"研究課題コード.*",
        r"プロジェクト番号.*",
    ]

    result = text
    for pattern in remove_patterns:
        result = re.sub(pattern, "", result, flags=re.MULTILINE)

    result = re.sub(r"\n\s*\n+", "\n", result)
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip()


def get_azure_client() -> AzureOpenAI:
    """Azure OpenAIクライアントを取得"""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not api_key or not endpoint:
        raise RuntimeError(".env に AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT が必要です。")

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def choose_best_with_azure(
    client: AzureOpenAI,
    deployment: str,
    claim: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """LLMで最も関連する1件を選択"""
    import json

    cand_prompt = [
        {
            "index": i,
            "title": c["title"],
            "abstract": c["abstract"][:800],
            "source_url": c["source_url"],
            "source": c["source"],
        }
        for i, c in enumerate(candidates)
    ]

    user_content = (
        "You are an assistant that selects the single best reference.\n"
        "You MUST return ONLY a JSON object like {\"index\": N}.\n"
        "The output must be valid JSON and nothing else.\n\n"
        "Patent claim text:\n"
        f"{claim}\n\n"
        "Here is a JSON array of candidate references (index, title, abstract, source_url, source):\n"
        f"{json.dumps(cand_prompt, ensure_ascii=False)}\n\n"
        "Return ONLY a JSON object with a single key \"index\" whose value is the integer index of "
        "the single most relevant reference. Do not output anything except JSON."
    )

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=256,
        )

        content = resp.choices[0].message.content or ""
        if not content:
            logger.warning("choose_best_with_azure: empty response from API")
            return candidates[0]

        json_text = extract_json_block(content)
        data = json.loads(json_text)
        idx = int(data.get("index", 0))
    except Exception as e:
        logger.warning(f"choose_best_with_azure failed: {e}")
        idx = 0

    if idx < 0 or idx >= len(candidates):
        idx = 0

    return candidates[idx]


def summarize_page_with_azure(
    client: AzureOpenAI,
    deployment: str,
    best: Dict[str, Any],
    page_info: Dict[str, str],
) -> Dict[str, Any]:
    """LLMでページ情報を日本語要約"""
    import json

    metadata_abstract = best.get("abstract", "")
    page_abstract = page_info.get("page_abstract", "")
    page_text = page_info.get("page_text_excerpt", "")

    available_content = []
    if metadata_abstract:
        available_content.append(f"【メタデータ抄録】\n{metadata_abstract}")
    if page_abstract:
        available_content.append(f"【ページ抄録】\n{page_abstract}")
    if page_text:
        available_content.append(f"【本文抜粋】\n{page_text[:2000]}")

    content_text = "\n\n".join(available_content) if available_content else "情報なし"

    title_candidate = (
        best.get("title")
        or page_info.get("page_title")
        or "タイトル不明"
    )

    user_prompt = f"""あなたは科学技術文献を日本語で要約する専門家です。

【タスク】
与えられた文献情報から、以下のJSON形式で出力してください：

{{"bibliographic": {{"title": "日本語タイトル"}}, "abstract": "技術的な要約（3〜5文）"}}

【abstractの書き方】
- 研究の目的、手法、主な成果を簡潔にまとめる
- 技術的な内容に焦点を当てる
- 日付、プロジェクト名、データベース情報などのメタデータは含めない
- 「〜と考えられる」「〜の可能性がある」などの推測表現は使わない
- 情報が不十分な場合は、タイトルから推測される研究内容を簡潔に説明する

【出力形式】
- 必ず有効なJSONのみを出力
- 余計なテキストや説明は一切付けない

==============================
以下の文献情報をもとに、日本語タイトルと技術的要約をJSON形式で作成してください。

【タイトル候補】
{title_candidate}

【利用可能な情報】
{content_text}

【出力】
JSONオブジェクトのみを出力してください。"""

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=1000,
        )
        raw = resp.choices[0].message.content or ""
        if raw:
            try:
                json_text = extract_json_block(raw)
                enriched = json.loads(json_text)
                if "abstract" in enriched:
                    enriched["abstract"] = clean_abstract_text(enriched["abstract"])

                title = (
                    enriched.get("bibliographic", {}).get("title")
                    or enriched.get("title")
                    or page_info.get("page_title")
                    or best.get("title")
                    or "関連資料"
                )

                abstract = (
                    enriched.get("abstract")
                    or page_info.get("page_abstract")
                    or page_info.get("page_text_excerpt", "")[:400]
                    or best.get("abstract")
                    or title
                )

                return {
                    "bibliographic": {"title": title},
                    "abstract": abstract,
                }
            except Exception as e:
                logger.warning(f"summarize_page_with_azure JSON parse failed: {e}")
    except Exception as e:
        logger.warning(f"summarize_page_with_azure failed: {e}")

    # フォールバック
    title = best.get("title") or page_info.get("page_title") or "関連技術文献"
    abstract = (
        best.get("abstract")
        or page_info.get("page_abstract")
        or page_info.get("page_text_excerpt", "")[:500]
    )
    abstract = clean_abstract_text(abstract)

    if not abstract or len(abstract) < 30:
        abstract = f"本文献は「{title}」に関する研究・技術資料である。詳細については原文を参照されたい。"

    return {
        "bibliographic": {"title": title},
        "abstract": abstract,
    }


async def search_web_references(claim1_text: str, azure_deployment: str = None) -> List[Dict[str, Any]]:
    """
    Web検索を実行し、関連文献を取得

    Args:
        claim1_text: 請求項1のテキスト
        azure_deployment: Azure OpenAIのデプロイメント名

    Returns:
        Web検索結果のリスト(最大10件)
        各要素: {
            "patent_id": "WEB_{source}_{index}",
            "title": "...",
            "summary": "...",
            "source_url": "...",
            "source": "arxiv|crossref|openalex",
            "is_web_result": True
        }
    """
    logger.info("Starting web search for related references...")

    # 候補取得
    all_records = await retrieve_candidates(claim1_text)
    if not all_records:
        logger.warning("No web search candidates found")
        return []

    logger.info(f"Retrieved {len(all_records)} candidates from web APIs")

    # Top10選定
    top10 = select_top10(all_records)
    logger.info(f"Selected top {len(top10)} candidates")

    # Azure OpenAIで要約・翻訳
    if not azure_deployment:
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if not azure_deployment:
        logger.warning("AZURE_OPENAI_DEPLOYMENT not set, skipping LLM enrichment")
        # LLMなしでそのまま返す
        results = []
        for idx, record in enumerate(top10):
            results.append({
                "patent_id": f"WEB_{record['source']}_{idx}",
                "title": record["title"] or "タイトル不明",
                "summary": record["abstract"] or "要約なし",
                "source_url": record["source_url"],
                "source": record["source"],
                "is_web_result": True,
            })
        return results

    # LLMで各候補を要約
    client = get_azure_client()
    results = []

    for idx, record in enumerate(top10):
        try:
            # ページ情報取得
            page_info = scrape_page_info(record.get("source_url", ""))

            # LLMで要約
            enriched = summarize_page_with_azure(client, azure_deployment, record, page_info)

            title = enriched.get("bibliographic", {}).get("title") or record["title"] or "関連資料"
            abstract = enriched.get("abstract") or record["abstract"] or title

            results.append({
                "patent_id": f"WEB_{record['source']}_{idx}",
                "title": title,
                "summary": abstract,
                "source_url": record["source_url"],
                "source": record["source"],
                "is_web_result": True,
            })

            logger.info(f"Processed web result {idx+1}/{len(top10)}: {title[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to process web result {idx}: {e}")
            # エラーでも最低限の情報は追加
            results.append({
                "patent_id": f"WEB_{record['source']}_{idx}",
                "title": record["title"] or "タイトル不明",
                "summary": record["abstract"] or "要約取得失敗",
                "source_url": record["source_url"],
                "source": record["source"],
                "is_web_result": True,
            })

    logger.info(f"Web search completed: {len(results)} results")
    return results

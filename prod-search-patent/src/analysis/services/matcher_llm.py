from __future__ import annotations
from typing import List, Dict, Any
import json
from ..models import EvidenceHit, PatentDoc, Span
from .aoai import AOAIClient
from .matcher_heuristic import _gather_sections

PROMPT_SYS = "You are a patent examiner analyzing novelty. Return STRICT JSON with evidence spans."
JSON_MODE = {"type": "json_object"}

def _build_section_dump(doc: PatentDoc):
    items = []
    for section, num, text in _gather_sections(doc.description):
        items.append({"section": section, "num": num or None, "text": text})
    return items

def _hits_from_json(obj: Dict[str, Any]) -> List[EvidenceHit]:
    hits: List[EvidenceHit] = []
    for h in obj.get("hits", []):
        spans = [Span(start=int(s["start"]), end=int(s["end"])) for s in h.get("spans", []) if "start" in s and "end" in s]
        hits.append(EvidenceHit(
            section=h["section"],
            num=h.get("num"),
            text=h["text"],
            spans=spans,
            support=1,
            score=float(h.get("score", 0.7))
        ))
    return hits

async def model_a_match(client: AOAIClient, src: PatentDoc, cand: PatentDoc, deployment: str) -> List[EvidenceHit]:
    """新規性判定に特化した分析"""
    content = {
        "task": "novelty_analysis",
        "source_claim1": src.claim1,
        "source_sections": _build_section_dump(src),
        "candidate_sections": _build_section_dump(cand),
        "instructions": """
         候補特許が元クレーム1の新規性を阻害する（新規性を否定する）かを分析する。
        クレーム1の全ての構成要素を候補文書内で見つけることに焦点を当てる。
        新規性拒絶のためには、全てのクレーム要素が単一の先行技術に存在する必要がある。
        
        以下を含むJSONを返す：
        1. "hits": 文字位置を含む一致テキストの配列
        2. "novelty_points": 新規性が否定される理由を説明する正確に3つの箇条書き
        3. "missing_elements": 候補文書に見つからなかったクレーム要素（もしあれば）
        
        重要: 請求項1の全ての構成要素が先行技術に開示されているかを判断
        """
    }
    
    resp = await client.chat(
        deployment=deployment,
        messages=[
            {"role": "system", "content": PROMPT_SYS},
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)}
        ],
        response_format=JSON_MODE,
        temperature=0.0,
    )
    
    raw = resp["choices"][0]["message"]["content"]
    result = json.loads(raw)
    
    # 新規性判定ポイントを取得
    novelty_points = result.get("novelty_points", [])
    missing_elements = result.get("missing_elements", [])
    
    # EvidenceHitに判定結果を含める
    hits = _hits_from_json(result)
    
    # 最初のヒットに新規性判定情報を付加（メタデータとして）
    if hits and novelty_points:
        hits[0].novelty_points = novelty_points
        hits[0].missing_elements = missing_elements
    
    return hits

async def model_b_match(client: AOAIClient, src: PatentDoc, cand: PatentDoc, deployment: str) -> List[EvidenceHit]:
    """進歩性の観点からの分析"""
    content = {
        "task": "inventive_step_analysis",
        "focus": ["technical-field", "background-art", "summary-of-invention", "advantageous-effects"],
        "source_claim1": src.claim1,
        "source_sections": _build_section_dump(src),
        "candidate_sections": _build_section_dump(cand),
        "instructions": """
        Analyze inventive step (non-obviousness) aspects.
        Find technical similarities and differences.
        
        Return JSON with:
        1. "hits": matching portions with character spans
        2. "technical_differences": key technical differences found
        3. "combination_hints": suggestions for combining with other references
        
        重要: 技術的な相違点と組み合わせの可能性を分析
        """
    }
    
    resp = await client.chat(
        deployment=deployment,
        messages=[
            {"role": "system", "content": PROMPT_SYS},
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)}
        ],
        response_format=JSON_MODE,
        temperature=0.0,
    )
    
    raw = resp["choices"][0]["message"]["content"]
    result = json.loads(raw)
    
    # 進歩性判定情報を取得
    technical_differences = result.get("technical_differences", [])
    combination_hints = result.get("combination_hints", [])
    
    hits = _hits_from_json(result)
    
    # メタデータを付加
    if hits:
        if technical_differences:
            hits[0].technical_differences = technical_differences
        if combination_hints:
            hits[0].combination_hints = combination_hints
    
    return hits

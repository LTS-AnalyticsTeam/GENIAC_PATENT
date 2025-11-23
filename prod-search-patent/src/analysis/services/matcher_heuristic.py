# app/services/matcher_heuristic.py
from __future__ import annotations
import itertools
import re
from typing import Iterable, List, Tuple
from ..models import EvidenceHit, PatentDoc, Span

TOKEN_RE = re.compile(r"[A-Za-z0-9一-龯ぁ-んァ-ンー]+")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text or "")

def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        return []
    return [tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens) - n + 1))]

def find_spans(text: str, phrase: str, limit: int = 5) -> List[Span]:
    if not text or not phrase:
        return []
    out: List[Span] = []
    start = 0
    taken = 0
    while taken < limit:
        idx = text.find(phrase, start)
        if idx < 0:
            break
        out.append(Span(start=idx, end=idx + len(phrase)))
        start = idx + len(phrase)
        taken += 1
    return out

def _flatten_text(obj) -> str:
    if isinstance(obj, dict):
        vals = []
        if "num" in obj and "text" in obj:
            vals.append(f'{obj.get("num")}: {obj.get("text")}')
        else:
            for v in obj.values():
                vals.append(_flatten_text(v))
        return " ".join(v for v in vals if v)
    if isinstance(obj, list):
        return " ".join(_flatten_text(v) for v in obj)
    if isinstance(obj, str):
        return obj
    return ""

def _gather_sections(desc) -> Iterable[Tuple[str, str | None, str]]:
    if not desc:
        return []
    for item in getattr(desc, "technical_field", []):
        yield "technical-field", item.num, item.text
    for item in getattr(desc, "background_art", []):
        yield "background-art", item.num, item.text
    for item in getattr(desc, "description_of_drawings", []):
        yield "description-of-drawings", item.num, item.text
    for sec_name in ["summary_of_invention", "description_of_embodiments", "advantageous_effects"]:
        for block in getattr(desc, sec_name, []):
            text = _flatten_text(block)
            yield sec_name.replace("_", "-"), None, text  # type: ignore

def _all_desc_text(doc: PatentDoc) -> str:
    if not doc.description:
        return ""
    parts = []
    for _, _, t in _gather_sections(doc.description):
        parts.append(t)
    return " ".join(parts)

def _dedup_preserve(seq: Iterable[str]) -> Iterable[str]:
    seen = set()
    for s in seq:
        if s in seen:
            continue
        seen.add(s)
        yield s

def unigram_matcher(src: PatentDoc, cand: PatentDoc) -> List[EvidenceHit]:
    src_tokens = tokenize(src.claim1 + " " + _all_desc_text(src))
    hits: List[EvidenceHit] = []
    for section, num, text in _gather_sections(cand.description):
        if not text:
            continue
        ctoks = tokenize(text)
        common = [t for t in ctoks if t in src_tokens]
        for t in itertools.islice(_dedup_preserve(common), 5):
            spans = find_spans(text, t, limit=3)
            if not spans:
                continue
            hits.append(EvidenceHit(
                section=section, num=num, text=text, spans=spans, support=1, score=min(1.0, 0.5 + len(spans)*0.1)
            ))
    return hits

def bigram_matcher(src: PatentDoc, cand: PatentDoc) -> List[EvidenceHit]:
    src_tokens = tokenize(src.claim1 + " " + _all_desc_text(src))
    src_bi = set(ngrams(src_tokens, 2))
    hits: List[EvidenceHit] = []
    for section, num, text in _gather_sections(cand.description):
        ctoks = tokenize(text)
        for bi in ngrams(ctoks, 2):
            phrase = " ".join(bi)
            if bi in src_bi and len(phrase) >= 4:
                spans = find_spans(text, phrase, limit=2)
                if spans:
                    hits.append(EvidenceHit(section=section, num=num, text=text, spans=spans, support=1, score=0.7))
    return hits

def char_shingle_matcher(src: PatentDoc, cand: PatentDoc, k: int = 8) -> List[EvidenceHit]:
    s = (src.claim1 or "") + " " + _all_desc_text(src)
    src_shingles = set(s[i:i+k] for i in range(0, max(0, len(s)-k+1)))
    hits: List[EvidenceHit] = []
    for section, num, text in _gather_sections(cand.description):
        shingles = [text[i:i+k] for i in range(0, max(0, len(text)-k+1))]
        matched = set(p for p in shingles if p in src_shingles)
        for phrase in itertools.islice(matched, 5):
            spans = find_spans(text, phrase, limit=1)
            if spans:
                hits.append(EvidenceHit(section=section, num=num, text=text, spans=spans, support=1, score=0.6))
    return hits

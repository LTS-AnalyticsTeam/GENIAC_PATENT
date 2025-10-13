from __future__ import annotations

import re
from typing import List, Sequence

from .models import AlphaInfo


def parse_alpha_text(raw_text: str) -> AlphaInfo:
    text = raw_text or ""
    normalized = _normalize(text)

    title = _extract_single_line(normalized, ["Title", "タイトル", "Subject"])
    pub_number = _extract_single_line(normalized, ["PubNumber", "Publication", "公報番号"])

    claims = _extract_claims(normalized)
    if not claims:
        claims = _fallback_blocks(normalized)

    if not claims:
        claims = ["デモ用テキストに請求項が見つかりませんでした。"]

    claim1 = claims[0]
    claims_rest = list(claims[1:]) if len(claims) > 1 else []

    return AlphaInfo(
        title=title or "特許α",
        pub_number=pub_number or "UNKNOWN",
        claim1=claim1,
        claims_rest=claims_rest,
    )


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    return text.strip()


def _extract_single_line(text: str, keys: Sequence[str]) -> str:
    for key in keys:
        pattern = rf"^{re.escape(key)}\s*[:：]\s*(.+)$"
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
    return ""


def _extract_claims(text: str) -> List[str]:
    patterns = [
        r"(?:請求項|Claim)\s*(\d+)\s*[:：]?\s*(.+?)(?=(?:請求項|Claim)\s*\d+[:：]?|\Z)",
    ]
    for pattern in patterns:
        matches = re.finditer(pattern, text, flags=re.S | re.I)
        values = [_clean_claim(match.group(2)) for match in matches]
        values = [value for value in values if value]
        if values:
            return values
    return []


def _clean_claim(value: str) -> str:
    value = re.sub(r"\n{2,}", "\n", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _fallback_blocks(text: str) -> List[str]:
    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    if not blocks:
        return []
    claim1 = blocks[0]
    claims_rest = blocks[1:4]
    claims = [re.sub(r"\s+", " ", claim1)]
    claims.extend(re.sub(r"\s+", " ", block) for block in claims_rest)
    return claims

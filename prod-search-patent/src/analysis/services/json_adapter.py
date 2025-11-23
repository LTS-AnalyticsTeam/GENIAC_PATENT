from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import Description, DescriptionItem, PatentDoc
from ..alpha_parser import build_patent_doc_from_inputs


def from_patent_json_v1(obj: Dict[str, Any]) -> PatentDoc:
    """
    質問で提示された特許JSONを PatentDoc に変換する。
    - title: bibliographic.title
    - pub_number: bibliographic.publication.doc_number
    - claim1: claims[0].text（なければ abstract を代用）
    - description: "advantageous-effects" までをそのまま渡す
    """
    biblio = obj.get("bibliographic", {}) or {}
    publication = biblio.get("publication", {}) or {}
    title = biblio.get("title")
    pub_number = publication.get("doc_number")

    claims = obj.get("claims") or []
    claim1 = ""
    if isinstance(claims, list) and len(claims) > 0:
        claim1 = (claims[0].get("text") or "").strip()
    if not claim1:
        claim1 = (obj.get("abstract") or "").strip()

    description_json = obj.get("description") or None
    # build_patent_doc_from_inputs が alias を解決して Description に詰めてくれる
    doc = build_patent_doc_from_inputs(
        title=title,
        pub_number=pub_number,
        claim1=claim1,
        description_json=description_json,
    )
    return doc

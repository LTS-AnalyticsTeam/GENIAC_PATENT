from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from .exceptions import IngestionError


def _clean(text: Optional[str]) -> str:
    return (text or "").strip()


def _iter_local(root: ET.Element, local_name: str):
    for elem in root.iter():
        if elem.tag.split("}")[-1] == local_name:
            yield elem


def _find_text(root: ET.Element, local_name: str) -> str:
    for elem in _iter_local(root, local_name):
        text = "".join(elem.itertext()).strip()
        if text:
            return text
    return ""


def _extract_ipc_codes(root: ET.Element) -> List[str]:
    codes: List[str] = []
    classification_nodes = [
        "PatentClassification",
        "MainClassification",
        "classification-ipc",
        "main-clsf",
        "classification-national",
    ]

    raw_values: List[str] = []
    for name in classification_nodes:
        for node in _iter_local(root, name):
            text = "".join(node.itertext()).strip()
            if text:
                raw_values.append(text)

    ipc_pattern = re.compile(r"([A-H][0-9]{2}[A-Z])\s*([0-9]{1,4})\s*/\s*([0-9]{1,4})")

    for value in raw_values:
        sanitized = value.replace("\xa0", " ").replace("　", " ")
        collapsed = " ".join(sanitized.split())
        matches = ipc_pattern.findall(collapsed)
        if not matches:
            no_spaces = collapsed.replace(" ", "")
            matches = ipc_pattern.findall(no_spaces)
        for section, main_group, sub_group in matches:
            normalized = f"{section}{main_group}/{sub_group}".replace(" ", "")
            codes.append(normalized)

    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for code in codes:
        if code not in seen:
            uniq.append(code)
            seen.add(code)
    return uniq


def _extract_claims(root: ET.Element) -> Dict[str, str]:
    claim_list: List[Dict[str, str]] = []
    for claim in _iter_local(root, "Claim"):
        num = claim.attrib.get("num") or claim.attrib.get("sequenceNumber") or claim.attrib.get("com:sequenceNumber") or ""
        text = "".join(claim.itertext()).strip()
        if text:
            claim_list.append({"num": num, "text": text})

    claim1_text = ""
    if claim_list:
        claim1_text = claim_list[0]["text"]
    else:
        potential = next(_iter_local(root, "Claims"), None)
        if potential is not None:
            claim1_text = "".join(potential.itertext()).strip().split("\n\n")[0]

    return {
        "claims_text": "\n\n".join(item["text"] for item in claim_list),
        "claims": claim_list,
        "claim1": claim1_text,
    }


def parse_xml_document(xml_bytes: bytes) -> Dict[str, object]:
    """Parse XML payload into normalized document fields."""
    try:
        root = ET.parse(io.BytesIO(xml_bytes)).getroot()
    except ET.ParseError as exc:
        raise IngestionError(f"Invalid XML: {exc}", status_code=400) from exc

    title = _clean(_find_text(root, "InventionTitle"))
    abstract = _clean(_find_text(root, "Abstract"))
    summary = abstract or title
    patent_id = _clean(_find_text(root, "doc-number"))

    claim_info = _extract_claims(root)
    ipc_codes = _extract_ipc_codes(root)

    if not ipc_codes:
        raise IngestionError("IPC classification required", status_code=422, detail={"reason": "missing_ipc"})

    document = {
        "patent_id": patent_id,
        "title": title,
        "summary": summary,
        "claim1": claim_info["claim1"],
        "claims_text": claim_info["claims_text"],
        "claims": claim_info["claims"],
        "classification_ipc": ipc_codes,
    }
    return document

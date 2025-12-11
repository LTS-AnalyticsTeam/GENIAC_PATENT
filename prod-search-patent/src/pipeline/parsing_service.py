from __future__ import annotations

import base64
import gzip
import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from .exceptions import IngestionError

NS = {
    "jp": "http://www.jpo.go.jp",
    "jppat": "http://www.jpo.go.jp/standards/XMLSchema/ST96/JPPatent",
    "pat": "http://www.wipo.int/standards/XMLSchema/ST96/Patent",
    "com": "http://www.wipo.int/standards/XMLSchema/ST96/Common",
    "jpcom": "http://www.jpo.go.jp/standards/XMLSchema/ST96/JPCommon",
}

DESC_KEY_OVERRIDES = {
    "TechnicalField": "technical-field",
    "BackgroundArt": "background-art",
    "Disclosure": "disclosure",
    "InventionSummary": "summary-of-invention",
    "IndustrialApplicability": "industrial-applicability",
    "EmbodimentDescription": "mode-for-invention",
    "ReferenceSignsList": "reference-signs-list",
    "CitationList": "citation-list",
    "CitationBag": "citation-list",
    "PatentCitationBag": "patent-literature",
    "NPLCitationBag": "non-patent-literature",
    "BestMode": "best-mode",
    "TechnicalProblem": "tech-problem",
    "SolutionToProblem": "tech-solution",
    "EffectOfInvention": "advantageous-effects",
    "AdvantageousEffects": "advantageous-effects",
    "ModeForInvention": "mode-for-invention",
    "EmbodimentExample": "embodiment-example",
    "ReferencePatentLiterature": "patent-literature",
    "ReferenceNonPatentLiterature": "non-patent-literature",
}

WS_RE = re.compile(r"[ \t\u3000\r\n]+")


def norm_text(value: str) -> str:
    return WS_RE.sub(" ", (value or "")).strip()


CAMEL_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
CAMEL_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


def camel_to_kebab(name: str) -> str:
    tmp = CAMEL_RE_1.sub(r"\1-\2", name)
    tmp = CAMEL_RE_2.sub(r"\1-\2", tmp)
    return tmp.replace("_", "-").lower()


def normalize_desc_key(name: str) -> str:
    if not name:
        return name
    if name in DESC_KEY_OVERRIDES:
        return DESC_KEY_OVERRIDES[name]
    if "-" in name or "_" in name or name.islower():
        return name
    return camel_to_kebab(name)


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def text_of(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return norm_text("".join(elem.itertext()))


def find_first(elem: ET.Element, paths) -> Optional[ET.Element]:
    for path in paths:
        found = elem.find(path, NS)
        if found is not None:
            return found
    return None


def elem_to_section_obj(elem: ET.Element) -> Any:
    paragraphs = []
    p_elements = elem.findall(".//p") + elem.findall(".//pat:P", NS) + elem.findall(".//com:P", NS)
    para_elements = elem.findall(".//pat:Paragraph", NS) + elem.findall(".//com:Paragraph", NS)
    all_p = p_elements + para_elements
    if all_p:
        for idx, p in enumerate(all_p):
            num = p.attrib.get("num") or p.attrib.get(f"{{{NS.get('com', '')}}}pNumber", "") or str(idx + 1).zfill(4)
            text = norm_text("".join(p.itertext()))
            if text:
                paragraphs.append({"num": num.strip(), "text": text})
        if paragraphs:
            return paragraphs

    text = norm_text("".join(elem.itertext()))
    if text:
        return [{"num": "0001", "text": text}]
    return []


def convert_xml_to_json(xml_text: str) -> Dict[str, Any]:
    parser = ET.XMLParser(target=ET.TreeBuilder(), encoding="utf-8")
    root = ET.fromstring(xml_text.encode("utf-8"), parser=parser)

    # Extract publication number (patent ID)
    # ST96 format: .//pat:PublicationNumber
    # ST36 format: ./bibliographic-data/publication-reference/document-id/doc-number
    patent_id = ""
    pub_number_node = find_first(root, [
        ".//pat:PublicationNumber",
        "./bibliographic-data/publication-reference/document-id/doc-number"
    ])
    if pub_number_node is not None:
        patent_id = text_of(pub_number_node)

    # ここでは省略版として最も重要なフィールドのみ抽出（詳細は元スクリプト同様に拡張可能）
    biblio_title = text_of(find_first(root, ["./bibliographic-data/invention-title", ".//pat:InventionTitle"]))
    claim_nodes = find_first(root, ["./claims", ".//claims"])
    claims: List[Dict[str, str]] = []
    if claim_nodes is not None:
        for claim in claim_nodes.findall(".//claim"):
            text = norm_text("".join(claim.itertext()))
            if text:
                claims.append({"num": claim.attrib.get("num", ""), "text": text})

    # IPC classification extraction - support both ST36 and ST96 formats
    ipc_codes = []

    # ST36 format: .//classification-ipc//main-clsf
    classification_nodes = root.findall(".//classification-ipc//main-clsf") or []
    for node in classification_nodes:
        code = norm_text("".join(node.itertext()))
        if code:
            ipc_codes.append(code)

    # ST96 format: .//jppat:SearchField/pat:SearchFieldText
    if not ipc_codes:
        search_field_texts = root.findall(".//jppat:SearchField/pat:SearchFieldText", NS) or []
        for node in search_field_texts:
            code = norm_text("".join(node.itertext()))
            # ST96 format uses full-width characters, normalize to half-width
            if code:
                # Convert full-width alphanumeric and symbols to half-width
                code = code.translate(str.maketrans(
                    'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９／',
                    'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/'
                ))
                # Remove extra spaces
                code = re.sub(r'\s+', '', code)
                if code:
                    ipc_codes.append(code)

    # ST96 format alternative: .//jppat:IPCClassification or .//pat:PatentClassification
    if not ipc_codes:
        ipc_nodes = root.findall(".//jppat:IPCClassification", NS) or root.findall(".//pat:PatentClassification", NS) or []
        for node in ipc_nodes:
            # Try to extract from MainClassification or similar elements
            main_class = node.find(".//pat:MainClassification", NS)
            if main_class is not None:
                code = norm_text("".join(main_class.itertext()))
                if code:
                    ipc_codes.append(code)

    return {
        "patent_id": patent_id,
        "bibliographic": {
            "title": biblio_title,
            "publication": {"doc_number": patent_id} if patent_id else {},
            "classification": {"ipc": [{"type": "main", "text": code} for code in ipc_codes]},
        },
        "claims": claims,
    }


def parse_text_document(text_bytes: bytes) -> Dict[str, Any]:
    """Parse uploaded text (XML) and return normalized structure."""
    try:
        xml_text = text_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"Invalid text encoding: {exc}", status_code=400) from exc

    document = convert_xml_to_json(xml_text)
    biblio = document.get("bibliographic") or {}
    title = biblio.get("title") or "UNKNOWN"

    claims = document.get("claims") or []
    claim1 = claims[0]["text"] if claims else ""
    claims_text = "\n\n".join(claim.get("text", "") for claim in claims)

    classification = biblio.get("classification") or {}
    ipc_entries = classification.get("ipc") or []
    classification_ipc = [entry.get("text") for entry in ipc_entries if entry.get("text")]

    if not classification_ipc:
        raise IngestionError("IPC classification required", status_code=422, detail={"reason": "missing_ipc"})

    return {
        "patent_id": document.get("patent_id") or "",
        "title": title,
        "summary": document.get("abstract") or title,
        "claim1": claim1,
        "claims_text": claims_text,
        "claims": claims,
        "classification_ipc": classification_ipc,
        "source_json": document,
    }


def load_json_document(json_bytes: bytes) -> Dict[str, Any]:
    try:
        document = json.loads(json_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IngestionError(f"Invalid JSON payload: {exc}", status_code=400) from exc

    biblio = document.get("bibliographic") or {}
    publication = biblio.get("publication") or {}

    patent_id = (
        publication.get("doc_number")
        or document.get("patent_id")
        or document.get("source_file")
        or ""
    )
    title = biblio.get("title") or document.get("title") or patent_id or "UNKNOWN"
    summary = document.get("summary") or document.get("abstract") or title

    claims = document.get("claims") or []
    claim1 = claims[0]["text"] if claims else document.get("claim1", "")
    claims_text = "\n\n".join(claim.get("text", "") for claim in claims if claim.get("text"))

    classification = biblio.get("classification") or {}
    ipc_entries = classification.get("ipc") or []
    classification_ipc = [entry.get("text") for entry in ipc_entries if entry.get("text")]

    if not classification_ipc:
        raise IngestionError("IPC classification required", status_code=422, detail={"reason": "missing_ipc"})

    return {
        "patent_id": patent_id,
        "title": title,
        "summary": summary,
        "claim1": claim1,
        "claims_text": claims_text,
        "claims": claims,
        "classification_ipc": classification_ipc,
        "source_json": document,
    }

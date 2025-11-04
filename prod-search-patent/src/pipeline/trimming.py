from __future__ import annotations

from datetime import datetime
from typing import Dict, List


def _parse_publication_date(doc: Dict) -> datetime:
    date_value = (
        doc.get("bibliographic", {})
        .get("publication", {})
        .get("date")
    )
    if date_value:
        try:
            return datetime.strptime(date_value, "%Y%m%d")
        except ValueError:
            pass
    timestamp = doc.get("_ts")
    if timestamp:
        try:
            return datetime.fromtimestamp(float(timestamp))
        except (ValueError, TypeError):
            pass
    return datetime.min


def sort_and_trim(documents: List[Dict], limit: int) -> List[Dict]:
    """Sort documents by publication date desc (fallback _ts) and trim to limit."""
    sorted_docs = sorted(
        documents,
        key=_parse_publication_date,
        reverse=True,
    )
    return sorted_docs[:limit]


"""
Utility functions for extracting summary and claims text for embedding generation.
"""
from __future__ import annotations

from typing import Any, Dict


def _to_clean_text(value: Any) -> str:
    """Convert value to stripped string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_to_clean_text(v) for v in value if v)
    return str(value).strip()


def extract_summary_text(document: Dict[str, Any]) -> str:
    """
    Extract summary text suitable for embedding.

    Prefers the `summary` field, falling back to `title` if absent.
    """
    summary_text = _to_clean_text(document.get("summary"))
    if summary_text:
        return summary_text

    # Fallback to title if summary is missing.
    title = _to_clean_text(document.get("title"))
    if title:
        return title

    # As a last resort, fall back to claims text.
    claims_text = _to_clean_text(document.get("claims_text"))
    if claims_text:
        return claims_text

    return _to_clean_text(document.get("patent_id"))


def extract_claims1_text(document: Dict[str, Any]) -> str:
    """
    Extract the first claim text suitable for embedding.

    Prefers claims[0].text, with graceful fallbacks.
    """
    claims = document.get("claims")
    first_claim = ""

    if isinstance(claims, list) and claims:
        first_entry = claims[0]
        if isinstance(first_entry, dict):
            first_claim = _to_clean_text(first_entry.get("text"))
        else:
            first_claim = _to_clean_text(first_entry)

    elif isinstance(claims, dict):
        first_claim = _to_clean_text(claims.get("text"))

    if first_claim:
        return first_claim

    # Fallback: derive from claims_text by splitting at sentence boundaries if available.
    claims_text = _to_clean_text(document.get("claims_text"))
    if claims_text:
        # Use the first paragraph/sentence as a surrogate for claim 1.
        for separator in ("\n\n", "\n", "。"):
            if separator in claims_text:
                candidate = claims_text.split(separator)[0].strip()
                if candidate:
                    return candidate
        return claims_text

    # Final fallback: reuse summary or title to avoid empty strings.
    summary = extract_summary_text(document)
    if summary:
        return summary

    return _to_clean_text(document.get("patent_id"))


__all__ = [
    "extract_summary_text",
    "extract_claims1_text",
]

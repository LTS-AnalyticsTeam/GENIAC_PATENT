from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .models import (
    AlphaInfo,
    AnalysisResponse,
    Candidate,
    Explanation,
    PatentDoc,
    RunLimits,
    Snippet,
)
from .services.json_adapter import from_patent_json_v1
from .services.matcher_heuristic import char_shingle_matcher, bigram_matcher, unigram_matcher
from .services.ensemble import ensemble_hits
from .stub_search import search_prior_art


@dataclass(slots=True)
class PatentWorkItem:
    patent_id: str
    title: Optional[str]
    source_json: Dict[str, Any]
    candidate_json: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class BatchAnalysisResult:
    patent_id: str
    title: str
    pub_number: str
    analysis: Optional[AnalysisResponse]
    status: str
    error_message: Optional[str]
    processed_at: str


class AnalysisService:
    """FastAPIに依存しない特許分析ロジック。"""

    async def analyze_single(
        self,
        *,
        patent_id: str,
        source_json: Dict[str, Any],
        title: Optional[str] = None,
        candidate_json: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResponse:
        source_doc = from_patent_json_v1(source_json)
        claims_2_to_5 = self._extract_claims_2_to_5(source_json)
        alpha = self._build_alpha_info(patent_id, source_doc, claims_2_to_5, title)

        ax_candidate: Optional[Candidate] = None
        if candidate_json:
            cand_doc = from_patent_json_v1(candidate_json)
            ax_candidate = self._build_candidate_from_doc(
                cand_doc,
                candidate_json,
                source_doc,
                alpha.claim1,
            )

        if not ax_candidate:
            ax_candidate, stub_ay = search_prior_art(
                claim1=alpha.claim1,
                claims_rest=alpha.claims_rest,
                source_patent=source_doc,
            )
        ay_candidates: List[Candidate] = []
        limits = RunLimits(max_total=1 + len(ay_candidates), Ay_min=len(ay_candidates))
        return AnalysisResponse(
            run_id=f"analysis-{patent_id}",
            alpha=alpha,
            Ax=ax_candidate,
            Ay=ay_candidates,
            limits=limits,
        )

    async def analyze_batch(
        self,
        items: Sequence[PatentWorkItem],
        *,
        shared_candidate_json: Optional[Dict[str, Any]] = None,
    ) -> List[BatchAnalysisResult]:
        results: List[BatchAnalysisResult] = []
        for item in items:
            candidate_json = item.candidate_json or shared_candidate_json
            try:
                analysis = await self.analyze_single(
                    patent_id=item.patent_id,
                    title=item.title,
                    source_json=item.source_json,
                    candidate_json=candidate_json,
                )
                results.append(
                    BatchAnalysisResult(
                        patent_id=item.patent_id,
                        title=analysis.alpha.title,
                        pub_number=analysis.alpha.pub_number,
                        analysis=analysis,
                        status="completed",
                        error_message=None,
                        processed_at=datetime.utcnow().isoformat(),
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                results.append(
                    BatchAnalysisResult(
                        patent_id=item.patent_id,
                        title=item.title or f"特許{item.patent_id}",
                        pub_number="ERROR",
                        analysis=None,
                        status="failed",
                        error_message=str(exc),
                        processed_at=datetime.utcnow().isoformat(),
                    )
                )
        return results

    @staticmethod
    def _extract_claims_2_to_5(source_json: Dict[str, Any]) -> List[Dict[str, str]]:
        claims = source_json.get("claims", []) or []
        claims_2_to_5: List[Dict[str, str]] = []
        for idx, claim in enumerate(claims[1:5], start=2):
            if not claim:
                continue
            claims_2_to_5.append(
                {
                    "num": idx,
                    "text": claim.get("text", ""),
                }
            )
        return claims_2_to_5

    @staticmethod
    def _build_alpha_info(
        patent_id: str,
        source_doc: PatentDoc,
        claims_2_to_5: Sequence[Dict[str, str]],
        title_override: Optional[str],
    ) -> AlphaInfo:
        claims_rest = [f"請求項{c['num']}: {c['text']}" for c in claims_2_to_5]
        return AlphaInfo(
            title=title_override or source_doc.title or f"特許{patent_id}",
            pub_number=source_doc.pub_number or "UNKNOWN",
            claim1=source_doc.claim1 or "",
            claims_rest=claims_rest,
        )

    def _build_candidate_from_doc(
        self,
        candidate_doc: PatentDoc,
        candidate_json: Dict[str, Any],
        source_doc: PatentDoc,
        claim1_snippet: str,
    ) -> Candidate:
        ipc_codes = self._extract_ipc_codes(candidate_json)
        doc_id = candidate_json.get("patent_id") or candidate_doc.pub_number or candidate_doc.title or "candidate"
        explanation = Explanation(
            summary="",
            why_match=[],
            examiner_hints=[],
        )
        comparison = self._compare_claims(source_doc.claim1, candidate_doc.claim1)
        explanation.summary = comparison["summary"]
        explanation.why_match = comparison["details"]
        candidate = Candidate(
            doc_id=str(doc_id),
            title=candidate_doc.title or "主引例（請求項1の新規性）",
            pub_number=candidate_doc.pub_number or "UNKNOWN",
            year=2019,
            ipc=ipc_codes or ["UNKNOWN"],
            score=0.75,
            snippets=[],
            explanation=explanation,
            source_url=None,
        )
        self._attach_evidence(candidate, source_doc, candidate_doc)
        return candidate

    @staticmethod
    def _build_ay_candidates(claims_2_to_5: Sequence[Dict[str, str]]) -> List[Candidate]:
        ay_candidates: List[Candidate] = []
        for idx, claim_info in enumerate(claims_2_to_5[:2], start=1):
            if not claim_info:
                continue
            explanation = Explanation(
                summary="副引例による進歩性判定",
                why_match=[
                    f"【請求項{claim_info['num']}の進歩性判定】\n"
                    "主引例と公知技術の組み合わせにより容易想到",
                    f"《請求項内容》\n{claim_info['text'][:200]}...",
                ],
                examiner_hints=[],
            )
            ay_candidates.append(
                Candidate(
                    doc_id=f"ay-claim{claim_info['num']}",
                    title=f"請求項{claim_info['num']}の進歩性判定",
                    pub_number=f"副引例{idx}",
                    year=2018,
                    ipc=["H04L 29/08"],
                    score=0.65,
                    snippets=[],
                    explanation=explanation,
                    source_url=None,
                )
            )
        return ay_candidates

    @staticmethod
    def _extract_ipc_codes(candidate_json: Dict[str, Any]) -> List[str]:
        ipc_codes: List[str] = []
        biblio = candidate_json.get("bibliographic") or {}
        classification = biblio.get("classification") or {}
        ipc_entries = classification.get("ipc") or []
        for entry in ipc_entries:
            if isinstance(entry, dict):
                text = entry.get("text") or entry.get("value")
            else:
                text = entry
            if text:
                text = str(text).strip()
                if text and text not in ipc_codes:
                    ipc_codes.append(text)
        top_level = candidate_json.get("classification_ipc")
        if isinstance(top_level, list):
            for text in top_level:
                if not text:
                    continue
                text = str(text).strip()
                if text and text not in ipc_codes:
                    ipc_codes.append(text)
        return ipc_codes

    def _attach_evidence(self, candidate: Candidate, source_doc: PatentDoc, candidate_doc: PatentDoc) -> None:
        if not candidate_doc.description or not source_doc.description:
            return
        hits1 = unigram_matcher(source_doc, candidate_doc)
        hits2 = bigram_matcher(source_doc, candidate_doc)
        hits3 = char_shingle_matcher(source_doc, candidate_doc)
        hits, judgment = ensemble_hits(source_doc, candidate_doc, [hits1, hits2, hits3])
        if hits:
            candidate.evidence_hits = hits
            candidate.judgment_basis = judgment
            snippets: List[Snippet] = []
            for idx, hit in enumerate(hits[:3]):
                first_span = hit.spans[0] if hit.spans else None
                snippets.append(
                    Snippet(
                        section=hit.section,
                        claim_no=1,
                        text=hit.text,
                        offset=first_span.start if first_span else 0,
                        len=(first_span.end - first_span.start) if first_span else min(len(hit.text), 80),
                        match_type="match",
                        score=round(min(0.95, 0.6 + 0.1 * hit.support - idx * 0.05), 2),
                    )
                )
            candidate.snippets = snippets

    @staticmethod
    def _compare_claims(alpha_claim1: str, candidate_claim1: str) -> Dict[str, List[str] | str]:
        summary = "請求項1の対比結果: "
        if not candidate_claim1:
            return {
                "summary": summary + "候補請求項のテキストが存在しません。",
                "details": ["候補文書に請求項1が無いため新規性比較ができません。"],
            }
        details: List[str] = []
        alpha_tokens = [t.strip() for t in alpha_claim1.split("、") if t.strip()]
        candidate_tokens = [t.strip() for t in candidate_claim1.split("、") if t.strip()]
        matches = [token for token in alpha_tokens if token and token in candidate_tokens]
        if matches:
            summary += "主要構成が候補請求項と一致しています。"
            for token in matches:
                details.append(f"構成要素「{token}」が候補請求項にも記載されており、新規性が否定される可能性があります。")
        else:
            summary += "一致する構成要素が見つかりませんでした。"
            details.append("候補文書に請求項1の主要構成要素が見つからず、新規性は維持される可能性があります。")
        return {"summary": summary, "details": details}

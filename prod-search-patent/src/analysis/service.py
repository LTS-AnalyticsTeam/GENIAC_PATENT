from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import (
    AlphaInfo,
    AnalysisResponse,
    AssessmentCandidate,
    ClaimAssessment,
    Evidence,
    PatentDoc,
    RunLimits,
)
from .services.aoai import AOAIClient
from pipeline.config import PipelineConfig
from .services.json_adapter import from_patent_json_v1


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
    """FastAPIに依存しない特許分析ロジック。LLMで請求項ごとに新規性/進歩性を判定する。"""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.aoai_client = AOAIClient(
            endpoint=self.config.azure_openai_endpoint,
            api_key=self.config.azure_openai_key,
            api_version=self.config.azure_openai_api_version,
            calls_per_minute=30,
        )
        self._chat_deployment = self.config.azure_openai_chat_deployment or self.config.azure_openai_deployment

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

        claim1_candidates, rest_candidates = self._prepare_candidates(
            source_doc,
            claims_2_to_5,
            candidate_json,
        )
        limits = RunLimits(max_total=len(claim1_candidates) + len(rest_candidates), Ay_min=len(rest_candidates))
        return AnalysisResponse(
            run_id=f"analysis-{patent_id}",
            alpha=alpha,
            claim1_candidates=claim1_candidates,
            rest_claim_candidates=rest_candidates,
            limits=limits,
        )

    async def analyze_candidates(
        self,
        *,
        alpha_json: Dict[str, Any],
        candidates_json: Sequence[Dict[str, Any]],
    ) -> AnalysisResponse:
        source_doc = from_patent_json_v1(alpha_json)
        claims_2_to_5 = self._extract_claims_2_to_5(alpha_json)
        alpha = self._build_alpha_info(alpha_json.get("patent_id", "alpha"), source_doc, claims_2_to_5, alpha_json.get("title"))

        claim1_candidates: List[AssessmentCandidate] = []
        rest_candidates: List[AssessmentCandidate] = []

        first_batch = list(candidates_json[:5])
        second_batch = list(candidates_json[5:10])

        for candidate_json in first_batch:
            cand_doc = from_patent_json_v1(candidate_json)
            assessment = await self._llm_assess_candidate(
                alpha=alpha,
                candidate_json=candidate_json,
                candidate_doc=cand_doc,
                target_claims=[(1, source_doc.claim1, False)],
            )
            if assessment:
                claim1_candidates.append(assessment)

        if claims_2_to_5:
            target_claims = [(c.get("num") or 0, c.get("text", ""), True) for c in claims_2_to_5 if c]
            for candidate_json in second_batch:
                cand_doc = from_patent_json_v1(candidate_json)
                assessment = await self._llm_assess_candidate(
                    alpha=alpha,
                    candidate_json=candidate_json,
                    candidate_doc=cand_doc,
                    target_claims=target_claims,
                )
                if assessment:
                    rest_candidates.append(assessment)

        limits = RunLimits(max_total=len(claim1_candidates) + len(rest_candidates), Ay_min=len(rest_candidates))
        return AnalysisResponse(
            run_id=f"analysis-{alpha.pub_number}",
            alpha=alpha,
            claim1_candidates=claim1_candidates[:5],
            rest_claim_candidates=rest_candidates[:5],
            limits=limits,
        )

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

    async def _llm_assess_candidate(
        self,
        *,
        alpha: AlphaInfo,
        candidate_json: Dict[str, Any],
        candidate_doc: PatentDoc,
        target_claims: Sequence[Tuple[int, str, bool]],
    ) -> Optional[AssessmentCandidate]:
        if not self._chat_deployment:
            return None

        doc_id = candidate_json.get("patent_id") or candidate_doc.pub_number or candidate_doc.title or "candidate"
        ipc_codes = self._extract_ipc_codes(candidate_json)
        prompt = self._build_prompt(alpha, candidate_json, target_claims)
        messages = [
            {
                "role": "system",
                "content": (
                    "あなたは特許審査の専門家（審査官）として行動します。"
                    "入力として与えられる特許α（対象特許）と候補先行技術を比較し、新規性・進歩性を否定できる根拠を最大限抽出し、JSONで返してください。"
                    "必ず指定フォーマットに従い、引用箇所は短く抜粋してください。"
                    "各エビデンスは『α請求項の該当部分（5〜30文字程度の短い片）』と『候補特許の該当部分（同程度の短い片）』のペアで必ず示してください。"
                    "【デフォルト姿勢】常に新規性・進歩性を否定できる根拠を最優先で探索する。以下の場合のみ例外的に supported / uncertain を使う。"
                    " supported: αの構成が候補に明確に欠落／矛盾し、否定できないと断定できる場合のみ。"
                    " uncertain: 情報不足で存在/欠落が判断不能の場合のみ。差異があるだけで supported にしてはならない。実質同等なら denied とする。"
                    "【判定基準】"
                    " 新規性: αの必須構成が候補に明示または実質同等なら denied。α要件が明確に欠落する場合のみ supported。情報不足は uncertain。"
                    " 進歩性: 候補単独＋公知技術で容易想到なら denied。非自明な差異を具体的に示せる場合のみ supported。不足は uncertain。"
                    "【エビデンス記載】各請求項ごとに必ず1ペア以上の証拠を返す。denied時は否定根拠を必ず含める。supportedの場合は『なぜ否定できないか』をwhyに明示。"
                    "whyには技術的理由を簡潔に記述し、不要な前置きは省く。"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        raw = await self.aoai_client.chat(
            deployment=self._chat_deployment,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_retries=5,
        )
        parsed = self._parse_llm_response(raw, target_claims)
        assessments: List[ClaimAssessment] = []
        for item in parsed:
            assessments.append(
                ClaimAssessment(
                    claim_no=item.get("claim_no", 0),
                    novelty=item.get("novelty", "uncertain"),
                    inventive_step=item.get("inventive_step"),
                    evidence=[
                        Evidence(
                            section=e.get("section", "unknown"),
                            quote=e.get("candidate_quote") or e.get("quote", ""),
                            why=e.get("why", ""),
                            offset=e.get("offset"),
                            length=e.get("length"),
                            alpha_fragment=e.get("alpha_fragment"),
                            candidate_quote=e.get("candidate_quote") or e.get("quote"),
                        )
                        for e in item.get("evidence", []) if e
                    ],
                    examiner_hints=item.get("examiner_hints", []) or [],
                )
            )

        return AssessmentCandidate(
            doc_id=str(doc_id),
            title=candidate_doc.title or "先行例",
            pub_number=candidate_doc.pub_number or "UNKNOWN",
            score=0.75,
            assessments=assessments,
            ipc=ipc_codes or ["UNKNOWN"],
            summary=candidate_json.get("summary") or candidate_json.get("abstract"),
            source_url=None,
        )

    def _build_prompt(
        self,
        alpha: AlphaInfo,
        candidate_json: Dict[str, Any],
        target_claims: Sequence[Tuple[int, str, bool]],
    ) -> str:
        summary = candidate_json.get("summary") or candidate_json.get("abstract") or ""
        claim1 = (candidate_json.get("claim1") or candidate_json.get("claims", [{}])[0].get("text", "")) if candidate_json.get("claims") else candidate_json.get("claim1", "")
        claims_texts: List[str] = []
        claims = candidate_json.get("claims") or []
        for cl in claims[:5]:
            if not cl:
                continue
            num = cl.get("num") or cl.get("claim_no") or ""
            text = cl.get("text") or ""
            if num:
                claims_texts.append(f"請求項{num}: {text}")
        target_claim_lines = []
        for num, text, include_inventive in target_claims:
            target_claim_lines.append(f"- 請求項{num} ({'進歩性も判定' if include_inventive else '新規性のみ'}): {text}")

        prompt_parts = [
            "【対象特許（α）】",
            f"タイトル: {alpha.title}",
            f"請求項1: {alpha.claim1}",
            f"請求項2以降: {' / '.join(alpha.claims_rest) if alpha.claims_rest else 'なし'}",
            "",
            "【比較候補】",
            f"タイトル: {candidate_json.get('title') or candidate_json.get('bibliographic', {}).get('title') or '不明'}",
            f"公報番号: {candidate_json.get('patent_id') or candidate_json.get('id') or candidate_json.get('pub_number') or '不明'}",
            f"要約: {summary}",
            f"請求項1(候補): {claim1}",
            f"主要な請求項抜粋: {' / '.join(claims_texts[:3]) if claims_texts else 'なし'}",
            "",
            "【判定対象の請求項（α）】",
            "\n".join(target_claim_lines),
            "",
            "以下のJSON形式で返してください。不要な文章は書かないこと。",
            "{",
            '  "assessments": [',
            '    {',
            '      "claim_no": <数値>,',
            '      "novelty": "denied|uncertain|supported",',
            '      "inventive_step": "denied|uncertain|supported|null",',
            '      "evidence": [',
            '        {',
            '          "alpha_fragment": "α請求項の該当部分",',
            '          "candidate_quote": "候補特許の該当部分",',
            '          "why": "なぜ新規性/進歩性が否定されるか",',
            '          "section": "claims|description|abstract|other",',
            '          "offset": null,',
            '          "length": null',
            '        }',
            '      ],',
            '      "examiner_hints": ["審査官への示唆"]',
            "    }",
            "  ]",
            "}",
            "根拠は短い引用で示し、必ず日本語で回答してください。",
        ]
        return "\n".join(prompt_parts)

    def _parse_llm_response(self, raw: Dict[str, Any], target_claims: Sequence[Tuple[int, str, bool]]) -> List[Dict[str, Any]]:
        content = None
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if choices and isinstance(choices, list):
            message = choices[0].get("message") if choices else None
            if message:
                content = message.get("content")
        if not content and isinstance(raw, dict):
            content = raw.get("content")
        if not content:
            return self._empty_assessments(target_claims)

        try:
            parsed = json.loads(content)
        except Exception:
            return self._empty_assessments(target_claims)

        assessments = parsed.get("assessments") if isinstance(parsed, dict) else None
        if not isinstance(assessments, list):
            return self._empty_assessments(target_claims)

        cleaned: List[Dict[str, Any]] = []
        for item in assessments:
            if not isinstance(item, dict):
                continue
            claim_no = item.get("claim_no")
            if claim_no is None:
                continue
            novelty = item.get("novelty", "uncertain")
            inventive = item.get("inventive_step")
            if inventive == "null":
                inventive = None
            evidence_raw = item.get("evidence") if isinstance(item.get("evidence"), list) else []
            evidence: List[Dict[str, Any]] = []
            for ev in evidence_raw:
                if not isinstance(ev, dict):
                    continue
                evidence.append(
                    {
                        "alpha_fragment": ev.get("alpha_fragment"),
                        "candidate_quote": ev.get("candidate_quote") or ev.get("quote"),
                        "why": ev.get("why"),
                        "section": ev.get("section"),
                        "offset": ev.get("offset"),
                        "length": ev.get("length"),
                        "quote": ev.get("candidate_quote") or ev.get("quote"),
                    }
                )
            hints = item.get("examiner_hints") if isinstance(item.get("examiner_hints"), list) else []
            cleaned.append(
                {
                    "claim_no": claim_no,
                    "novelty": novelty,
                    "inventive_step": inventive,
                    "evidence": evidence,
                    "examiner_hints": hints,
                }
            )

        if not cleaned:
            return self._empty_assessments(target_claims)
        return cleaned

    @staticmethod
    def _empty_assessments(target_claims: Sequence[Tuple[int, str, bool]]) -> List[Dict[str, Any]]:
        payload = []
        for num, _text, include_inventive in target_claims:
            payload.append(
                {
                    "claim_no": num,
                    "novelty": "uncertain",
                    "inventive_step": "uncertain" if include_inventive else None,
                    "evidence": [],
                    "examiner_hints": [],
                }
            )
        return payload

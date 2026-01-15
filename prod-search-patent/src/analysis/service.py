from __future__ import annotations

import json
import os
import re
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
            # 50k TPM を安全に使うため、呼び出し回数を抑制
            calls_per_minute=12,
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

        def _tokenize(text: str) -> List[str]:
            if not text:
                return []
            tokens = re.findall(r"[A-Za-z0-9]+|[\u3040-\u30ff\u4e00-\u9fff]+", text)
            return [tok.lower() for tok in tokens if len(tok) > 1]

        def _extract_claim1(candidate: Dict[str, Any]) -> str:
            claim1 = candidate.get("claim1")
            if isinstance(claim1, str) and claim1:
                return claim1
            claims = candidate.get("claims")
            if isinstance(claims, list) and claims:
                first = claims[0]
                if isinstance(first, dict):
                    return str(first.get("text") or first.get("claim_text") or "")
                return str(first)
            return ""

        alpha_claim_tokens = set(_tokenize(source_doc.claim1))
        alpha_summary = alpha_json.get("summary") or alpha_json.get("abstract") or ""
        alpha_context_tokens = set(_tokenize(" ".join([alpha.title or "", str(alpha_summary)])))

        scored_candidates: List[Tuple[int, int, Dict[str, Any]]] = []
        for idx, candidate in enumerate(candidates_json):
            title = candidate.get("title") or candidate.get("bibliographic", {}).get("title") or ""
            summary = candidate.get("summary") or candidate.get("abstract") or ""
            claim1 = _extract_claim1(candidate)
            cand_claim_tokens = set(_tokenize(claim1))
            cand_context_tokens = set(_tokenize(" ".join([str(title), str(summary)])))
            claim_overlap = len(alpha_claim_tokens & cand_claim_tokens)
            context_overlap = len(alpha_context_tokens & cand_context_tokens)
            score = claim_overlap * 2 + context_overlap
            scored_candidates.append((score, idx, candidate))

        scored_candidates.sort(key=lambda item: (-item[0], item[1]))
        candidates_slice = [item[2] for item in scored_candidates[:50]]

        # なるべく否定根拠の強いものを優先しつつ、必ず各カテゴリで1件以上返す。
        def score_assessment(assess: AssessmentCandidate, consider_inventive: bool) -> float:
            nov_denied = sum(1 for a in assess.assessments if a.novelty == "denied")
            inv_denied = sum(1 for a in assess.assessments if consider_inventive and a.inventive_step == "denied")
            evidence_total = sum(len(a.evidence) for a in assess.assessments)
            return nov_denied * 2.0 + inv_denied * 1.5 + evidence_total * 0.1

        def build_placeholder_assessments(target_claims: Sequence[Tuple[int, str, bool]]) -> List[ClaimAssessment]:
            assessments: List[ClaimAssessment] = []
            for num, _text, include_inventive in target_claims:
                assessments.append(
                    ClaimAssessment(
                        claim_no=num,
                        novelty="uncertain",
                        inventive_step="uncertain" if include_inventive else None,
                        evidence=[],
                        examiner_hints=["LLM evaluation unavailable; placeholder"],
                    )
                )
            return assessments

        def build_placeholder_candidate(candidate_json: Dict[str, Any], target_claims: Sequence[Tuple[int, str, bool]]) -> AssessmentCandidate | None:
            cand_doc = from_patent_json_v1(candidate_json)
            doc_id = candidate_json.get("patent_id") or cand_doc.pub_number or cand_doc.title
            if not doc_id:
                return None
            is_web_result = candidate_json.get("is_web_result", False)
            source_url = candidate_json.get("source_url") if is_web_result else None
            pub_number = None if is_web_result else (cand_doc.pub_number or candidate_json.get("patent_id"))
            return AssessmentCandidate(
                doc_id=str(doc_id),
                title=cand_doc.title or "先行例",
                pub_number=pub_number,
                score=0.0,
                assessments=build_placeholder_assessments(target_claims),
                ipc=self._extract_ipc_codes(candidate_json) or ["UNKNOWN"],
                summary=candidate_json.get("summary") or candidate_json.get("abstract"),
                source_url=source_url,
                is_web_result=is_web_result,
            )

        claim1_pool: List[Dict[str, Any]] = []
        rest_pool: List[Dict[str, Any]] = []
        MAX_EVALS = min(50, len(candidates_slice))  # レート制限を避けるため評価上限を設定（50k TPM想定）

        for idx, candidate_json in enumerate(candidates_slice[:MAX_EVALS]):
            cand_doc = from_patent_json_v1(candidate_json)

            # 請求項1
            try:
                assess1 = await self._llm_assess_candidate(
                    alpha=alpha,
                    candidate_json=candidate_json,
                    candidate_doc=cand_doc,
                    target_claims=[(1, source_doc.claim1, False)],
                )
            except Exception as exc:  # rate limit や一時的エラーに備えてスキップ
                assess1 = None
            if assess1:
                claim1_pool.append(
                    {
                        "assessment": assess1,
                        "score": score_assessment(assess1, consider_inventive=False),
                        "rank": idx,
                        "doc_id": assess1.doc_id,
                    }
                )

            # 請求項2以降
            if claims_2_to_5:
                target_claims = [(c.get("num") or 0, c.get("text", ""), True) for c in claims_2_to_5 if c]
                try:
                    assess_rest = await self._llm_assess_candidate(
                        alpha=alpha,
                        candidate_json=candidate_json,
                        candidate_doc=cand_doc,
                        target_claims=target_claims,
                    )
                except Exception:
                    assess_rest = None
                if assess_rest:
                    rest_pool.append(
                        {
                            "assessment": assess_rest,
                            "score": score_assessment(assess_rest, consider_inventive=True),
                            "rank": idx,
                            "doc_id": assess_rest.doc_id,
                        }
                    )

            # 早期終了: 十分に候補が溜まったら打ち切り
            if len(claim1_pool) >= 8 and (not claims_2_to_5 or len(rest_pool) >= 8):
                break

        def pick_top(pool: List[Dict[str, Any]], max_items: int, require_denied: bool, used_ids: set[str]) -> List[AssessmentCandidate]:
            filtered = []
            for entry in pool:
                assess = entry["assessment"]
                has_denial = any(a.novelty == "denied" or a.inventive_step == "denied" for a in assess.assessments)
                if require_denied and not has_denial:
                    continue
                if assess.doc_id in used_ids:
                    continue
                filtered.append(entry)
            if not filtered and not require_denied:
                filtered = [e for e in pool if e["assessment"].doc_id not in used_ids]
            sorted_pool = sorted(filtered, key=lambda e: (-e["score"], e["rank"]))
            selected: List[AssessmentCandidate] = []
            for entry in sorted_pool:
                if len(selected) >= max_items:
                    break
                assess = entry["assessment"]
                used_ids.add(assess.doc_id)
                selected.append(assess)
            return selected

        used_ids: set[str] = set()
        max_ax = 5

        # 請求項1: 否定あり優先（最大5件）、なければ上位1件のみ
        claim1_selected = pick_top(claim1_pool, max_ax, require_denied=True, used_ids=used_ids)
        if not claim1_selected and claim1_pool:
            claim1_selected = pick_top(claim1_pool, 1, require_denied=False, used_ids=used_ids)

        # 請求項2以降: 否定あり優先、なければスコア上位から最低1件（請求項が存在するときのみ）
        rest_selected: List[AssessmentCandidate] = []
        if claims_2_to_5:
            rest_selected = pick_top(rest_pool, 5, require_denied=True, used_ids=used_ids)
            if not rest_selected and rest_pool:
                rest_selected = pick_top(rest_pool, 1, require_denied=False, used_ids=used_ids)

        # フォールバック: LLM評価が得られなかった場合でもAx/Ayを最低1件ずつ返す
        if not claim1_selected and candidates_slice:
            placeholder = None
            for cand_json in candidates_slice:
                candidate = build_placeholder_candidate(cand_json, [(1, source_doc.claim1, False)])
                if candidate and candidate.doc_id not in used_ids:
                    used_ids.add(candidate.doc_id)
                    placeholder = candidate
                    break
            if not placeholder:
                fallback_candidate = build_placeholder_candidate(candidates_slice[0], [(1, source_doc.claim1, False)])
                if fallback_candidate and fallback_candidate.doc_id not in used_ids:
                    used_ids.add(fallback_candidate.doc_id)
                    placeholder = fallback_candidate
            if placeholder:
                claim1_selected = [placeholder]

        if claims_2_to_5 and not rest_selected and candidates_slice:
            target_claims = [(c.get("num") or 0, c.get("text", ""), True) for c in claims_2_to_5 if c]
            placeholder = None
            for cand_json in candidates_slice:
                candidate = build_placeholder_candidate(cand_json, target_claims)
                if candidate and candidate.doc_id not in used_ids:
                    placeholder = candidate
                    break
            if not placeholder:
                placeholder = build_placeholder_candidate(candidates_slice[0], target_claims)
            if placeholder:
                used_ids.add(placeholder.doc_id)
                rest_selected = [placeholder]

        limits = RunLimits(max_total=len(claim1_selected) + len(rest_selected), Ay_min=len(rest_selected))
        return AnalysisResponse(
            run_id=f"analysis-{alpha.pub_number}",
            alpha=alpha,
            claim1_candidates=claim1_selected[:5] if claim1_selected else claim1_selected,
            rest_claim_candidates=rest_selected[:5] if rest_selected else rest_selected,
            limits=limits,
        )

    async def select_top_candidates(
        self,
        *,
        combined_candidates: List[Dict[str, Any]],
        analysis_result: Optional[Dict[str, Any]],
        alpha_info: Dict[str, Any],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        分析済み候補からLLMを使用して上位N件を選択

        Args:
            combined_candidates: 全候補のリスト
            analysis_result: analyze_candidatesの結果
            alpha_info: 出願特許の情報（title, summary, claim1）
            top_n: 選択する候補数

        Returns:
            選択された上位候補のリスト
        """
        # 候補数が少ない場合は全て返す
        if len(combined_candidates) <= top_n:
            return combined_candidates

        # 分析結果がない場合は先頭N件を返す
        if not analysis_result:
            return combined_candidates[:top_n]

        # 分析結果からclaim1の評価を抽出
        claim1_candidates = analysis_result.get("claim1_candidates", [])

        # LLM用に候補リストを準備
        candidates_for_llm = []
        for idx, candidate in enumerate(combined_candidates):
            # 対応する分析結果を探す
            matching_assessment = None
            candidate_id = candidate.get("patent_id", "")
            for assessed in claim1_candidates:
                if assessed.get("doc_id") == candidate_id:
                    matching_assessment = assessed
                    break

            # 分析結果のサマリーを作成
            assessment_summary = ""
            if matching_assessment:
                assessments = matching_assessment.get("assessments", [])
                if assessments:
                    first_assessment = assessments[0]
                    novelty = first_assessment.get("novelty", "uncertain")
                    inventive_step = first_assessment.get("inventive_step", "")
                    evidence_count = len(first_assessment.get("evidence", []))
                    assessment_summary = f"新規性:{novelty}, 進歩性:{inventive_step}, 根拠数:{evidence_count}"

            candidates_for_llm.append({
                "index": idx,
                "patent_id": candidate.get("patent_id"),
                "title": candidate.get("title", "")[:100],
                "is_web_result": candidate.get("is_web_result", False),
                "assessment": assessment_summary,
            })

        user_prompt = f"""あなたは特許審査の専門家です。以下の分析済み候補から、最も重要な先行技術候補を{top_n}件選択してください。

【出願特許】
タイトル: {alpha_info.get("title", "")}
要約: {alpha_info.get("summary", "")[:300]}
請求項1: {alpha_info.get("claim1", "")[:300]}

【分析済み候補一覧】
{json.dumps(candidates_for_llm, ensure_ascii=False, indent=2)}

【選択基準】
1. 新規性が「denied」（否定）の候補を優先
2. 進歩性も「denied」の候補はさらに高優先
3. 根拠数が多い候補を優先
4. 特許データベースとWeb検索結果をバランスよく含める

【出力形式】
以下のJSON形式で、選択した候補のindexリストを返してください：
{{"selected_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}}

必ず有効なJSONのみを出力し、余計なテキストは含めないでください。
"""

        try:
            response = await self.aoai_client.chat(
                deployment=self._chat_deployment,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
            )

            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                raise ValueError("Empty response from LLM")

            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_text = content[start:end+1]
                data = json.loads(json_text)
                selected_indices = data.get("selected_indices", [])

                # Validate and select
                final_candidates = []
                for idx in selected_indices[:top_n]:
                    if 0 <= idx < len(combined_candidates):
                        final_candidates.append(combined_candidates[idx])

                if len(final_candidates) < top_n:
                    # Fill with remaining candidates if LLM didn't select enough
                    for idx, candidate in enumerate(combined_candidates):
                        if idx not in selected_indices and len(final_candidates) < top_n:
                            final_candidates.append(candidate)

                return final_candidates
            else:
                raise ValueError("No valid JSON found in LLM response")

        except Exception as e:
            # Fallback: return first N candidates
            return combined_candidates[:top_n]

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
        pub_number = source_doc.pub_number or patent_id or None
        return AlphaInfo(
            title=title_override or source_doc.title or f"特許{patent_id or 'α'}",
            pub_number=pub_number,
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

    @staticmethod
    def _flatten_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if "text" in value:
                return str(value.get("text") or "")
            for key in ("ja", "en", "value", "content"):
                if key in value and value.get(key):
                    return str(value.get(key))
            return ""
        if isinstance(value, list):
            parts = [AnalysisService._flatten_text(item) for item in value]
            return " ".join(part for part in parts if part)
        return str(value)

    @staticmethod
    def _extract_description_section(description: Any, key: str) -> str:
        if isinstance(description, dict):
            raw = description.get(key) or description.get(key.replace("_", "-"))
        else:
            raw = None
        return AnalysisService._flatten_text(raw).strip()

    @staticmethod
    def _split_claim_elements(text: str, limit: int = 12) -> List[str]:
        if not text:
            return []
        cleaned = re.sub(r"\s+", " ", str(text)).strip()
        if not cleaned:
            return []
        chunks = re.split(r"[。;；]\s*", cleaned)
        elements: List[str] = []
        for chunk in chunks:
            if not chunk:
                continue
            if "、" in chunk:
                elements.extend([part.strip() for part in chunk.split("、") if part.strip()])
            else:
                elements.append(chunk.strip())
        seen: set[str] = set()
        ordered: List[str] = []
        for elem in elements:
            if elem in seen:
                continue
            seen.add(elem)
            ordered.append(elem)
            if len(ordered) >= limit:
                break
        return ordered

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
                    "【評価の軸】α請求項1の要素分解リストを最優先で参照し、候補が全要素を満たすかを判定する。"
                    "【デフォルト姿勢】否定根拠を優先して探索するが、以下の条件で supported / uncertain も柔軟に用いてよい。"
                    " supported: αの必須構成に対し候補に明確な欠落・矛盾・非同等の差異があるとき。"
                    " uncertain: 記載不足や曖昧表現で一致/不一致が判断できないとき（差異が小さい場合は denied も検討するが、無理に否定しない）。"
                    "【判定基準】"
                    " 新規性: αの必須構成が候補に明示または実質同等なら denied。α要件が明確に欠落する場合のみ supported。情報不足は uncertain。"
                    " 進歩性: 候補単独＋公知技術で容易想到なら denied。非自明な差異を具体的に示せる場合のみ supported。不足は uncertain。"
                    "【必須要件】denied を出す場合は、αの要素ごとに候補側の対応記載があることを確認し、examiner_hints に対応/欠落の要約を書く。"
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
        is_web_result = candidate_json.get("is_web_result", False)

        assessments: List[ClaimAssessment] = []
        for item in parsed:
            # Web検索結果の場合、sectionを「web」に統一
            evidence_list = []
            for e in item.get("evidence", []):
                if not e:
                    continue
                section = e.get("section", "unknown")
                # Web検索結果の場合、sectionを「web」に統一（細かい区分は不要）
                if is_web_result:
                    section = "web"

                evidence_list.append(
                    Evidence(
                        section=section,
                        quote=e.get("candidate_quote") or e.get("quote", ""),
                        why=e.get("why", ""),
                        offset=e.get("offset"),
                        length=e.get("length"),
                        alpha_fragment=e.get("alpha_fragment"),
                        candidate_quote=e.get("candidate_quote") or e.get("quote"),
                    )
                )

            assessments.append(
                ClaimAssessment(
                    claim_no=item.get("claim_no", 0),
                    novelty=item.get("novelty", "uncertain"),
                    inventive_step=item.get("inventive_step"),
                    evidence=evidence_list,
                    examiner_hints=item.get("examiner_hints", []) or [],
                )
            )

        source_url = candidate_json.get("source_url") if is_web_result else None
        pub_number = None if is_web_result else candidate_doc.pub_number

        return AssessmentCandidate(
            doc_id=str(doc_id),
            title=candidate_doc.title or "先行例",
            pub_number=pub_number,
            score=0.75,
            assessments=assessments,
            ipc=ipc_codes or ["UNKNOWN"],
            summary=candidate_json.get("summary") or candidate_json.get("abstract"),
            source_url=source_url,
            is_web_result=is_web_result,
        )

    def _build_prompt(
        self,
        alpha: AlphaInfo,
        candidate_json: Dict[str, Any],
        target_claims: Sequence[Tuple[int, str, bool]],
    ) -> str:
        summary = candidate_json.get("summary") or candidate_json.get("abstract") or ""
        claim1 = (
            (candidate_json.get("claim1") or candidate_json.get("claims", [{}])[0].get("text", ""))
            if candidate_json.get("claims")
            else candidate_json.get("claim1", "")
        )
        candidate_description = candidate_json.get("description")
        technical_field = self._extract_description_section(candidate_description, "technical-field")
        alpha_elements = self._split_claim_elements(alpha.claim1)
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

        # Web検索結果の場合はpage_contentを追加
        is_web_result = candidate_json.get("is_web_result", False)
        page_content = candidate_json.get("page_content", "")

        # プロンプトパーツを構築
        prompt_parts_list = [
            "【対象特許（α）】",
            f"タイトル: {alpha.title}",
            f"請求項1: {alpha.claim1}",
            "請求項1の必須構成（要素分解）:",
            "\n".join([f"  - {elem}" for elem in alpha_elements]) if alpha_elements else "  - （要素抽出なし）",
            f"請求項2以降: {' / '.join(alpha.claims_rest) if alpha.claims_rest else 'なし'}",
            "",
            "【比較候補】",
            f"タイトル: {candidate_json.get('title') or candidate_json.get('bibliographic', {}).get('title') or '不明'}",
            f"公報番号: {candidate_json.get('patent_id') or candidate_json.get('id') or candidate_json.get('pub_number') or '不明'}",
            f"技術分野: {technical_field or '不明'}",
            f"要約: {summary}",
            f"請求項1(候補): {claim1}",
            f"主要な請求項抜粋: {' / '.join(claims_texts[:3]) if claims_texts else 'なし'}",
        ]

        # Web検索結果の場合は本文内容も追加
        if is_web_result and page_content:
            prompt_parts_list.extend([
                "",
                "【参考：Web資料の本文抜粋】",
                page_content[:3000],  # 最大3000文字
            ])

        # 残りのプロンプト
        prompt_parts_list.extend([
            "",
            "【判定対象の請求項（α）】",
            "\n".join(target_claim_lines),
            "",
            "【判定の手順】",
            "1) αの要素分解リストを参照し、候補に各要素の対応記載があるか確認する。",
            "2) 全要素が候補に記載されていれば新規性は denied。",
            "3) 1つでも欠落・非同等があれば supported、判断不能は uncertain。",
            "4) examiner_hints に対応要素/欠落要素の要約を書く。",
            "",
        ])

        # Web検索結果の場合は引用ルールを明示
        if is_web_result:
            prompt_parts_list.extend([
                "【重要】この候補はWeb検索結果（論文・技術資料）です：",
                "- 請求項は存在しないため、引用元sectionは「abstract」または「description」を使用してください",
                "- candidate_quoteは「要約」または「Web資料の本文抜粋」から引用してください",
                "- 技術的な内容が一致する場合のみ「denied」と判定してください",
                "",
            ])

        prompt_parts_list.extend([
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
        ])
        return "\n".join(prompt_parts_list)

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

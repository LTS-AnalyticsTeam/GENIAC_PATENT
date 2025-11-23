from __future__ import annotations

import hashlib
import itertools
import re
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import (
    Candidate,
    Explanation,
    EvidenceHit,
    PatentDoc,
    Snippet,
)
from .services.matcher_heuristic import (
    unigram_matcher, bigram_matcher, char_shingle_matcher
)
from .services.ensemble import ensemble_hits

AX_POOL = [
    {
        "doc_id": "ax-001",
        "title": "Prior Art X – Secure IoT Gateway",
        "pub_number": "JP1234567A",
        "year": 2019,
        "ipc": ["H04L 29/08"],
        "base_score": 0.88,
        "section_templates": [
            ("claims", 1, "A gateway controller embeds {term} to negotiate cryptographic sessions with field devices."),
            ("description", 0, "In the orchestrated pipeline, {term} cooperates with adaptive routing to stabilize uplink bursts."),
            ("figures", 0, "Figure 3 highlights {term} aligned with buffering logic to preserve telemetry integrity."),
        ],
        "examiner_hints": [
            "進歩性の観点で通信制御モジュールとの組合せを検討",
            "暗号化ハンドシェイクの限定が差異点として残る可能性",
        ],
        # デモ用の簡易 description
        "description_demo": {
            "technical-field": [{"num": "0001", "text": "本発明は、超音波計測装置の通信ゲートウェイ制御に関する。"}],
            "background-art": [{"num": "0002", "text": "従来の通信路では経路混雑時に測定精度が低下する問題がある。"}],
            "summary-of-invention": [[
                {"num": "0005", "text": "経路毎の振幅正規化によりセンサ感度の影響を相殺する。"}
            ]],
            "advantageous-effects": [[
                {"num": "0008", "text": "感度に依存しない計測が可能となる。"}
            ]],
        },
    },
    {
        "doc_id": "ax-002",
        "title": "Prior Art X – Adaptive Edge Router",
        "pub_number": "US2019123456A1",
        "year": 2018,
        "ipc": ["H04W 84/12"],
        "base_score": 0.83,
        "section_templates": [
            ("claims", 1, "{term} drives a rule engine that classifies sensor frames before transmission."),
            ("description", 0, "The router reconfigures throughput when {term} indicates congestion thresholds."),
            ("summary", 0, "Operational analytics visualize {term} for operators to tune policies."),
        ],
        "examiner_hints": [
            "Xとの単独引用で請求項1の構成を充足する可能性",
            "データ前処理の限定が相違点として残存",
        ],
        "description_demo": {
            "technical-field": [{"num": "0001", "text": "本発明は、エッジ側での特徴量抽出と超音波計測に関する。"}],
            "background-art": [{"num": "0002", "text": "従来手法はセンサ受信波形の振幅変化に依存していた。"}],
            "summary-of-invention": [[
                {"num": "0006", "text": "異なる経路を通る受信波の振幅比に基づき感度ばらつきを打ち消す。"}
            ]],
            "advantageous-effects": [[
                {"num": "0008", "text": "センサ寿命に対して頑健で、長期使用が可能。"}
            ]],
        },
    },
]

AY_POOL = [
    {
        "doc_id": "ay-001",
        "title": "Prior Art Y1 – Analytics Pipeline Optimizer",
        "pub_number": "US2020123456A1",
        "year": 2020,
        "ipc": ["G06F 16/90"],
        "base_score": 0.81,
        "section_templates": [
            ("claims", 2, "{term} orchestrates batched normalization for downstream inference modules."),
            ("description", 0, "Processing nodes allocate cache lines where {term} hotspots appear."),
            ("effects", 0, "The optimizer reports that {term} reduces drift by 12%."),
        ],
        "examiner_hints": [
            "A(x)との組合せで進歩性×の指摘が想定される",
            "追加の正規化工程での差異を要確認",
        ],
        "description_demo": {
            "technical-field": [{"num": "0001", "text": "データ正規化と計測解析に関する。"}],
            "background-art": [{"num": "0002", "text": "振幅依存の計測は変動に弱い。"}],
            "summary-of-invention": [[
                {"num": "0007", "text": "受信波の異経路振幅比により減衰率変化のみを抽出する。"}
            ]],
            "advantageous-effects": [[
                {"num": "0008", "text": "感度特性に依存しない圧力計測を実現する。"}
            ]],
        },
    },
    {
        "doc_id": "ay-002",
        "title": "Prior Art Y2 – Contextual Feature Extractor",
        "pub_number": "JP2020554321A",
        "year": 2021,
        "ipc": ["G06K 9/62"],
        "base_score": 0.79,
        "section_templates": [
            ("claims", 3, "Feature vectors include {term} to bind spatial relations across modalities."),
            ("description", 0, "The extractor caches {term} at the edge to cut aggregation latency."),
            ("summary", 0, "{term} assists reviewers in interpreting similar patent corpora."),
        ],
        "examiner_hints": [
            "解決手段の平衡化で限定を付す余地あり",
            "引用例の組合せで構造的相違が縮小する見込み",
        ],
        "description_demo": {
            "technical-field": [{"num": "0001", "text": "特徴抽出器とセンサデータ解析に関する。"}],
            "background-art": [{"num": "0003", "text": "従来は校正コストが高い。"}],
            "summary-of-invention": [[
                {"num": "0006", "text": "複数経路の受信波を比較し感度起因の差分を除去する。"}
            ]],
            "advantageous-effects": [[
                {"num": "0008", "text": "長期使用時の誤検知を低減する。"}
            ]],
        },
    },
    {
        "doc_id": "ay-003",
        "title": "Prior Art Y3 – Feedback Control Ledger",
        "pub_number": "EP3456789A1",
        "year": 2017,
        "ipc": ["G05B 13/02"],
        "base_score": 0.76,
        "section_templates": [
            ("claims", 2, "{term} logs actuator events for subsequent adaptive tuning cycles."),
            ("description", 0, "Supervisory nodes broadcast {term} to synchronize controllers."),
            ("summary", 0, "{term} mitigates oscillations in legacy plants."),
        ],
        "examiner_hints": [
            "フィードバック制御との併用で比較検討が必要",
            "適用範囲の限定が差異点として残る",
        ],
        "description_demo": {
            "technical-field": [{"num": "0001", "text": "フィードバック制御に関する。"}],
            "background-art": [{"num": "0002", "text": "本件の圧力計測とは非直接的。"}],
            "summary-of-invention": [[{"num":"0005", "text":"同期機構の説明。"}]],
        },
    },
    {
        "doc_id": "ay-004",
        "title": "Prior Art Y4 – Explainable Scoring Engine",
        "pub_number": "WO2020555123A1",
        "year": 2022,
        "ipc": ["G06N 3/08"],
        "base_score": 0.74,
        "section_templates": [
            ("claims", 2, "The scoring engine exposes {term} to trace model attribution paths."),
            ("description", 0, "Dashboard widgets surface {term} for compliance reviews."),
            ("effects", 0, "By surfacing {term}, analysts compress review cycles."),
        ],
        "examiner_hints": [
            "説明可能性要素で審査官指摘が予想",
            "公開年が新しく引用可能性が高い",
        ],
        "description_demo": {
            "technical-field": [{"num":"0001","text":"説明可能性モデルに関する。"}],
        },
    },
    {
        "doc_id": "ay-005",
        "title": "Prior Art Y5 – Distributed Token Manager",
        "pub_number": "KR2018456123A",
        "year": 2018,
        "ipc": ["H04L 9/32"],
        "base_score": 0.75,
        "section_templates": [
            ("claims", 4, "{term} rotates credentials across segmented gateways."),
            ("description", 0, "A watchdog compares {term} events before issuing revocation."),
            ("summary", 0, "{term} mitigates replay vectors in distributed deployments."),
        ],
        "examiner_hints": [
            "セキュリティ観点で組合せ引用の根拠になり得る",
            "差異点は認証トークンの配列方法に集中",
        ],
        "description_demo": {
            "technical-field": [{"num":"0001","text":"認証トークン管理に関する。"}],
        },
    },
]


def search_prior_art(
    claim1: str,
    claims_rest: Sequence[str],
    *,
    max_total: int = 10,
    ay_min: int = 2,

    source_patent: PatentDoc | None = None,
) -> Tuple[Candidate, List[Candidate]]:
    terms_claim1 = _extract_terms(claim1)
    terms_rest = _extract_terms(" ".join(claims_rest) if claims_rest else claim1)

    summary = _build_alpha_summary(claim1, claims_rest)

    ax_candidates = [
        _build_candidate(raw, terms_claim1, summary, focus_claim=1, seed_text=claim1, source_patent=source_patent)
        for raw in AX_POOL
    ]
    ax_candidates.sort(key=lambda c: c.score, reverse=True)
    ax_top = ax_candidates[0]

    ay_candidates = [
        _build_candidate(raw, terms_rest, summary, focus_claim=2, seed_text=" ".join(claims_rest) or claim1, source_patent=source_patent)
        for raw in AY_POOL
    ]
    ay_candidates.sort(key=lambda c: c.score, reverse=True)

    remaining = max(max_total - 1, 0)
    ay_count = min(len(ay_candidates), max(ay_min, remaining))
    ay_selected = ay_candidates[:ay_count]

    return ax_top, ay_selected


def _build_candidate(
    raw: Dict,
    terms: Sequence[str],
    summary: str,
    *,
    focus_claim: int,
    seed_text: str,
    source_patent: PatentDoc | None,
) -> Candidate:
    score = _score(seed_text, raw["doc_id"], raw["base_score"])
    snippets = _build_snippets(raw["section_templates"], terms, focus_claim, raw["doc_id"])
    explanation = Explanation(
        summary=summary,
        why_match=_build_why_match(terms, raw["title"]),
        examiner_hints=raw["examiner_hints"],
    )
    cand = Candidate(
        doc_id=raw["doc_id"],
        title=raw["title"],
        pub_number=raw["pub_number"],
        year=raw["year"],
        ipc=raw["ipc"],
        score=score,
        snippets=snippets,
        explanation=explanation,
    )


    if source_patent is not None:
        cand_patent = _candidate_patent_from_demo(cand.title, raw.get("description_demo"))
        if cand_patent:
            model1 = unigram_matcher(source_patent, cand_patent)
            model2 = bigram_matcher(source_patent, cand_patent)
            model3 = char_shingle_matcher(source_patent, cand_patent)
            hits, judgment = ensemble_hits(source_patent, cand_patent, [model1, model2, model3])
            cand.evidence_hits = hits
            cand.judgment_basis = judgment

    return cand


def _candidate_patent_from_demo(title: str, desc_json) -> PatentDoc | None:
    if not desc_json:
        return None
    # デモのため、claim1 は titleを代用（本来は候補の請求項を入れる）
    from .alpha_parser import build_patent_doc_from_inputs
    return build_patent_doc_from_inputs(
        title=title,
        pub_number=None,
        claim1=title,
        description_json=desc_json,
    )


def _build_snippets(
    templates: Iterable[Tuple[str, int, str]],
    terms: Sequence[str],
    focus_claim: int,
    doc_id: str,
) -> List[Snippet]:
    snippets: List[Snippet] = []
    fallback_terms = ["主要要素"]
    use_terms = list(terms) or fallback_terms

    for idx, (section, claim_no, template) in enumerate(templates):
        term = use_terms[idx % len(use_terms)]
        text = template.format(term=term)
        offset = _safe_index(text.lower(), term.lower())
        snippet_score = _snippet_score(doc_id, idx)
        snippets.append(
            Snippet(
                section=section,
                claim_no=claim_no or focus_claim,
                text=text,
                offset=offset,
                len=len(term),
                match_type="phrase" if len(term.split()) > 1 else "term",
                score=snippet_score,
            )
        )
    return snippets


def _build_why_match(terms: Sequence[str], title: str) -> List[str]:
    if not terms:
        return [f"{title} の主要構成が請求項要素に概ね一致"]
    messages = []
    for term in itertools.islice(terms, 3):
        messages.append(f"{term} が対応要素として一致")
    if len(messages) < 3:
        messages.append(f"{title} による構造が請求項の要件を補強")
    return messages


def _build_alpha_summary(claim1: str, claims_rest: Sequence[str]) -> str:
    claim1_short = claim1[:140] + ("…" if len(claim1) > 140 else "")
    rest_short = " / ".join(claim[:60] + ("…" if len(claim) > 60 else "") for claim in itertools.islice(claims_rest, 2))
    if rest_short:
        return f"課題: {claim1_short} | 手段/効果: {rest_short}"
    return f"課題: {claim1_short}"


def _extract_terms(text: str, limit: int = 6) -> List[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9一-龯ぁ-んァ-ンー]+", text or ""):
        if len(token) <= 1:
            continue
        normalized = token.strip()
        if normalized not in tokens:
            tokens.append(normalized)
        if len(tokens) >= limit:
            break
    return tokens


def _score(seed_text: str, doc_id: str, base_score: float) -> float:
    digest = hashlib.sha256(f"{seed_text}:{doc_id}".encode("utf-8")).hexdigest()
    jitter = (int(digest[:4], 16) / 0xFFFF) * 0.1 - 0.05
    raw = base_score + jitter
    return round(min(max(raw, 0.5), 0.96), 2)


def _snippet_score(doc_id: str, index: int) -> float:
    digest = hashlib.md5(f"{doc_id}:{index}".encode("utf-8")).hexdigest()
    jitter = (int(digest[:2], 16) / 0xFF) * 0.15
    base = 0.78 - index * 0.07
    raw = base + jitter
    return round(min(max(raw, 0.5), 0.95), 2)


def _safe_index(text: str, term: str) -> int:
    pos = text.find(term)
    return pos if pos >= 0 else 0
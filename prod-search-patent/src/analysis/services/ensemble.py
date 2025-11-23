from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from ..models import EvidenceHit, JudgmentBasis, ModelVote, EnsembleDecision, PatentDoc


def _hit_key(hit: EvidenceHit) -> Tuple[str, str | None, str]:
    return (hit.section, hit.num, hit.text)


def ensemble_hits(
    source: PatentDoc,
    candidate: PatentDoc,
    hit_groups: Sequence[Sequence[EvidenceHit]],
) -> Tuple[List[EvidenceHit], JudgmentBasis]:
    """ざっくりとしたアンサンブル判定。

    1. 各モデルのヒットをマージして support を積算
    2. モデルごとの vote score を算出し、平均スコアを final_score として返す
    3. reasoning / rationale に簡単な要約を書き込む
    """

    combined: dict[Tuple[str, str | None, str], EvidenceHit] = {}
    votes: List[ModelVote] = []

    for idx, hits in enumerate(hit_groups):
        weight = min(1.0, 0.5 + 0.1 * len(hits))
        votes.append(ModelVote(model_name=f"model_{idx + 1}", score=round(weight, 2)))
        for hit in hits:
            key = _hit_key(hit)
            entry = combined.setdefault(
                key,
                EvidenceHit(
                    section=hit.section,
                    num=hit.num,
                    text=hit.text,
                    spans=list(hit.spans),
                    support=0,
                    score=hit.score,
                ),
            )
            entry.support += 1
            entry.score = max(entry.score, hit.score)
            existing_spans = {(span.start, span.end) for span in entry.spans}
            for span in hit.spans:
                coords = (span.start, span.end)
                if coords not in existing_spans:
                    entry.spans.append(span)
                    existing_spans.add(coords)

    merged_hits = sorted(
        combined.values(),
        key=lambda h: (-h.support, -h.score),
    )

    final_score = round(
        sum(vote.score for vote in votes) / len(votes), 2
    ) if votes else 0.0

    top_sections = ", ".join(
        f"{hit.section}(support={hit.support})" for hit in merged_hits[:3]
    ) or "一致箇所なし"

    reasoning = [
        f"クレーム1と候補明細書で{len(merged_hits)}箇所の一致を検出",
        f"最も支持の高いセクション: {top_sections}",
    ]
    rationale = [
        "support値と個別モデルの投票を平均化した便宜的な信頼度",
    ]

    decision = EnsembleDecision(votes=votes, final_score=final_score, rationale=rationale)
    judgment = JudgmentBasis(reasoning=reasoning, decision=decision)
    return merged_hits, judgment

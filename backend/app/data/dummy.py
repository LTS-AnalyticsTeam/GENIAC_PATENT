from __future__ import annotations

from typing import List

from ..models import AlphaInfo, Candidate, Explanation, Snippet


def build_dummy_alpha() -> AlphaInfo:
    return AlphaInfo(
        title="特許αデモ",
        pub_number="DUMMY-0001",
        claim1="請求項1: エッジ装置から取得したデータを暗号化し、遅延閾値に応じて通信経路を切り替える制御方式。",
        claims_rest=[
            "請求項2: 収集データの系列を解析し、異常検知用のベクトルを生成する処理。",
        ],
    )


def build_dummy_ax() -> Candidate:
    return Candidate(
        doc_id="ax-dummy",
        title="エッジ通信制御装置",
        pub_number="JP2023123456A",
        year=2022,
        ipc=["H04L 29/08"],
        score=0.9,
        snippets=[
            Snippet(
                section="内部資料",
                claim_no=1,
                text="暗号化ハンドシェイクと遅延閾値制御を組み合わせたゲートウェイ制御方式の例示。",
                offset=0,
                len=6,
                match_type="phrase",
                score=0.82,
            )
        ],
        explanation=Explanation(
            summary="端末ごとの暗号化ハンドシェイク情報を保持し、遅延閾値を超えた際に通信経路を切り替えるゲートウェイ装置の要約。",
            why_match=[
                "暗号化ハンドシェイクの流れが請求項1と一致",
                "閾値ベースの制御切替が対応",
            ],
            examiner_hints=[
                "単独引用で課題の大部分を充足する可能性",
                "ログ管理の差分を補う引用例を検討",
            ],
        ),
    )


def build_dummy_ay_candidates() -> List[Candidate]:
    return [
        Candidate(
            doc_id="ay-dummy-1",
            title="ログ解析パイプライン（web検索）",
            pub_number=None,
            year=None,
            ipc=[],
            score=0.84,
            snippets=[
                Snippet(
                    section="web検索",
                    claim_no=2,
                    text="（web検索）イベント系列から異常検知ベクトルを生成し、ダッシュボードへ連携する処理フロー。",
                    offset=0,
                    len=4,
                    match_type="phrase",
                    score=0.8,
                )
            ],
            explanation=Explanation(
                summary="（web検索）異常検知パイプラインのダミー例。請求項2で述べる解析・生成処理と対応する。",
                why_match=[
                    "イベント解析エンジンの構造が一致",
                    "異常検知ベクトルの生成が対応",
                ],
                examiner_hints=[
                    "A(x)と組み合わせると進歩性指摘の可能性あり",
                    "特徴量保持手段の差異を精査",
                ],
            ),
            source_url="https://example.com/log-analytics",
        ),
        Candidate(
            doc_id="ay-dummy-2",
            title="ダッシュボード連携基盤",
            pub_number="US2021001122A1",
            year=2021,
            ipc=["G06Q 50/00"],
            score=0.8,
            snippets=[
                Snippet(
                    section="図面",
                    claim_no=2,
                    text="解析済みイベントを可視化パネルへ送信し、レビュー担当者が異常値を即座に確認できるワークフロー。",
                    offset=0,
                    len=4,
                    match_type="phrase",
                    score=0.77,
                ),
                Snippet(
                    section="実施例",
                    claim_no=2,
                    text="アラート生成と承認コメントの履歴を保持し、組織内に提示するダッシュボード機能。",
                    offset=0,
                    len=4,
                    match_type="term",
                    score=0.73,
                ),
            ],
            explanation=Explanation(
                summary="レビュー作業を効率化するダッシュボード連携基盤。生成した異常検知ベクトルをパネル化して共有する例。",
                why_match=[
                    "異常検知結果を可視化しレビューに活用する点が一致",
                    "通知とコメント管理が補完的に作用",
                ],
                examiner_hints=[
                    "通知設計が差異点となる可能性",
                    "アクセス権管理の限定を追加検討",
                ],
            ),
        ),
        Candidate(
            doc_id="ay-dummy-3",
            title="フィードバック制御ログ集約装置",
            pub_number="EP2020123456A1",
            year=2020,
            ipc=["G05B 13/02"],
            score=0.78,
            snippets=[
                Snippet(
                    section="請求項",
                    claim_no=2,
                    text="制御ログを時系列に蓄積し、異常兆候を解析して外部システムへ通知する仕組み。",
                    offset=0,
                    len=4,
                    match_type="term",
                    score=0.75,
                ),
                Snippet(
                    section="概要",
                    claim_no=3,
                    text="解析結果を監視端末へ配信し、運用者の判断を支援するための概要提示機能。",
                    offset=0,
                    len=4,
                    match_type="phrase",
                    score=0.72,
                ),
            ],
            explanation=Explanation(
                summary="フィードバック制御ログを集約し、異常検知結果を外部へ提示する装置。請求項2〜3のロギング処理に対応。",
                why_match=[
                    "ログ蓄積と異常検知の流れが符合",
                    "外部提示によるレビュー支援が請求項3と合致",
                ],
                examiner_hints=[
                    "ログ同期手段の限定が差異として残る",
                    "A(x)との組み合わせ引用で進歩性判断が想定される",
                ],
            ),
        ),
    ]

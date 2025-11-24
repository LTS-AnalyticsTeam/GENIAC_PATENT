import React, { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { ChevronLeft, FileText, AlertCircle } from "lucide-react";
import {
  AnalysisResponse,
  AssessmentCandidate,
  ClaimAssessment,
} from "../types";
import "./BatchResults.css";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ??
  import.meta.env.VITE_API_BASE ??
  "http://localhost:8080";
const RESULT_CACHE_PREFIX = "ps-result-cache";
const buildResultCacheKey = (jobId: string) =>
  `${RESULT_CACHE_PREFIX}:${jobId}`;

type GraphResult = {
  patent_id: string;
  title?: string;
  summary?: string;
  classification_ipc?: string[];
  graph_score?: number;
  analysis_status?: string;
  analysis_error?: string;
  analysis?: AnalysisResponse;
};

interface PipelineResultResponse {
  job_id: string;
  completed_at: string;
  results: GraphResult[];
  pipeline_stats: Record<string, unknown>;
}

interface LocationState {
  batchResult: PipelineResultResponse;
}

const statusLabel = (
  value: string | undefined,
  kind: "novelty" | "inventive"
) => {
  const head = kind === "novelty" ? "新規性" : "進歩性";
  if (value === "denied") return `${head}: 否定（先行例が充足/容易想到）`;
  if (value === "supported") return `${head}: 支持（差異あり）`;
  return `${head}: 要検討`;
};

const statusClass = (value: string | undefined) => {
  if (value === "denied") return "danger";
  if (value === "supported") return "success";
  return "warning";
};

const BatchResults: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const state = location.state as LocationState | null;
  const stateBatchResult = state?.batchResult;
  const [batchResult, setBatchResult] = useState<PipelineResultResponse | null>(
    stateBatchResult ?? null
  );
  const [activeCandidateIdx, setActiveCandidateIdx] = useState(0);
  const [loading, setLoading] = useState(!stateBatchResult);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const jobIdFromQuery = searchParams.get("jobId");

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (stateBatchResult) {
      setBatchResult(stateBatchResult);
      setLoading(false);
      setErrorMessage(null);
      return;
    }

    if (!jobIdFromQuery) {
      setBatchResult(null);
      setLoading(false);
      setErrorMessage("結果データがありません");
      return;
    }

    const cacheKey = buildResultCacheKey(jobIdFromQuery);
    const cached = sessionStorage.getItem(cacheKey);
    if (cached) {
      try {
        setBatchResult(JSON.parse(cached));
        setLoading(false);
        setErrorMessage(null);
        return;
      } catch {
        sessionStorage.removeItem(cacheKey);
      }
    }

    const fetchResult = async () => {
      setLoading(true);
      setErrorMessage(null);
      try {
        const res = await fetch(`${API_BASE}/result/${jobIdFromQuery}`);
        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(errorText || "結果取得に失敗しました");
        }
        const payload = await res.json();
        setBatchResult(payload);
        sessionStorage.setItem(cacheKey, JSON.stringify(payload));
      } catch (err) {
        console.error("Failed to load result", err);
        setErrorMessage("結果取得に失敗しました");
      } finally {
        setLoading(false);
      }
    };

    fetchResult();
  }, [jobIdFromQuery, stateBatchResult]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (batchResult?.job_id) {
      try {
        sessionStorage.setItem(
          buildResultCacheKey(batchResult.job_id),
          JSON.stringify(batchResult)
        );
      } catch (err) {
        console.warn("Failed to persist result cache", err);
      }
    }
  }, [batchResult]);

  const results = batchResult?.results ?? [];
  const activeResult = useMemo(() => results[0], [results]);
  const analysis = activeResult?.analysis;
  const claim1Candidates = analysis?.claim1_candidates ?? [];
  const restCandidates = analysis?.rest_claim_candidates ?? [];
  const candidateTabs = useMemo(() => {
    const first = claim1Candidates.map((c) => ({
      ...c,
      _kind: "claim1" as const,
    }));
    const rest = restCandidates.map((c) => ({ ...c, _kind: "rest" as const }));
    return [...first, ...rest].slice(0, 10);
  }, [claim1Candidates, restCandidates]);

  useEffect(() => {
    if (activeCandidateIdx >= candidateTabs.length) {
      setActiveCandidateIdx(0);
    }
  }, [candidateTabs.length, activeCandidateIdx]);

  if (loading && !batchResult) {
    return (
      <div className="no-results">
        <AlertCircle size={48} />
        <p>結果を読み込んでいます...</p>
        <button onClick={() => navigate("/")}>戻る</button>
      </div>
    );
  }

  if (errorMessage && !batchResult) {
    return (
      <div className="no-results">
        <AlertCircle size={48} />
        <p>{errorMessage}</p>
        <button onClick={() => navigate("/")}>戻る</button>
      </div>
    );
  }

  if (!batchResult || results.length === 0 || !activeResult) {
    return (
      <div className="no-results">
        <AlertCircle size={48} />
        <p>結果データがありません</p>
        <button onClick={() => navigate("/")}>戻る</button>
      </div>
    );
  }

  const renderAssessment = (assessment: ClaimAssessment) => (
    <div key={assessment.claim_no} className="assessment-block">
      <div className="assessment-header">
        <span className="badge">
          {assessment.claim_no === 1
            ? "請求項1"
            : `請求項${assessment.claim_no}`}
        </span>
        <span className={`badge ${statusClass(assessment.novelty)}`}>
          {statusLabel(assessment.novelty, "novelty")}
        </span>
        {assessment.inventive_step && (
          <span className={`badge ${statusClass(assessment.inventive_step)}`}>
            {statusLabel(assessment.inventive_step, "inventive")}
          </span>
        )}
      </div>
      {assessment.evidence.length > 0 ? (
        <>
          <h5>参照箇所表示</h5>
          <ul className="evidence-list">
            {assessment.evidence.map((ev, idx) => (
              <li key={idx} className="evidence-item">
                <div className="evidence-pair">
                  <div className="evidence-col alpha">
                    <div className="evidence-label">α該当部分</div>
                    <div className="evidence-text">
                      {ev.alpha_fragment || "記載なし"}
                    </div>
                  </div>
                  <div className="evidence-col candidate">
                    <div className="evidence-label">候補引用</div>
                    <div className="evidence-text">
                      {ev.candidate_quote || ev.quote || "記載なし"}
                    </div>
                  </div>
                </div>
                <div className="evidence-why">{ev.why}</div>
                {ev.section && (
                  <div className="evidence-meta">出典: {ev.section}</div>
                )}
              </li>
            ))}
          </ul>
        </>
      ) : (
        <p className="muted">エビデンスがありません（要再判定）</p>
      )}
      {assessment.examiner_hints.length > 0 && (
        <div className="examiner-hints">
          <h5>審査官への示唆</h5>
          <ul>
            {assessment.examiner_hints.map((hint, idx) => (
              <li key={idx}>{hint}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );

  const renderCandidate = (
    candidate: AssessmentCandidate & { _kind: "claim1" | "rest" }
  ) => {
    const label = candidate._kind === "claim1" ? "Ax" : "Ay";
    const assessments = candidate.assessments;
    const restCombined =
      candidate._kind === "rest"
        ? {
            noveltyLines: assessments.map((a) => ({
              claimNo: a.claim_no,
              novelty: a.novelty,
              inventive: a.inventive_step,
            })),
            evidence: assessments.flatMap((a) =>
              a.evidence.map((ev) => ({ ...ev, claimNo: a.claim_no }))
            ),
            hints: assessments.flatMap((a) =>
              (a.examiner_hints || []).map((h) => `請求項${a.claim_no}: ${h}`)
            ),
          }
        : null;
    return (
      <div className="candidate-card" key={candidate.doc_id}>
        <div className="candidate-header">
          <div>
            <div className="badge primary">{label}</div>
            <h3>{candidate.title}</h3>
            <div className="candidate-meta">
              <span>特許番号: {candidate.pub_number ?? "不明"}</span>
            </div>
          </div>
        </div>
        {candidate.summary && (
          <div className="candidate-summary">
            <div className="evidence-label">要約</div>
            <p>{candidate.summary}</p>
          </div>
        )}
        {candidate._kind === "claim1" ? (
          assessments.map(renderAssessment)
        ) : (
          <div className="assessment-block">
            <div className="assessment-header">
              <span className="badge">請求項2以降まとめ</span>
            </div>
            <div className="claims-summary-grid">
              {restCombined?.noveltyLines.map((line) => (
                <div key={line.claimNo} className="claim-summary-row">
                  <span className="badge subtle">請求項{line.claimNo}</span>
                  <span className={`badge ${statusClass(line.novelty)}`}>
                    {statusLabel(line.novelty, "novelty")}
                  </span>
                  {line.inventive && (
                    <span className={`badge ${statusClass(line.inventive)}`}>
                      {statusLabel(line.inventive, "inventive")}
                    </span>
                  )}
                </div>
              ))}
            </div>
            {restCombined?.evidence && restCombined.evidence.length > 0 ? (
              <>
                <h5>参照箇所表示</h5>
                <ul className="evidence-list">
                  {restCombined.evidence.map((ev, idx) => (
                    <li key={idx} className="evidence-item">
                      <div className="evidence-pair">
                        <div className="evidence-col alpha">
                          <div className="evidence-label">α該当部分</div>
                          <div className="evidence-text">
                            {ev.alpha_fragment || "記載なし"}
                          </div>
                        </div>
                        <div className="evidence-col candidate">
                          <div className="evidence-label">候補引用</div>
                          <div className="evidence-text">
                            {ev.candidate_quote || ev.quote || "記載なし"}
                          </div>
                        </div>
                      </div>
                      <div className="evidence-why">{ev.why}</div>
                      {ev.section && (
                        <div className="evidence-meta">出典: {ev.section}</div>
                      )}
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="muted">参照箇所表示はありません。</p>
            )}
            {restCombined?.hints && restCombined.hints.length > 0 && (
              <div className="examiner-hints">
                <h5>審査官への示唆</h5>
                <ul>
                  {restCombined.hints.map((hint, idx) => (
                    <li key={idx}>{hint}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="batch-results-container">
      {/* ヘッダー */}
      <div className="results-header">
        <button className="back-btn" onClick={() => navigate("/")}>
          <ChevronLeft size={20} />
          戻る
        </button>
        <h1>特許分析結果</h1>
        <div className="batch-summary">
          <span>特許ID: {analysis?.alpha.pub_number || "不明"}</span>
          <span>タイトル: {analysis?.alpha.title || "不明"}</span>
          <span>
            完了日時:{" "}
            {new Date(batchResult.completed_at).toLocaleString("ja-JP")}
          </span>
        </div>
      </div>

      {/* タブナビゲーション */}
      <div className="tabs-container">
        <div className="tabs-nav">
          {candidateTabs.map((c, idx) => {
            const kindLabel = c._kind === "claim1" ? "Ax" : "Ay";
            const kindClass = c._kind === "claim1" ? "tab-ax" : "tab-ay";
            return (
              <button
                key={`${c._kind}-${c.doc_id}`}
                className={`tab-btn ${activeCandidateIdx === idx ? "active" : ""} ${kindClass}`}
                onClick={() => setActiveCandidateIdx(idx)}
              >
                <FileText size={16} />
                <span className="tab-title">
                  {kindLabel} : {c.title}
                  {c.pub_number && (
                    <span className="tab-subtitle">{c.pub_number}</span>
                  )}
                </span>
              </button>
            );
          })}
        </div>

        <div className="tab-content">
          {activeResult.analysis_status === "failed" ? (
            <div className="error-message">
              <AlertCircle size={48} />
              <h3>分析エラー</h3>
              <p>{activeResult.analysis_error ?? "分析に失敗しました"}</p>
            </div>
          ) : (
            <div className="analysis-content">
              {analysis && candidateTabs.length > 0 ? (
                renderCandidate(candidateTabs[activeCandidateIdx])
              ) : (
                <div className="info-message">
                  <AlertCircle size={32} />
                  <p>分析結果がまだありません。別のジョブをお試しください。</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BatchResults;

// frontend/src/components/BatchResults.tsx
import React, { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { ChevronLeft, FileText, AlertCircle, CheckCircle, XCircle } from "lucide-react";
import "./BatchResults.css";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ??
  import.meta.env.VITE_API_BASE ??
  "http://localhost:8080";
const RESULT_CACHE_PREFIX = "ps-result-cache";
const buildResultCacheKey = (jobId: string) => `${RESULT_CACHE_PREFIX}:${jobId}`;

type AnalysisPayload = {
  run_id: string;
  alpha: {
    title: string;
    pub_number: string;
    claim1: string;
    claims_rest: string[];
  };
  Ax: {
    doc_id: string;
    title: string;
    pub_number: string;
    year?: number;
    ipc: string[];
    score: number;
    explanation: {
      summary: string;
      why_match: string[];
      examiner_hints: string[];
    };
  };
  Ay: Array<{
    doc_id: string;
    title: string;
    pub_number: string;
    year?: number;
    ipc: string[];
    score: number;
    explanation: {
      summary: string;
      why_match: string[];
      examiner_hints: string[];
    };
  }>;
};

type GraphResult = {
  patent_id: string;
  title?: string;
  summary?: string;
  classification_ipc?: string[];
  graph_score?: number;
  analysis_status?: string;
  analysis_error?: string;
  analysis?: AnalysisPayload;
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

const BatchResults: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const state = location.state as LocationState | null;
  const stateBatchResult = state?.batchResult;
  const [batchResult, setBatchResult] = useState<PipelineResultResponse | null>(stateBatchResult ?? null);
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(!stateBatchResult);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const jobIdFromQuery = searchParams.get("jobId");

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
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
    if (typeof window === "undefined") {
      return;
    }
    if (batchResult?.job_id) {
      try {
        sessionStorage.setItem(buildResultCacheKey(batchResult.job_id), JSON.stringify(batchResult));
      } catch (err) {
        console.warn("Failed to persist result cache", err);
      }
    }
  }, [batchResult]);

  useEffect(() => {
    if (!batchResult) {
      if (activeTab !== 0) {
        setActiveTab(0);
      }
      return;
    }
    if (activeTab >= (batchResult.results?.length ?? 0)) {
      setActiveTab(0);
    }
  }, [batchResult, activeTab]);

  const results = batchResult?.results ?? [];
  const activeResult = useMemo(() => results[activeTab], [results, activeTab]);

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

  return (
    <div className="batch-results-container">
      {/* ヘッダー */}
      <div className="results-header">
        <button className="back-btn" onClick={() => navigate('/')}>
          <ChevronLeft size={20} />
          戻る
        </button>
        <h1>特許分析結果</h1>
        <div className="batch-summary">
          <span>ジョブID: {batchResult.job_id}</span>
          <span>完了日時: {new Date(batchResult.completed_at).toLocaleString("ja-JP")}</span>
          <span>結果件数: {results.length}</span>
        </div>
      </div>

      {/* タブナビゲーション */}
      <div className="tabs-container">
        <div className="tabs-nav">
          {results.map((result, index) => (
            <button
              key={result.patent_id}
              className={`tab-btn ${activeTab === index ? 'active' : ''} ${
                result.analysis_status === 'failed' ? 'error' : ''
              }`}
              onClick={() => setActiveTab(index)}
            >
              <FileText size={16} />
              <span className="tab-title">
                {(result.title || result.patent_id || `特許 ${index + 1}`)}
                {result.patent_id && result.title && (
                  <span className="tab-subtitle">{result.patent_id}</span>
                )}
              </span>
              {result.analysis_status === 'completed' ? (
                <CheckCircle size={16} className="status-icon success" />
              ) : result.analysis_status === 'failed' ? (
                <XCircle size={16} className="status-icon error" />
              ) : (
                <AlertCircle size={16} className="status-icon pending" />
              )}
            </button>
          ))}
        </div>

        {/* タブコンテンツ */}
        <div className="tab-content">
          {activeResult.analysis_status === "failed" ? (
            <div className="error-message">
              <AlertCircle size={48} />
              <h3>分析エラー</h3>
              <p>{activeResult.analysis_error ?? "分析に失敗しました"}</p>
            </div>
          ) : (
            <div className="analysis-content">
              <div className="patent-info">
                <h2>{activeResult.title || activeResult.patent_id}</h2>
                {activeResult.graph_score !== undefined && (
                  <p className="pub-number">Graph Score: {activeResult.graph_score.toFixed(2)}</p>
                )}
              </div>

              <div className="section">
                <h3>要約</h3>
                <div className="claim-text">{activeResult.summary ?? "要約情報がありません。"}</div>
              </div>

              {activeResult.classification_ipc && activeResult.classification_ipc.length > 0 && (
                <div className="section">
                  <h3>IPC分類</h3>
                  <div className="claims-list">
                    {activeResult.classification_ipc.map((code) => (
                      <div key={code} className="claim-item">
                        {code}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activeResult.analysis ? (
                <>
                  <div className="section">
                    <h3>請求項1</h3>
                    <div className="claim-text">{activeResult.analysis.alpha.claim1}</div>
                  </div>
                  {activeResult.analysis.alpha.claims_rest.length > 0 && (
                    <div className="section">
                      <h3>請求項2-5</h3>
                      <div className="claims-list">
                        {activeResult.analysis.alpha.claims_rest.map((claim, idx) => (
                          <div key={idx} className="claim-item">
                            {claim}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="section novelty-section">
                    <h3>Ax: 新規性判定（請求項1）</h3>
                    <div className="prior-art-card">
                      <div className="card-header">
                        <span className="doc-number">{activeResult.analysis.Ax.pub_number}</span>
                        <span className="score">スコア: {activeResult.analysis.Ax.score.toFixed(2)}</span>
                      </div>
                      <h4>{activeResult.analysis.Ax.title}</h4>
                      <div className="explanation">
                        {activeResult.analysis.Ax.explanation.why_match.map((reason, idx) => (
                          <div key={idx} className="reason-block">
                            {reason.split("\n").map((line, lineIdx) => (
                              <p key={lineIdx}>{line}</p>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {activeResult.analysis.Ay && activeResult.analysis.Ay.length > 0 && (
                    <div className="section inventive-section">
                      <h3>Ay: 進歩性判定（請求項2-5）</h3>
                      {activeResult.analysis.Ay.map((ay, idx) => (
                        <div key={idx} className="prior-art-card">
                          <div className="card-header">
                            <span className="doc-number">{ay.pub_number}</span>
                            <span className="score">スコア: {ay.score.toFixed(2)}</span>
                          </div>
                          <h4>{ay.title}</h4>
                          <div className="summary">{ay.explanation.summary}</div>
                          <div className="explanation">
                            {ay.explanation.why_match.map((reason, reasonIdx) => (
                              <div key={reasonIdx} className="reason-block">
                                {reason.split("\n").map((line, lineIdx) => (
                                  <p key={lineIdx}>{line}</p>
                                ))}
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <div className="info-message">
                  <AlertCircle size={32} />
                  <p>Ax/Ay 解析はまだ完了していません。Graph結果のみ表示しています。</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* フッター統計 */}
        <div className="results-footer">
          <div className="stats">
            <span>処理時刻: {new Date(batchResult.completed_at).toLocaleString("ja-JP")}</span>
          </div>
      </div>
    </div>
  );
};

export default BatchResults;

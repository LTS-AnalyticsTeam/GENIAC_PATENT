import { useCallback, useEffect, useMemo, useState } from "react";
import StageProgress from "./components/StageProgress";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8080";

const STORAGE_KEY = "ps-job-state";

const STAGES = [
  { id: "parsing", label: "① テキスト解析" },
  { id: "cosmos_query", label: "② Cosmos検索" },
  { id: "trimming", label: "③ トリミング" },
  { id: "embedding", label: "④ 埋め込み生成" },
  { id: "stage1_indexing", label: "⑤ Stage1インデックス" },
  { id: "vector_search", label: "⑥ ベクトル検索" },
  { id: "stage2_indexing", label: "⑦ Neo4j登録" },
  { id: "graph_rag", label: "⑧ Graph-RAG" },
];

type JobStatus = {
  status: string;
  detail?: Record<string, unknown> & {};
};

type ResultEntry = {
  patent_id: string;
  title?: string;
  summary?: string;
  classification_ipc?: string[];
  graph_score?: number;
};

const POLL_INTERVAL_MS = 3000;

const App = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [results, setResults] = useState<ResultEntry[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [pollActive, setPollActive] = useState(false);

  const candidateCount = useMemo(() => {
    if (!jobStatus?.detail) return undefined;
    const detail = jobStatus.detail as Record<string, unknown>;
    return detail["candidate_count"] as number | undefined;
  }, [jobStatus]);

  const stagesDetail = useMemo(() => {
    return (jobStatus?.detail as Record<string, unknown> | undefined) ?? {};
  }, [jobStatus]);

  const axCandidate = useMemo(() => (results.length > 0 ? results[0] : null), [results]);
  const ayCandidates = useMemo(() => (results.length > 1 ? results.slice(1) : []), [results]);

  const reset = () => {
    setJobId(null);
    setJobStatus(null);
    setResults([]);
    setError(null);
    setSuccessMsg(null);
    setPollActive(false);
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files || event.target.files.length === 0) {
      setSelectedFile(null);
      return;
    }
    setSelectedFile(event.target.files[0]);
  };

  const fetchStatus = useCallback(async () => {
    if (!jobId) return;
    try {
      const res = await fetch(`${API_BASE_URL}/status/${jobId}`);
      if (res.status === 404) {
        setError("ジョブが見つかりませんでした。");
        return;
      }
      const payload = await res.json();
      setJobStatus(payload);
      if (payload.status === "completed") {
        setSuccessMsg("処理が完了しました。結果を取得しています...");
        setPollActive(false);
        const resultRes = await fetch(`${API_BASE_URL}/result/${jobId}`);
        if (resultRes.status === 200) {
          const data = await resultRes.json();
          setResults(data.results ?? []);
          setSuccessMsg("結果を取得しました。");
        } else if (resultRes.status !== 202) {
          setError("結果取得時にエラーが発生しました。");
        }
      } else if (payload.status === "failed") {
        setError("ジョブが失敗しました。詳細を確認してください。");
        setPollActive(false);
      }
    } catch (err) {
      console.error(err);
      setError("ステータス確認時にエラーが発生しました。");
      setPollActive(false);
    }
  }, [jobId]);

  useEffect(() => {
    if (!jobId || !pollActive) return;
    fetchStatus();
    const timer = window.setInterval(fetchStatus, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [jobId, pollActive, fetchStatus]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try {
      const saved = JSON.parse(raw) as {
        jobId?: string;
        jobStatus?: JobStatus | null;
        results?: ResultEntry[];
      };
      if (saved.jobId) {
        setJobId(saved.jobId);
        if (saved.jobStatus) {
          setJobStatus(saved.jobStatus);
        }
        if (saved.results) {
          setResults(saved.results);
        }
        setPollActive(true);
      }
    } catch (err) {
      console.error("failed to hydrate job state", err);
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!jobId) {
      window.localStorage.removeItem(STORAGE_KEY);
      return;
    }
    const payload = {
      jobId,
      jobStatus,
      results,
    };
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (err) {
      console.error("failed to persist job state", err);
    }
  }, [jobId, jobStatus, results]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setSuccessMsg(null);
    setResults([]);

    if (!selectedFile) {
      setError("テキストファイル（.txt）を選択してください。");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    setIsSubmitting(true);
    try {
      const response = await fetch(`${API_BASE_URL}/ingest`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        setError(payload.detail ?? "ジョブ投入に失敗しました。");
        return;
      }

      const payload = await response.json();
      setJobId(payload.job_id);
      setJobStatus({ status: "queued", detail: payload.detail ?? {} });
      setSuccessMsg("ジョブを受け付けました。処理を開始します。");
      setPollActive(true);
    } catch (err) {
      console.error(err);
      setError("ジョブ投入時にエラーが発生しました。");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="page">
      <section className="card hero-card">
        <h1>特許α 先行技術探索モック</h1>
        <p className="muted">
          テキストファイル（XML 含む）をアップロードすると、Stage1 ベクトル検索と Neo4j Graph-RAG を経て A(x) / A(y) の候補を提示します。
        </p>
      </section>

      <section className="card input-card">
        <form onSubmit={handleSubmit}>
          <label className="input-block">
            <span>テキストファイル</span>
            <input type="file" accept=".txt,text/plain" onChange={handleFileChange} />
            {selectedFile && <span className="muted filename">{selectedFile.name}</span>}
          </label>
          <div className="actions">
            <button type="button" className="secondary-btn" onClick={reset}>
              リセット
            </button>
            <button type="submit" className="primary-btn" disabled={!selectedFile || isSubmitting}>
              {isSubmitting ? "送信中..." : "解析を実行"}
            </button>
          </div>
        </form>

        {jobId && (
          <section className="status-block">
            <header className="status-header">
              <div>
                <span className="status-label">ジョブ ID</span>
                <p className="status-value">{jobId}</p>
              </div>
              <span className="status-pill">現在: {jobStatus?.status ?? "queued"}</span>
            </header>
            <StageProgress stages={STAGES} detail={stagesDetail as any} />
            {candidateCount !== undefined && (
              <div className="meta-grid">
                <div className="meta-row">
                  <span>Cosmos 戻り件数</span>
                  <span>{candidateCount.toLocaleString()} 件</span>
                </div>
              </div>
            )}
          </section>
        )}

        {error && <div className="toast error-banner">{error}</div>}
        {successMsg && <div className="toast success-banner">{successMsg}</div>}
      </section>

      {results.length > 0 && (
        <div className="results-stack">
          {axCandidate && (
            <section className="section-card ax-card">
              <header className="section-title">
                <span className="badge-ax">A(x)</span>
                {axCandidate.graph_score !== undefined && (
                  <span className="score-tag">Score {axCandidate.graph_score.toFixed(2)}</span>
                )}
              </header>
              <h3 className="candidate-title">{axCandidate.title ?? "タイトル未設定"}</h3>
              <div className="candidate-meta">
                <div>
                  <span className="label">Patent ID</span>
                  <p>{axCandidate.patent_id}</p>
                </div>
              </div>
              {axCandidate.classification_ipc && axCandidate.classification_ipc.length > 0 && (
                <div className="ipc-chips">
                  {axCandidate.classification_ipc.slice(0, 6).map((ipc) => (
                    <span key={ipc} className="chip">
                      {ipc}
                    </span>
                  ))}
                </div>
              )}
              <div className="summary-card">
                <span className="label">要約</span>
                <p>{axCandidate.summary ?? "要約情報がありません。"}</p>
              </div>
            </section>
          )}

          <section className="section-card ay-card">
            <header className="section-title">
              <span className="badge-ay">A(y)</span>
              <span className="muted">{ayCandidates.length} 件</span>
            </header>
            <div className="ay-list">
              {ayCandidates.map((candidate, index) => (
                <article key={candidate.patent_id ?? index} className="ay-item">
                  <header className="ay-item__header">
                    <h4>{candidate.title ?? "タイトル未設定"}</h4>
                    {candidate.graph_score !== undefined && (
                      <span className="score-tag small">Score {candidate.graph_score.toFixed(2)}</span>
                    )}
                  </header>
                  <div className="ay-meta">
                    <div>
                      <span className="label">Patent ID</span>
                      <p>{candidate.patent_id}</p>
                    </div>
                  </div>
                  {candidate.classification_ipc && candidate.classification_ipc.length > 0 && (
                    <div className="ipc-chips">
                      {candidate.classification_ipc.slice(0, 5).map((ipc) => (
                        <span key={ipc} className="chip chip-soft">
                          {ipc}
                        </span>
                      ))}
                    </div>
                  )}
                  <p className="summary-card summary-inline">
                    {(candidate.summary ?? "要約情報がありません。").slice(0, 260)}
                    {candidate.summary && candidate.summary.length > 260 ? "…" : ""}
                  </p>
                </article>
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  );
};

export default App;

const buildTerms = (entry: ResultEntry) => {
  const terms = new Set<string>();

  const addText = (text?: string) => {
    if (!text) return;
    extractTokens(text).forEach((token) => {
      if (token.length > 1) {
        terms.add(token);
      }
    });
  };

  addText(entry.summary);
  entry.classification_ipc?.forEach((code) => terms.add(code));

  return Array.from(terms).slice(0, 10);
};

const extractTokens = (text: string) =>
  (text.match(/[A-Za-z0-9一-龯ぁ-んァ-ンー]+/g) ?? []).map((token) => token.trim());

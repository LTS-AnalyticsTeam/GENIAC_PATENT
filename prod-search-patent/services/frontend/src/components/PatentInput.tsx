// frontend/src/components/PatentInput.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Upload, FileText, Plus, Trash2, Send, ExternalLink, FileText as FileIcon } from "lucide-react";
import "./PatentInput.css";
import StageProgress from "./StageProgress";
import { StageDetail } from "../types";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ??
  import.meta.env.VITE_API_BASE ??
  "http://localhost:8080";
const STORAGE_KEY = "ps-job-state";
const RESULT_CACHE_PREFIX = "ps-result-cache";
const POLL_INTERVAL_MS = 3000;

const STAGES = [
  { id: "parsing", label: "① テキスト解析" },
  { id: "cosmos_query", label: "② Cosmos検索" },
  { id: "trimming", label: "③ トリミング" },
  { id: "embedding", label: "④ 埋め込み生成" },
  { id: "stage1_indexing", label: "⑤ Stage1インデックス" },
  { id: "vector_search", label: "⑥ ベクトル検索" },
  { id: "stage2_indexing", label: "⑦ Neo4j登録" },
  { id: "graph_rag", label: "⑧ Graph-RAG" },
  { id: "analysis", label: "⑨ 特許分析" },
];

interface PatentSlot {
  id: string;
  file: File | null;
}

type JobResultPayload = {
  job_id: string;
  completed_at: string;
  results: Array<{ patent_id: string; analysis_status?: string }>;
  pipeline_stats: Record<string, unknown>;
};

interface JobInfo {
  slotId: string;
  jobId: string;
  fileName: string;
  status: string;
  detail?: StageDetail;
  pollActive: boolean;
  candidateCount?: number;
  resultPayload?: JobResultPayload;
  error?: string;
}

const resultCacheKey = (jobId: string) => `${RESULT_CACHE_PREFIX}:${jobId}`;

const cacheResultPayload = (jobId: string, payload: JobResultPayload) => {
  if (typeof window === "undefined") return;
  try {
    sessionStorage.setItem(resultCacheKey(jobId), JSON.stringify(payload));
  } catch (err) {
    console.warn("Failed to cache result payload", err);
  }
};

const openResultsTab = (jobId: string) => {
  if (typeof window === "undefined") return;
  const url = `${window.location.origin}/results?jobId=${jobId}`;
  window.open(url, "_blank", "noopener,noreferrer");
};

const persistJobs = (jobList: JobInfo[]) => {
  if (jobList.length === 0) {
    localStorage.removeItem(STORAGE_KEY);
    return;
  }
  const payload = {
    jobs: jobList.map((job) => ({
      slotId: job.slotId,
      jobId: job.jobId,
      fileName: job.fileName,
    })),
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
};

const PatentInput: React.FC = () => {
  const [patents, setPatents] = useState<PatentSlot[]>([
    { id: "patent_1", file: null },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const jobsRef = useRef<JobInfo[]>([]);

  // 特許を追加（最大5件）
  const addPatent = () => {
    if (patents.length < 5) {
      setPatents([...patents, { id: `patent_${patents.length + 1}`, file: null }]);
    }
  };

  // 特許を削除
  const removePatent = (index: number) => {
    if (patents.length > 1) {
      setPatents(patents.filter((_, i) => i !== index));
    }
  };

  // ファイルアップロード処理
  const handleFileUpload = async (index: number, file: File) => {
    if (file && file.type === "text/plain") {
      const updated = [...patents];
      updated[index].file = file;
      setPatents(updated);
    } else {
      alert(".txt (XML) ファイルを選択してください");
    }
  };

  const resetAll = async () => {
    const jobIds = Array.from(new Set(jobs.map((job) => job.jobId)));
    if (jobIds.length > 0) {
      const results = await Promise.all(
        jobIds.map(async (jobId) => {
          try {
            const res = await fetch(`${API_BASE}/cancel/${jobId}`, { method: "POST" });
            if (!res.ok) {
              console.error("Failed to cancel job", jobId, await res.text());
              return false;
            }
            return true;
          } catch (error) {
            console.error("Failed to cancel job", jobId, error);
            return false;
          }
        })
      );
      if (results.some((ok) => !ok)) {
        alert("一部のジョブのキャンセルに失敗しました。数秒後に再度ご確認ください。");
      }
    }
    setPatents([{ id: "patent_1", file: null }]);
    setJobs([]);
    persistJobs([]);
    localStorage.removeItem(STORAGE_KEY);
  };

  const handleBatchAnalysis = async () => {
    const textFiles = patents.filter((p) => p.file !== null) as {
      id: string;
      file: File;
    }[];

    if (textFiles.length === 0) {
      alert("少なくとも1つのテキストファイルをアップロードしてください");
      return;
    }

    const newJobs: JobInfo[] = [];

    setIsLoading(true);
    try {
      for (const slot of textFiles) {
        const formData = new FormData();
        formData.append("file", slot.file);
        const response = await fetch(`${API_BASE}/ingest-text`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          alert(`${slot.file.name}: ${payload.detail ?? "ジョブ投入に失敗しました"}`);
          continue;
        }

        const payload = await response.json();
        newJobs.push({
          slotId: slot.id,
          jobId: payload.job_id,
          fileName: slot.file.name,
          status: "queued",
          detail: payload.detail ?? {},
          pollActive: true,
        });
      }

      if (newJobs.length > 0) {
        setJobs((prev) => [...newJobs, ...prev]);
        alert(`${newJobs.length} 件のジョブを受け付けました。進捗は下部カードで確認できます。`);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("エラーが発生しました");
    } finally {
      setIsLoading(false);
    }
  };

  const updateJobsFromStatus = async (jobList: JobInfo[]) => {
    const updates: Record<string, Partial<JobInfo>> = {};
    for (const job of jobList) {
      try {
        const res = await fetch(`${API_BASE}/status/${job.jobId}`);
        if (res.status === 404) {
          updates[job.jobId] = { pollActive: false, error: "ジョブが見つかりません" };
          continue;
        }
        const payload = await res.json();
        const detail = payload.detail as StageDetail | undefined;
        const errorMessage =
          payload.status === "failed"
            ? payload.detail?.message ?? "処理に失敗しました"
            : payload.status === "cancelled"
            ? payload.detail?.message ?? "ユーザーによってキャンセルされました"
            : job.error;
        const baseUpdate: Partial<JobInfo> = {
          status: payload.status,
          detail: payload.detail,
          candidateCount: detail?.candidate_count ?? job.candidateCount,
          error: errorMessage,
        };

        if (payload.status === "cancelled") {
          baseUpdate.pollActive = false;
        } else if (payload.status === "failed") {
          baseUpdate.pollActive = false;
        } else if (payload.status === "completed") {
          const resultRes = await fetch(`${API_BASE}/result/${job.jobId}`);
          if (resultRes.status === 200) {
            const resultPayload = await resultRes.json();
            baseUpdate.resultPayload = resultPayload;
            baseUpdate.pollActive = false;
          } else if (resultRes.status === 202) {
            baseUpdate.pollActive = true;
          } else {
            baseUpdate.pollActive = true;
            baseUpdate.error = "結果取得に失敗しました";
          }
        } else {
          baseUpdate.pollActive = true;
        }
        updates[job.jobId] = baseUpdate;
      } catch (err) {
        console.error("Failed to fetch status", err);
      }
    }

    if (Object.keys(updates).length > 0) {
      setJobs((prev) =>
        prev.map((item) => (updates[item.jobId] ? { ...item, ...updates[item.jobId] } : item))
      );
    }
  };

  const statusBadgeClass = (status: string) => {
    if (status === "completed") return "status-badge success";
    if (status === "failed") return "status-badge danger";
    if (status === "cancelled") return "status-badge cancelled";
    return "status-badge process";
  };

  const handleViewResult = async (job: JobInfo) => {
    if (job.status !== "completed") return;
    const launchResultView = (payload: JobResultPayload) => {
      cacheResultPayload(job.jobId, payload);
      openResultsTab(job.jobId);
    };
    if (!job.resultPayload) {
      try {
        const res = await fetch(`${API_BASE}/result/${job.jobId}`);
        if (res.status === 200) {
          const payload = await res.json();
          setJobs((prev) =>
            prev.map((item) => (item.jobId === job.jobId ? { ...item, resultPayload: payload } : item))
          );
          launchResultView(payload);
        }
      } catch (err) {
        console.error("Failed to fetch result", err);
        alert("結果取得に失敗しました");
      }
      return;
    }
    launchResultView(job.resultPayload);
  };

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return;
    try {
      const parsed = JSON.parse(stored) as { jobs?: Array<{ slotId: string; jobId: string; fileName: string }> };
      if (parsed.jobs && Array.isArray(parsed.jobs) && parsed.jobs.length > 0) {
        const restored = parsed.jobs.map<JobInfo>((meta) => ({
          slotId: meta.slotId,
          jobId: meta.jobId,
          fileName: meta.fileName,
          status: "queued",
          pollActive: true,
        }));
        setJobs(restored);
        updateJobsFromStatus(restored);
      }
    } catch (err) {
      console.error("Failed to restore jobs", err);
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    jobsRef.current = jobs;
    persistJobs(jobs);
  }, [jobs]);

  useEffect(() => {
    const timer = window.setInterval(async () => {
      const activeJobs = jobsRef.current.filter((job) => job.pollActive);
      if (activeJobs.length === 0) {
        return;
      }
      updateJobsFromStatus(activeJobs);
    }, POLL_INTERVAL_MS);

    return () => window.clearInterval(timer);
  }, []);

  const jobCountText = useMemo(() => {
    const active = jobs.filter((job) => job.pollActive).length;
    const completed = jobs.filter((job) => job.status === "completed").length;
    return `${active} 実行中・${completed} 完了`;
  }, [jobs]);

  const isResultReady = (job: JobInfo) => job.status === "completed" && Boolean(job.resultPayload);

  return (
    <div className="patent-input-container">
      <div className="header-section">
        <h1>特許分析システム</h1>
        <p className="subtitle">JSONファイルをアップロードして特許の新規性・進歩性を分析します</p>
      </div>
      
      {/* 出願特許の入力 */}
      <div className="patents-section">
        <h3>
          <FileText size={24} />
          出願特許（最大5件）
        </h3>
        
        {patents.map((patent, index) => (
          <div key={patent.id} className="patent-input-item">
            <div className="patent-header">
              <span className="patent-label">特許 {index + 1}</span>
              {patents.length > 1 && (
                <button
                  className="remove-btn"
                  onClick={() => removePatent(index)}
                  aria-label="削除"
                >
                  <Trash2 size={16} />
                </button>
              )}
            </div>
            
            <div className="patent-content">
              <div className="file-upload-area">
                <input
                  type="file"
                  accept=".txt"
                  onChange={(e) => e.target.files && handleFileUpload(index, e.target.files[0])}
                  id={`patent-file-${index}`}
                />
                <label htmlFor={`patent-file-${index}`} className="file-upload-label">
                  <Upload size={20} />
                  <span>.txt（XML）ファイルをアップロード</span>
                </label>
                {patent.file && (
                  <div className="file-info">
                    <span className="check-icon">✓</span>
                    <span className="file-name">{patent.file.name}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        
        {patents.length < 5 && (
          <button className="add-patent-btn" onClick={addPatent}>
            <Plus size={20} />
            特許を追加
          </button>
        )}
      </div>

      {/* 分析実行ボタン */}
      <div className="action-section">
        <div className="action-buttons">
          <button className="reset-btn" onClick={resetAll} disabled={isLoading && patents.every((p) => !p.file)}>
            リセット
          </button>
          <button
            className="analyze-btn"
            onClick={handleBatchAnalysis}
            disabled={isLoading || patents.every((p) => !p.file)}
          >
            {isLoading ? (
              <span className="loading">
                <span className="spinner"></span>
                処理中...
              </span>
            ) : (
              <>
                <Send size={20} />
                分析を実行
              </>
            )}
          </button>
        </div>
      </div>

      {jobs.length > 0 && (
        <section className="job-section">
          <div className="job-section__header">
            <h3>処理ステータス</h3>
            <span className="job-count">{jobCountText}</span>
          </div>
          <div className="job-grid">
            {jobs.map((job) => (
              <article className="job-card" key={job.jobId}>
                <header className="job-card__header">
                  <div className="job-card__title">
                    <FileIcon size={18} />
                    <div>
                      <p className="job-file">{job.fileName}</p>
                      <small>ID: {job.jobId}</small>
                    </div>
                  </div>
                  <span className={statusBadgeClass(job.status)}>{job.status}</span>
                </header>
                <StageProgress stages={STAGES} detail={job.detail} />
                <div className="job-meta">
                  {job.candidateCount !== undefined && (
                    <div>
                      <span>Cosmos 戻り件数</span>
                      <strong>{job.candidateCount.toLocaleString()} 件</strong>
                    </div>
                  )}
                  {job.detail?.current_stage && (
                    <div>
                      <span>現在ステージ</span>
                      <strong>{job.detail.current_stage}</strong>
                    </div>
                  )}
                </div>
                {job.error && <p className="job-error">{job.error}</p>}
                <div className="job-actions">
                  <button
                    className="result-btn"
                    disabled={!isResultReady(job)}
                    onClick={() => handleViewResult(job)}
                  >
                    <ExternalLink size={18} />
                    {job.status === "failed"
                      ? "失敗"
                      : job.status === "cancelled"
                      ? "キャンセル済み"
                      : isResultReady(job)
                      ? "結果を表示"
                      : job.status === "completed"
                      ? "解析集計中"
                      : "解析中"}
                  </button>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
};

export default PatentInput;

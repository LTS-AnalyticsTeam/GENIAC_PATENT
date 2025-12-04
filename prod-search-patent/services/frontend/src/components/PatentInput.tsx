// frontend/src/components/PatentInput.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Upload,
  FileText,
  Plus,
  Trash2,
  Send,
  ExternalLink,
  FileText as FileIcon,
  X,
} from "lucide-react";
import "./PatentInput.css";
import StageProgress from "./StageProgress";
import { StageDetail, SearchResultItem } from "../types";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ??
  import.meta.env.VITE_API_BASE ??
  "http://localhost:8080";
const STORAGE_KEY = "ps-job-state";
const RESULT_CACHE_PREFIX = "ps-result-cache";
const POLL_INTERVAL_MS = 3000;

const STAGES = [
  { id: "parsing", label: "① テキスト解析" },
  { id: "keyword_search", label: "② キーワード検索" },
  { id: "embedding", label: "③ 埋め込み生成" },
  { id: "vector_search", label: "④ ベクトル検索" },
  { id: "stage2_indexing", label: "⑤ Graph登録" },
  { id: "graph_rag", label: "⑥ Graph検索" },
  { id: "analysis", label: "⑦ 特許分析" },
];

// ステージIDを日本語ラベルに変換
const STAGE_LABELS: Record<string, string> = {
  parsing: "① テキスト解析",
  keyword_search: "② キーワード検索",
  embedding: "③ 埋め込み生成",
  stage1_indexing: "④ ベクトル検索",
  vector_search: "④ ベクトル検索",
  stage2_indexing: "⑤ Graph登録",
  graph_rag: "⑥ Graph検索",
  analysis: "⑦ 特許分析",
};

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

// 特許番号を統一フォーマット（JP{number}A）に変換する関数
const formatPatentNumber = (patentId: string): string => {
  if (!patentId) return "";
  // 既にJPで始まりAで終わる場合はそのまま返す
  if (patentId.startsWith("JP") && patentId.endsWith("A")) {
    return patentId;
  }
  // 数字のみの場合はJP{number}Aの形式にする
  if (/^\d+$/.test(patentId)) {
    return `JP${patentId}A`;
  }
  // その他の場合はそのまま返す（WO番号など）
  return patentId;
};

const PatentInput: React.FC = () => {
  const [patents, setPatents] = useState<PatentSlot[]>([
    { id: "patent_1", file: null },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const jobsRef = useRef<JobInfo[]>([]);

  // キーワード検索結果モーダル用state
  const [showPatentListModal, setShowPatentListModal] = useState(false);
  const [patentListData, setPatentListData] = useState<{
    jobId: string;
    patentIds: string[];
    searchResults: SearchResultItem[];
    totalCount: number;
    pipelineStats: Record<string, unknown>;
  } | null>(null);
  // ベクトル検索結果モーダル用state
  const [showVectorListModal, setShowVectorListModal] = useState(false);
  const [vectorListData, setVectorListData] = useState<{
    jobId: string;
    topPatentIds: string[];
    uniqueHits?: number;
    generatedQueries?: string[];
    hitsPerQuery?: number[];
  } | null>(null);
  // Graph検索結果モーダル用state
  const [showGraphListModal, setShowGraphListModal] = useState(false);
  const [graphListData, setGraphListData] = useState<{
    jobId: string;
    results: Array<{
      patent_id: string;
      title?: string;
      graph_score?: number;
      vector_score?: number;
    }>;
    totalCount?: number;
  } | null>(null);

  // キーワード検索結果を取得してモーダルを表示
  const handleShowPatentList = async (jobId: string) => {
    try {
      const res = await fetch(`${API_BASE}/keyword-search-result/${jobId}`);
      if (res.ok) {
        const data = await res.json();
        console.log("Keyword search result data:", data);
        console.log("Search results with scores:", data.search_results);
        setPatentListData({
          jobId: data.job_id,
          patentIds: data.patent_ids,
          searchResults: data.search_results || [],
          totalCount: data.total_count,
          pipelineStats: data.pipeline_stats,
        });
        setShowPatentListModal(true);
      } else {
        alert("キーワード検索結果の取得に失敗しました");
      }
    } catch (err) {
      console.error("Failed to fetch keyword search result", err);
      alert("キーワード検索結果の取得に失敗しました");
    }
  };

  const readStageDetail = (
    detail: StageDetail | undefined,
    stageId: string
  ): Record<string, unknown> | undefined => {
    return (detail?.stages?.[stageId] as Record<string, unknown>) ?? undefined;
  };

  const handleShowVectorList = (jobId: string, detail?: StageDetail) => {
    const stageDetail = readStageDetail(detail, "vector_search");
    if (!stageDetail) {
      alert("ベクトル検索結果がまだありません。");
      return;
    }
    const topPatentIds = Array.isArray(stageDetail.vector_patent_ids)
      ? (stageDetail.vector_patent_ids as unknown[]).map(String)
      : Array.isArray(stageDetail.top_patent_ids)
      ? (stageDetail.top_patent_ids as unknown[]).map(String)
      : [];
    const generatedQueries = Array.isArray(stageDetail.generated_queries)
      ? (stageDetail.generated_queries as unknown[]).map(String)
      : undefined;
    const hitsPerQuery = Array.isArray(stageDetail.hits_per_query)
      ? (stageDetail.hits_per_query as unknown[]).map((n) =>
          typeof n === "number" ? n : Number(n)
        )
      : undefined;

    setVectorListData({
      jobId,
      topPatentIds,
      uniqueHits:
        typeof stageDetail.unique_hits === "number"
          ? stageDetail.unique_hits
          : undefined,
      generatedQueries,
      hitsPerQuery,
    });
    setShowVectorListModal(true);
  };

  const handleShowGraphList = (
    jobId: string,
    detail?: StageDetail,
    resultPayload?: JobResultPayload
  ) => {
    const stageDetail = readStageDetail(detail, "graph_rag");
    const fromStage = Array.isArray(stageDetail?.graph_patent_results)
      ? (stageDetail?.graph_patent_results as Array<Record<string, unknown>>).map(
          (item) => ({
            patent_id: String(item.patent_id ?? item.patentId ?? ""),
            title: typeof item.title === "string" ? item.title : undefined,
            graph_score:
              typeof item.graph_score === "number"
                ? item.graph_score
                : undefined,
            vector_score:
              typeof item.vector_score === "number"
                ? item.vector_score
                : undefined,
          })
        )
      : Array.isArray(stageDetail?.graph_top_results)
      ? (stageDetail?.graph_top_results as Array<Record<string, unknown>>).map(
          (item) => ({
            patent_id: String(item.patent_id ?? item.patentId ?? ""),
            title: typeof item.title === "string" ? item.title : undefined,
            graph_score:
              typeof item.graph_score === "number"
                ? item.graph_score
                : undefined,
            vector_score:
              typeof item.vector_score === "number"
                ? item.vector_score
                : undefined,
          })
        )
      : null;

    const fromResults =
      resultPayload && Array.isArray(resultPayload.results)
        ? resultPayload.results.map((item) => ({
            patent_id: String(item.patent_id),
            title:
              typeof (item as Record<string, unknown>).title === "string"
                ? ((item as Record<string, unknown>).title as string)
                : undefined,
            graph_score:
              typeof (item as Record<string, unknown>).graph_score === "number"
                ? ((item as Record<string, unknown>).graph_score as number)
                : undefined,
            vector_score:
              typeof (item as Record<string, unknown>).vector_score === "number"
                ? ((item as Record<string, unknown>).vector_score as number)
                : undefined,
          }))
        : null;

    const results = fromStage ?? fromResults ?? [];
    if (results.length === 0) {
      alert("Graph検索結果がまだありません。");
      return;
    }

    setGraphListData({
      jobId,
      results,
      totalCount:
        typeof stageDetail?.graph_results === "number"
          ? stageDetail.graph_results
          : results.length,
    });
    setShowGraphListModal(true);
  };

  // 特許を追加（最大30件）
  const addPatent = () => {
    if (patents.length < 30) {
      setPatents([
        ...patents,
        { id: `patent_${patents.length + 1}`, file: null },
      ]);
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
            const res = await fetch(`${API_BASE}/cancel/${jobId}`, {
              method: "POST",
            });
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
        alert(
          "一部のジョブのキャンセルに失敗しました。数秒後に再度ご確認ください。"
        );
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
          alert(
            `${slot.file.name}: ${payload.detail ?? "ジョブ投入に失敗しました"}`
          );
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
        alert(
          `${newJobs.length} 件のジョブを受け付けました。進捗は下部カードで確認できます。`
        );
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
          updates[job.jobId] = {
            pollActive: false,
            error: "ジョブが見つかりません",
          };
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
          candidateCount: detail?.narrowed_count ?? job.candidateCount,
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
            // 完了済みジョブでもcandidateCountを設定（pipeline_statsから取得）
            if (!baseUpdate.candidateCount && resultPayload.pipeline_stats) {
              baseUpdate.candidateCount = resultPayload.pipeline_stats.stage2_keyword_filter_results;
            }
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
        prev.map((item) =>
          updates[item.jobId] ? { ...item, ...updates[item.jobId] } : item
        )
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
            prev.map((item) =>
              item.jobId === job.jobId
                ? { ...item, resultPayload: payload }
                : item
            )
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
      const parsed = JSON.parse(stored) as {
        jobs?: Array<{ slotId: string; jobId: string; fileName: string }>;
      };
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

  const isResultReady = (job: JobInfo) =>
    job.status === "completed" && Boolean(job.resultPayload);

  return (
    <div className="patent-input-container">
      <div className="header-section">
        <h1>特許分析システム</h1>
        <p className="subtitle">
          XMLファイルをアップロードして特許の新規性・進歩性を分析します
        </p>
      </div>

      {/* 出願特許の入力 */}
      <div className="patents-section">
        <h3>
          <FileText size={24} />
          出願特許（最大30件）
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
                  onChange={(e) =>
                    e.target.files && handleFileUpload(index, e.target.files[0])
                  }
                  id={`patent-file-${index}`}
                />
                <label
                  htmlFor={`patent-file-${index}`}
                  className="file-upload-label"
                >
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

        {patents.length < 30 && (
          <button className="add-patent-btn" onClick={addPatent}>
            <Plus size={20} />
            特許を追加
          </button>
        )}
      </div>

      {/* 分析実行ボタン */}
      <div className="action-section">
        <div className="action-buttons">
          <button
            className="reset-btn"
            onClick={resetAll}
            disabled={isLoading && patents.every((p) => !p.file)}
          >
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
                  <span className={statusBadgeClass(job.status)}>
                    {job.status}
                  </span>
                </header>
                <StageProgress
                  stages={STAGES}
                  detail={job.detail}
                  onShowPatentList={() => handleShowPatentList(job.jobId)}
                  keywordSearchCount={job.candidateCount}
                  onShowVectorList={() =>
                    handleShowVectorList(job.jobId, job.detail)
                  }
                  onShowGraphList={() =>
                    handleShowGraphList(job.jobId, job.detail, job.resultPayload)
                  }
                />
                <div className="job-meta">
                  {job.detail?.current_stage && (
                    <div>
                      <span>現在ステージ</span>
                      <strong>
                        {STAGE_LABELS[job.detail.current_stage] ||
                          job.detail.current_stage}
                      </strong>
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

      {/* キーワード検索結果モーダル */}
      {showPatentListModal && patentListData && (
        <div
          className="modal-overlay"
          onClick={() => setShowPatentListModal(false)}
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
          }}
        >
          <div
            className="modal-content"
            onClick={(e: React.MouseEvent) => e.stopPropagation()}
            style={{
              backgroundColor: "white",
              borderRadius: "8px",
              padding: "24px",
              maxWidth: "600px",
              width: "90%",
              maxHeight: "80vh",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "16px",
              }}
            >
              <h3 style={{ margin: 0, fontSize: "18px" }}>
                キーワード検索結果 ({patentListData.totalCount.toLocaleString()}
                件)
              </h3>
              <button
                onClick={() => setShowPatentListModal(false)}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  padding: "4px",
                }}
              >
                <X size={20} />
              </button>
            </div>
            <div
              style={{ marginBottom: "12px", fontSize: "12px", color: "#666" }}
            >
              <p style={{ margin: "4px 0" }}>
                STAGE1 IPC候補:{" "}
                {(
                  patentListData.pipelineStats.stage1_IPC_candidates as number
                )?.toLocaleString() ?? "-"}
                件
              </p>
              <p style={{ margin: "4px 0" }}>
                STAGE2 キーワード絞込:{" "}
                {(
                  patentListData.pipelineStats
                    .stage2_keyword_filter_results as number
                )?.toLocaleString() ?? "-"}
                件
              </p>
            </div>
            <div
              style={{
                flex: 1,
                overflowY: "auto",
                border: "1px solid #e5e7eb",
                borderRadius: "4px",
                padding: "8px",
                fontFamily: "monospace",
                fontSize: "12px",
              }}
            >
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr
                    style={{
                      position: "sticky",
                      top: 0,
                      backgroundColor: "#f9fafb",
                    }}
                  >
                    <th
                      style={{
                        padding: "8px 4px",
                        textAlign: "left",
                        borderBottom: "1px solid #e5e7eb",
                        width: "60px",
                      }}
                    >
                      順位
                    </th>
                    <th
                      style={{
                        padding: "8px 4px",
                        textAlign: "left",
                        borderBottom: "1px solid #e5e7eb",
                      }}
                    >
                      特許番号
                    </th>
                    <th
                      style={{
                        padding: "8px 4px",
                        textAlign: "right",
                        borderBottom: "1px solid #e5e7eb",
                        width: "100px",
                      }}
                    >
                      スコア
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {patentListData.searchResults && patentListData.searchResults.length > 0 ? (
                    patentListData.searchResults.map(
                      (result: SearchResultItem, index: number) => (
                        <tr
                          key={result.doc_number}
                          style={{ borderBottom: "1px solid #f3f4f6" }}
                        >
                          <td style={{ padding: "4px", color: "#6b7280" }}>
                            {index + 1}
                          </td>
                          <td style={{ padding: "4px" }}>{formatPatentNumber(result.doc_number)}</td>
                          <td style={{ padding: "4px", textAlign: "right", fontWeight: "500" }}>
                            {result.score.toFixed(1)}
                          </td>
                        </tr>
                      )
                    )
                  ) : (
                    patentListData.patentIds.map(
                      (patentId: string, index: number) => (
                        <tr
                          key={patentId}
                          style={{ borderBottom: "1px solid #f3f4f6" }}
                        >
                          <td style={{ padding: "4px", color: "#6b7280" }}>
                            {index + 1}
                          </td>
                          <td style={{ padding: "4px" }}>{formatPatentNumber(patentId)}</td>
                          <td style={{ padding: "4px", textAlign: "right", color: "#9ca3af" }}>
                            -
                          </td>
                        </tr>
                      )
                    )
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ベクトル検索結果モーダル */}
      {showVectorListModal && vectorListData && (
        <div
          className="modal-overlay"
          onClick={() => setShowVectorListModal(false)}
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
          }}
        >
          <div
            className="modal-content"
            onClick={(e: React.MouseEvent) => e.stopPropagation()}
            style={{
              backgroundColor: "white",
              borderRadius: "8px",
              padding: "24px",
              maxWidth: "760px",
              width: "95%",
              maxHeight: "80vh",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "16px",
              }}
            >
              <div>
                <h3 style={{ margin: 0, fontSize: "18px" }}>
                  ベクトル検索結果{" "}
                  ({(vectorListData.uniqueHits ?? vectorListData.topPatentIds.length).toLocaleString()}
                  件)
                </h3>
                <p style={{ margin: "4px 0", color: "#6b7280", fontSize: "12px" }}>
                  ジョブID: {vectorListData.jobId}
                </p>
              </div>
              <button
                onClick={() => setShowVectorListModal(false)}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  padding: "4px",
                }}
              >
                <X size={20} />
              </button>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                gap: "8px",
                marginBottom: "12px",
                fontSize: "12px",
                color: "#444",
              }}
            >
              <div>
                <strong>ユニークヒット数</strong>
                <div>{vectorListData.uniqueHits ?? vectorListData.topPatentIds.length} 件</div>
              </div>
              {vectorListData.generatedQueries && (
                <div>
                  <strong>生成クエリ</strong>
                  <div>{vectorListData.generatedQueries.length} 件</div>
                </div>
              )}
            </div>

            {vectorListData.generatedQueries && (
              <div
                style={{
                  marginBottom: "12px",
                  padding: "8px",
                  background: "#f9fafb",
                  border: "1px solid #e5e7eb",
                  borderRadius: "4px",
                  fontSize: "12px",
                }}
              >
                <strong style={{ display: "block", marginBottom: "4px" }}>
                  生成クエリ一覧
                </strong>
                <ul style={{ paddingLeft: "16px", margin: 0 }}>
                  {vectorListData.generatedQueries.map((q, idx) => (
                    <li key={`${q}-${idx}`} style={{ marginBottom: "4px" }}>
                      {idx + 1}. {q}
                      {vectorListData.hitsPerQuery &&
                        typeof vectorListData.hitsPerQuery[idx] === "number" && (
                          <span style={{ color: "#6b7280", marginLeft: "6px" }}>
                            ({vectorListData.hitsPerQuery[idx]}件ヒット)
                          </span>
                        )}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div
              style={{
                flex: 1,
                overflowY: "auto",
                border: "1px solid #e5e7eb",
                borderRadius: "4px",
                padding: "8px",
                fontFamily: "monospace",
                fontSize: "12px",
              }}
            >
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr
                    style={{
                      position: "sticky",
                      top: 0,
                      backgroundColor: "#f9fafb",
                    }}
                  >
                    <th
                      style={{
                        padding: "8px 4px",
                        textAlign: "left",
                        borderBottom: "1px solid #e5e7eb",
                      }}
                    >
                      順位
                    </th>
                    <th
                      style={{
                        padding: "8px 4px",
                        textAlign: "left",
                        borderBottom: "1px solid #e5e7eb",
                      }}
                    >
                      特許ID
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {vectorListData.topPatentIds.map(
                    (patentId: string, index: number) => (
                      <tr
                        key={`${patentId}-${index}`}
                        style={{ borderBottom: "1px solid #f3f4f6" }}
                      >
                        <td style={{ padding: "4px", color: "#6b7280" }}>
                          {index + 1}
                        </td>
                        <td style={{ padding: "4px" }}>{formatPatentNumber(patentId)}</td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Graph検索結果モーダル */}
      {showGraphListModal && graphListData && (
        <div
          className="modal-overlay"
          onClick={() => setShowGraphListModal(false)}
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
          }}
        >
          <div
            className="modal-content"
            onClick={(e: React.MouseEvent) => e.stopPropagation()}
            style={{
              backgroundColor: "white",
              borderRadius: "8px",
              padding: "24px",
              maxWidth: "860px",
              width: "95%",
              maxHeight: "80vh",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "16px",
              }}
            >
              <div>
                <h3 style={{ margin: 0, fontSize: "18px" }}>
                  Graph検索結果 (
                  {(graphListData.totalCount ?? graphListData.results.length).toLocaleString()}
                  件)
                </h3>
                <p style={{ margin: "4px 0", color: "#6b7280", fontSize: "12px" }}>
                  ジョブID: {graphListData.jobId}
                </p>
              </div>
              <button
                onClick={() => setShowGraphListModal(false)}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  padding: "4px",
                }}
              >
                <X size={20} />
              </button>
            </div>

            <div
              style={{
                flex: 1,
                overflowY: "auto",
                border: "1px solid #e5e7eb",
                borderRadius: "4px",
                padding: "8px",
                fontFamily: "monospace",
                fontSize: "12px",
              }}
            >
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr
                    style={{
                      position: "sticky",
                      top: 0,
                      backgroundColor: "#f9fafb",
                    }}
                  >
                    <th
                      style={{
                        padding: "8px 4px",
                        textAlign: "left",
                        borderBottom: "1px solid #e5e7eb",
                      }}
                    >
                      順位
                    </th>
                    <th
                      style={{
                        padding: "8px 4px",
                        textAlign: "left",
                        borderBottom: "1px solid #e5e7eb",
                      }}
                    >
                      特許ID
                    </th>
                    <th
                      style={{
                        padding: "8px 4px",
                        textAlign: "left",
                        borderBottom: "1px solid #e5e7eb",
                      }}
                    >
                      GraphScore
                    </th>
                    <th
                      style={{
                        padding: "8px 4px",
                        textAlign: "left",
                        borderBottom: "1px solid #e5e7eb",
                      }}
                    >
                      VectorScore
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {graphListData.results.map((item, index) => (
                    <tr
                      key={`${item.patent_id}-${index}`}
                      style={{ borderBottom: "1px solid #f3f4f6" }}
                    >
                      <td style={{ padding: "4px", color: "#6b7280" }}>
                        {index + 1}
                      </td>
                      <td style={{ padding: "4px" }}>{formatPatentNumber(item.patent_id)}</td>
                      <td style={{ padding: "4px" }}>
                        {item.graph_score !== undefined
                          ? item.graph_score.toFixed(3)
                          : "-"}
                      </td>
                      <td style={{ padding: "4px" }}>
                        {item.vector_score !== undefined
                          ? item.vector_score.toFixed(3)
                          : "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PatentInput;

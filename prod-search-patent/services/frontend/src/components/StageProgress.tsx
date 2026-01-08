type StageMeta = {
  id: string;
  label: string;
};

type StageStatus = "pending" | "in_progress" | "completed";

interface StageProgressProps {
  stages: StageMeta[];
  detail?: {
    current_stage?: string;
    stages?: Record<string, Record<string, unknown>>;
    narrowed_count?: number;
  };
  onShowPatentList?: () => void;
  keywordSearchCount?: number;
  onShowVectorList?: () => void;
  onShowRerankList?: () => void;
  onShowFusionList?: () => void;
}

const statusColor = (status: StageStatus) => {
  switch (status) {
    case "completed":
      return "completed";
    case "in_progress":
      return "active";
    default:
      return "";
  }
};

const resolveStatus = (
  stageId: string,
  detail?: StageProgressProps["detail"]
): StageStatus => {
  const info = detail?.stages?.[stageId] as Record<string, unknown> | undefined;
  const rawStatus =
    typeof info?.status === "string"
      ? (info.status as string)
      : typeof info?.["status"] === "string"
      ? (info["status"] as string)
      : undefined;
  const state = rawStatus as StageStatus | undefined;
  if (state === "completed" || state === "in_progress") {
    return state;
  }
  return "pending";
};

const getStatusText = (status: StageStatus): string => {
  switch (status) {
    case "completed":
      return "完了";
    case "in_progress":
      return "実行中";
    default:
      return "待機中";
  }
};

const StageProgress = ({
  stages,
  detail,
  onShowPatentList,
  keywordSearchCount,
  onShowVectorList,
  onShowRerankList,
  onShowFusionList,
}: StageProgressProps) => {
  const completedCount = stages.filter(
    (stage) => resolveStatus(stage.id, detail) === "completed"
  ).length;
  const progressPercent = Math.floor((completedCount / stages.length) * 100);

  return (
    <div className="progress-wrapper">
      <div className="progress-bar">
        <span className="pill">進捗 {progressPercent}%</span>
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>
      <div className="stage-list">
        {stages.map((stage) => {
          const status = resolveStatus(stage.id, detail);
          const className = ["stage-item", statusColor(status)]
            .filter(Boolean)
            .join(" ");
          const statusText = getStatusText(status);

          // キーワード検索完了時に件数リンクを表示（ジョブ完了後も表示）
          const isKeywordSearchCompleted =
            stage.id === "keyword_search" && status === "completed";
          const keywordStageDetail = detail?.stages
            ?.keyword_search as Record<string, unknown> | undefined;
          const stageNarrowedCount = keywordStageDetail?.narrowed_count as
            | number
            | undefined;
          const narrowedCount =
            keywordSearchCount ?? detail?.narrowed_count ?? stageNarrowedCount;

          const isVectorSearchCompleted =
            stage.id === "vector_search" && status === "completed";
          const vectorStageDetail = detail?.stages
            ?.vector_search as Record<string, unknown> | undefined;
          const vectorHitCount =
            (typeof vectorStageDetail?.unique_hits === "number"
              ? vectorStageDetail.unique_hits
              : Array.isArray(vectorStageDetail?.vector_patent_ids)
              ? (vectorStageDetail?.vector_patent_ids as unknown[]).length
              : Array.isArray(vectorStageDetail?.top_patent_ids)
              ? (vectorStageDetail?.top_patent_ids as unknown[]).length
              : undefined) ?? undefined;

          const isRerankCompleted =
            stage.id === "rerank" && status === "completed";
          const rerankStageDetail = detail?.stages
            ?.rerank as Record<string, unknown> | undefined;
          const rerankResultCount =
            (typeof rerankStageDetail?.rerank_results === "number"
              ? rerankStageDetail.rerank_results
              : Array.isArray(rerankStageDetail?.rerank_patent_results)
              ? (rerankStageDetail?.rerank_patent_results as unknown[]).length
              : Array.isArray(rerankStageDetail?.rerank_top_patent_ids)
              ? (rerankStageDetail?.rerank_top_patent_ids as unknown[]).length
              : undefined) ?? undefined;

          const isFusionCompleted =
            stage.id === "fusion" && status === "completed";
          const fusionStageDetail = detail?.stages
            ?.fusion as Record<string, unknown> | undefined;
          const fusionResultCount =
            (typeof fusionStageDetail?.fusion_top === "number"
              ? fusionStageDetail.fusion_top
              : Array.isArray(fusionStageDetail?.fusion_patent_results)
              ? (fusionStageDetail?.fusion_patent_results as unknown[]).length
              : undefined) ?? undefined;

          return (
            <div key={stage.id} className={className}>
              <span>
                {stage.label}
                {statusText}
                {isKeywordSearchCompleted &&
                  narrowedCount !== undefined &&
                  onShowPatentList && (
                    <a
                      href="#"
                      onClick={(e) => {
                        e.preventDefault();
                        onShowPatentList();
                      }}
                      style={{
                        marginLeft: "8px",
                        color: "#2563eb",
                        textDecoration: "underline",
                        fontSize: "12px",
                      }}
                    >
                      ({narrowedCount.toLocaleString()}件)
                    </a>
                  )}
                {isVectorSearchCompleted &&
                  vectorHitCount !== undefined &&
                  onShowVectorList && (
                    <a
                      href="#"
                      onClick={(e) => {
                        e.preventDefault();
                        onShowVectorList();
                      }}
                      style={{
                        marginLeft: "8px",
                        color: "#2563eb",
                        textDecoration: "underline",
                        fontSize: "12px",
                      }}
                    >
                      ({vectorHitCount.toLocaleString()}件)
                    </a>
                  )}
                {isRerankCompleted &&
                  rerankResultCount !== undefined &&
                  onShowRerankList && (
                    <a
                      href="#"
                      onClick={(e) => {
                        e.preventDefault();
                        onShowRerankList();
                      }}
                      style={{
                        marginLeft: "8px",
                        color: "#2563eb",
                        textDecoration: "underline",
                        fontSize: "12px",
                      }}
                    >
                      ({rerankResultCount.toLocaleString()}件)
                    </a>
                  )}
                {isFusionCompleted &&
                  fusionResultCount !== undefined &&
                  onShowFusionList && (
                    <a
                      href="#"
                      onClick={(e) => {
                        e.preventDefault();
                        onShowFusionList();
                      }}
                      style={{
                        marginLeft: "8px",
                        color: "#2563eb",
                        textDecoration: "underline",
                        fontSize: "12px",
                      }}
                    >
                      ({fusionResultCount.toLocaleString()}件)
                    </a>
                  )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default StageProgress;

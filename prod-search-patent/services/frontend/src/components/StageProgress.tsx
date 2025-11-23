type StageMeta = {
  id: string;
  label: string;
};

type StageStatus = "pending" | "in_progress" | "completed";

interface StageProgressProps {
  stages: StageMeta[];
  detail?: {
    current_stage?: string;
    stages?: Record<string, Record<string, string>>;
    narrowed_count?: number;
  };
  onShowPatentList?: () => void;
  keywordSearchCount?: number;
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
  const info = detail?.stages?.[stageId];
  const state = info?.status as StageStatus | undefined;
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

const StageProgress = ({ stages, detail, onShowPatentList, keywordSearchCount }: StageProgressProps) => {
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
          const isKeywordSearchCompleted = stage.id === "keyword_search" && status === "completed";
          const narrowedCount = keywordSearchCount ?? detail?.narrowed_count;

          return (
            <div key={stage.id} className={className}>
              <span>
                {stage.label}{statusText}
                {isKeywordSearchCompleted && narrowedCount !== undefined && onShowPatentList && (
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
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default StageProgress;

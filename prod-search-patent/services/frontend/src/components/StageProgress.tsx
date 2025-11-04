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
  };
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

const StageProgress = ({ stages, detail }: StageProgressProps) => {
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
          return (
            <div key={stage.id} className={className}>
              <strong>{stage.label}</strong>
              <small>
                {status === "completed" && "完了"}
                {status === "in_progress" && "実行中"}
                {status === "pending" && "待機中"}
              </small>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default StageProgress;

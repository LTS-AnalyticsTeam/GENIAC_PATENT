import { Candidate } from "../types";
import SnippetItem from "./Snippet";
import { highlightText } from "../lib/highlight";

type Props = {
  candidate: Candidate;
  kind: "Ax" | "Ay";
  terms: string[];
};

const ResultCard = ({ candidate, kind, terms }: Props) => {
  const isWeb = Boolean(candidate.source_url);
  const hasPubNumber = Boolean(candidate.pub_number);
  const hasYear = typeof candidate.year === "number";
  const hasIpc = candidate.ipc.length > 0;

  return (
    <div className="result-card">
      <div className="result-card-header">
        <span className={`badge ${kind === "Ax" ? "badge-ax" : "badge-ay"}`}>{kind}</span>
        <span className="score">Score {candidate.score.toFixed(2)}</span>
      </div>

      <h3 className="result-title">{candidate.title}</h3>

      {isWeb ? (
        <div className="web-source">
          <span className="label">参照URL</span>
          {candidate.source_url ? (
            <a href={candidate.source_url} target="_blank" rel="noopener noreferrer">
              {candidate.source_url}
            </a>
          ) : (
            <span className="muted">URL情報なし</span>
          )}
        </div>
      ) : (
        <>
          {(hasPubNumber || hasYear) && (
            <div className="patent-meta">
              {hasPubNumber && (
                <div>
                  <span className="label">特許番号</span>
                  <p>{candidate.pub_number}</p>
                </div>
              )}
              {hasYear && (
                <div>
                  <span className="label">公開年</span>
                  <p>{candidate.year}</p>
                </div>
              )}
            </div>
          )}

          {hasIpc && (
            <div className="ipc-list">
              {candidate.ipc.map((code) => (
                <span key={code} className="chip">
                  {code}
                </span>
              ))}
            </div>
          )}

          <div className="result-section">
            <h4>特許要約</h4>
            <p className="patent-summary">{candidate.explanation.summary}</p>
          </div>
        </>
      )}

      <div className="result-section">
        <h4>参照箇所表示</h4>
        <div className="snippet-list">
          {candidate.snippets.map((snippet, index) => (
            <SnippetItem key={`${candidate.doc_id}-${index}`} snippet={snippet} terms={terms} />
          ))}
        </div>
      </div>

      <div className="result-section">
        <h4>判断根拠</h4>
        <div className="explanation-block">
          <strong>一致理由</strong>
          <ul>
            {candidate.explanation.why_match.map((reason, index) => (
              <li key={`${candidate.doc_id}-why-${index}`}>{reason}</li>
            ))}
          </ul>
        </div>
        <div className="explanation-block">
          <strong>審査官示唆</strong>
          <ul>
            {candidate.explanation.examiner_hints.map((hint, index) => (
              <li key={`${candidate.doc_id}-hint-${index}`}>{hint}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ResultCard;

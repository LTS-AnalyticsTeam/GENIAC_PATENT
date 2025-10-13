import { Snippet } from "../types";
import { highlightText } from "../lib/highlight";

type Props = {
  snippet: Snippet;
  terms: string[];
};

const SnippetItem = ({ snippet, terms }: Props) => {
  return (
    <div className="snippet-card">
      <div className="snippet-header">
        <span className="badge">{snippet.section}</span>
        <span className="muted">
          Claim {snippet.claim_no} · {snippet.match_type} · score {snippet.score.toFixed(2)}
        </span>
      </div>
      <p
        className="snippet-text"
        dangerouslySetInnerHTML={{ __html: highlightText(snippet.text, terms) }}
      />
    </div>
  );
};

export default SnippetItem;

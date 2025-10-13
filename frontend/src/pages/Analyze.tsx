import { FormEvent, useState } from "react";
import ResultCard from "../components/ResultCard";
import { AnalysisResponse, Candidate } from "../types";
import "./Analyze.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

const Analyze = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) {
      setError("テキストファイルをアップロードしてください。");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "解析に失敗しました。");
      }
      const data: AnalysisResponse = await response.json();
      setResult(data);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("不明なエラーが発生しました。");
      }
    } finally {
      setLoading(false);
    }
  };

  const resetInput = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="analyze-page">
      <header className="page-header">
        <div>
          <h1>特許α 先行技術探索モック</h1>
          <p className="muted">
            請求項テキストファイルをアップロードすると A(x) と A(y) のダミー結果を表示します。
          </p>
        </div>
      </header>

      <section className="input-section">
        <form onSubmit={handleSubmit}>
          <div className="input-grid">
            <label className="input-block">
              <span>テキストファイル</span>
              <input
                type="file"
                accept=".txt,text/plain"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              {file && <span className="muted">{file.name}</span>}
            </label>
          </div>
          <div className="button-row">
            <button type="button" onClick={resetInput} className="secondary">
              リセット
            </button>
            <button type="submit" disabled={loading}>
              {loading ? "解析中..." : "解析を実行"}
            </button>
          </div>
        </form>
        {error && <div className="toast error">{error}</div>}
      </section>

      {result && (
        <section className="results-section">
          <div className="alpha-info">
            <h2>特許α 情報</h2>
            <div className="alpha-meta">
              <div>
                <span className="label">タイトル</span>
                <p>{result.alpha.title}</p>
              </div>
              <div>
                <span className="label">公報番号</span>
                <p>{result.alpha.pub_number}</p>
              </div>
            </div>
            <div className="claims">
              <div>
                <span className="label">請求項1</span>
                <p>{result.alpha.claim1}</p>
              </div>
              {result.alpha.claims_rest.length > 0 && (
                <div>
                  <span className="label">請求項2以降</span>
                  <ul>
                    {result.alpha.claims_rest.map((claim, index) => (
                      <li key={index}>{claim}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          <section className="card-group">
            <header className="section-header">
              <h2>A(x) 候補</h2>
            </header>
            <ResultCard candidate={result.Ax} kind="Ax" terms={buildTerms(result.Ax)} />
          </section>

          <section className="card-group">
            <header className="section-header">
              <h2>A(y) 候補</h2>
            </header>
            <div className="ay-list">
              {result.Ay.map((candidate) => (
                <ResultCard
                  key={candidate.doc_id}
                  candidate={candidate}
                  kind="Ay"
                  terms={buildTerms(candidate)}
                />
              ))}
            </div>
          </section>
        </section>
      )}
    </div>
  );
};

const buildTerms = (candidate: Candidate) => {
  const termSet = new Set<string>();
  const collect = (value: string) => {
    extractTokens(value).forEach((token) => {
      if (token.length > 1) {
        termSet.add(token);
      }
    });
  };

  collect(candidate.explanation.summary);
  candidate.explanation.why_match.forEach(collect);
  candidate.explanation.examiner_hints.forEach(collect);
  candidate.snippets.forEach((snippet) => collect(snippet.text));

  return Array.from(termSet).slice(0, 12);
};

const extractTokens = (text: string) =>
  (text.match(/[A-Za-z0-9一-龯ぁ-んァ-ンー]+/g) ?? []).map((token) => token.trim());

export default Analyze;

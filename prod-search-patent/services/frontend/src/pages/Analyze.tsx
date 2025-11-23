import { FormEvent, useState } from "react";
import ResultCard from "../components/ResultCard";
import { AnalysisResponse, Candidate } from "../types";
import "./Analyze.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

const Analyze = () => {
  const [patentFile, setPatentFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    
    if (!patentFile) {
      setError("特許JSONファイルをアップロードしてください。");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // JSONファイルを読み込む
      const fileText = await patentFile.text();
      
      // JSONとしてパース
      let patentJson;
      try {
        patentJson = JSON.parse(fileText);
      } catch (e) {
        throw new Error("JSONファイルの形式が正しくありません。");
      }
      
      // APIに送信（source_jsonのみ送信、candidate_jsonはバックエンドが自動で探す）
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          source_json: patentJson,
          candidate_json: null,  // バックエンドが自動的に類似特許を探す
          raw_text: null
        })
      });
      
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "解析に失敗しました。");
      }
      
      const data = await response.json();
      // レスポンスは { response: AnalysisResponse } の形式
      setResult(data.response);
      
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
    setPatentFile(null);
    setResult(null);
    setError(null);
  };

  // デモ用: サンプルデータで実行
  const runDemo = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // デモ用：バックエンドの.env設定のデータを使用
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          source_json: null,     // .envのGOLD_JSON_PATHを使用
          candidate_json: null,  // .envのCAND_JSON_PATHを使用
          raw_text: null
        })
      });
      
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "デモ実行に失敗しました。");
      }
      
      const data = await response.json();
      setResult(data.response);
      
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

  return (
    <div className="analyze-page">
      <header className="page-header">
        <div>
          <h1>特許類似性解析システム</h1>
          <p className="muted">
            特許JSONファイルをアップロードすると、類似する先行技術を自動的に探索し、
            類似箇所をハイライト表示します。
          </p>
        </div>
      </header>

      <section className="input-section">
        <form onSubmit={handleSubmit}>
          <div className="input-grid">
            <label className="input-block">
              <span>解析対象の特許JSON</span>
              <input
                type="file"
                accept=".json,application/json"
                onChange={(e) => setPatentFile(e.target.files?.[0] ?? null)}
              />
              {patentFile && (
                <span className="muted">📄 {patentFile.name}</span>
              )}
            </label>
          </div>
          
          <div className="info-box">
            <p>💡 アップロードされた特許に類似する先行技術を自動的に検索し、どの部分が類似しているかを解析します。</p>
          </div>
          
          <div className="button-row">
            <button type="button" onClick={resetInput} className="secondary">
              リセット
            </button>
            <button type="button" onClick={runDemo} className="secondary">
              サンプルデータで試す
            </button>
            <button type="submit" disabled={loading || !patentFile}>
              {loading ? "🔍 類似特許を探索中..." : "解析を開始"}
            </button>
          </div>
        </form>
        {error && <div className="toast error">{error}</div>}
        {loading && (
          <div className="toast info">
            Azure OpenAIで特許文書を解析しています...
          </div>
        )}
      </section>

      {result && (
        <section className="results-section">
          <div className="alpha-info">
            <h2>📋 入力特許情報</h2>
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
              <h2>🎯 最も類似している先行技術</h2>
              <span className="muted">A(x): 最も関連性の高い特許</span>
            </header>
            <ResultCard candidate={result.Ax} kind="Ax" terms={buildTerms(result.Ax)} />
          </section>

          {result.Ay && result.Ay.length > 0 && (
            <section className="card-group">
              <header className="section-header">
                <h2>📚 その他の関連先行技術</h2>
                <span className="muted">A(y): 関連性のある特許群</span>
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
          )}
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
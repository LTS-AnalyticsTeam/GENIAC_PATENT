export type Snippet = {
  section: string;
  claim_no: number;
  text: string;
  offset: number;
  len: number;
  match_type: string;
  score: number;
};

export type Explanation = {
  summary: string;
  why_match: string[];
  examiner_hints: string[];
};

export type Candidate = {
  doc_id: string;
  title: string;
  pub_number?: string | null;
  year?: number | null;
  ipc: string[];
  score: number;
  snippets: Snippet[];
  explanation: Explanation;
  source_url?: string;
};

export type AlphaInfo = {
  title: string;
  pub_number: string;
  claim1: string;
  claims_rest: string[];
};

export type RunLimits = {
  max_total: number;
  Ay_min: number;
};

export type AnalysisResponse = {
  run_id: string;
  alpha: AlphaInfo;
  Ax: Candidate;
  Ay: Candidate[];
  limits: RunLimits;
};

export type StageDetail = {
  current_stage?: string;
  stages?: Record<string, Record<string, string>>;
  completed_stages?: string[];
  candidate_count?: number;
  [key: string]: unknown;
};

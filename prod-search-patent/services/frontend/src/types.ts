export type Evidence = {
  section: string;
  quote: string;
  why: string;
  offset?: number;
  length?: number;
  alpha_fragment?: string | null;
  candidate_quote?: string | null;
};

export type ClaimAssessment = {
  claim_no: number;
  novelty: "denied" | "uncertain" | "supported";
  inventive_step?: "denied" | "uncertain" | "supported";
  evidence: Evidence[];
  examiner_hints: string[];
};

export type AssessmentCandidate = {
  doc_id: string;
  title: string;
  pub_number?: string | null;
  score: number;
  assessments: ClaimAssessment[];
  summary?: string | null;
  ipc?: string[];
  source_url?: string;
  year?: number | null;
  is_web_result?: boolean;
  source?: string;
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
  claim1_candidates: AssessmentCandidate[];
  rest_claim_candidates: AssessmentCandidate[];
  limits: RunLimits;
};

export type StageDetail = {
  current_stage?: string;
  stages?: Record<string, Record<string, unknown>>;
  completed_stages?: string[];
  candidate_count?: number;
  narrowed_count?: number;
  [key: string]: unknown;
};

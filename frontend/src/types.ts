export type Scalar = string | number | boolean | null;
export type Row = Record<string, unknown>;
export type Step = string | number;
export interface Source {
  path: string;
  format: string;
  canonical_sha256: string;
  files: { path: string; sha256: string; size_bytes: Step }[];
}
export interface Run {
  id: string;
  run: string;
  status: "queued" | "loading" | "analyzing" | "ready" | "failed";
  error?: string | null;
  counts?: Row | null;
  source?: Source | null;
  elapsed_seconds?: number | null;
}
export interface Example extends Row {
  sample_id?: string | null;
  status?: string;
  step?: Step;
}
export interface Member {
  feature: string;
  kind: string;
  status?: string;
  onset_step?: Step | null;
  alert_step?: Step | null;
  direction?: string;
  evidence: Row;
  exemplars: Example[];
}
export interface Event {
  rank: number;
  event_id?: string;
  representative_feature: string;
  ranking_status: string;
  onset_start?: Step | null;
  onset_end?: Step | null;
  onset_step: Step | null;
  alert_step: Step | null;
  n_members: number;
  ranking: Row;
  members: Member[];
}
export interface EventDetail {
  event: Event;
  evidence: Row[];
  prompt_shifts: Row[];
}
export interface Point {
  step: Step;
  mean: number | null;
  se: number | null;
  n_rollouts: Step;
  n_prompts: Step;
  n_sampling_units: Step;
}
export interface Series {
  feature: string;
  tier: string;
  kind: string;
  points: Point[];
  evidence: Row | null;
  persistence: Row[];
  prompt_values: Row[];
  prompt_shifts: Row[];
}
export interface Citation {
  citation: Row;
  source: Source;
  response_sha256: string;
}
export interface Comparison {
  id: string;
  status: "queued" | "comparing" | "ready" | "failed";
  error?: string | null;
  report?: ComparisonReport | null;
}
export interface Estimate extends Row {
  mean?: number | null;
  estimate?: number | null;
  se?: number | null;
  status: string;
  ci_low?: number | null;
  ci_high?: number | null;
}
export interface ComparisonFeature {
  feature: string;
  tier: string;
  kind: string;
  direction: string;
  a: Estimate;
  b: Estimate;
  difference: Estimate;
  paired: Estimate;
  citations: {
    a: { status: string; items: Example[] };
    b: { status: string; items: Example[] };
  };
}
export interface ComparisonReport {
  mode: string;
  sides: Row[];
  prompt_mix: Row;
  features: ComparisonFeature[];
}
export interface SavedAnalysisRun {
  run: string;
  status: string;
  counts: Record<string, Step | null>;
}
export interface SavedComparisonSide {
  side: string;
  run: string;
  window: Step[];
  observed_steps: Step[];
  n_rollouts: Step;
  n_prompts: Step;
  n_sampling_units: Step;
  source?: Row;
}
export interface SavedPromptMix extends Row {
  status: string;
  shared_prompts: Step;
  only_a_prompts: Step;
  only_b_prompts: Step;
  prompt_shift: Row[];
}
export interface EvidenceSummary extends Row {
  runs?: SavedAnalysisRun[];
  mode?: string;
  sides?: SavedComparisonSide[];
  feature_count?: Step;
  prompt_mix?: SavedPromptMix;
}
export interface Evidence {
  kind: string;
  path: string;
  sha256: string;
  verification: string;
  source_link: { status: string; reason: string };
  identity: Row | null;
  summary: EvidenceSummary;
  conditions: Row[];
  hypothesis: string | null;
  predictions: Row[];
  edit: { old: string; new: string; occurrence: number } | null;
  source_diff: string | null;
  mechanism_status: string;
}

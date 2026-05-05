// Shared types — kept in sync with `packages/api/pps_api/schemas/*` by hand.
// The API publishes an OpenAPI schema at /openapi.json; we can codegen later
// but for ~6 types the friction of a generator outweighs the benefit.

export type JobStatus = "queued" | "running" | "completed" | "failed";

export type StageReport = {
  name: string;
  applied: boolean;
  skipped: boolean;
  error: string | null;
  duration_ms: number;
  warnings: Array<[string, string]>;
  metrics: Record<string, number>;
  reason: string;
};

export type Report = {
  job_id: string;
  duration_ms: number;
  halted: boolean;
  stages: StageReport[];
};

export type JobOut = {
  job_id: string;
  status: JobStatus;
  error: string | null;
  report: Report | null;
  result_url: string | null;
};

export type JobCreate = {
  stages: string[];
  params?: Record<string, unknown>;
  seed?: number;
  metadata?: Record<string, string>;
};

// Pipeline stage catalog — surfaced on /upload as toggles.
export type StageInfo = {
  id: string;
  label: string;
  description: string;
  default: boolean;
};

export const AVAILABLE_STAGES: StageInfo[] = [
  {
    id: "preflight",
    label: "Preflight QC",
    description: "Blur / exposure / colour-cast detection. Halts the pipeline if the input is unsalvageable.",
    default: true,
  },
  {
    id: "perspective",
    label: "Vertical correction",
    description: "Detects converging verticals (Hough) and rotates so walls / windows are plumb.",
    default: true,
  },
  {
    id: "real_estate",
    label: "Real-estate enhance",
    description: "Sky replace + lawn boost + window pull, scene-aware (interior vs exterior vs aerial).",
    default: true,
  },
  {
    id: "enhance_studio",
    label: "Studio finish",
    description: "Robust white-balance, CLAHE, halo-free local detail, vibrance with skin protection.",
    default: true,
  },
  {
    id: "twilight",
    label: "Virtual twilight",
    description: "Daylight → golden-hour transform with sky tint and window glow. Opt-in.",
    default: false,
  },
];

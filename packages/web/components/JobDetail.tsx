"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { AlertTriangle, Download, Loader2, RefreshCw } from "lucide-react";
import type { JobOut, StageReport } from "@/lib/types";
import { JobStatusBadge } from "./JobStatusBadge";
import { BeforeAfterSlider } from "./BeforeAfterSlider";

type Props = {
  initial: JobOut;
  beforeImageUrl: string | null;
};

const POLL_INTERVAL_MS = 1500;
const TERMINAL = new Set(["completed", "failed"]);

export function JobDetail({ initial, beforeImageUrl }: Props) {
  const [job, setJob] = useState<JobOut>(initial);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const resp = await fetch(`/api/jobs/${encodeURIComponent(job.job_id)}`, {
        cache: "no-store",
      });
      if (!resp.ok) {
        setError(`Refresh failed: HTTP ${resp.status}`);
        return;
      }
      const next = (await resp.json()) as JobOut;
      setJob(next);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Refresh failed");
    } finally {
      setRefreshing(false);
    }
  }, [job.job_id]);

  useEffect(() => {
    if (TERMINAL.has(job.status)) return;
    const id = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [job.status, refresh]);

  const afterUrl = `/api/jobs/${encodeURIComponent(job.job_id)}/result`;

  return (
    <div className="space-y-8">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <code className="font-mono text-sm text-[var(--color-fg-muted)]">
              {job.job_id}
            </code>
            <JobStatusBadge status={job.status} />
          </div>
          <h1 className="mt-2 text-3xl font-semibold">Job detail</h1>
        </div>
        <div className="flex gap-2">
          {job.status === "completed" && (
            <a
              href={afterUrl}
              download={`pps-${job.job_id}.jpg`}
              className="flex items-center gap-2 rounded-md bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-black transition hover:bg-[var(--color-accent-hover)]"
            >
              <Download className="h-4 w-4" />
              Download
            </a>
          )}
          <button
            type="button"
            onClick={refresh}
            className="flex items-center gap-2 rounded-md border border-[var(--color-border)] px-4 py-2 text-sm transition hover:bg-[var(--color-surface)]"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
            Refresh
          </button>
          <Link
            href="/jobs"
            className="rounded-md border border-[var(--color-border)] px-4 py-2 text-sm transition hover:bg-[var(--color-surface)]"
          >
            All jobs
          </Link>
        </div>
      </header>

      {error && (
        <div className="rounded-md border border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 px-4 py-3 text-sm text-[var(--color-danger)]">
          {error}
        </div>
      )}

      {job.status === "failed" && job.error && (
        <div className="flex items-start gap-3 rounded-md border border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 p-4 text-sm">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-[var(--color-danger)]" />
          <div>
            <div className="font-medium text-[var(--color-danger)]">Job failed</div>
            <pre className="mt-1 whitespace-pre-wrap text-xs text-[var(--color-fg-muted)]">
              {job.error}
            </pre>
          </div>
        </div>
      )}

      {!TERMINAL.has(job.status) && (
        <div className="flex items-center gap-3 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] p-4 text-sm text-[var(--color-fg-muted)]">
          <Loader2 className="h-4 w-4 animate-spin text-[var(--color-accent)]" />
          {job.status === "queued"
            ? "Waiting for a worker. Auto-refreshing every 1.5s."
            : "Pipeline running. Auto-refreshing every 1.5s."}
        </div>
      )}

      {job.status === "completed" && beforeImageUrl ? (
        <section>
          <h2 className="mb-3 text-base font-medium text-[var(--color-fg-muted)]">
            Before / after — drag the slider
          </h2>
          <BeforeAfterSlider beforeSrc={beforeImageUrl} afterSrc={afterUrl} />
        </section>
      ) : job.status === "completed" ? (
        <section>
          <h2 className="mb-3 text-base font-medium text-[var(--color-fg-muted)]">
            Result
          </h2>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={afterUrl}
            alt="Rendered result"
            className="w-full rounded-lg border border-[var(--color-border)] object-contain"
          />
          <p className="mt-2 text-xs text-[var(--color-fg-muted)]">
            Original photo is not retained, so the before/after slider is unavailable.
            Submit through the web upload form to see it side-by-side.
          </p>
        </section>
      ) : null}

      {job.report && <ReportPanel report={job.report} />}
    </div>
  );
}

function ReportPanel({
  report,
}: {
  report: NonNullable<JobOut["report"]>;
}) {
  return (
    <section className="space-y-4">
      <div className="flex items-baseline justify-between">
        <h2 className="text-base font-medium text-[var(--color-fg-muted)]">
          Pipeline report
        </h2>
        <div className="text-sm text-[var(--color-fg-muted)]">
          Total: <span className="font-mono text-[var(--color-fg)]">
            {(report.duration_ms / 1000).toFixed(2)}s
          </span>
          {report.halted && (
            <span className="ml-3 rounded bg-[var(--color-danger)]/15 px-2 py-0.5 text-xs text-[var(--color-danger)]">
              halted
            </span>
          )}
        </div>
      </div>

      <ul className="overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)]">
        {report.stages.map((s, i) => (
          <StageRow key={`${s.name}-${i}`} stage={s} />
        ))}
      </ul>
    </section>
  );
}

function StageRow({ stage }: { stage: StageReport }) {
  const tone = stage.error
    ? "text-[var(--color-danger)]"
    : stage.skipped
      ? "text-[var(--color-fg-muted)]"
      : stage.applied
        ? "text-[var(--color-success)]"
        : "text-[var(--color-fg-muted)]";

  const verdict = stage.error
    ? "error"
    : stage.skipped
      ? "skipped"
      : stage.applied
        ? "applied"
        : "—";

  return (
    <li className="grid grid-cols-[180px_70px_1fr_90px] items-center gap-3 border-b border-[var(--color-border)] px-4 py-3 text-sm last:border-b-0">
      <div className="font-medium">{stage.name}</div>
      <div className={`font-mono text-xs ${tone}`}>{verdict}</div>
      <div className="min-w-0 truncate text-xs text-[var(--color-fg-muted)]">
        {stage.error ? (
          <span className="text-[var(--color-danger)]">{stage.error}</span>
        ) : stage.warnings.length > 0 ? (
          stage.warnings.map(([sev, msg]) => `${sev}: ${msg}`).join(" · ")
        ) : stage.reason ? (
          stage.reason
        ) : (
          Object.entries(stage.metrics)
            .slice(0, 3)
            .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(2) : v}`)
            .join(" · ") || "—"
        )}
      </div>
      <div className="text-right font-mono text-xs text-[var(--color-fg-muted)]">
        {stage.duration_ms.toFixed(1)}ms
      </div>
    </li>
  );
}

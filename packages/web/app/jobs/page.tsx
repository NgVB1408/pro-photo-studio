import Link from "next/link";
import type { Metadata } from "next";
import { listJobs } from "@/lib/api";
import { JobStatusBadge } from "@/components/JobStatusBadge";

export const metadata: Metadata = { title: "Recent jobs" };
export const dynamic = "force-dynamic";
export const revalidate = 0;

function summarise(stages: { name: string; applied: boolean }[]): string {
  const applied = stages.filter((s) => s.applied).map((s) => s.name);
  if (applied.length === 0) return "no stages applied";
  if (applied.length <= 3) return applied.join(" · ");
  return `${applied.slice(0, 3).join(" · ")} +${applied.length - 3}`;
}

export default async function JobsPage() {
  let jobs = await listJobs(50).catch(() => null);
  if (jobs === null) {
    return (
      <section className="mx-auto max-w-5xl px-6 py-14">
        <h1 className="text-3xl font-semibold">Recent jobs</h1>
        <p className="mt-4 rounded-md border border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 px-4 py-3 text-sm text-[var(--color-danger)]">
          Couldn't reach the API. Make sure <code>uvicorn pps_api.main:app</code> is
          running and <code>PPS_API_URL</code> is correct in <code>.env.local</code>.
        </p>
      </section>
    );
  }

  return (
    <section className="mx-auto max-w-5xl px-6 py-14">
      <div className="mb-8 flex items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold md:text-4xl">Recent jobs</h1>
          <p className="mt-2 text-[var(--color-fg-muted)]">
            {jobs.length === 0
              ? "Nothing here yet — submit a photo to see it appear."
              : `Showing the most recent ${jobs.length} job${jobs.length === 1 ? "" : "s"}.`}
          </p>
        </div>
        <Link
          href="/upload"
          className="rounded-md bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-black transition hover:bg-[var(--color-accent-hover)]"
        >
          New upload
        </Link>
      </div>

      {jobs.length === 0 ? (
        <EmptyState />
      ) : (
        <ul className="divide-y divide-[var(--color-border)] overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)]">
          {jobs.map((job) => (
            <li key={job.job_id}>
              <Link
                href={`/jobs/${job.job_id}`}
                className="grid grid-cols-[1fr_auto] items-center gap-4 px-5 py-4 transition hover:bg-[var(--color-surface-elevated)]"
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-3">
                    <code className="truncate font-mono text-sm text-[var(--color-fg)]">
                      {job.job_id.slice(0, 12)}…
                    </code>
                    <JobStatusBadge status={job.status} />
                  </div>
                  <div className="mt-1 truncate text-xs text-[var(--color-fg-muted)]">
                    {job.report
                      ? `${summarise(job.report.stages)} · ${(job.report.duration_ms / 1000).toFixed(2)}s`
                      : job.error
                        ? `Error: ${job.error.slice(0, 80)}`
                        : "Pending stage report…"}
                  </div>
                </div>
                <div className="text-right text-xs text-[var(--color-fg-muted)]">
                  View →
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function EmptyState() {
  return (
    <div className="rounded-lg border border-dashed border-[var(--color-border)] bg-[var(--color-surface)] p-12 text-center">
      <div className="text-base font-medium">No jobs yet</div>
      <p className="mt-2 text-sm text-[var(--color-fg-muted)]">
        Drop a photo on the upload page and we'll process it through the pipeline.
      </p>
      <Link
        href="/upload"
        className="mt-6 inline-block rounded-md bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-black transition hover:bg-[var(--color-accent-hover)]"
      >
        Upload a photo
      </Link>
    </div>
  );
}

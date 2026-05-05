import { CheckCircle2, Clock, Loader2, XCircle } from "lucide-react";
import type { JobStatus } from "@/lib/types";

const STYLES: Record<
  JobStatus,
  { label: string; tone: string; icon: typeof Clock }
> = {
  queued: {
    label: "Queued",
    tone: "border-[var(--color-border)] bg-[var(--color-surface-elevated)] text-[var(--color-fg-muted)]",
    icon: Clock,
  },
  running: {
    label: "Running",
    tone: "border-[var(--color-accent)]/40 bg-[var(--color-accent)]/10 text-[var(--color-accent)]",
    icon: Loader2,
  },
  completed: {
    label: "Completed",
    tone: "border-[var(--color-success)]/40 bg-[var(--color-success)]/10 text-[var(--color-success)]",
    icon: CheckCircle2,
  },
  failed: {
    label: "Failed",
    tone: "border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 text-[var(--color-danger)]",
    icon: XCircle,
  },
};

export function JobStatusBadge({ status }: { status: JobStatus }) {
  const s = STYLES[status];
  const Icon = s.icon;
  const animate = status === "running" ? "animate-spin" : "";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${s.tone}`}
    >
      <Icon className={`h-3.5 w-3.5 ${animate}`} />
      {s.label}
    </span>
  );
}

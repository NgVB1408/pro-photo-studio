"use client";

import { useState } from "react";
import {
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Sparkles,
  TriangleAlert,
  XCircle,
} from "lucide-react";
import type { AgentReport, ChecklistStatus, StudioReport } from "@/lib/types";
import { cn } from "@/lib/cn";

const GRADE_TONE: Record<StudioReport["grade"], string> = {
  S: "border-[var(--color-accent)] text-[var(--color-accent)] bg-[var(--color-accent)]/10",
  A: "border-[var(--color-success)] text-[var(--color-success)] bg-[var(--color-success)]/10",
  B: "border-[var(--color-success)] text-[var(--color-success)] bg-[var(--color-success)]/5",
  C: "border-yellow-500 text-yellow-400 bg-yellow-500/10",
  D: "border-[var(--color-danger)] text-[var(--color-danger)] bg-[var(--color-danger)]/10",
};

const STATUS_ICON: Record<ChecklistStatus, typeof CheckCircle2> = {
  pass: CheckCircle2,
  warn: TriangleAlert,
  fail: XCircle,
};

const STATUS_TONE: Record<ChecklistStatus, string> = {
  pass: "text-[var(--color-success)]",
  warn: "text-yellow-400",
  fail: "text-[var(--color-danger)]",
};

export function StudioScorecard({ report }: { report: StudioReport }) {
  const applicable = report.agents.filter(
    (a) => a.after.metrics?.applicable !== 0,
  );
  const intervened = report.agents.filter((a) => a.apply.applied);

  return (
    <section className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold">Post-production scorecard</h2>
          <p className="mt-1 text-sm text-[var(--color-fg-muted)]">{report.summary}</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge tone={GRADE_TONE[report.grade]} large>
            {report.grade}
          </Badge>
          <div className="text-right">
            <div className="font-mono text-2xl font-semibold tabular-nums">
              {report.overall_after.toFixed(1)}
              <span className="text-sm text-[var(--color-fg-muted)]"> / 10</span>
            </div>
            <div className="text-xs text-[var(--color-fg-muted)]">
              up from {report.overall_before.toFixed(1)} · {(report.duration_ms / 1000).toFixed(1)}s
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-wrap items-center gap-4 text-xs text-[var(--color-fg-muted)]">
        <span className="inline-flex items-center gap-1.5">
          <Sparkles className="h-3.5 w-3.5 text-[var(--color-accent)]" />
          Scene: <span className="font-medium text-[var(--color-fg)]">{report.scene}</span>
        </span>
        <span>
          {applicable.length} specialists reviewed · {intervened.length} intervened
        </span>
      </div>

      <ul className="space-y-3">
        {report.agents.map((agent) => (
          <AgentRow key={agent.name} agent={agent} />
        ))}
      </ul>
    </section>
  );
}

function AgentRow({ agent }: { agent: AgentReport }) {
  const [expanded, setExpanded] = useState(false);
  const applicable = agent.after.metrics?.applicable !== 0;
  const score = agent.after.score;
  const delta = agent.after.score - agent.before.score;

  const tone =
    score >= 9
      ? "text-[var(--color-success)]"
      : score >= 7.5
        ? "text-[var(--color-fg)]"
        : score >= 6
          ? "text-yellow-400"
          : "text-[var(--color-danger)]";

  return (
    <li className="overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)]">
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        className="grid w-full grid-cols-[1fr_auto] items-center gap-4 px-5 py-4 text-left transition hover:bg-[var(--color-surface-elevated)]"
      >
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold">{agent.name}</span>
            {agent.apply.applied && (
              <Badge tone="border-[var(--color-accent)]/40 text-[var(--color-accent)] bg-[var(--color-accent)]/10">
                intervened
              </Badge>
            )}
            {!applicable && (
              <Badge tone="border-[var(--color-border)] text-[var(--color-fg-muted)]">
                n/a
              </Badge>
            )}
          </div>
          <div className="mt-0.5 truncate text-xs text-[var(--color-fg-muted)]">
            {agent.role}
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right">
            <div className={cn("font-mono text-lg font-semibold tabular-nums", tone)}>
              {applicable ? score.toFixed(1) : "—"}
            </div>
            {applicable && Math.abs(delta) >= 0.1 && (
              <div className="text-[10px] text-[var(--color-fg-muted)]">
                {delta > 0 ? "+" : ""}
                {delta.toFixed(1)} vs before
              </div>
            )}
          </div>
          {expanded ? (
            <ChevronUp className="h-4 w-4 text-[var(--color-fg-muted)]" />
          ) : (
            <ChevronDown className="h-4 w-4 text-[var(--color-fg-muted)]" />
          )}
        </div>
      </button>

      {expanded && (
        <div className="border-t border-[var(--color-border)] px-5 py-4 text-sm">
          {applicable ? (
            <>
              <div className="mb-3 text-xs text-[var(--color-fg-muted)]">
                {agent.after.summary}
              </div>
              <ChecklistTable items={agent.after.checklist} />
              {agent.apply.applied && (
                <div className="mt-4 rounded-md border border-[var(--color-accent)]/30 bg-[var(--color-accent)]/5 px-3 py-2 text-xs">
                  <div className="font-medium text-[var(--color-accent)]">Actions</div>
                  <ul className="mt-1 list-disc pl-4 text-[var(--color-fg-muted)]">
                    {agent.apply.actions.map((action) => (
                      <li key={action}>{action}</li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          ) : (
            <div className="text-xs text-[var(--color-fg-muted)]">
              {agent.before.summary || "Not applicable to this scene."}
            </div>
          )}
        </div>
      )}
    </li>
  );
}

function ChecklistTable({
  items,
}: {
  items: AgentReport["after"]["checklist"];
}) {
  if (items.length === 0) {
    return null;
  }
  return (
    <ul className="space-y-2">
      {items.map((item) => {
        const Icon = STATUS_ICON[item.status];
        return (
          <li
            key={`${item.label}-${item.detail}`}
            className="grid grid-cols-[20px_1fr_auto] items-start gap-3"
          >
            <Icon className={cn("mt-0.5 h-4 w-4", STATUS_TONE[item.status])} />
            <div>
              <div className="font-medium">{item.label}</div>
              <div className="text-xs text-[var(--color-fg-muted)]">{item.detail}</div>
            </div>
            <span
              className={cn(
                "rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider",
                item.status === "pass"
                  ? "bg-[var(--color-success)]/15 text-[var(--color-success)]"
                  : item.status === "warn"
                    ? "bg-yellow-500/15 text-yellow-400"
                    : "bg-[var(--color-danger)]/15 text-[var(--color-danger)]",
              )}
            >
              {item.status}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function Badge({
  children,
  tone,
  large,
}: {
  children: React.ReactNode;
  tone: string;
  large?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center justify-center rounded-md border font-medium",
        large ? "h-12 w-12 text-xl" : "px-2 py-0.5 text-[10px] uppercase tracking-wider",
        tone,
      )}
    >
      {children}
    </span>
  );
}

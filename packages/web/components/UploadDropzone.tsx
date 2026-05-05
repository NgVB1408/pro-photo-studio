"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Sparkles, UploadCloud, Wrench, X } from "lucide-react";
import { AVAILABLE_STAGES } from "@/lib/types";
import type { JobCreate, JobOut, StageInfo } from "@/lib/types";
import { cn } from "@/lib/cn";

type Status = "idle" | "uploading" | "error";
type Mode = "autopilot" | "manual";

const ACCEPT = ["image/jpeg", "image/png", "image/webp"] as const;
const MAX_BYTES = 50 * 1024 * 1024;

export function UploadDropzone() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const stagesId = useId();

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("autopilot");
  const [stages, setStages] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(AVAILABLE_STAGES.map((s) => [s.id, s.default])),
  );
  const [seed, setSeed] = useState<number>(42);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onSelect = useCallback((f: File | null) => {
    if (!f) {
      setFile(null);
      return;
    }
    if (!ACCEPT.includes(f.type as (typeof ACCEPT)[number])) {
      setError("Only JPEG, PNG, or WebP photos are accepted.");
      return;
    }
    if (f.size > MAX_BYTES) {
      setError(`Photo is ${(f.size / 1_048_576).toFixed(1)} MB; the limit is 50 MB.`);
      return;
    }
    setError(null);
    setFile(f);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files?.[0];
      if (f) onSelect(f);
    },
    [onSelect],
  );

  const submit = useCallback(async () => {
    if (!file) return;
    setStatus("uploading");
    setError(null);

    let endpoint = "/api/auto";
    const fd = new FormData();
    fd.append("image", file, file.name);

    if (mode === "manual") {
      const enabled = AVAILABLE_STAGES.filter((s) => stages[s.id]).map((s) => s.id);
      if (enabled.length === 0) {
        setError("Pick at least one stage.");
        setStatus("error");
        return;
      }
      const body: JobCreate = {
        stages: enabled,
        seed,
        metadata: { source: "web-portal", filename: file.name, mode: "manual" },
      };
      fd.append("body", JSON.stringify(body));
      endpoint = "/api/jobs";
    } else {
      fd.append("seed", String(seed));
    }

    try {
      const resp = await fetch(endpoint, { method: "POST", body: fd });
      if (!resp.ok) {
        const detail = await resp
          .json()
          .then((j) =>
            typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail),
          )
          .catch(() => `HTTP ${resp.status}`);
        setStatus("error");
        setError(detail);
        return;
      }
      const job = (await resp.json()) as JobOut;
      router.push(`/jobs/${job.job_id}`);
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : "Upload failed");
    }
  }, [file, mode, router, seed, stages]);

  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_360px]">
      <div className="space-y-4">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={cn(
            "relative flex min-h-[420px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed bg-[var(--color-surface)] p-6 text-center transition",
            dragOver
              ? "border-[var(--color-accent)] bg-[var(--color-surface-elevated)]"
              : "border-[var(--color-border)] hover:border-[var(--color-fg-muted)]",
          )}
        >
          {preview ? (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={preview}
                alt={file?.name ?? "preview"}
                className="max-h-[380px] max-w-full rounded object-contain"
              />
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                }}
                className="absolute right-3 top-3 rounded-full bg-black/60 p-1.5 text-white transition hover:bg-black/80"
                aria-label="Remove photo"
              >
                <X className="h-4 w-4" />
              </button>
            </>
          ) : (
            <>
              <UploadCloud
                className="mb-4 h-12 w-12 text-[var(--color-fg-muted)]"
                strokeWidth={1.4}
              />
              <div className="text-base font-medium">
                Drop a photo, or click to choose
              </div>
              <div className="mt-1 text-sm text-[var(--color-fg-muted)]">
                JPEG, PNG, or WebP. Up to 50 MB. Original is not retained.
              </div>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT.join(",")}
            className="hidden"
            onChange={(e) => onSelect(e.target.files?.[0] ?? null)}
          />
        </div>

        {error && (
          <div className="rounded-md border border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 px-4 py-3 text-sm text-[var(--color-danger)]">
            {error}
          </div>
        )}
      </div>

      <aside className="space-y-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
        <div>
          <h2 className="text-base font-semibold">Mode</h2>
          <p className="mt-1 text-xs text-[var(--color-fg-muted)]">
            Auto-pilot is the recommended default. Switch to Manual only when you
            need to control individual stages.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-2 rounded-md border border-[var(--color-border)] bg-[var(--color-canvas)] p-1">
          <ModeButton
            active={mode === "autopilot"}
            onClick={() => setMode("autopilot")}
            icon={Sparkles}
            label="Auto-pilot"
          />
          <ModeButton
            active={mode === "manual"}
            onClick={() => setMode("manual")}
            icon={Wrench}
            label="Manual"
          />
        </div>

        {mode === "autopilot" ? (
          <div className="rounded-md border border-[var(--color-accent)]/30 bg-[var(--color-accent)]/5 p-4 text-xs text-[var(--color-fg-muted)]">
            <div className="mb-2 font-medium text-[var(--color-fg)]">
              The studio runs every specialist
            </div>
            <ul className="grid grid-cols-1 gap-1.5">
              {[
                "Auto-detects scene (interior / exterior / aerial)",
                "Picks the right baseline pipeline per scene",
                "Routes through 9 specialists with checklists",
                "Rolls back any change that lowers a category score",
                "Returns a 0–10 scorecard you can audit",
              ].map((it) => (
                <li key={it} className="flex items-center gap-1.5">
                  <span className="h-1 w-1 rounded-full bg-[var(--color-accent)]" />
                  <span>{it}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <>
            <div>
              <h3 className="text-sm font-semibold">Pipeline stages</h3>
              <p className="mt-1 text-xs text-[var(--color-fg-muted)]">
                Toggle off steps you don't want. Order is fixed.
              </p>
            </div>
            <ul className="space-y-3">
              {AVAILABLE_STAGES.map((s: StageInfo) => (
                <li key={s.id}>
                  <label
                    htmlFor={`${stagesId}-${s.id}`}
                    className="flex cursor-pointer items-start gap-3 rounded-md border border-transparent p-2 transition hover:border-[var(--color-border)] hover:bg-[var(--color-surface-elevated)]"
                  >
                    <input
                      id={`${stagesId}-${s.id}`}
                      type="checkbox"
                      className="mt-1 h-4 w-4 accent-[var(--color-accent)]"
                      checked={stages[s.id] ?? false}
                      onChange={(e) =>
                        setStages((cur) => ({ ...cur, [s.id]: e.target.checked }))
                      }
                    />
                    <div>
                      <div className="text-sm font-medium">{s.label}</div>
                      <div className="text-xs leading-relaxed text-[var(--color-fg-muted)]">
                        {s.description}
                      </div>
                    </div>
                  </label>
                </li>
              ))}
            </ul>
          </>
        )}

        <div>
          <label htmlFor="seed" className="text-sm font-medium">
            Seed
          </label>
          <input
            id="seed"
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number.parseInt(e.target.value, 10) || 0)}
            className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-canvas)] px-3 py-2 text-sm focus:border-[var(--color-accent)] focus:outline-none"
          />
          <p className="mt-1 text-xs text-[var(--color-fg-muted)]">
            Same seed + same input = byte-identical output.
          </p>
        </div>

        <button
          type="button"
          onClick={submit}
          disabled={!file || status === "uploading"}
          className={cn(
            "flex w-full items-center justify-center gap-2 rounded-md py-3 text-sm font-medium transition",
            !file || status === "uploading"
              ? "cursor-not-allowed bg-[var(--color-surface-elevated)] text-[var(--color-fg-muted)]"
              : "bg-[var(--color-accent)] text-black hover:bg-[var(--color-accent-hover)]",
          )}
        >
          {status === "uploading" ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Submitting…
            </>
          ) : mode === "autopilot" ? (
            <>
              <Sparkles className="h-4 w-4" />
              Run auto-pilot
            </>
          ) : (
            "Process photo"
          )}
        </button>
      </aside>
    </div>
  );
}

function ModeButton({
  active,
  onClick,
  icon: Icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: typeof Sparkles;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "flex items-center justify-center gap-2 rounded-md px-3 py-2 text-xs font-medium transition",
        active
          ? "bg-[var(--color-accent)] text-black"
          : "text-[var(--color-fg-muted)] hover:bg-[var(--color-surface-elevated)] hover:text-[var(--color-fg)]",
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
    </button>
  );
}

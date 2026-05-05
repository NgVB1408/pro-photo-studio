import Link from "next/link";
import type { Metadata } from "next";
import { CheckCircle2, Sparkles, Timer } from "lucide-react";
import { DEMO_PHOTOS } from "@/lib/demo";
import { BeforeAfterSlider } from "@/components/BeforeAfterSlider";

export const metadata: Metadata = {
  title: "Live demo gallery",
  description:
    "Real before/after pairs rendered by the Pro Photo Studio production pipeline. No login required.",
};

export default function DemoPage() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-14">
      <header className="mb-10 max-w-3xl">
        <div className="text-xs uppercase tracking-widest text-[var(--color-accent)]">
          Live demo
        </div>
        <h1 className="mt-2 text-4xl font-semibold leading-tight md:text-5xl">
          Real photos, real pipeline, real timings.
        </h1>
        <p className="mt-3 text-[var(--color-fg-muted)]">
          Every pair below was produced by the same v2 pipeline that runs in
          production — same stages, same seed, byte-identical to what you'd
          get if you uploaded the original. Drag the slider to compare.
        </p>
      </header>

      <ul className="space-y-16">
        {DEMO_PHOTOS.map((photo) => (
          <li
            key={photo.id}
            className="grid gap-6 lg:grid-cols-[1fr_320px] lg:gap-10"
          >
            <div>
              <BeforeAfterSlider
                beforeSrc={photo.before_src}
                afterSrc={photo.after_src}
              />
              <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-[var(--color-fg-muted)]">
                <span className="rounded-full border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-0.5">
                  {photo.scene}
                </span>
                <span className="inline-flex items-center gap-1">
                  <Timer className="h-3 w-3" />
                  {(photo.duration_ms / 1000).toFixed(2)}s
                </span>
                <span className="inline-flex items-center gap-1">
                  <Sparkles className="h-3 w-3 text-[var(--color-accent)]" />
                  {photo.stages.join(" · ")}
                </span>
                <code className="font-mono text-[10px]">{photo.id}</code>
              </div>
            </div>

            <aside className="space-y-4">
              <h2 className="text-lg font-semibold leading-tight">{photo.title}</h2>
              <ul className="space-y-2 text-sm">
                {photo.highlights.map((h) => (
                  <li key={h} className="flex items-start gap-2">
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[var(--color-success)]" />
                    <span className="text-[var(--color-fg-muted)]">{h}</span>
                  </li>
                ))}
              </ul>
              <a
                href={photo.report_src}
                className="inline-block rounded-md border border-[var(--color-border)] px-3 py-1.5 text-xs text-[var(--color-fg-muted)] transition hover:bg-[var(--color-surface)] hover:text-[var(--color-fg)]"
                target="_blank"
                rel="noreferrer"
              >
                View raw stage report (JSON)
              </a>
            </aside>
          </li>
        ))}
      </ul>

      <div className="mt-20 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-8 text-center">
        <h2 className="text-2xl font-semibold">Ready to run yours?</h2>
        <p className="mt-2 text-[var(--color-fg-muted)]">
          Upload a single photo or wire up the API for a full listing.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-3">
          <Link
            href="/upload"
            className="rounded-md bg-[var(--color-accent)] px-5 py-2.5 text-sm font-medium text-black transition hover:bg-[var(--color-accent-hover)]"
          >
            Upload a photo
          </Link>
          <Link
            href="/#pricing"
            className="rounded-md border border-[var(--color-border)] px-5 py-2.5 text-sm transition hover:bg-[var(--color-surface-elevated)]"
          >
            See pricing
          </Link>
        </div>
      </div>
    </section>
  );
}

import Link from "next/link";
import {
  Sparkles,
  Wand2,
  Layers,
  Zap,
  CheckCircle2,
  ArrowRight,
  Sun,
  Image as ImageIcon,
  Camera,
  Sofa,
} from "lucide-react";
import { publicEnv } from "@/lib/env";

const NET_NEW = [
  {
    icon: Sofa,
    title: "Virtual staging",
    body: "Generate furnished interiors from empty rooms in seconds. SD3.5 + IPAdapter, brand-consistent across the listing.",
    badge: "Net new",
  },
  {
    icon: Camera,
    title: "Multi-angle synthesis",
    body: "One photo of a room becomes three angles for the listing carousel. Saves a return trip to the property.",
    badge: "Net new",
  },
  {
    icon: Wand2,
    title: "Instruction editing",
    body: "Type 'brighten the kitchen, warmer wood floor' — the model edits with intent, not toggles. Powered by Qwen-Image-Lightning.",
    badge: "Net new",
  },
];

const PARITY = [
  { label: "Sky replace (procedural + ControlNet LoRA)" },
  { label: "HDR fusion with deghost + alignment" },
  { label: "Window-pull + selective lawn boost" },
  { label: "Vertical correction (Hough)" },
  { label: "Lens distortion correction" },
  { label: "Auto privacy (faces + plates)" },
  { label: "RAW / DNG decode" },
  { label: "Batch tone coherency across listing" },
  { label: "SUPIR upscale x2 / x4" },
];

const PRICING_TIERS = [
  {
    name: "Studio",
    price: "$0.45",
    unit: "per photo",
    blurb: "Full pipeline. Pay-as-you-go.",
    features: [
      "All 11 enhancement stages",
      "REST + webhook delivery",
      "Volume discount > 1k / month",
      "Email support",
    ],
  },
  {
    name: "Brokerage",
    price: "$80",
    unit: "per listing",
    blurb: "20–40 photos, virtual staging incl.",
    features: [
      "Studio + virtual staging",
      "Multi-angle synthesis",
      "Tone coherency across listing",
      "White-label portal",
      "Slack delivery",
    ],
    highlight: true,
  },
  {
    name: "API",
    price: "Custom",
    unit: "",
    blurb: "Volume + dedicated GPU pool.",
    features: [
      "Dedicated GPU autoscale",
      "SLA 99.9% uptime",
      "Custom LoRAs + brand styles",
      "Single-tenant DB option",
    ],
  },
];

export default function HomePage() {
  return (
    <>
      {/* HERO */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 -z-10 bg-gradient-to-b from-[#1a0f08] via-[var(--color-canvas)] to-[var(--color-canvas)]" />
        <div className="absolute right-1/3 top-32 -z-10 h-96 w-96 rounded-full bg-[var(--color-accent)]/15 blur-3xl" />

        <div className="mx-auto max-w-6xl px-6 py-24 md:py-32">
          <div className="flex max-w-3xl flex-col gap-6">
            <div className="flex items-center gap-2 self-start rounded-full border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1 text-xs">
              <Sparkles className="h-3.5 w-3.5 text-[var(--color-accent)]" />
              <span className="text-[var(--color-fg-muted)]">
                {publicEnv.brandName} v2 — production beta
              </span>
            </div>

            <h1 className="text-4xl font-semibold leading-tight tracking-tight md:text-6xl">
              Real-estate photos,
              <br />
              <span className="text-[var(--color-accent)]">automated to listing-ready.</span>
            </h1>

            <p className="max-w-2xl text-lg text-[var(--color-fg-muted)]">
              {publicEnv.brandTagline} Sky replace, HDR fusion, virtual staging, and
              instruction-based editing — all behind one API. Process a 30-photo listing in
              under 4 minutes, with consistent tone across the whole set.
            </p>

            <div className="mt-2 flex flex-wrap items-center gap-3">
              <Link
                href="/demo"
                className="flex items-center gap-2 rounded-md bg-[var(--color-accent)] px-5 py-3 text-sm font-medium text-black transition hover:bg-[var(--color-accent-hover)]"
              >
                See live demo <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/upload"
                className="flex items-center gap-2 rounded-md border border-[var(--color-border)] px-5 py-3 text-sm transition hover:bg-[var(--color-surface)]"
              >
                Upload your photo
              </Link>
            </div>

            <ul className="mt-6 grid grid-cols-2 gap-x-6 gap-y-2 text-sm text-[var(--color-fg-muted)] md:grid-cols-3">
              {[
                "Async REST + webhooks",
                "Sub-4-minute listings",
                "GDPR delete-on-request",
                "S3 / R2 / MinIO",
                "Stripe billing",
                "White-label portal",
              ].map((f) => (
                <li key={f} className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-[var(--color-success)]" />
                  {f}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* WHY DIFFERENT */}
      <section id="features" className="border-t border-[var(--color-border)] py-20">
        <div className="mx-auto max-w-6xl px-6">
          <div className="mb-12 max-w-2xl">
            <div className="text-xs uppercase tracking-widest text-[var(--color-accent)]">
              Why we win
            </div>
            <h2 className="mt-2 text-3xl font-semibold md:text-4xl">
              Three features the incumbents don't ship.
            </h2>
            <p className="mt-3 text-[var(--color-fg-muted)]">
              AutoEnhance and Manuka match us on color and HDR. We pull ahead on the work
              agents and brokerages actually pay extra for elsewhere.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-3">
            {NET_NEW.map((f) => (
              <div
                key={f.title}
                className="group relative rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-6 transition hover:border-[var(--color-accent)]/40"
              >
                <div className="absolute right-4 top-4 rounded-full bg-[var(--color-accent)]/15 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-[var(--color-accent)]">
                  {f.badge}
                </div>
                <f.icon
                  className="h-7 w-7 text-[var(--color-accent)]"
                  strokeWidth={1.5}
                />
                <h3 className="mt-4 text-lg font-semibold">{f.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-[var(--color-fg-muted)]">
                  {f.body}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-16 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-8">
            <div className="flex items-center gap-3">
              <Layers className="h-5 w-5 text-[var(--color-accent)]" />
              <h3 className="text-lg font-semibold">
                Plus everything the established tools do.
              </h3>
            </div>
            <p className="mt-2 text-sm text-[var(--color-fg-muted)]">
              We're at parity on the entire AutoEnhance + Manuka feature set.
            </p>
            <ul className="mt-6 grid grid-cols-1 gap-x-6 gap-y-3 text-sm md:grid-cols-3">
              {PARITY.map((p) => (
                <li key={p.label} className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-[var(--color-success)]" />
                  <span className="text-[var(--color-fg-muted)]">{p.label}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* PIPELINE / HOW IT WORKS */}
      <section className="border-t border-[var(--color-border)] py-20">
        <div className="mx-auto max-w-6xl px-6">
          <div className="mb-12 max-w-2xl">
            <div className="text-xs uppercase tracking-widest text-[var(--color-accent)]">
              How it works
            </div>
            <h2 className="mt-2 text-3xl font-semibold md:text-4xl">
              One pipeline, fully deterministic.
            </h2>
            <p className="mt-3 text-[var(--color-fg-muted)]">
              Every stage is observable. Every job is reproducible from a seed. When a
              warning fires, you know exactly which stage and why.
            </p>
          </div>

          <ol className="grid gap-3 md:grid-cols-2">
            {[
              { n: 1, t: "Preflight QC", d: "Blur, exposure, colour cast — halt early on unsalvageable input." },
              { n: 2, t: "Lens + perspective", d: "Brown-Conrady + Hough verticals. Walls plumb." },
              { n: 3, t: "Bracket fuse", d: "Auto-detect sets via EXIF. Mertens / Debevec on demand." },
              { n: 4, t: "Scene classify", d: "Interior vs exterior vs aerial — gates which stages run." },
              { n: 5, t: "Object removal", d: "SAM 2 click-anywhere mask + LaMa inpaint." },
              { n: 6, t: "Real-estate enhance", d: "Sky replace, lawn boost, window pull. Tag-aware." },
              { n: 7, t: "Virtual staging", d: "SD3.5 + IPAdapter for empty-room → furnished." },
              { n: 8, t: "Twilight transform", d: "Daylight to golden hour with sky tint + window glow." },
              { n: 9, t: "Tone coherency", d: "LAB anchor across listings — no jarring colour shifts." },
              { n: 10, t: "SUPIR upscale", d: "x2 / x4 photoreal upscaling, with Real-ESRGAN ncnn fallback." },
              { n: 11, t: "Encode + EXIF", d: "JPEG quality preserved, EXIF round-tripped (or stripped on request)." },
            ].map((s) => (
              <li
                key={s.n}
                className="flex gap-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4"
              >
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-[var(--color-accent)]/15 font-mono text-sm text-[var(--color-accent)]">
                  {String(s.n).padStart(2, "0")}
                </span>
                <div>
                  <div className="font-medium">{s.t}</div>
                  <div className="mt-1 text-sm text-[var(--color-fg-muted)]">{s.d}</div>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* PRICING */}
      <section id="pricing" className="border-t border-[var(--color-border)] py-20">
        <div className="mx-auto max-w-6xl px-6">
          <div className="mb-12 max-w-2xl">
            <div className="text-xs uppercase tracking-widest text-[var(--color-accent)]">
              Pricing
            </div>
            <h2 className="mt-2 text-3xl font-semibold md:text-4xl">
              Pay per photo or per listing.
            </h2>
            <p className="mt-3 text-[var(--color-fg-muted)]">
              No subscription floor. We make money when you ship listings — same as you.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-3">
            {PRICING_TIERS.map((t) => (
              <div
                key={t.name}
                className={
                  t.highlight
                    ? "relative rounded-lg border border-[var(--color-accent)] bg-[var(--color-surface-elevated)] p-6 ring-1 ring-[var(--color-accent)]/20"
                    : "rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-6"
                }
              >
                {t.highlight && (
                  <div className="absolute -top-3 left-6 rounded-full bg-[var(--color-accent)] px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-black">
                    Most popular
                  </div>
                )}
                <div className="text-sm font-medium uppercase tracking-wider text-[var(--color-fg-muted)]">
                  {t.name}
                </div>
                <div className="mt-3 flex items-baseline gap-2">
                  <span className="text-3xl font-semibold">{t.price}</span>
                  {t.unit && (
                    <span className="text-sm text-[var(--color-fg-muted)]">{t.unit}</span>
                  )}
                </div>
                <p className="mt-2 text-sm text-[var(--color-fg-muted)]">{t.blurb}</p>
                <ul className="mt-6 space-y-2 text-sm">
                  {t.features.map((f) => (
                    <li key={f} className="flex items-start gap-2">
                      <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[var(--color-success)]" />
                      <span>{f}</span>
                    </li>
                  ))}
                </ul>
                <Link
                  href={t.name === "API" ? "mailto:hello@propho.studio" : "/upload"}
                  className={
                    t.highlight
                      ? "mt-6 block w-full rounded-md bg-[var(--color-accent)] py-2 text-center text-sm font-medium text-black transition hover:bg-[var(--color-accent-hover)]"
                      : "mt-6 block w-full rounded-md border border-[var(--color-border)] py-2 text-center text-sm transition hover:bg-[var(--color-surface-elevated)]"
                  }
                >
                  {t.name === "API" ? "Talk to us" : "Get started"}
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-[var(--color-border)] py-20">
        <div className="mx-auto max-w-3xl px-6 text-center">
          <Sun className="mx-auto h-10 w-10 text-[var(--color-accent)]" strokeWidth={1.5} />
          <h2 className="mt-4 text-3xl font-semibold md:text-4xl">
            Ship a listing in the time it takes to make coffee.
          </h2>
          <p className="mt-4 text-[var(--color-fg-muted)]">
            Drop a photo. We'll show you the report, the timing, and the result.
          </p>
          <div className="mt-8 flex justify-center gap-3">
            <Link
              href="/upload"
              className="flex items-center gap-2 rounded-md bg-[var(--color-accent)] px-6 py-3 text-sm font-medium text-black transition hover:bg-[var(--color-accent-hover)]"
            >
              <ImageIcon className="h-4 w-4" />
              Upload a photo
            </Link>
            <Link
              href="/jobs"
              className="flex items-center gap-2 rounded-md border border-[var(--color-border)] px-6 py-3 text-sm transition hover:bg-[var(--color-surface)]"
            >
              <Zap className="h-4 w-4" />
              See recent jobs
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}

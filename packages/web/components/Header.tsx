import Link from "next/link";
import { Aperture } from "lucide-react";
import { publicEnv } from "@/lib/env";

const NAV = [
  { href: "/demo", label: "Live demo" },
  { href: "/upload", label: "Upload" },
  { href: "/jobs", label: "Jobs" },
  { href: "/#features", label: "Features" },
  { href: "/#pricing", label: "Pricing" },
];

export function Header() {
  return (
    <header className="sticky top-0 z-40 border-b border-[var(--color-border)] bg-[var(--color-canvas)]/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link
          href="/"
          className="flex items-center gap-2 text-lg font-semibold tracking-tight"
        >
          <Aperture className="h-6 w-6 text-[var(--color-accent)]" strokeWidth={1.6} />
          <span>{publicEnv.brandName}</span>
        </Link>

        <nav className="hidden items-center gap-7 text-sm md:flex">
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="text-[var(--color-fg-muted)] transition hover:text-[var(--color-fg)]"
            >
              {item.label}
            </Link>
          ))}
        </nav>

        <Link
          href="/upload"
          className="rounded-md bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-black transition hover:bg-[var(--color-accent-hover)]"
        >
          Try a photo
        </Link>
      </div>
    </header>
  );
}

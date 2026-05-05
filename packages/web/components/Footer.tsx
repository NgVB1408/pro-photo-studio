import Link from "next/link";
import { publicEnv } from "@/lib/env";

export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="border-t border-[var(--color-border)] bg-[var(--color-canvas)] py-10">
      <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-6 px-6 text-sm text-[var(--color-fg-muted)] md:flex-row md:items-center">
        <div>
          <div className="font-semibold text-[var(--color-fg)]">{publicEnv.brandName}</div>
          <div>© {year} — production-grade real-estate photo enhancement.</div>
        </div>
        <nav className="flex flex-wrap gap-6">
          <Link href="/jobs" className="hover:text-[var(--color-fg)]">
            Recent jobs
          </Link>
          <Link href="/upload" className="hover:text-[var(--color-fg)]">
            New upload
          </Link>
          <a
            href="https://github.com/NgVB1408/pro-photo-studio"
            className="hover:text-[var(--color-fg)]"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
          <a
            href="mailto:hello@propho.studio"
            className="hover:text-[var(--color-fg)]"
          >
            Contact
          </a>
        </nav>
      </div>
    </footer>
  );
}

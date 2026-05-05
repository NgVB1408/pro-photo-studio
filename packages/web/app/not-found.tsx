import Link from "next/link";

export default function NotFound() {
  return (
    <section className="mx-auto max-w-2xl px-6 py-32 text-center">
      <div className="text-xs uppercase tracking-widest text-[var(--color-accent)]">
        404
      </div>
      <h1 className="mt-2 text-4xl font-semibold">Nothing here.</h1>
      <p className="mt-3 text-[var(--color-fg-muted)]">
        The link is wrong, the job has been deleted, or you typed the URL by hand.
      </p>
      <Link
        href="/"
        className="mt-6 inline-block rounded-md bg-[var(--color-accent)] px-5 py-2.5 text-sm font-medium text-black transition hover:bg-[var(--color-accent-hover)]"
      >
        Back to home
      </Link>
    </section>
  );
}

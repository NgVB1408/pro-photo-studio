import type { Metadata } from "next";
import { UploadDropzone } from "@/components/UploadDropzone";

export const metadata: Metadata = {
  title: "Upload",
  description:
    "Upload a real-estate photo to run through the Pro Photo Studio pipeline.",
};

export default function UploadPage() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-14">
      <div className="mb-10 max-w-2xl">
        <div className="text-xs uppercase tracking-widest text-[var(--color-accent)]">
          New job
        </div>
        <h1 className="mt-2 text-3xl font-semibold md:text-4xl">
          Drop a photo. Watch the pipeline run.
        </h1>
        <p className="mt-3 text-[var(--color-fg-muted)]">
          The original is decoded once and discarded. Only the rendered output is stored.
          Jobs typically finish in 5–60 seconds depending on which stages you enable.
        </p>
      </div>

      <UploadDropzone />
    </section>
  );
}

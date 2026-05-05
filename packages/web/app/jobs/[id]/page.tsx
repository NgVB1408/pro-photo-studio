import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { ApiError, getJob } from "@/lib/api";
import { JobDetail } from "@/components/JobDetail";

export const dynamic = "force-dynamic";
export const revalidate = 0;

type RouteParams = { id: string };

export async function generateMetadata({
  params,
}: {
  params: Promise<RouteParams>;
}): Promise<Metadata> {
  const { id } = await params;
  return { title: `Job ${id.slice(0, 8)}` };
}

export default async function JobPage({
  params,
}: {
  params: Promise<RouteParams>;
}) {
  const { id } = await params;
  let job;
  try {
    job = await getJob(id);
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) {
      notFound();
    }
    throw e;
  }

  return (
    <section className="mx-auto max-w-5xl px-6 py-14">
      {/* `before` image is unavailable from the server (the API does not
          retain originals) — pass null and the slider falls back to a single
          full-bleed result image. The /upload flow is wired to post the file
          and immediately route here, so for live demos the user sees the
          rendered output and report. */}
      <JobDetail initial={job} beforeImageUrl={null} />
    </section>
  );
}

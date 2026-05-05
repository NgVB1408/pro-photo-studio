// Server-side proxy: browser → /api/jobs → FastAPI /v1/jobs.
// The API key lives only on the server so we can put this app on a public
// URL without exposing it.

import { NextRequest, NextResponse } from "next/server";
import { ApiError, createJob, listJobs } from "@/lib/api";
import type { JobCreate } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024; // 50 MB

export async function GET(req: NextRequest): Promise<NextResponse> {
  const limit = Number.parseInt(req.nextUrl.searchParams.get("limit") ?? "50", 10);
  const safeLimit = Number.isFinite(limit) ? Math.max(1, Math.min(500, limit)) : 50;
  try {
    const jobs = await listJobs(safeLimit);
    return NextResponse.json(jobs);
  } catch (e) {
    return errorResponse(e);
  }
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  const ct = req.headers.get("content-type") ?? "";
  if (!ct.includes("multipart/form-data")) {
    return NextResponse.json(
      { detail: "Expected multipart/form-data" },
      { status: 400 },
    );
  }

  const form = await req.formData();
  const image = form.get("image");
  const bodyRaw = form.get("body");

  if (!(image instanceof Blob)) {
    return NextResponse.json({ detail: "Missing image field" }, { status: 400 });
  }
  if (image.size === 0) {
    return NextResponse.json({ detail: "Image upload is empty" }, { status: 400 });
  }
  if (image.size > MAX_UPLOAD_BYTES) {
    return NextResponse.json(
      { detail: `Image exceeds 50 MB limit (${image.size} bytes)` },
      { status: 413 },
    );
  }

  let body: JobCreate;
  try {
    body = JSON.parse(typeof bodyRaw === "string" ? bodyRaw : "{}") as JobCreate;
  } catch {
    return NextResponse.json({ detail: "Invalid body JSON" }, { status: 400 });
  }
  if (!Array.isArray(body.stages) || body.stages.length === 0) {
    return NextResponse.json(
      { detail: "body.stages must be a non-empty list" },
      { status: 400 },
    );
  }

  try {
    const created = await createJob(image, body);
    return NextResponse.json(created, { status: 202 });
  } catch (e) {
    return errorResponse(e);
  }
}

function errorResponse(e: unknown): NextResponse {
  if (e instanceof ApiError) {
    let parsed: unknown = e.body;
    try {
      parsed = JSON.parse(e.body);
    } catch {
      // pass through string body
    }
    return NextResponse.json({ detail: parsed }, { status: e.status });
  }
  console.error("[/api/jobs] upstream error", e);
  return NextResponse.json(
    { detail: "Upstream API unavailable" },
    { status: 502 },
  );
}

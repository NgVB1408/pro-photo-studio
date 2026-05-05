// Streams the rendered image bytes through the Next server so the browser
// never needs the API key. Cached for 60s — results are immutable per job.

import { NextRequest, NextResponse } from "next/server";
import { ApiError, streamJobResult } from "@/lib/api";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
): Promise<NextResponse> {
  const { id } = await params;
  try {
    const { bytes, contentType } = await streamJobResult(id);
    return new NextResponse(bytes, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "private, max-age=60",
        "Content-Disposition": `inline; filename="${id}.jpg"`,
      },
    });
  } catch (e) {
    if (e instanceof ApiError) {
      return NextResponse.json({ detail: e.body }, { status: e.status });
    }
    console.error("[/api/jobs/:id/result] upstream error", e);
    return NextResponse.json({ detail: "Upstream unavailable" }, { status: 502 });
  }
}

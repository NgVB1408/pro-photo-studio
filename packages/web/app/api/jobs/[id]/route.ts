import { NextRequest, NextResponse } from "next/server";
import { ApiError, getJob } from "@/lib/api";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
): Promise<NextResponse> {
  const { id } = await params;
  try {
    const job = await getJob(id);
    return NextResponse.json(job);
  } catch (e) {
    if (e instanceof ApiError) {
      let parsed: unknown = e.body;
      try {
        parsed = JSON.parse(e.body);
      } catch {
        // string body
      }
      return NextResponse.json({ detail: parsed }, { status: e.status });
    }
    console.error("[/api/jobs/:id] upstream error", e);
    return NextResponse.json({ detail: "Upstream unavailable" }, { status: 502 });
  }
}

// Server-side proxy for the FastAPI /v1/auto endpoint.
// Mirrors /api/jobs but with simpler form-data — no body JSON, just an image
// plus optional query params. The browser receives a JobOut and starts polling.

import { NextRequest, NextResponse } from "next/server";
import { ApiError, autoEnhance } from "@/lib/api";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
const VALID_SCENES = new Set(["interior", "exterior", "aerial"]);

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
  if (!(image instanceof Blob) || image.size === 0) {
    return NextResponse.json({ detail: "Missing or empty image" }, { status: 400 });
  }
  if (image.size > MAX_UPLOAD_BYTES) {
    return NextResponse.json(
      { detail: `Image exceeds 50 MB limit (${image.size} bytes)` },
      { status: 413 },
    );
  }
  const seedRaw = form.get("seed");
  const seed = typeof seedRaw === "string" ? Number.parseInt(seedRaw, 10) : 42;
  const sceneRaw = form.get("scene");
  let scene: string | undefined;
  if (typeof sceneRaw === "string" && sceneRaw.length > 0) {
    if (!VALID_SCENES.has(sceneRaw)) {
      return NextResponse.json(
        { detail: `scene must be one of: ${[...VALID_SCENES].join(", ")}` },
        { status: 400 },
      );
    }
    scene = sceneRaw;
  }
  const twilight = form.get("twilight") === "true";

  try {
    const created = await autoEnhance(image, {
      seed: Number.isFinite(seed) ? seed : 42,
      scene,
      twilight,
    });
    return NextResponse.json(created, { status: 202 });
  } catch (e) {
    if (e instanceof ApiError) {
      let parsed: unknown = e.body;
      try {
        parsed = JSON.parse(e.body);
      } catch {
        // pass through string body
      }
      return NextResponse.json({ detail: parsed }, { status: e.status });
    }
    console.error("[/api/auto] upstream error", e);
    return NextResponse.json(
      { detail: "Upstream API unavailable" },
      { status: 502 },
    );
  }
}

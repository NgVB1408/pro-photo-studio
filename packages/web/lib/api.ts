// Server-side API client for the FastAPI backend.
// Anything importing this module runs on the Node runtime and the API key
// is never shipped to the browser.
import "server-only";

import { env } from "./env";
import type { JobCreate, JobOut } from "./types";

const DEFAULT_TIMEOUT_MS = 30_000;

class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: string,
  ) {
    super(`API ${status}: ${body.slice(0, 200)}`);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  init: RequestInit & { timeoutMs?: number } = {},
): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, headers, ...rest } = init;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const resp = await fetch(`${env.apiUrl}${path}`, {
      ...rest,
      headers: {
        "X-API-Key": env.apiKey,
        ...headers,
      },
      signal: controller.signal,
      cache: "no-store",
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new ApiError(resp.status, text);
    }

    const ct = resp.headers.get("content-type") ?? "";
    if (ct.includes("application/json")) {
      return (await resp.json()) as T;
    }
    return (await resp.text()) as unknown as T;
  } finally {
    clearTimeout(timer);
  }
}

// ---- Public API surface used by route handlers + server components ----

export async function listJobs(limit = 50): Promise<JobOut[]> {
  return request<JobOut[]>(`/v1/jobs?limit=${limit}`, { method: "GET" });
}

export async function getJob(jobId: string): Promise<JobOut> {
  return request<JobOut>(`/v1/jobs/${encodeURIComponent(jobId)}`, { method: "GET" });
}

export async function createJob(image: Blob, body: JobCreate): Promise<JobOut> {
  const fd = new FormData();
  fd.append("image", image, "upload.jpg");
  fd.append("body", JSON.stringify(body));
  return request<JobOut>("/v1/jobs", {
    method: "POST",
    body: fd,
    timeoutMs: 60_000,
  });
}

export async function streamJobResult(jobId: string): Promise<{
  bytes: ArrayBuffer;
  contentType: string;
}> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 60_000);
  try {
    const resp = await fetch(
      `${env.apiUrl}/v1/jobs/${encodeURIComponent(jobId)}/result`,
      {
        method: "GET",
        headers: { "X-API-Key": env.apiKey },
        signal: controller.signal,
        cache: "no-store",
      },
    );
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new ApiError(resp.status, text);
    }
    return {
      bytes: await resp.arrayBuffer(),
      contentType: resp.headers.get("content-type") ?? "image/jpeg",
    };
  } finally {
    clearTimeout(timer);
  }
}

export { ApiError };

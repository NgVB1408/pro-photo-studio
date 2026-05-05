// Server-only env access. Importing this module from a client component will
// fail at build time because of the `server-only` import below.
import "server-only";

function required(name: string): string {
  const v = process.env[name];
  if (!v || !v.trim()) {
    throw new Error(
      `Missing required env var ${name}. Set it in .env.local (see .env.example).`,
    );
  }
  return v;
}

export const env = {
  apiUrl: (process.env.PPS_API_URL ?? "http://localhost:8000").replace(/\/+$/, ""),
  apiKey: required("PPS_API_KEY"),
  publicCdn: process.env.PPS_PUBLIC_CDN ?? "",
};

export const publicEnv = {
  brandName: process.env.NEXT_PUBLIC_BRAND_NAME ?? "Pro Photo Studio",
  brandTagline:
    process.env.NEXT_PUBLIC_BRAND_TAGLINE ?? "Real-estate photo enhancement, automated.",
};

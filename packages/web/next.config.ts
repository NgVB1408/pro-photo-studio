import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // Emit a self-contained server bundle in `.next/standalone/` so the prod
  // Docker image can copy a single tree and run with `node server.js` —
  // no `pnpm install` step in the runtime stage.
  output: "standalone",
  // Image domains: allow our own API + S3-compatible CDN domains.
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost" },
      { protocol: "http", hostname: "127.0.0.1" },
      { protocol: "https", hostname: "*.r2.dev" },
      { protocol: "https", hostname: "*.amazonaws.com" },
      { protocol: "https", hostname: "cdn.propho.studio" },
    ],
  },
  experimental: {
    // Server actions are used for the upload proxy so the API key never reaches the browser bundle.
    serverActions: { bodySizeLimit: "50mb" },
  },
};

export default config;

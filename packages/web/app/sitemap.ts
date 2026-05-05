import type { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  const base =
    process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/+$/, "") ?? "https://propho.studio";
  const now = new Date();
  return [
    { url: `${base}/`, lastModified: now, changeFrequency: "weekly", priority: 1 },
    { url: `${base}/upload`, lastModified: now, changeFrequency: "monthly", priority: 0.8 },
    { url: `${base}/jobs`, lastModified: now, changeFrequency: "weekly", priority: 0.6 },
    { url: `${base}/demo`, lastModified: now, changeFrequency: "weekly", priority: 0.9 },
  ];
}

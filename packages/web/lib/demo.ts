// Curated public demo gallery — real before/after pairs rendered by the
// production pipeline. The asset files live in `public/demo/<id>/{before,after}.jpg`
// so they're served as plain static content and need no API key.

export type DemoPhoto = {
  id: string;
  title: string;
  scene: "Interior" | "Exterior" | "Aerial";
  duration_ms: number;
  stages: string[];
  highlights: string[];
  before_src: string;
  after_src: string;
  report_src: string;
};

export const DEMO_PHOTOS: DemoPhoto[] = [
  {
    id: "0C1A5044",
    title: "Open-plan living, west-facing windows",
    scene: "Interior",
    duration_ms: 3711,
    stages: ["preflight", "perspective", "real_estate", "enhance_studio"],
    highlights: [
      "Window pull recovers blown highlights without halo",
      "Wood-floor warmth preserved while brightening shadows",
      "Halo-free local detail via guided filter, not unsharp",
    ],
    before_src: "/demo/0C1A5044/before.jpg",
    after_src: "/demo/0C1A5044/after.jpg",
    report_src: "/demo/0C1A5044/report.json",
  },
  {
    id: "0C1A5029",
    title: "Master bedroom — backlit windows",
    scene: "Interior",
    duration_ms: 3580,
    stages: ["preflight", "real_estate", "enhance_studio"],
    highlights: [
      "Bedding texture retained at 4K",
      "Skin-safe vibrance avoids the orange-cast competitors ship",
      "LAB-anchor tone keeps the listing colour-consistent",
    ],
    before_src: "/demo/0C1A5029/before.jpg",
    after_src: "/demo/0C1A5029/after.jpg",
    report_src: "/demo/0C1A5029/report.json",
  },
  {
    id: "0C1A5035",
    title: "Kitchen — mixed lighting, pendants on",
    scene: "Interior",
    duration_ms: 3420,
    stages: ["preflight", "real_estate", "enhance_studio"],
    highlights: [
      "Tungsten / daylight mix neutralised cleanly",
      "Stone counter texture sharper than competitor B",
      "Stage report shows preflight passed at 0.4s",
    ],
    before_src: "/demo/0C1A5035/before.jpg",
    after_src: "/demo/0C1A5035/after.jpg",
    report_src: "/demo/0C1A5035/report.json",
  },
  {
    id: "0C1A5014",
    title: "Hallway — flat ambient light",
    scene: "Interior",
    duration_ms: 3290,
    stages: ["preflight", "perspective", "real_estate", "enhance_studio"],
    highlights: [
      "Vertical correction nudges walls plumb (Hough)",
      "Shadow lift opens the dark end of the hallway",
      "Identical seed = byte-identical output across runs",
    ],
    before_src: "/demo/0C1A5014/before.jpg",
    after_src: "/demo/0C1A5014/after.jpg",
    report_src: "/demo/0C1A5014/report.json",
  },
];

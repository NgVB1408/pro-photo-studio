import type { Metadata, Viewport } from "next";
import { publicEnv } from "@/lib/env";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: `${publicEnv.brandName} — ${publicEnv.brandTagline}`,
    template: `%s — ${publicEnv.brandName}`,
  },
  description:
    "Pro Photo Studio enhances real-estate photos at scale. Sky replace, HDR fusion, virtual staging, instruction-based editing.",
  applicationName: publicEnv.brandName,
  authors: [{ name: "Pro Photo Studio" }],
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  themeColor: "#0b0d10",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  );
}

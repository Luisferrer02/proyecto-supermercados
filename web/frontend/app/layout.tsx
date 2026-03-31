import type { Metadata } from "next";
import { Inter, DM_Sans } from "next/font/google";
import { Geist_Mono } from "next/font/google";
import "./globals.css";
import { TopNav } from "@/components/TopNav";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const dmSans = DM_Sans({
  variable: "--font-dm-sans",
  subsets: ["latin"],
  weight: ["700", "800"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ShelfOpt — MLOps Dashboard",
  description: "Supermarket shelf optimization pipeline",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${dmSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        <TopNav />
        <main className="flex-1 w-full max-w-5xl mx-auto px-6 py-10">{children}</main>
      </body>
    </html>
  );
}

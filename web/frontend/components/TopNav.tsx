"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Upload,
  Cog,
  BarChart3,
  Database,
  Sparkles,
} from "lucide-react";

const nav = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/upload", label: "Upload", icon: Upload },
  { href: "/train", label: "Train", icon: Cog },
  { href: "/evaluate", label: "Evaluate", icon: BarChart3 },
  { href: "/ingest", label: "Ingest", icon: Database },
  { href: "/predict", label: "Predict", icon: Sparkles },
];

export function TopNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <div className="max-w-6xl mx-auto flex h-16 items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2">
          <span className="font-heading text-xl font-extrabold tracking-tight text-primary">
            SHELFOPT
          </span>
          <span className="text-xs text-muted-foreground hidden sm:inline">
            MLOps Pipeline
          </span>
        </Link>

        <nav className="flex items-center gap-1">
          {nav.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  active
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent"
                }`}
              >
                <Icon size={16} />
                <span className="hidden md:inline">{label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="text-xs text-muted-foreground hidden lg:block">
          Mercadona &middot; 4,772 products
        </div>
      </div>
    </header>
  );
}

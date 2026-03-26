"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const nav = [
  { href: "/", label: "Dashboard", icon: "⬛" },
  { href: "/upload", label: "Upload CSV", icon: "⬆" },
  { href: "/train", label: "Train Models", icon: "⚙" },
  { href: "/evaluate", label: "Evaluate", icon: "📊" },
  { href: "/ingest", label: "Ingest", icon: "🗄" },
  { href: "/predict", label: "Predict", icon: "🔮" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-52 shrink-0 border-r border-border bg-sidebar flex flex-col h-screen sticky top-0">
      <div className="p-4 border-b border-border">
        <h1 className="font-bold text-sm text-sidebar-foreground">ShelfOpt</h1>
        <p className="text-xs text-muted-foreground mt-0.5">MLOps Pipeline</p>
      </div>
      <nav className="flex-1 p-2 space-y-0.5">
        {nav.map(({ href, label, icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-md text-sm transition-colors ${
                active
                  ? "bg-sidebar-primary text-sidebar-primary-foreground"
                  : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              }`}
            >
              <span className="text-base leading-none">{icon}</span>
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="p-3 border-t border-border">
        <p className="text-xs text-muted-foreground">Mercadona · 4,772 products</p>
      </div>
    </aside>
  );
}

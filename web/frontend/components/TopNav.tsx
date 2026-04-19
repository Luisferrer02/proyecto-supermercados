"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useRef, useEffect } from "react";
import {
  Sparkles,
  ChevronDown,
  Upload,
  Cog,
  BarChart3,
  Database,
  Wand2,
} from "lucide-react";

// Main flow — what a non-technical user will touch
const MAIN_NAV = [
  { href: "/", label: "Optimizar", icon: Sparkles },
];

// Everything below is hidden under "Avanzado" because it exposes
// training metrics, epochs, MSE… useful for defence but noise for a
// supermarket manager.
const ADVANCED_NAV = [
  { href: "/upload",   label: "Subir archivos (avanzado)", icon: Upload },
  { href: "/ingest",   label: "Ingesta manual",            icon: Database },
  { href: "/train",    label: "Entrenamiento comparado",   icon: Cog },
  { href: "/evaluate", label: "Gráficas de evaluación",    icon: BarChart3 },
  { href: "/predict",  label: "Predicción paso a paso",    icon: Wand2 },
];

export function TopNav() {
  const pathname = usePathname();
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const advancedRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (advancedRef.current && !advancedRef.current.contains(e.target as Node)) {
        setAdvancedOpen(false);
      }
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const isAdvancedActive = ADVANCED_NAV.some(n => n.href === pathname);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <div className="max-w-6xl mx-auto flex h-16 items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2">
          <span className="font-heading text-xl font-extrabold tracking-tight text-primary">
            SHELFOPT
          </span>
          <span className="text-xs text-muted-foreground hidden sm:inline">
            Optimización de estanterías
          </span>
        </Link>

        <nav className="flex items-center gap-1">
          {MAIN_NAV.map(({ href, label, icon: Icon }) => {
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

          <div ref={advancedRef} className="relative">
            <button
              onClick={() => setAdvancedOpen(o => !o)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                isAdvancedActive
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent"
              }`}
            >
              <span>Avanzado</span>
              <ChevronDown size={14} className={`transition-transform ${advancedOpen ? "rotate-180" : ""}`} />
            </button>

            {advancedOpen && (
              <div className="absolute right-0 mt-2 w-64 rounded-lg border border-border bg-background shadow-lg py-1">
                {ADVANCED_NAV.map(({ href, label, icon: Icon }) => {
                  const active = pathname === href;
                  return (
                    <Link
                      key={href}
                      href={href}
                      onClick={() => setAdvancedOpen(false)}
                      className={`flex items-center gap-2 px-3 py-2 text-sm transition-colors ${
                        active
                          ? "bg-accent text-foreground"
                          : "text-muted-foreground hover:text-foreground hover:bg-accent/60"
                      }`}
                    >
                      <Icon size={15} />
                      <span>{label}</span>
                    </Link>
                  );
                })}
              </div>
            )}
          </div>
        </nav>

        <div className="text-xs text-muted-foreground hidden lg:block">
          Mercadona &middot; 4,772 productos
        </div>
      </div>
    </header>
  );
}

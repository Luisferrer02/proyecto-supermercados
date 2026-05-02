"use client";

import { useCallback, useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LiveLog } from "@/components/LiveLog";
import { api } from "@/lib/api";
import { X } from "lucide-react";

interface ChartInfo {
  name: string;
  exists: boolean;
  url: string;
}

const CHART_LABELS: Record<string, string> = {
  "mse_comparison.png": "Precisión de los modelos (error típico)",
  "profit_comparison.png": "Beneficio optimizado por modelo",
  "rack_comparison.png": "Mejora por estantería",
  "alluvial_diagram.png": "Movimiento de productos entre baldas",
};

interface ExpandedChart { url: string; title: string }

export default function EvaluatePage() {
  const [charts, setCharts] = useState<ChartInfo[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [streamUrl, setStreamUrl] = useState("");
  const [showLog, setShowLog] = useState(false);
  const [cacheBust, setCacheBust] = useState(() => Date.now());
  const [expanded, setExpanded] = useState<ExpandedChart | null>(null);

  const loadStatus = async () => {
    try {
      const d = await api.evaluateStatus();
      setCharts(d.charts ?? []);
    } catch {}
  };

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { loadStatus(); }, []);

  const runEvaluate = () => {
    setStreamUrl(api.evaluateStreamUrl());
    setStreaming(true);
    setShowLog(true);
  };

  const closeExpanded = useCallback(() => setExpanded(null), []);

  // Escape key + body scroll lock — only when the modal is mounted, so the
  // page behaves normally otherwise.
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") closeExpanded(); };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [expanded, closeExpanded]);

  const existingCharts = charts.filter((c) => c.exists);

  return (
    <div className="space-y-8 max-w-4xl relative">
      <div
        className="fixed inset-0 bg-cover bg-center opacity-[0.09] blur-sm pointer-events-none -z-10"
        style={{ backgroundImage: "url(/evaluate.png)" }}
      />
      <div className="animate-fade-in-up">
        <h1 className="text-3xl font-heading font-extrabold uppercase tracking-tight text-secondary">
          Gráficas de evaluación
        </h1>
        <p className="text-muted-foreground text-sm mt-2">
          Ejecuta <code className="bg-muted px-1.5 py-0.5 rounded text-xs">03_evaluate.py</code> para generar las
          gráficas comparativas a partir de los resultados de entrenamiento.
        </p>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={runEvaluate}
          disabled={streaming}
          className="bg-primary text-primary-foreground hover:bg-primary/90 px-5 py-2 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
        >
          {streaming ? "Generando gráficas…" : "Regenerar gráficas"}
        </button>
        {streaming && (
          <button
            onClick={() => { api.evaluateStop(); setStreaming(false); }}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            Cancelar
          </button>
        )}
        <Badge variant={existingCharts.length > 0 ? "default" : "secondary"}>
          {existingCharts.length}/{charts.length} disponibles
        </Badge>
      </div>

      {showLog && streamUrl && (
        <LiveLog
          url={streamUrl}
          onDone={() => { setStreaming(false); loadStatus(); setCacheBust(Date.now()); }}
        />
      )}

      {existingCharts.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {existingCharts.map((chart) => {
            const title = CHART_LABELS[chart.name] ?? chart.name;
            const url = `${api.resultUrl(chart.name)}?t=${cacheBust}`;
            return (
              <Card
                key={chart.name}
                role="button"
                tabIndex={0}
                aria-label={`Ampliar ${title}`}
                onClick={() => setExpanded({ url, title })}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setExpanded({ url, title });
                  }
                }}
                className="shadow-sm cursor-zoom-in transition-all hover:shadow-md hover:ring-2 hover:ring-primary/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary group"
              >
                <CardHeader className="pb-2">
                  <CardTitle className="text-xs text-muted-foreground flex items-center justify-between gap-2">
                    <span>{title}</span>
                    <span className="text-[10px] uppercase tracking-wide text-primary opacity-0 group-hover:opacity-100 transition-opacity">
                      Pulsa para ampliar
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {/* Cache-bust with timestamp to reload after regeneration */}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={url}
                    alt={chart.name}
                    className="w-full rounded-lg border border-border transition-transform group-hover:scale-[1.01]"
                  />
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {charts.length > 0 && existingCharts.length === 0 && !streaming && (
        <div className="text-sm text-muted-foreground">
          Aún no hay gráficas. Pulsa <span className="font-medium">«Regenerar gráficas»</span>, o
          ejecuta antes el entrenamiento.
        </div>
      )}

      {/*
        Lightweight image lightbox. No portal, no extra deps — the overlay
        sits at z-50 above the sticky TopNav (z-50 too) thanks to render
        order, the body scroll lock keeps the page behind it still while
        open, and the click-on-backdrop / Escape / X all close it.
      */}
      {expanded && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center p-4 sm:p-8 bg-black/70 backdrop-blur-sm animate-fade-in-up"
          onClick={closeExpanded}
          role="dialog"
          aria-modal="true"
          aria-label={expanded.title}
        >
          <div
            className="relative max-w-[95vw] max-h-[92vh] flex flex-col bg-background rounded-xl shadow-2xl border border-border overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between gap-4 px-5 py-3 border-b border-border bg-card">
              <h2 className="text-sm font-semibold truncate">{expanded.title}</h2>
              <button
                type="button"
                onClick={closeExpanded}
                aria-label="Cerrar"
                className="shrink-0 inline-flex items-center justify-center h-8 w-8 rounded-md text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
              >
                <X size={18} />
              </button>
            </div>
            <div className="flex-1 overflow-auto p-4 bg-muted/30">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={expanded.url}
                alt={expanded.title}
                className="block mx-auto max-w-full max-h-[80vh] object-contain rounded-md"
              />
            </div>
            <div className="px-5 py-2 text-[11px] text-muted-foreground border-t border-border bg-card">
              Pulsa <kbd className="px-1.5 py-0.5 rounded bg-muted text-[10px] font-mono">Esc</kbd>,
              haz clic fuera o pulsa la X para cerrar.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

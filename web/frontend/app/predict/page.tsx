"use client";

import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LiveLog } from "@/components/LiveLog";
import { ShelfMap, type Product } from "@/components/ShelfMap";
import { ShelfSankey } from "@/components/ShelfSankey";
import { api } from "@/lib/api";

interface RackSummary {
  original: number;
  optimized: number;
  products: number;
}

interface Reason {
  code: string;
  text: string;
}

interface Explanation {
  product: string;
  old_shelf: number;
  new_shelf: number;
  margin_pct: number;
  monthly_sales: number;
  profit_score: number;
  reasons: Reason[];
}

interface ExplanationsPayload {
  n_products_moved: number;
  by_rack: Record<string, Explanation[]>;
}

interface PredictResult {
  products: Product[];
  forecast: Record<string, number>;
  forecastSource?: string | null;
  rackSummary: Record<string, RackSummary>;
  explanations?: ExplanationsPayload | null;
}

interface Movement { from: number; to: number; count: number }

interface OptimizeAggregate {
  movements?: Movement[];
}

export default function PredictPage() {
  const [month, setMonth] = useState("2026-01");
  const [category, setCategory] = useState("");
  const [dryRun, setDryRun] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamUrl, setStreamUrl] = useState("");
  const [showLog, setShowLog] = useState(false);
  const [predError, setPredError] = useState<string | null>(null);
  const [results, setResults] = useState<PredictResult | null>(null);
  const [aggregate, setAggregate] = useState<OptimizeAggregate | null>(null);
  const [selectedRack, setSelectedRack] = useState("");
  const [pastMonths, setPastMonths] = useState<string[]>([]);

  useEffect(() => {
    api.predictList().then((d) => setPastMonths(d.months ?? [])).catch(() => {});
  }, []);

  const startPrediction = () => {
    setStreamUrl(api.predictStreamUrl(month, category || undefined, dryRun));
    setStreaming(true);
    setShowLog(true);
    setPredError(null);
    setResults(null);
    setAggregate(null);
  };

  const loadResults = async () => {
    try {
      const d = await api.predictResults(month);
      if (!d || d.error || !Array.isArray(d.products)) {
        setPredError(d?.error ?? "La predicción terminó pero no se guardaron resultados. Mira el log de arriba para más detalles.");
        return;
      }
      setPredError(null);
      setResults(d);
      const racks = [...new Set(d.products.map((p: Product) => p.rack_id ?? p.Category ?? ""))].filter(Boolean);
      setSelectedRack((racks[0] as string) ?? "");

      // Reuse the aggregate endpoint that powers the home page sankey.
      // It reads the same optimised CSV that 05_predict.py just wrote, so
      // movements/KPIs are available without duplicating logic here.
      try {
        const agg = await api.optimizeResults(month);
        if (agg && Array.isArray(agg.movements)) setAggregate(agg);
      } catch { /* aggregate is best-effort, the page still works without it */ }
    } catch {
      setPredError("No se pudieron cargar los resultados del servidor.");
    }
  };

  const selectedProducts = results?.products?.filter(
    (p) => (p.rack_id ?? p.Category ?? "") === selectedRack
  ) ?? [];

  // For each unique rack present in the results, look up a representative
  // category name. This way the dropdown shows e.g. "Rack 47 — Aves y jamón
  // cocido" instead of the bare numeric id, which made the long list
  // impossible to scan.
  const rackOptions = useMemo(() => {
    if (!results?.products) return [] as { id: string; label: string }[];
    const m = new Map<string, string>();
    for (const p of results.products) {
      const id = (p.rack_id ?? p.Category ?? "").toString();
      if (!id || m.has(id)) continue;
      const cat = (p.Category ?? "").toString().trim();
      m.set(id, cat ? `Rack ${id} — ${cat}` : `Rack ${id}`);
    }
    return [...m.entries()]
      .sort((a, b) => Number(a[0]) - Number(b[0]) || a[0].localeCompare(b[0]))
      .map(([id, label]) => ({ id, label }));
  }, [results]);

  const topForecast = results?.forecast
    ? Object.entries(results.forecast)
        .sort(([, a], [, b]) => Math.abs(b - 1) - Math.abs(a - 1))
        .slice(0, 12)
    : [];

  const rackSummary = selectedRack && results?.rackSummary?.[selectedRack];

  // Map rack_id → category (first category seen for that rack), so the
  // summary table shows readable names instead of raw rack ids.
  const rackCategoryMap = useMemo(() => {
    const map = new Map<string, string>();
    if (results?.products) {
      for (const p of results.products) {
        const id = (p.rack_id ?? p.Category ?? "").toString();
        if (!id || map.has(id)) continue;
        if (p.Category) map.set(id, p.Category.toString());
      }
    }
    return map;
  }, [results]);

  return (
    <div className="space-y-8 max-w-5xl relative">
      <div
        className="fixed inset-0 bg-cover bg-center opacity-[0.09] blur-sm pointer-events-none -z-10"
        style={{ backgroundImage: "url(/predict.png)" }}
      />
      <div className="animate-fade-in-up">
        <h1 className="text-3xl font-heading font-extrabold uppercase tracking-tight text-secondary">
          Predicción avanzada
        </h1>
        <p className="text-muted-foreground text-sm mt-2">
          Ejecuta el script <code className="bg-muted px-1.5 py-0.5 rounded text-xs">05_predict.py</code>:
          recuperación RAG → predicción LLM → optimización en ensemble.
        </p>
      </div>

      {/* Config form */}
      <Card className="shadow-sm">
        <CardContent className="pt-4 space-y-4">
          <div className="flex flex-wrap gap-4">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Mes objetivo</span>
              <input
                type="month"
                value={month}
                onChange={(e) => setMonth(e.target.value)}
                className="bg-input border border-border rounded-lg px-3 py-1.5 text-sm focus:ring-2 focus:ring-primary/30 focus:border-primary outline-none transition-shadow"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Categoría (opcional)</span>
              <input
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                placeholder="ej. Fruta"
                className="bg-input border border-border rounded-lg px-3 py-1.5 text-sm w-40 focus:ring-2 focus:ring-primary/30 focus:border-primary outline-none transition-shadow"
              />
            </label>
            <label
              className="flex items-center gap-2 self-end pb-1"
              title="Cuando está activado, no se consulta al modelo de lenguaje; se usan multiplicadores estacionales predefinidos. Útil si la API está caída o para reproducibilidad."
            >
              <input
                type="checkbox"
                checked={dryRun}
                onChange={(e) => setDryRun(e.target.checked)}
                className="rounded accent-primary"
              />
              <span className="text-sm">Modo sin IA (usar temporada fija)</span>
            </label>
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={startPrediction}
              disabled={streaming}
              className="bg-primary text-primary-foreground hover:bg-primary/90 px-5 py-2 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
            >
              {streaming ? "Ejecutando predicción…" : "Lanzar predicción"}
            </button>
            {streaming && (
              <button
                onClick={() => { api.predictStop(); setStreaming(false); }}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
              >
                Cancelar
              </button>
            )}

            {pastMonths.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Cargar resultado anterior:</span>
                <select
                  className="bg-input border border-border rounded-lg px-2 py-1 text-xs focus:ring-2 focus:ring-primary/30 focus:border-primary outline-none transition-shadow"
                  onChange={(e) => { setMonth(e.target.value); setTimeout(loadResults, 100); }}
                  defaultValue=""
                >
                  <option value="" disabled>Selecciona mes</option>
                  {pastMonths.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {showLog && streamUrl && (
        <LiveLog
          url={streamUrl}
          onDone={() => { setStreaming(false); loadResults(); }}
        />
      )}

      {predError && !streaming && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
          {predError}
        </div>
      )}

      {results && (
        <div className="space-y-6">
          {/* Global movement Sankey — same component used on the home
              page so Avanzado shows the same visual grammar. The card is
              hidden when the optimize aggregate has no movement data. */}
          {aggregate?.movements && aggregate.movements.length > 0 && (
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle className="text-sm">
                  Reorganización de productos por balda
                  <span className="block text-xs font-normal text-muted-foreground mt-0.5">
                    Vista global de cuántos productos cambian de balda en toda la tienda
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ShelfSankey movements={aggregate.movements} />
              </CardContent>
            </Card>
          )}

          {/* Forecast multipliers */}
          {topForecast.length > 0 && (
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle className="text-sm">Ajustes estacionales por categoría (top 12)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                  {topForecast.map(([cat, mult]) => (
                    <div key={cat} className="bg-muted rounded-lg p-2.5 text-xs">
                      <div className="font-medium truncate">{cat}</div>
                      <div className={`text-base font-bold mt-0.5 ${mult > 1 ? "text-green-600" : "text-red-600"}`}>
                        ×{mult.toFixed(2)}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Explanations for moved products in the selected rack */}
          {results.explanations?.by_rack?.[selectedRack]?.length ? (
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle className="text-sm">
                  ¿Por qué se han movido estos productos?
                  <span className="block text-xs font-normal text-muted-foreground mt-0.5">
                    Explicación de cada reubicación en esta estantería
                    ({results.explanations.by_rack[selectedRack].length} productos)
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3 text-sm">
                  {results.explanations.by_rack[selectedRack].slice(0, 8).map((e) => (
                    <li key={e.product} className="border-l-2 border-primary/40 pl-3">
                      <div className="font-medium">{e.product}</div>
                      <div className="text-xs text-muted-foreground mt-0.5">
                        Balda {e.old_shelf} → Balda {e.new_shelf}
                        {" · "}margen {e.margin_pct}%
                        {" · "}{e.monthly_sales.toLocaleString()} ventas/mes
                      </div>
                      <ul className="mt-1 space-y-0.5 text-xs text-foreground/80">
                        {e.reasons.map((r, i) => (
                          <li key={i}>• {r.text}</li>
                        ))}
                      </ul>
                    </li>
                  ))}
                </ul>
                {results.explanations.by_rack[selectedRack].length > 8 && (
                  <p className="text-xs text-muted-foreground mt-3">
                    + {results.explanations.by_rack[selectedRack].length - 8} productos
                    más reubicados en esta estantería.
                  </p>
                )}
              </CardContent>
            </Card>
          ) : null}

          {/* Per-rack detail — Shelf Map */}
          <Card className="shadow-sm">
            <CardHeader>
              <div className="flex items-center gap-3 flex-wrap">
                <CardTitle className="text-sm">Detalle por estantería</CardTitle>
                <select
                  value={selectedRack}
                  onChange={(e) => setSelectedRack(e.target.value)}
                  className="bg-input border border-border rounded-lg px-2 py-1 text-xs focus:ring-2 focus:ring-primary/30 focus:border-primary outline-none transition-shadow max-w-xs"
                >
                  {rackOptions.map((o) => (
                    <option key={o.id} value={o.id}>{o.label}</option>
                  ))}
                </select>
                <span className="text-xs text-muted-foreground">
                  {rackOptions.length} estanterías
                </span>
                {rackSummary && (
                  <span className="text-xs text-muted-foreground">
                    {rackSummary.products} productos ·{" "}
                    <span className={rackSummary.optimized >= rackSummary.original ? "text-green-600" : "text-red-600"}>
                      {rackSummary.optimized >= rackSummary.original ? "+" : ""}€{Math.round(rackSummary.optimized - rackSummary.original).toLocaleString()} mejora
                    </span>
                  </span>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <ShelfMap products={selectedProducts} rackId={selectedRack} />
            </CardContent>
          </Card>

          {/* Rack summary table */}
          {results.rackSummary && Object.keys(results.rackSummary).length > 0 && (
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle className="text-sm">
                  Resumen de beneficio por estantería (top 20)
                  <span className="block text-xs font-normal text-muted-foreground mt-0.5">
                    Pulsa una fila para ver el detalle de esa estantería
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border text-muted-foreground">
                      <th className="text-left pb-2">Estantería / Categoría</th>
                      <th className="text-right pb-2">Productos</th>
                      <th className="text-right pb-2">Antes (€)</th>
                      <th className="text-right pb-2">Optimizado (€)</th>
                      <th className="text-right pb-2">Mejora</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(results.rackSummary)
                      .sort(([, a], [, b]) => (b.optimized - b.original) - (a.optimized - a.original))
                      .slice(0, 20)
                      .map(([rack, s]) => {
                        const lift = s.optimized - s.original;
                        const cat = rackCategoryMap.get(rack);
                        const label = cat ? `Rack ${rack} — ${cat}` : `Rack ${rack}`;
                        return (
                          <tr
                            key={rack}
                            className={`border-b border-border last:border-0 cursor-pointer hover:bg-accent/30 ${selectedRack === rack ? "bg-accent/30" : ""}`}
                            onClick={() => setSelectedRack(rack)}
                          >
                            <td className="py-1.5 font-medium">{label}</td>
                            <td className="text-right text-muted-foreground">{s.products}</td>
                            <td className="text-right">€{Math.round(s.original).toLocaleString()}</td>
                            <td className="text-right">€{Math.round(s.optimized).toLocaleString()}</td>
                            <td className={`text-right font-semibold ${lift >= 0 ? "text-green-600" : "text-red-600"}`}>
                              {lift >= 0 ? "+" : ""}€{Math.round(lift).toLocaleString()}
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {!results && !streaming && !showLog && (
        <div className="text-sm text-muted-foreground">
          Lanza la ingesta primero para construir la base de conocimiento; después ejecuta una predicción.
        </div>
      )}
    </div>
  );
}

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, BASE } from "@/lib/api";
import { ShelfSankey } from "@/components/ShelfSankey";
import { parseLine, type FriendlyProgress } from "@/lib/progressParser";

type Phase = "upload" | "running" | "results";

interface UploadedFile { name: string; size: number; }
interface Kpi {
  profitOriginal: number;
  profitOptimized: number;
  profitLiftEur: number;
  profitLiftPct: number;
  productsMoved: number;
  totalProducts: number;
  racksImproved: number;
}
interface Movement { from: number; to: number; count: number; }
interface RackRow { rack: string; products: number; original: number; optimized: number; lift: number; }
interface OptimizeResults {
  kpi: Kpi;
  movements: Movement[];
  racks: RackRow[];
  multipliers: Record<string, number>;
  forecastSource: string | null;
}

export default function HomePage() {
  const [phase, setPhase] = useState<Phase>("upload");
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [month, setMonth] = useState<string>("2026-01");
  const [dryRun, setDryRun] = useState(false);
  const [progress, setProgress] = useState<FriendlyProgress>({ stage: "idle", label: "" });
  const [currentStep, setCurrentStep] = useState<{ index: number; total: number; name: string } | null>(null);
  const [results, setResults] = useState<OptimizeResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);

  /* ---------- Load initial state ------------------------------------- */

  const refreshFiles = useCallback(async () => {
    try {
      const d = await api.listFiles();
      setFiles(d.files ?? []);
    } catch { /* backend may be down */ }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshFiles();
    api.optimizeDefaultMonth().then(d => {
      if (d?.month) setMonth(d.month);
    }).catch(() => {});
  }, [refreshFiles]);

  /* ---------- Upload -------------------------------------------------- */

  const uploadFiles = async (list: FileList | File[]) => {
    const form = new FormData();
    Array.from(list).forEach(f => form.append("files", f));
    try {
      await fetch(`${BASE}/api/upload`, { method: "POST", body: form });
      await refreshFiles();
    } catch (err) {
      setError(`No se pudieron subir los archivos: ${err}`);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dropRef.current?.classList.remove("ring-2", "ring-primary");
    if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  };
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    dropRef.current?.classList.add("ring-2", "ring-primary");
  };
  const onDragLeave = () => {
    dropRef.current?.classList.remove("ring-2", "ring-primary");
  };

  const deleteFile = async (name: string) => {
    try {
      await api.deleteFile(name);
      refreshFiles();
    } catch { /* ignore */ }
  };

  /* ---------- Run optimize ------------------------------------------- */

  const runOptimize = () => {
    if (files.length === 0) {
      setError("Primero sube los CSVs mensuales.");
      return;
    }
    setError(null);
    setResults(null);
    setPhase("running");
    setProgress({ stage: "ingest", label: "Arrancando…" });
    setCurrentStep(null);

    const url = api.optimizeStreamUrl(month, dryRun);
    const es = new EventSource(url);
    esRef.current = es;

    let state: { stage: FriendlyProgress["stage"]; labelHint?: string } = {
      stage: "ingest",
    };

    es.addEventListener("step", (e) => {
      const payload = JSON.parse((e as MessageEvent).data);
      const stage: FriendlyProgress["stage"] =
        payload.name === "predict" ? "predict" : "ingest";
      state = { stage };
      setCurrentStep({ index: payload.index, total: payload.total, name: payload.name });
      setProgress({
        stage,
        label: payload.name === "predict"
          ? "Calculando la nueva disposición…"
          : "Preparando datos y entrenando modelos…",
      });
    });

    es.addEventListener("log", (e) => {
      const payload = JSON.parse((e as MessageEvent).data);
      const friendly = parseLine(state, payload.message ?? "");
      state.labelHint = friendly.label;
      setProgress(friendly);
    });

    es.addEventListener("error", (e) => {
      try {
        const payload = JSON.parse((e as MessageEvent).data ?? "{}");
        if (payload?.message) setError(payload.message);
      } catch { /* network error — EventSource sends empty */ }
    });

    es.addEventListener("done", async (e) => {
      es.close();
      esRef.current = null;
      const payload = JSON.parse((e as MessageEvent).data);
      if (!payload.ok) {
        setError(`Fallo en el paso "${payload.failedStep}" (código ${payload.code}).`);
        setPhase("upload");
        return;
      }
      try {
        const r = await api.optimizeResults(month);
        if (r?.error) {
          setError(r.error);
          setPhase("upload");
          return;
        }
        setResults(r);
        setPhase("results");
        setProgress({ stage: "done", label: "Optimización completada" });
      } catch (err) {
        setError(`No se pudieron cargar los resultados: ${err}`);
        setPhase("upload");
      }
    });
  };

  const stopOptimize = async () => {
    if (esRef.current) { esRef.current.close(); esRef.current = null; }
    await api.optimizeStop();
    setPhase("upload");
    setProgress({ stage: "idle", label: "" });
    setCurrentStep(null);
  };

  /* ---------- Render -------------------------------------------------- */

  return (
    <div className="max-w-5xl space-y-8">
      <div className="animate-fade-in-up">
        <h1 className="text-3xl font-heading font-extrabold uppercase tracking-tight text-secondary">
          Optimiza tu supermercado
        </h1>
        <p className="text-muted-foreground mt-2 text-sm">
          Sube los datos de ventas mensuales y obtén la nueva disposición recomendada con el beneficio que generaría.
        </p>
      </div>

      {/* Error bar */}
      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Phase 1: upload + configure */}
      {phase === "upload" && (
        <div className="space-y-6 animate-fade-in-up">
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-base">1 · Sube los datos de ventas</CardTitle>
              <p className="text-xs text-muted-foreground mt-1">
                Arrastra los archivos <code className="bg-muted px-1 rounded">sales_*.csv</code> (uno por mes).
              </p>
            </CardHeader>
            <CardContent>
              <div
                ref={dropRef}
                onDrop={onDrop}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-border rounded-xl p-8 text-center cursor-pointer hover:bg-accent/30 transition-colors"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".csv"
                  className="hidden"
                  onChange={(e) => e.target.files && uploadFiles(e.target.files)}
                />
                <div className="text-sm font-medium">
                  Arrastra archivos aquí o haz clic para seleccionar
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  Formato: sales_YYYY_MM_*.csv
                </div>
              </div>

              {files.length > 0 && (
                <div className="mt-4 space-y-1.5 max-h-56 overflow-y-auto">
                  {files.map((f) => (
                    <div key={f.name}
                         className="flex items-center justify-between text-sm bg-muted rounded px-3 py-1.5">
                      <span className="truncate">{f.name}</span>
                      <div className="flex items-center gap-3 shrink-0">
                        <span className="text-xs text-muted-foreground">
                          {(f.size / 1024).toFixed(1)} KB
                        </span>
                        <button
                          onClick={() => deleteFile(f.name)}
                          className="text-xs text-red-600 hover:underline"
                        >
                          Eliminar
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {files.length === 0 && (
                <p className="text-xs text-muted-foreground mt-3">
                  Ningún archivo subido todavía.
                </p>
              )}
            </CardContent>
          </Card>

          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-base">2 · Elige el mes a optimizar</CardTitle>
              <p className="text-xs text-muted-foreground mt-1">
                Por defecto el siguiente al último CSV. Al pulsar «Optimizar» se entrenan los modelos y se calcula la nueva disposición automáticamente.
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-4 items-end flex-wrap">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">Mes objetivo</span>
                  <input
                    type="month"
                    value={month}
                    onChange={(e) => setMonth(e.target.value)}
                    className="bg-input border border-border rounded-lg px-3 py-1.5 text-sm"
                  />
                </label>
                <label className="flex items-center gap-2 pb-1 self-end"
                       title="Usa reglas estacionales fijas en vez de consultar la IA. Útil si OpenRouter está caído.">
                  <input
                    type="checkbox"
                    checked={dryRun}
                    onChange={(e) => setDryRun(e.target.checked)}
                    className="rounded accent-primary"
                  />
                  <span className="text-sm">Modo sin IA (reglas fijas)</span>
                </label>
              </div>

              <button
                onClick={runOptimize}
                disabled={files.length === 0}
                className="bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed px-6 py-2.5 rounded-lg text-sm font-semibold transition-colors"
              >
                Optimizar
              </button>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Phase 2: running */}
      {phase === "running" && (
        <Card className="shadow-sm animate-fade-in-up">
          <CardHeader>
            <CardTitle className="text-base">Optimizando tu supermercado…</CardTitle>
            {currentStep && (
              <p className="text-xs text-muted-foreground mt-1">
                Paso {currentStep.index} de {currentStep.total}:{" "}
                {currentStep.name === "predict" ? "optimización final" : "entrenamiento y catálogo"}
              </p>
            )}
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="flex items-center gap-3">
              <div className="w-2.5 h-2.5 rounded-full bg-primary animate-pulse" />
              <div>
                <div className="text-sm font-medium">{progress.label}</div>
                {progress.detail && (
                  <div className="text-xs text-muted-foreground">{progress.detail}</div>
                )}
              </div>
            </div>

            {typeof progress.percent === "number" && (
              <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${progress.percent}%` }}
                />
              </div>
            )}

            <div className="text-xs text-muted-foreground">
              Esto puede tardar unos minutos la primera vez.
              No cierres la pestaña.
            </div>

            <button
              onClick={stopOptimize}
              className="text-xs text-red-600 hover:underline"
            >
              Cancelar
            </button>
          </CardContent>
        </Card>
      )}

      {/* Phase 3: results */}
      {phase === "results" && results && (
        <div className="space-y-6 animate-fade-in-up">
          {/* KPI cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <KpiCard
              label="Beneficio extra al mes"
              value={eur(results.kpi.profitLiftEur)}
              positive={results.kpi.profitLiftEur > 0}
            />
            <KpiCard
              label="% mejora"
              value={`${results.kpi.profitLiftPct > 0 ? "+" : ""}${results.kpi.profitLiftPct.toFixed(1)}%`}
              positive={results.kpi.profitLiftPct > 0}
            />
            <KpiCard
              label="Productos reubicados"
              value={`${results.kpi.productsMoved.toLocaleString()} / ${results.kpi.totalProducts.toLocaleString()}`}
            />
            <KpiCard
              label="Estanterías mejoradas"
              value={`${results.kpi.racksImproved}`}
            />
          </div>

          {/* Sankey */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-base">Reorganización de productos por balda</CardTitle>
              <p className="text-xs text-muted-foreground mt-1">
                Cada cinta representa productos que pasan de una balda a otra.
                Las baldas a la altura de los ojos (3–5) son las más rentables.
              </p>
            </CardHeader>
            <CardContent>
              <ShelfSankey movements={results.movements} />
              <div className="mt-4 grid grid-cols-2 gap-4 text-xs text-muted-foreground">
                <div>
                  <span className="inline-block w-3 h-3 rounded-sm mr-1.5 align-middle"
                        style={{ background: "rgba(9,84,61,0.55)" }} />
                  Promovido a la altura de los ojos (mejor visibilidad)
                </div>
                <div>
                  <span className="inline-block w-3 h-3 rounded-sm mr-1.5 align-middle"
                        style={{ background: "rgba(244,92,36,0.45)" }} />
                  Liberado de eye-level para dejar sitio a más rentables
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Top racks */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-base">Estanterías con mayor mejora</CardTitle>
            </CardHeader>
            <CardContent>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-muted-foreground text-xs">
                    <th className="text-left pb-2">Estantería</th>
                    <th className="text-right pb-2">Productos</th>
                    <th className="text-right pb-2">Antes (€)</th>
                    <th className="text-right pb-2">Después (€)</th>
                    <th className="text-right pb-2">Mejora</th>
                  </tr>
                </thead>
                <tbody>
                  {results.racks.slice(0, 10).map((r) => (
                    <tr key={r.rack} className="border-b border-border last:border-0">
                      <td className="py-1.5 font-medium">{r.rack}</td>
                      <td className="text-right text-muted-foreground">{r.products}</td>
                      <td className="text-right">{eur(r.original)}</td>
                      <td className="text-right">{eur(r.optimized)}</td>
                      <td className={`text-right font-semibold ${r.lift >= 0 ? "text-green-600" : "text-red-600"}`}>
                        {r.lift >= 0 ? "+" : ""}{eur(r.lift)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>

          {/* Forecast source chip */}
          {results.forecastSource && (
            <div className="text-xs text-muted-foreground">
              Fuente de la predicción estacional:{" "}
              <span className={`inline-block px-2 py-0.5 rounded ${results.forecastSource === "llm" ? "bg-primary/10 text-primary" : "bg-muted"}`}>
                {results.forecastSource === "llm" ? "IA (LLM)" : "Reglas estacionales"}
              </span>
            </div>
          )}

          <button
            onClick={() => {
              setPhase("upload");
              setResults(null);
              setProgress({ stage: "idle", label: "" });
            }}
            className="text-sm text-muted-foreground hover:text-foreground"
          >
            ← Optimizar otro mes
          </button>
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */

function KpiCard({ label, value, positive }: { label: string; value: string; positive?: boolean }) {
  return (
    <Card className="shadow-sm">
      <CardContent className="pt-4">
        <div className={`text-2xl font-bold ${
          positive === true ? "text-green-600" :
          positive === false ? "text-red-600" : "text-primary"
        }`}>
          {value}
        </div>
        <div className="text-xs text-muted-foreground mt-0.5">{label}</div>
      </CardContent>
    </Card>
  );
}

function eur(n: number): string {
  const rounded = Math.round(n);
  return `€${rounded.toLocaleString("es-ES")}`;
}

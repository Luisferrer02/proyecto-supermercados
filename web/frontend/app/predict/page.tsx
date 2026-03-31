"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LiveLog } from "@/components/LiveLog";
import { ShelfMap, type Product } from "@/components/ShelfMap";
import { api } from "@/lib/api";

interface RackSummary {
  original: number;
  optimized: number;
  products: number;
}

interface PredictResult {
  products: Product[];
  forecast: Record<string, number>;
  rackSummary: Record<string, RackSummary>;
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
  };

  const loadResults = async () => {
    try {
      const d = await api.predictResults(month);
      if (!d || d.error || !Array.isArray(d.products)) {
        setPredError(d?.error ?? "Prediction finished but no results were saved. Check the log above for details.");
        return;
      }
      setPredError(null);
      setResults(d);
      const racks = [...new Set(d.products.map((p: Product) => p.rack_id ?? p.Category ?? ""))].filter(Boolean);
      setSelectedRack((racks[0] as string) ?? "");
    } catch {
      setPredError("Could not load results from the server.");
    }
  };

  const selectedProducts = results?.products?.filter(
    (p) => (p.rack_id ?? p.Category ?? "") === selectedRack
  ) ?? [];

  const rackIds = results?.products
    ? ([...new Set(results.products.map((p) => p.rack_id ?? p.Category ?? ""))].filter(Boolean) as string[])
    : [];

  const topForecast = results?.forecast
    ? Object.entries(results.forecast)
        .sort(([, a], [, b]) => Math.abs(b - 1) - Math.abs(a - 1))
        .slice(0, 12)
    : [];

  const rackSummary = selectedRack && results?.rackSummary?.[selectedRack];

  return (
    <div className="space-y-8 max-w-5xl relative">
      <div
        className="fixed inset-0 bg-cover bg-center opacity-[0.09] blur-sm pointer-events-none -z-10"
        style={{ backgroundImage: "url(/predict.png)" }}
      />
      <div className="animate-fade-in-up">
        <h1 className="text-3xl font-heading font-extrabold uppercase tracking-tight text-secondary">
          Predict
        </h1>
        <p className="text-muted-foreground text-sm mt-2">
          Runs <code className="bg-muted px-1.5 py-0.5 rounded text-xs">05_predict.py</code> — RAG retrieval → LLM forecast → ensemble optimization.
        </p>
      </div>

      {/* Config form */}
      <Card className="shadow-sm">
        <CardContent className="pt-4 space-y-4">
          <div className="flex flex-wrap gap-4">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Target Month</span>
              <input
                type="month"
                value={month}
                onChange={(e) => setMonth(e.target.value)}
                className="bg-input border border-border rounded-lg px-3 py-1.5 text-sm focus:ring-2 focus:ring-primary/30 focus:border-primary outline-none transition-shadow"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Category (optional)</span>
              <input
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                placeholder="e.g. Fruta"
                className="bg-input border border-border rounded-lg px-3 py-1.5 text-sm w-40 focus:ring-2 focus:ring-primary/30 focus:border-primary outline-none transition-shadow"
              />
            </label>
            <label className="flex items-center gap-2 self-end pb-1">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={(e) => setDryRun(e.target.checked)}
                className="rounded accent-primary"
              />
              <span className="text-sm">Dry-run (skip LLM)</span>
            </label>
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={startPrediction}
              disabled={streaming}
              className="bg-primary text-primary-foreground hover:bg-primary/90 px-5 py-2 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
            >
              {streaming ? "Running prediction…" : "Run Prediction"}
            </button>
            {streaming && (
              <button
                onClick={() => { api.predictStop(); setStreaming(false); }}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
              >
                Stop
              </button>
            )}

            {pastMonths.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Load past result:</span>
                <select
                  className="bg-input border border-border rounded-lg px-2 py-1 text-xs focus:ring-2 focus:ring-primary/30 focus:border-primary outline-none transition-shadow"
                  onChange={(e) => { setMonth(e.target.value); setTimeout(loadResults, 100); }}
                  defaultValue=""
                >
                  <option value="" disabled>Select month</option>
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
          ⚠ {predError}
        </div>
      )}

      {results && (
        <div className="space-y-6">
          {/* Forecast multipliers */}
          {topForecast.length > 0 && (
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle className="text-sm">Sales Forecast Adjustments (top categories)</CardTitle>
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

          {/* Shelf Map */}
          <Card className="shadow-sm">
            <CardHeader>
              <div className="flex items-center gap-3 flex-wrap">
                <CardTitle className="text-sm">Shelf Map</CardTitle>
                <select
                  value={selectedRack}
                  onChange={(e) => setSelectedRack(e.target.value)}
                  className="bg-input border border-border rounded-lg px-2 py-1 text-xs focus:ring-2 focus:ring-primary/30 focus:border-primary outline-none transition-shadow"
                >
                  {rackIds.map((id) => (
                    <option key={id} value={id}>{id}</option>
                  ))}
                </select>
                {rackSummary && (
                  <span className="text-xs text-muted-foreground">
                    {rackSummary.products} products ·{" "}
                    <span className="text-green-600">
                      +€{Math.round(rackSummary.optimized - rackSummary.original).toLocaleString()} lift
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
                <CardTitle className="text-sm">Profit Summary by Rack (top 20)</CardTitle>
              </CardHeader>
              <CardContent>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border text-muted-foreground">
                      <th className="text-left pb-2">Rack / Category</th>
                      <th className="text-right pb-2">Products</th>
                      <th className="text-right pb-2">Original €</th>
                      <th className="text-right pb-2">Optimized €</th>
                      <th className="text-right pb-2">Lift</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(results.rackSummary)
                      .sort(([, a], [, b]) => (b.optimized - b.original) - (a.optimized - a.original))
                      .slice(0, 20)
                      .map(([rack, s]) => {
                        const lift = s.optimized - s.original;
                        return (
                          <tr
                            key={rack}
                            className={`border-b border-border last:border-0 cursor-pointer hover:bg-accent/30 ${selectedRack === rack ? "bg-accent/30" : ""}`}
                            onClick={() => setSelectedRack(rack)}
                          >
                            <td className="py-1.5 font-medium">{rack}</td>
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
          Run ingestion first to build the knowledge base, then run a prediction.
        </div>
      )}
    </div>
  );
}

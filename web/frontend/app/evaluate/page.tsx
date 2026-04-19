"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LiveLog } from "@/components/LiveLog";
import { api } from "@/lib/api";

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

export default function EvaluatePage() {
  const [charts, setCharts] = useState<ChartInfo[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [streamUrl, setStreamUrl] = useState("");
  const [showLog, setShowLog] = useState(false);

  const loadStatus = async () => {
    try {
      const d = await api.evaluateStatus();
      setCharts(d.charts ?? []);
    } catch {}
  };

  useEffect(() => { loadStatus(); }, []);

  const runEvaluate = () => {
    setStreamUrl(api.evaluateStreamUrl());
    setStreaming(true);
    setShowLog(true);
  };

  const existingCharts = charts.filter((c) => c.exists);

  return (
    <div className="space-y-8 max-w-4xl relative">
      <div
        className="fixed inset-0 bg-cover bg-center opacity-[0.09] blur-sm pointer-events-none -z-10"
        style={{ backgroundImage: "url(/evaluate.png)" }}
      />
      <div className="animate-fade-in-up">
        <h1 className="text-3xl font-heading font-extrabold uppercase tracking-tight text-secondary">
          Evaluate
        </h1>
        <p className="text-muted-foreground text-sm mt-2">
          Runs <code className="bg-muted px-1.5 py-0.5 rounded text-xs">03_evaluate.py</code> to generate comparison charts from training results.
        </p>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={runEvaluate}
          disabled={streaming}
          className="bg-primary text-primary-foreground hover:bg-primary/90 px-5 py-2 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
        >
          {streaming ? "Generating charts…" : "Regenerate Charts"}
        </button>
        {streaming && (
          <button
            onClick={() => { api.evaluateStop(); setStreaming(false); }}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            Stop
          </button>
        )}
        <Badge variant={existingCharts.length > 0 ? "default" : "secondary"}>
          {existingCharts.length}/{charts.length} charts available
        </Badge>
      </div>

      {showLog && streamUrl && (
        <LiveLog
          url={streamUrl}
          onDone={() => { setStreaming(false); loadStatus(); }}
        />
      )}

      {existingCharts.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {existingCharts.map((chart) => (
            <Card key={chart.name} className="shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-muted-foreground">
                  {CHART_LABELS[chart.name] ?? chart.name}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {/* Cache-bust with timestamp to reload after regeneration */}
                <img
                  src={`${api.resultUrl(chart.name)}?t=${Date.now()}`}
                  alt={chart.name}
                  className="w-full rounded-lg border border-border"
                />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {charts.length > 0 && existingCharts.length === 0 && !streaming && (
        <div className="text-sm text-muted-foreground">
          No charts found. Click &quot;Regenerate Charts&quot; or run training first.
        </div>
      )}
    </div>
  );
}

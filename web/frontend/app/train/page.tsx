"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LiveLog } from "@/components/LiveLog";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { api } from "@/lib/api";

interface ModelResult {
  mse?: number;
  original_profit?: number;
  optimized_profit?: number;
}

type TrainResults = Record<string, ModelResult>;

export default function TrainPage() {
  const [streaming, setStreaming] = useState(false);
  const [streamUrl, setStreamUrl] = useState("");
  const [showLog, setShowLog] = useState(false);
  const [results, setResults] = useState<TrainResults | null>(null);

  const loadResults = async () => {
    try {
      const d = await api.trainResults();
      setResults(d);
    } catch {}
  };

  useEffect(() => { loadResults(); }, []);

  const startTraining = () => {
    setStreamUrl(api.trainStreamUrl());
    setStreaming(true);
    setShowLog(true);
  };

  const chartData = results
    ? Object.entries(results).map(([model, m]) => ({
        model,
        "Optimized Profit": Math.round(m.optimized_profit ?? 0),
        "Original Profit": Math.round(m.original_profit ?? 0),
        MSE: m.mse != null ? Math.round(m.mse) : null,
      }))
    : [];

  return (
    <div className="space-y-8 max-w-4xl relative">
      <div
        className="fixed inset-0 bg-cover bg-center opacity-[0.09] blur-sm pointer-events-none -z-10"
        style={{ backgroundImage: "url(/trainmodels.png)" }}
      />
      <div className="animate-fade-in-up">
        <h1 className="text-3xl font-heading font-extrabold uppercase tracking-tight text-secondary">
          Train Models
        </h1>
        <p className="text-muted-foreground text-sm mt-2">
          Runs <code className="bg-muted px-1.5 py-0.5 rounded text-xs">02_train_models.py</code> — trains MLP, LSTM, Transformer, and PPO on the uploaded CSV data.
        </p>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={startTraining}
          disabled={streaming}
          className="bg-primary text-primary-foreground hover:bg-primary/90 px-5 py-2 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
        >
          {streaming ? "Training in progress…" : "Start Training"}
        </button>
        {streaming && (
          <button
            onClick={() => { api.trainStop(); setStreaming(false); }}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            Stop
          </button>
        )}
      </div>

      {showLog && streamUrl && (
        <LiveLog
          url={streamUrl}
          onDone={() => { setStreaming(false); loadResults(); }}
        />
      )}

      {results && (
        <div className="space-y-6">
          {/* Profit chart */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-sm">Optimized vs Original Profit (sample rack)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
                  <XAxis dataKey="model" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Original Profit" fill="#461e10" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Optimized Profit" fill="#09543d" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* MSE chart */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-sm">Test MSE (lower is better)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={chartData.filter((d) => d.MSE != null)} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
                  <XAxis dataKey="model" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="MSE" fill="#f45c24" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Results table */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-sm">Model Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-muted-foreground text-xs">
                    <th className="text-left pb-2">Model</th>
                    <th className="text-right pb-2">MSE</th>
                    <th className="text-right pb-2">Original €</th>
                    <th className="text-right pb-2">Optimized €</th>
                    <th className="text-right pb-2">Lift</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(results).map(([model, m]) => {
                    const lift = (m.optimized_profit ?? 0) - (m.original_profit ?? 0);
                    return (
                      <tr key={model} className="border-b border-border last:border-0">
                        <td className="py-2 font-medium">{model}</td>
                        <td className="text-right text-muted-foreground">{m.mse != null ? m.mse.toFixed(1) : "—"}</td>
                        <td className="text-right">€{m.original_profit?.toLocaleString() ?? "—"}</td>
                        <td className="text-right">€{m.optimized_profit?.toLocaleString() ?? "—"}</td>
                        <td className={`text-right font-semibold ${lift >= 0 ? "text-green-600" : "text-red-600"}`}>
                          {lift >= 0 ? "+" : ""}€{lift.toLocaleString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>
      )}

      {!results && !streaming && (
        <div className="text-sm text-muted-foreground">
          No training results yet. Click &quot;Start Training&quot; to begin.
        </div>
      )}
    </div>
  );
}

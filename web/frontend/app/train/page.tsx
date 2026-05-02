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
  mse_eur2?: number;
  rmse_eur?: number;
  mae_eur?: number;
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

  // eslint-disable-next-line react-hooks/set-state-in-effect
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
          Comparación de modelos
        </h1>
        <p className="text-muted-foreground text-sm mt-2">
          Ejecuta <code className="bg-muted px-1.5 py-0.5 rounded text-xs">02_train_models.py</code>: entrena MLP, LSTM, Transformer y PPO sobre los CSVs subidos y muestra cómo se comporta cada uno.
        </p>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={startTraining}
          disabled={streaming}
          className="bg-primary text-primary-foreground hover:bg-primary/90 px-5 py-2 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
        >
          {streaming ? "Entrenando…" : "Lanzar entrenamiento"}
        </button>
        {streaming && (
          <button
            onClick={() => { api.trainStop(); setStreaming(false); }}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            Cancelar
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
              <CardTitle className="text-sm">
                Beneficio antes vs. después por modelo
                <span className="block text-xs font-normal text-muted-foreground mt-0.5">
                  Calculado sobre una estantería representativa para comparar modelos en condiciones idénticas
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 8 }}>
                  <XAxis dataKey="model" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `€${v.toLocaleString("es-ES")}`} />
                  <Tooltip formatter={(value) => [
                    `€${Number(value ?? 0).toLocaleString("es-ES")}`,
                    "",
                  ]} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Original Profit" name="Antes" fill="#461e10" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Optimized Profit" name="Después" fill="#09543d" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* MSE chart */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-sm">
                Error de predicción por modelo
                <span className="text-xs font-normal text-muted-foreground block mt-0.5">
                  MSE en €² — cuanto menor, más preciso
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={chartData.filter((d) => d.MSE != null)} margin={{ top: 8, right: 12, bottom: 4, left: 8 }}>
                  <XAxis dataKey="model" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `${v.toLocaleString("es-ES")}`} />
                  <Tooltip formatter={(value) => [
                    `${Number(value ?? 0).toLocaleString("es-ES")} €²`,
                    "MSE",
                  ]} />
                  <Bar dataKey="MSE" fill="#f45c24" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Results table */}
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-sm">Comparativa de modelos</CardTitle>
            </CardHeader>
            <CardContent>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-muted-foreground text-xs">
                    <th className="text-left pb-2">Modelo</th>
                    <th className="text-right pb-2" title="Error cuadrático medio (€²)">MSE (€²)</th>
                    <th className="text-right pb-2" title="Raíz del MSE: error típico interpretable en €">RMSE (€)</th>
                    <th className="text-right pb-2">Antes (€)</th>
                    <th className="text-right pb-2">Después (€)</th>
                    <th className="text-right pb-2">Mejora</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(results)
                    .filter(([model]) => !model.startsWith("_"))
                    .map(([model, m]) => {
                      const lift = (m.optimized_profit ?? 0) - (m.original_profit ?? 0);
                      const mse = m.mse_eur2 ?? m.mse;
                      const rmse = m.rmse_eur ?? (mse != null ? Math.sqrt(mse) : null);
                      return (
                        <tr key={model} className="border-b border-border last:border-0">
                          <td className="py-2 font-medium">{model}</td>
                          <td className="text-right text-muted-foreground">{mse != null ? mse.toFixed(1) : "—"}</td>
                          <td className="text-right text-muted-foreground">{rmse != null ? rmse.toFixed(2) : "—"}</td>
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
              <p className="text-xs text-muted-foreground mt-3">
                MSE en €² (errores al cuadrado); RMSE en € es el error típico interpretable por producto.
                Los <em>baselines</em> de predicción (Identidad = predice 0; Aleatorio = ruido gaussiano) se guardan en{" "}
                <code className="bg-muted px-1 rounded text-[11px]">training_results.json</code>.
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {!results && !streaming && (
        <div className="text-sm text-muted-foreground">
          Aún no hay resultados de entrenamiento. Pulsa <span className="font-medium">«Lanzar entrenamiento»</span> para empezar.
        </div>
      )}
    </div>
  );
}

"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LiveLog } from "@/components/LiveLog";
import { api } from "@/lib/api";

export default function IngestPage() {
  const [streaming, setStreaming] = useState(false);
  const [streamUrl, setStreamUrl] = useState("");
  const [showLog, setShowLog] = useState(false);
  const [done, setDone] = useState<boolean | null>(null);

  const startIngest = () => {
    setStreamUrl(api.ingestStreamUrl());
    setStreaming(true);
    setShowLog(true);
    setDone(null);
  };

  return (
    <div className="space-y-8 max-w-3xl relative">
      <div
        className="fixed inset-0 bg-cover bg-center opacity-[0.09] blur-sm pointer-events-none -z-10"
        style={{ backgroundImage: "url(/ingest.png)" }}
      />
      <div className="animate-fade-in-up">
        <h1 className="text-3xl font-heading font-extrabold uppercase tracking-tight text-secondary">
          Ingest
        </h1>
        <p className="text-muted-foreground text-sm mt-2">
          Runs <code className="bg-muted px-1.5 py-0.5 rounded text-xs">04_ingest.py</code> in two parallel threads:
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <Card className="shadow-sm border-t-2 border-t-[oklch(0.75_0.15_220)]">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Thread 1 — Embeddings</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground space-y-1">
            <p>Reads all sales CSVs by category</p>
            <p>Generates natural-language summaries</p>
            <p>Embeds with sentence-transformers</p>
            <p>Stores in ChromaDB vector database</p>
          </CardContent>
        </Card>
        <Card className="shadow-sm border-t-2 border-t-[oklch(0.65_0.2_45)]">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Thread 2 — Production Models</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground space-y-1">
            <p>Genera casos sintéticos de entrenamiento</p>
            <p>Entrena el modelo rápido (MLP, 80 iteraciones)</p>
            <p>Entrena el modelo preciso (Transformer, 120 iteraciones)</p>
            <p>Guarda los modelos versionados en results/models/</p>
          </CardContent>
        </Card>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={startIngest}
          disabled={streaming}
          className="bg-primary text-primary-foreground hover:bg-primary/90 px-5 py-2 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
        >
          {streaming ? "Ingesting…" : "Start Ingestion"}
        </button>
        {streaming && (
          <button
            onClick={() => { api.ingestStop(); setStreaming(false); }}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            Stop
          </button>
        )}
        {done === true && (
          <span className="text-sm text-green-600">✓ Ingestion complete — knowledge base ready</span>
        )}
        {done === false && (
          <span className="text-sm text-red-600">⚠ Ingestion finished with errors</span>
        )}
      </div>

      {showLog && streamUrl && (
        <LiveLog
          url={streamUrl}
          onDone={(success) => {
            setStreaming(false);
            setDone(success);
          }}
        />
      )}

      {!streaming && done === null && (
        <div className="text-sm text-muted-foreground">
          Make sure you have uploaded CSV files before ingesting.
          This step can take several minutes depending on the number of files.
        </div>
      )}
    </div>
  );
}

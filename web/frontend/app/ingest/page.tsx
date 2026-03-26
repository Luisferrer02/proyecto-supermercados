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
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold">Ingest</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Runs <code className="bg-muted px-1 rounded text-xs">04_ingest.py</code> in two parallel threads:
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Card>
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
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Thread 2 — Production Models</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground space-y-1">
            <p>Generates synthetic training samples</p>
            <p>Trains MLP (80 epochs)</p>
            <p>Trains Transformer (120 epochs)</p>
            <p>Saves mlp.pth + transformer.pth</p>
          </CardContent>
        </Card>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={startIngest}
          disabled={streaming}
          className="bg-primary text-primary-foreground hover:bg-primary/90 px-5 py-2 rounded-md text-sm font-medium disabled:opacity-50 transition-colors"
        >
          {streaming ? "Ingesting…" : "Start Ingestion"}
        </button>
        {streaming && (
          <button
            onClick={() => { api.ingestStop(); setStreaming(false); }}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90 px-4 py-2 rounded-md text-sm font-medium transition-colors"
          >
            Stop
          </button>
        )}
        {done === true && (
          <span className="text-sm text-green-400">✓ Ingestion complete — knowledge base ready</span>
        )}
        {done === false && (
          <span className="text-sm text-red-400">⚠ Ingestion finished with errors</span>
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

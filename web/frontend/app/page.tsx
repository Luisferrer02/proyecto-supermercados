"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

interface StepStatus {
  label: string;
  description: string;
  check: () => Promise<boolean>;
  href: string;
  image: string;
}

const steps: StepStatus[] = [
  {
    label: "1. Upload CSVs",
    description: "Upload sales_*.csv files to data/monthly/",
    check: async () => {
      const d = await api.listFiles();
      return d.files?.length > 0;
    },
    href: "/upload",
    image: "/uploadcsv.png",
  },
  {
    label: "2. Train Models",
    description: "Run 02_train_models.py to train MLP, LSTM, Transformer, PPO",
    check: async () => {
      try {
        await api.trainResults();
        return true;
      } catch {
        return false;
      }
    },
    href: "/train",
    image: "/trainmodels.png",
  },
  {
    label: "3. Evaluate",
    description: "Generate comparison charts with 03_evaluate.py",
    check: async () => {
      const d = await api.evaluateStatus();
      return d.charts?.some((c: { exists: boolean }) => c.exists);
    },
    href: "/evaluate",
    image: "/evaluate.png",
  },
  {
    label: "4. Ingest",
    description: "Build ChromaDB embeddings + production models with 04_ingest.py",
    check: async () => false,
    href: "/ingest",
    image: "/ingest.png",
  },
  {
    label: "5. Predict",
    description: "Generate optimized shelf layout for a target month with 05_predict.py",
    check: async () => {
      const d = await api.predictList();
      return d.months?.length > 0;
    },
    href: "/predict",
    image: "/predict.png",
  },
];

export default function Dashboard() {
  const [statuses, setStatuses] = useState<boolean[]>(new Array(steps.length).fill(false));
  const [fileCount, setFileCount] = useState(0);

  useEffect(() => {
    async function load() {
      const results = await Promise.allSettled(steps.map((s) => s.check()));
      setStatuses(results.map((r) => r.status === "fulfilled" && r.value));
      try {
        const d = await api.listFiles();
        setFileCount(d.files?.length ?? 0);
      } catch {}
    }
    load();
  }, []);

  return (
    <div className="space-y-10 max-w-5xl">
      <div className="animate-fade-in-up">
        <h1 className="text-3xl font-heading font-extrabold uppercase tracking-tight text-secondary">
          ShelfOpt Dashboard
        </h1>
        <p className="text-muted-foreground mt-2 text-sm">
          Supermarket shelf optimization pipeline — Mercadona dataset, 4,772 products, 149 categories.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6 sm:grid-cols-4 animate-fade-in-up" style={{ animationDelay: "0.1s" }}>
        {[
          { label: "Products", value: "4,772" },
          { label: "Categories", value: "149" },
          { label: "CSV files uploaded", value: fileCount },
          { label: "Profit lift (historical)", value: "+16.2%" },
        ].map((s) => (
          <Card key={s.label} className="shadow-sm">
            <CardContent className="pt-4">
              <div className="text-2xl font-bold text-primary">{s.value}</div>
              <div className="text-xs text-muted-foreground mt-0.5">{s.label}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
        <h2 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide mb-4">
          Pipeline Steps
        </h2>
        <div className="grid gap-4">
        {steps.map((step, i) => (
          <a key={i} href={step.href}>
            <Card className="hover:bg-accent/30 transition-colors cursor-pointer shadow-sm overflow-hidden">
              <div className="flex items-center">
                <CardHeader className="py-3 px-4 flex-row items-center gap-3 space-y-0 flex-1">
                  <Badge
                    variant={statuses[i] ? "default" : "secondary"}
                    className={statuses[i] ? "bg-primary" : ""}
                  >
                    {statuses[i] ? "Done" : "Pending"}
                  </Badge>
                  <div>
                    <CardTitle className="text-sm">{step.label}</CardTitle>
                    <p className="text-xs text-muted-foreground mt-0.5">{step.description}</p>
                  </div>
                </CardHeader>
                <img
                  src={step.image}
                  alt={step.label}
                  className="h-20 w-28 object-cover rounded-r-xl shrink-0 hidden sm:block"
                />
              </div>
            </Card>
          </a>
        ))}
        </div>
      </div>
    </div>
  );
}

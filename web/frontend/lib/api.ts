export const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3001";

export const api = {
  // Upload
  uploadFiles: (form: FormData) =>
    fetch(`${BASE}/api/upload`, { method: "POST", body: form }).then((r) => r.json()),
  listFiles: () =>
    fetch(`${BASE}/api/upload/files`).then((r) => r.json()),
  deleteFile: (name: string) =>
    fetch(`${BASE}/api/upload/files/${encodeURIComponent(name)}`, { method: "DELETE" }).then((r) => r.json()),

  // Train
  trainStatus: () =>
    fetch(`${BASE}/api/train/status`).then((r) => r.json()),
  trainResults: () =>
    fetch(`${BASE}/api/train/results`).then((r) => r.json()),
  trainStreamUrl: () => `${BASE}/api/train/stream`,
  trainStop: () =>
    fetch(`${BASE}/api/train/stop`, { method: "POST" }).then((r) => r.json()),

  // Evaluate
  evaluateStatus: () =>
    fetch(`${BASE}/api/evaluate/status`).then((r) => r.json()),
  evaluateStreamUrl: () => `${BASE}/api/evaluate/stream`,
  evaluateStop: () =>
    fetch(`${BASE}/api/evaluate/stop`, { method: "POST" }).then((r) => r.json()),

  // Ingest
  ingestStatus: () =>
    fetch(`${BASE}/api/ingest/status`).then((r) => r.json()),
  ingestStreamUrl: () => `${BASE}/api/ingest/stream`,
  ingestStop: () =>
    fetch(`${BASE}/api/ingest/stop`, { method: "POST" }).then((r) => r.json()),

  // Predict
  predictStatus: () =>
    fetch(`${BASE}/api/predict/status`).then((r) => r.json()),
  predictStop: () =>
    fetch(`${BASE}/api/predict/stop`, { method: "POST" }).then((r) => r.json()),
  predictStreamUrl: (month: string, category?: string, dryRun?: boolean) => {
    const p = new URLSearchParams({ month });
    if (category) p.set("category", category);
    if (dryRun) p.set("dryRun", "true");
    return `${BASE}/api/predict/stream?${p}`;
  },
  predictResults: (month: string) =>
    fetch(`${BASE}/api/predict/results?month=${month}`).then((r) => r.json()),
  predictList: () =>
    fetch(`${BASE}/api/predict/list`).then((r) => r.json()),

  // Optimize — unified upload → ingest → predict flow
  optimizeStatus: () =>
    fetch(`${BASE}/api/optimize/status`).then((r) => r.json()),
  optimizeDefaultMonth: () =>
    fetch(`${BASE}/api/optimize/default-month`).then((r) => r.json()),
  optimizeStreamUrl: (month: string, dryRun?: boolean) => {
    const p = new URLSearchParams({ month });
    if (dryRun) p.set("dryRun", "true");
    return `${BASE}/api/optimize/run?${p}`;
  },
  optimizeStop: () =>
    fetch(`${BASE}/api/optimize/stop`, { method: "POST" }).then((r) => r.json()),
  optimizeResults: (month: string) =>
    fetch(`${BASE}/api/optimize/results?month=${month}`).then((r) => r.json()),

  // Static result files
  resultUrl: (filename: string) => `${BASE}/results/${filename}`,
};

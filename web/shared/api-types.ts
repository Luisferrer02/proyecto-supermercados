export interface UploadedFile {
  name: string;
  size: number;
  modified?: string;
}

export interface UploadFilesResponse {
  files: UploadedFile[];
}

export interface UploadResult {
  name: string;
  size: number;
  valid: boolean;
  report: string;
}

export interface UploadResponse {
  uploaded: UploadResult[];
}

export interface ModelResult {
  mse?: number;
  mse_eur2?: number;
  rmse_eur?: number;
  mae_eur?: number;
  original_profit?: number;
  optimized_profit?: number;
}

export interface TrainResults {
  [model: string]: ModelResult;
}

export interface ChartInfo {
  name: string;
  exists: boolean;
  url: string;
}

export interface EvaluateStatusResponse {
  charts: ChartInfo[];
  running: boolean;
}

export interface PredictListResponse {
  months: string[];
}

export interface RackSummaryEntry {
  original: number;
  optimized: number;
  products: number;
}

export interface PredictResultsResponse {
  products: Record<string, string>[];
  forecast: Record<string, number>;
  forecastSource: string | null;
  rackSummary: Record<string, RackSummaryEntry>;
  explanations: unknown;
}

export interface OptimizeDefaultMonthResponse {
  month: string;
}

export interface Kpi {
  profitOriginal: number;
  profitOptimized: number;
  profitLiftEur: number;
  profitLiftPct: number;
  productsMoved: number;
  totalProducts: number;
  racksImproved: number;
}

export interface Movement {
  from: number;
  to: number;
  count: number;
}

export interface RackRow {
  rack: string;
  products: number;
  original: number;
  optimized: number;
  lift: number;
}

export interface OptimizeResultsResponse {
  kpi: Kpi;
  movements: Movement[];
  racks: RackRow[];
  multipliers: Record<string, number>;
  forecastSource: string | null;
  explanations: unknown;
}

// SSE event payloads
export interface SseLogEvent {
  message: string;
}

export interface SseDoneEvent {
  message?: string;
  ok?: boolean;
  failedStep?: string;
  code?: number;
}

export interface SseStepEvent {
  name: string;
  index: number;
  total: number;
}

export interface SseErrorEvent {
  message: string;
  step?: string;
}

// Status responses (shared pattern)
export interface StatusResponse {
  running: boolean;
}

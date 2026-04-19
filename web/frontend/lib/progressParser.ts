/**
 * Translate the raw Python log stream into short, non-technical progress
 * messages for end users.
 *
 * The log lines come from two scripts run back-to-back:
 *   1. 04_ingest.py    — builds the knowledge base and trains production models
 *   2. 05_predict.py   — forecasts sales and optimises the layout
 *
 * Each stage of the pipeline emits recognisable substrings we match with
 * cheap indexOf checks (no regex). When a match is found we expose a
 * friendly Spanish label, an optional numeric progress (0-100) and a
 * step key so the UI can show a stepper.
 */

export interface FriendlyProgress {
  stage: "upload" | "ingest" | "predict" | "done" | "idle";
  label: string;
  detail?: string;
  /** Integer 0-100 when we can infer it from the log line. */
  percent?: number;
}

const DEFAULT: FriendlyProgress = {
  stage: "idle",
  label: "Listo para empezar",
};

const INGEST_RULES: Array<[string, string]> = [
  ["SHELF OPTIMIZER -- DATA INGESTION", "Preparando la ingesta de datos"],
  ["Schema validation",                 "Validando archivos CSV"],
  ["Embeddings:",                       "Indexando productos por categoría"],
  ["Training MLP",                      "Entrenando modelo rápido"],
  ["Training Transformer",              "Entrenando modelo preciso"],
  ["MLP saved",                         "Modelo rápido guardado"],
  ["Transformer saved",                 "Modelo preciso guardado"],
  ["Models:",                           "Modelos listos"],
];

const PREDICT_RULES: Array<[string, string]> = [
  ["Step 1: RAG",                       "Buscando datos históricos relevantes"],
  ["Step 2:",                           "Cargando catálogo base"],
  ["Step 3:",                           "Prediciendo tendencias de venta"],
  ["Using heuristic",                   "Calculando con reglas de temporada"],
  ["Querying LLM",                      "Consultando la IA"],
  ["🤖",                                "Consultando modelo de IA"],
  ["Got forecasts",                     "Ajuste estacional listo"],
  ["falling back to heuristic",         "Usando reglas de temporada (la IA no responde)"],
  ["Step 4:",                           "Aplicando ajustes a las ventas"],
  ["Step 5:",                           "Calculando la nueva disposición"],
  ["ensemble optimization",             "Optimizando las estanterías"],
  ["Rack",                              "Recolocando productos"],
  ["Explanations",                      "Generando explicaciones"],
  ["OPTIMIZATION RESULTS",              "Guardando resultados"],
];

interface ParserState {
  stage: FriendlyProgress["stage"];
  step?: string;
  labelHint?: string;
}

export function parseLine(state: ParserState, line: string): FriendlyProgress {
  // Epoch progress  →  extract a percentage if present
  const epochMatch = line.match(/Epoch\s+(\d+)\/(\d+)/i);
  if (epochMatch && state.stage === "ingest") {
    const cur = Number(epochMatch[1]);
    const tot = Number(epochMatch[2]);
    const pct = Math.round((cur / tot) * 100);
    const isTrans = /transformer/i.test(line);
    return {
      stage: "ingest",
      label: isTrans ? "Entrenando modelo preciso" : "Entrenando modelo rápido",
      detail: `Iteración ${cur} de ${tot}`,
      percent: pct,
    };
  }

  const rules = state.stage === "predict" ? PREDICT_RULES : INGEST_RULES;
  for (const [needle, friendly] of rules) {
    if (line.includes(needle)) {
      return { stage: state.stage, label: friendly, detail: undefined };
    }
  }

  // Unknown line — keep last label
  return { stage: state.stage, label: state.labelHint ?? "Procesando…" };
}

export const INITIAL: FriendlyProgress = DEFAULT;

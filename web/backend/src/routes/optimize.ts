import { Router, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { runPythonChainWithSSE, isRunning, killProcess } from '../services/pythonRunner';

const router = Router();

const MONTH_NAMES: Record<string, string> = {
  '01': 'january', '02': 'february', '03': 'march', '04': 'april',
  '05': 'may', '06': 'june', '07': 'july', '08': 'august',
  '09': 'september', '10': 'october', '11': 'november', '12': 'december',
};

const monthlyDir = (): string =>
  path.join(process.env.MLOPS_DIR!, 'data', 'monthly');

const resultsDir = (): string =>
  process.env.RESULTS_DIR || path.join(process.env.MLOPS_DIR!, 'results');

/** Parse `sales_YYYY_MM_*.csv` filename into a [year, month] tuple. */
function parseSalesFilename(name: string): [number, number] | null {
  const m = name.match(/^sales_(\d{4})_(\d{2})_/);
  return m ? [Number(m[1]), Number(m[2])] : null;
}

/** Compute the "next" month after the newest uploaded CSV. */
function defaultTargetMonth(): string {
  const dir = monthlyDir();
  if (!fs.existsSync(dir)) return '2026-01';
  const months = fs.readdirSync(dir)
    .map(parseSalesFilename)
    .filter((x): x is [number, number] => x !== null)
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (months.length === 0) return '2026-01';
  let [year, month] = months[months.length - 1];
  month += 1;
  if (month > 12) { month = 1; year += 1; }
  return `${year}-${String(month).padStart(2, '0')}`;
}

// GET /api/optimize/status — is a run in progress?
router.get('/status', (_req: Request, res: Response) => {
  res.json({ running: isRunning('optimize') });
});

// GET /api/optimize/default-month — suggest the target month for the UI
router.get('/default-month', (_req: Request, res: Response) => {
  res.json({ month: defaultTargetMonth() });
});

// GET /api/optimize/run?month=YYYY-MM
// SSE stream that chains 04_ingest.py → 05_predict.py so the user gets a
// single "Optimize" button instead of three separate ones.
router.get('/run', async (req: Request, res: Response) => {
  const month = typeof req.query.month === 'string' && req.query.month
    ? req.query.month
    : defaultTargetMonth();
  const dryRun = req.query.dryRun === 'true';

  const predictArgs = ['--month', month];
  if (dryRun) predictArgs.push('--dry-run');

  await runPythonChainWithSSE(res, 'optimize', [
    { name: 'ingest',  script: '04_ingest.py',  args: ['data/monthly'] },
    { name: 'predict', script: '05_predict.py', args: predictArgs },
  ]);
});

// POST /api/optimize/stop
router.post('/stop', (_req: Request, res: Response) => {
  killProcess('optimize');
  res.json({ ok: true });
});

// GET /api/optimize/results?month=YYYY-MM
// Aggregated payload for the results page — KPIs, movement matrix for the
// Sankey, and the top racks by profit lift. Computed from the CSVs the
// pipeline just produced.
router.get('/results', (req: Request, res: Response) => {
  const month = String(req.query.month || '');
  const [year, mon] = month.split('-');
  if (!year || !mon || !MONTH_NAMES[mon]) {
    res.status(400).json({ error: 'month must be YYYY-MM format' });
    return;
  }

  const monthName = MONTH_NAMES[mon];
  const rd = resultsDir();
  const optimizedCsv = path.join(rd, `optimized_${year}_${mon}_${monthName}.csv`);
  const forecastJson = path.join(rd, `forecast_${year}_${mon}.json`);
  const explanationsJson = path.join(rd, `explanations_${year}_${mon}.json`);

  if (!fs.existsSync(optimizedCsv)) {
    res.status(404).json({ error: `No optimized CSV yet for ${month}. Run optimize first.` });
    return;
  }

  // Parse the optimized CSV
  const optimized = parseCsv(fs.readFileSync(optimizedCsv, 'utf-8'));

  // Find the baseline CSV: the newest CSV in data/monthly/ at the time of
  // the run. We pick the one with the largest (year, month) ≤ target.
  const baseline = newestMonthlyCsv(before(year, mon));

  // Compute KPIs and movements by joining on product name
  const { movements, kpi, racks } = aggregate(baseline, optimized);

  const forecast = fs.existsSync(forecastJson)
    ? JSON.parse(fs.readFileSync(forecastJson, 'utf-8')) : null;
  let forecastSource: string | null = null;
  let multipliers: Record<string, number> = {};
  if (forecast) {
    if (forecast.multipliers) {
      multipliers = forecast.multipliers;
      forecastSource = forecast._source ?? null;
    } else {
      multipliers = forecast;
    }
  }

  const explanations = fs.existsSync(explanationsJson)
    ? JSON.parse(fs.readFileSync(explanationsJson, 'utf-8')) : null;

  res.json({ kpi, movements, racks, multipliers, forecastSource, explanations });
});

/* -------------------------------------------------------------------------- */
/*  Helpers                                                                   */
/* -------------------------------------------------------------------------- */

function before(year: string, mon: string): [number, number] {
  let y = Number(year);
  let m = Number(mon) - 1;
  if (m < 1) { m = 12; y -= 1; }
  return [y, m];
}

function newestMonthlyCsv(upTo?: [number, number]): Record<string, string>[] {
  const dir = monthlyDir();
  if (!fs.existsSync(dir)) return [];
  const files = fs.readdirSync(dir)
    .map(f => ({ name: f, yearMonth: parseSalesFilename(f) }))
    .filter(x => x.yearMonth !== null)
    .sort((a, b) => {
      const [ay, am] = a.yearMonth!;
      const [by, bm] = b.yearMonth!;
      return ay - by || am - bm;
    });
  const candidates = upTo
    ? files.filter(x => {
        const [y, m] = x.yearMonth!;
        return y < upTo[0] || (y === upTo[0] && m <= upTo[1]);
      })
    : files;
  if (candidates.length === 0 && files.length === 0) return [];
  const pick = candidates[candidates.length - 1] ?? files[files.length - 1];
  return parseCsv(fs.readFileSync(path.join(dir, pick.name), 'utf-8'));
}

function parseCsv(text: string): Record<string, string>[] {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const values = line.match(/(".*?"|[^,]+)(?=,|$)/g) || line.split(',');
    const rec: Record<string, string> = {};
    headers.forEach((h, i) => {
      rec[h] = (values[i] || '').replace(/^"|"$/g, '').trim();
    });
    return rec;
  });
}

interface Movement { from: number; to: number; count: number; }
interface Kpi {
  profitOriginal: number;
  profitOptimized: number;
  profitLiftEur: number;
  profitLiftPct: number;
  productsMoved: number;
  totalProducts: number;
  racksImproved: number;
}
interface RackRow {
  rack: string;
  products: number;
  original: number;
  optimized: number;
  lift: number;
}

function productProfit(row: Record<string, string>): number {
  const price = parseFloat(row.price_numeric || row.price || '0') || 0;
  const margin = (parseFloat(row.profit_margin_percentage || '0') || 0) / 100;
  const sales = parseFloat(row.estimated_monthly_sales || '0') || 0;
  const shelf = parseInt(row.shelf_level || '1', 10) || 1;
  // Same shelf multipliers as retail_physics.py
  const mult = [3, 4, 5].includes(shelf) ? 1.2 : 0.7;
  return price * margin * sales * mult;
}

function aggregate(
  baseline: Record<string, string>[],
  optimized: Record<string, string>[],
): { movements: Movement[]; kpi: Kpi; racks: RackRow[] } {
  const origByName = new Map<string, Record<string, string>>();
  baseline.forEach(r => { if (r.name) origByName.set(r.name, r); });

  const matrix: Record<string, number> = {};
  const rackMap: Record<string, { orig: number; opt: number; products: number }> = {};

  let totalOrig = 0;
  let totalOpt = 0;
  let moved = 0;
  let total = 0;

  optimized.forEach(o => {
    total += 1;
    const orig = origByName.get(o.name);
    const origShelf = orig ? parseInt(orig.shelf_level || '1', 10) : parseInt(o.shelf_level || '1', 10);
    const newShelf = parseInt(o.shelf_level || '1', 10) || 1;
    const from = Math.max(1, Math.min(7, origShelf));
    const to = Math.max(1, Math.min(7, newShelf));
    const key = `${from}->${to}`;
    matrix[key] = (matrix[key] || 0) + 1;
    if (from !== to) moved += 1;

    const oProfit = orig ? productProfit({ ...orig }) : productProfit({ ...o, shelf_level: String(origShelf) });
    const nProfit = productProfit(o);
    totalOrig += oProfit;
    totalOpt += nProfit;

    const rack = String(o.rack_id ?? o.Category ?? 'Unknown');
    if (!rackMap[rack]) rackMap[rack] = { orig: 0, opt: 0, products: 0 };
    rackMap[rack].orig += oProfit;
    rackMap[rack].opt += nProfit;
    rackMap[rack].products += 1;
  });

  const movements: Movement[] = Object.entries(matrix)
    .map(([k, count]) => {
      const [from, to] = k.split('->').map(Number);
      return { from, to, count };
    })
    .sort((a, b) => b.count - a.count);

  const racks: RackRow[] = Object.entries(rackMap)
    .map(([rack, v]) => ({
      rack, products: v.products,
      original: v.orig, optimized: v.opt, lift: v.opt - v.orig,
    }))
    .sort((a, b) => b.lift - a.lift);

  const racksImproved = racks.filter(r => r.lift > 0).length;

  const kpi: Kpi = {
    profitOriginal: totalOrig,
    profitOptimized: totalOpt,
    profitLiftEur: totalOpt - totalOrig,
    profitLiftPct: totalOrig > 0 ? ((totalOpt - totalOrig) / totalOrig) * 100 : 0,
    productsMoved: moved,
    totalProducts: total,
    racksImproved,
  };

  return { movements, kpi, racks };
}

export { router as optimizeRouter };

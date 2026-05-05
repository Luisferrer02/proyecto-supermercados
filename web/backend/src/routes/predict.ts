import { Router, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { runPythonWithSSE, isRunning, killProcess } from '../services/pythonRunner';
import { parseCsv } from '../services/csv';

const router = Router();

const MONTH_NAMES: Record<string, string> = {
  '01': 'january', '02': 'february', '03': 'march', '04': 'april',
  '05': 'may', '06': 'june', '07': 'july', '08': 'august',
  '09': 'september', '10': 'october', '11': 'november', '12': 'december',
};

// GET /api/predict/status
router.get('/status', (_req: Request, res: Response) => {
  res.json({ running: isRunning('predict') });
});

// GET /api/predict/stream?month=YYYY-MM&category=X&dryRun=true
router.get('/stream', (req: Request, res: Response) => {
  const { month, category, dryRun } = req.query;

  if (!month || typeof month !== 'string' || !/^\d{4}-\d{2}$/.test(month)) {
    res.status(400).json({ error: 'month query param required (YYYY-MM format)' });
    return;
  }

  const args: string[] = ['--month', month];
  if (category && typeof category === 'string') {
    if (!/^[A-Za-zÀ-ÿ0-9 _-]+$/.test(category)) {
      res.status(400).json({ error: 'Invalid category format' });
      return;
    }
    args.push('--category', category);
  }
  if (dryRun === 'true') args.push('--dry-run');

  runPythonWithSSE(res, 'predict', '05_predict.py', args);
});

// POST /api/predict/stop
router.post('/stop', (_req: Request, res: Response) => {
  killProcess('predict');
  res.json({ ok: true });
});

// GET /api/predict/results?month=YYYY-MM
router.get('/results', (req: Request, res: Response) => {
  const month = String(req.query.month || '');
  const [year, mon] = month.split('-');

  if (!year || !mon || !MONTH_NAMES[mon]) {
    res.status(400).json({ error: 'month must be YYYY-MM format' });
    return;
  }

  const monthName = MONTH_NAMES[mon];
  const resultsDir = process.env.RESULTS_DIR!;
  const csvPath = path.join(resultsDir, `optimized_${year}_${mon}_${monthName}.csv`);
  const forecastPath = path.join(resultsDir, `forecast_${year}_${mon}.json`);
  const explanationsPath = path.join(resultsDir, `explanations_${year}_${mon}.json`);

  if (!fs.existsSync(csvPath)) {
    res.status(404).json({ error: `No results found for ${month}. Run prediction first.` });
    return;
  }

  const products = parseCsv(fs.readFileSync(csvPath, 'utf-8'));

  // Forecast JSON comes in two shapes depending on the version of
  // 05_predict.py that produced it. Normalize to a flat { category: mult }
  // dict plus an optional source flag ("llm" | "heuristic").
  let forecast: Record<string, number> = {};
  let forecastSource: string | null = null;
  if (fs.existsSync(forecastPath)) {
    const raw = JSON.parse(fs.readFileSync(forecastPath, 'utf-8'));
    if (raw && typeof raw === 'object' && raw.multipliers) {
      forecast = raw.multipliers;
      forecastSource = raw._source ?? null;
    } else {
      forecast = raw;
    }
  }

  // Mirrors mlops/utils/retail_physics.py SHELF_MULTIPLIERS.
  const SHELF_MULTIPLIERS: Record<number, number> = {
    1: 0.60, 2: 0.80, 3: 0.95, 4: 1.15, 5: 1.00, 6: 0.75, 7: 0.50,
  };

  // Compute per-rack profit summary
  const rackMap: Record<string, { original: number; optimized: number; products: number }> = {};
  products.forEach(p => {
    const rack = p.rack_id || p.Category || 'Unknown';
    if (!rackMap[rack]) rackMap[rack] = { original: 0, optimized: 0, products: 0 };
    const price = parseFloat(p.price_numeric || p.price || '0');
    const margin = parseFloat(p.profit_margin_percentage || '0') / 100;
    const sales = parseFloat(p.estimated_monthly_sales || '0');
    const shelf = parseInt(p.shelf_level || '1');
    const shelfMult = SHELF_MULTIPLIERS[shelf] ?? 0.60;
    rackMap[rack].optimized += price * margin * sales * shelfMult;
    rackMap[rack].original += price * margin * sales * 0.95; // baseline approximation
    rackMap[rack].products += 1;
  });

  // Optional per-product explanations (see mlops/utils/explainability.py).
  // Missing file just means an older run — return null and let the UI hide
  // the explanation panel.
  let explanations: unknown = null;
  if (fs.existsSync(explanationsPath)) {
    try {
      explanations = JSON.parse(fs.readFileSync(explanationsPath, 'utf-8'));
    } catch {
      explanations = null;
    }
  }

  res.json({ products, forecast, forecastSource, rackSummary: rackMap, explanations });
});

// GET /api/predict/list — list all available prediction result months
router.get('/list', (_req: Request, res: Response) => {
  const resultsDir = process.env.RESULTS_DIR!;
  if (!fs.existsSync(resultsDir)) {
    res.json({ months: [] });
    return;
  }
  const months = fs.readdirSync(resultsDir)
    .filter(f => f.match(/^optimized_\d{4}_\d{2}_\w+\.csv$/))
    .map(f => {
      const m = f.match(/^optimized_(\d{4})_(\d{2})_/);
      return m ? `${m[1]}-${m[2]}` : null;
    })
    .filter(Boolean);
  res.json({ months });
});

export { router as predictRouter };

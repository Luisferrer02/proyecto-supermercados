import { Router, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { runPythonWithSSE, isRunning, killProcess } from '../services/pythonRunner';

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

  if (!month || typeof month !== 'string') {
    res.status(400).json({ error: 'month query param required (YYYY-MM)' });
    return;
  }

  const args: string[] = ['--month', month];
  if (category && typeof category === 'string') args.push('--category', category);
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

  if (!fs.existsSync(csvPath)) {
    res.status(404).json({ error: `No results found for ${month}. Run prediction first.` });
    return;
  }

  // Parse CSV → JSON
  const csvText = fs.readFileSync(csvPath, 'utf-8');
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  const products = lines.slice(1).map(line => {
    // Handle quoted fields with commas inside
    const values = line.match(/(".*?"|[^,]+)(?=,|$)/g) || line.split(',');
    const record: Record<string, string> = {};
    headers.forEach((h, i) => {
      record[h] = (values[i] || '').replace(/^"|"$/g, '').trim();
    });
    return record;
  });

  const forecast = fs.existsSync(forecastPath)
    ? JSON.parse(fs.readFileSync(forecastPath, 'utf-8'))
    : {};

  // Compute per-rack profit summary
  const rackMap: Record<string, { original: number; optimized: number; products: number }> = {};
  products.forEach(p => {
    const rack = p.rack_id || p.Category || 'Unknown';
    if (!rackMap[rack]) rackMap[rack] = { original: 0, optimized: 0, products: 0 };
    const price = parseFloat(p.price_numeric || p.price || '0');
    const margin = parseFloat(p.profit_margin_percentage || '0') / 100;
    const sales = parseFloat(p.estimated_monthly_sales || '0');
    const shelfMult = [3, 4, 5].includes(parseInt(p.shelf_level || '1')) ? 1.2 : 0.8;
    rackMap[rack].optimized += price * margin * sales * shelfMult;
    rackMap[rack].original += price * margin * sales * 0.95; // baseline approximation
    rackMap[rack].products += 1;
  });

  res.json({ products, forecast, rackSummary: rackMap });
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

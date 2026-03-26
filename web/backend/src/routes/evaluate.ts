import { Router, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { runPythonWithSSE, isRunning, killProcess } from '../services/pythonRunner';

const router = Router();

const CHARTS = [
  'mse_comparison.png',
  'profit_comparison.png',
  'rack_comparison.png',
  'alluvial_diagram.png',
];

// GET /api/evaluate/status — check which charts exist
router.get('/status', (_req: Request, res: Response) => {
  const resultsDir = process.env.RESULTS_DIR!;
  const charts = CHARTS.map(name => ({
    name,
    exists: fs.existsSync(path.join(resultsDir, name)),
    url: `/results/${name}`,
  }));
  res.json({ charts, running: isRunning('evaluate') });
});

// GET /api/evaluate/stream — SSE: run 03_evaluate.py
router.get('/stream', (req: Request, res: Response) => {
  runPythonWithSSE(res, 'evaluate', '03_evaluate.py', []);
});

// POST /api/evaluate/stop
router.post('/stop', (_req: Request, res: Response) => {
  killProcess('evaluate');
  res.json({ ok: true });
});

export { router as evaluateRouter };

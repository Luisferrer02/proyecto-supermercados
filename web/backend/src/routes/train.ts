import { Router, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { runPythonWithSSE, isRunning, killProcess } from '../services/pythonRunner';

const router = Router();

// GET /api/train/status
router.get('/status', (_req: Request, res: Response) => {
  res.json({ running: isRunning('train') });
});

// GET /api/train/stream — SSE: run 02_train_models.py
router.get('/stream', (req: Request, res: Response) => {
  runPythonWithSSE(res, 'train', '02_train_models.py', []);
});

// POST /api/train/stop
router.post('/stop', (_req: Request, res: Response) => {
  killProcess('train');
  res.json({ ok: true });
});

// GET /api/train/results — return training_results.json
router.get('/results', (_req: Request, res: Response) => {
  const resultsPath = path.join(process.env.MLOPS_DIR!, 'results', 'training_results.json');
  if (!fs.existsSync(resultsPath)) {
    res.status(404).json({ error: 'No training results yet. Run training first.' });
    return;
  }
  const data = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));
  res.json(data);
});

export { router as trainRouter };

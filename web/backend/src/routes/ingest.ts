import { Router, Request, Response } from 'express';
import { runPythonWithSSE, isRunning, killProcess } from '../services/pythonRunner';

const router = Router();

// GET /api/ingest/status
router.get('/status', (_req: Request, res: Response) => {
  res.json({ running: isRunning('ingest') });
});

// GET /api/ingest/stream — SSE: run 04_ingest.py data/monthly/
router.get('/stream', (req: Request, res: Response) => {
  const args: string[] = ['data/monthly'];
  runPythonWithSSE(res, 'ingest', '04_ingest.py', args);
});

// POST /api/ingest/stop
router.post('/stop', (_req: Request, res: Response) => {
  killProcess('ingest');
  res.json({ ok: true });
});

export { router as ingestRouter };

import express from 'express';
import cors from 'cors';
import path from 'path';
import dotenv from 'dotenv';

dotenv.config();

import { uploadRouter } from './routes/upload';
import { trainRouter } from './routes/train';
import { evaluateRouter } from './routes/evaluate';
import { ingestRouter } from './routes/ingest';
import { predictRouter } from './routes/predict';
import { optimizeRouter } from './routes/optimize';

const app = express();
const PORT = parseInt(process.env.PORT || '3001', 10);

// Security headers
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const helmet = require('helmet');
  app.use(helmet());
} catch { /* helmet not installed — skip in dev */ }

// Rate limiting
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const rateLimit = require('express-rate-limit');
  app.use('/api/', rateLimit({ windowMs: 60_000, max: 100 }));
} catch { /* express-rate-limit not installed — skip in dev */ }

// CORS from environment
const corsOrigins = (process.env.CORS_ORIGINS || 'http://localhost:3000,http://127.0.0.1:3000')
  .split(',').map(s => s.trim());
app.use(cors({ origin: corsOrigins }));

// Body size limit
app.use(express.json({ limit: '1mb' }));

// Serve generated chart PNGs and result CSVs as static files
const resultsDir = process.env.RESULTS_DIR || path.join(process.env.MLOPS_DIR || '', 'results');
app.use('/results', express.static(resultsDir));

// Routes
app.use('/api/upload', uploadRouter);
app.use('/api/train', trainRouter);
app.use('/api/evaluate', evaluateRouter);
app.use('/api/ingest', ingestRouter);
app.use('/api/predict', predictRouter);
app.use('/api/optimize', optimizeRouter);

// Health check
app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    mlopsDir: process.env.MLOPS_DIR,
    python: process.env.PYTHON_PATH || 'python3',
  });
});

app.listen(PORT, () => {
  console.log(`Backend running at http://localhost:${PORT}`);
  console.log(`MLOPS_DIR: ${process.env.MLOPS_DIR}`);
});

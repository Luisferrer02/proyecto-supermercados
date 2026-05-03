import express from 'express';
import os from 'os';
import path from 'path';
import fs from 'fs';

export function createTmpDirs() {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'mlops-test-'));
  const monthly = path.join(base, 'data', 'monthly');
  const results = path.join(base, 'results');
  fs.mkdirSync(monthly, { recursive: true });
  fs.mkdirSync(results, { recursive: true });
  process.env.MLOPS_DIR = base;
  process.env.RESULTS_DIR = results;
  return { base, monthly, results };
}

export function cleanupTmpDirs(base: string) {
  fs.rmSync(base, { recursive: true, force: true });
}

export function buildApp(router: express.Router, prefix: string) {
  const app = express();
  app.use(express.json());
  app.use(prefix, router);
  return app;
}

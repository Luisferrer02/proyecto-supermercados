/**
 * Contract tests — validate that backend route responses conform to the
 * TypeScript interfaces in @shared/api-types.ts. These catch drift between
 * the backend JSON and what the frontend expects at compile time.
 */
import request from 'supertest';
import path from 'path';
import fs from 'fs';
import express from 'express';
import os from 'os';
import type {
  UploadFilesResponse,
  TrainResults,
  EvaluateStatusResponse,
  StatusResponse,
  PredictListResponse,
  OptimizeDefaultMonthResponse,
} from '@shared/api-types';

// --- Helpers ----------------------------------------------------------------

function createTmpDirs() {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'contract-test-'));
  const monthly = path.join(base, 'data', 'monthly');
  const results = path.join(base, 'results');
  fs.mkdirSync(monthly, { recursive: true });
  fs.mkdirSync(results, { recursive: true });
  process.env.MLOPS_DIR = base;
  process.env.RESULTS_DIR = results;
  return { base, monthly, results };
}

function assertKeysAndTypes(obj: any, spec: Record<string, string | string[]>) {
  for (const [key, expectedType] of Object.entries(spec)) {
    expect(obj).toHaveProperty(key);
    const types = Array.isArray(expectedType) ? expectedType : [expectedType];
    expect(types).toContain(typeof obj[key]);
  }
}

// --- Mocks ------------------------------------------------------------------

jest.mock('../src/services/pythonRunner', () => ({
  isRunning: jest.fn(() => false),
  killProcess: jest.fn(),
  runPythonWithSSE: jest.fn((res: any) => { res.status(200).end(); }),
  runPythonChainWithSSE: jest.fn((res: any) => { res.status(200).end(); }),
}));

jest.mock('child_process', () => ({
  spawn: jest.fn(() => {
    const { EventEmitter } = require('events');
    const proc: any = new EventEmitter();
    proc.stdout = new EventEmitter();
    proc.stderr = new EventEmitter();
    proc.exitCode = null;
    process.nextTick(() => {
      proc.stdout.emit('data', Buffer.from('OK\n'));
      proc.exitCode = 0;
      proc.emit('close', 0);
    });
    return proc;
  }),
}));

// --- Setup ------------------------------------------------------------------

let dirs: ReturnType<typeof createTmpDirs>;

beforeEach(() => {
  dirs = createTmpDirs();
});

afterEach(() => {
  fs.rmSync(dirs.base, { recursive: true, force: true });
});

// --- Upload contracts -------------------------------------------------------

describe('Contract: Upload', () => {
  let app: express.Express;

  beforeEach(async () => {
    const { uploadRouter } = await import('../src/routes/upload');
    app = express();
    app.use(express.json());
    app.use('/api/upload', uploadRouter);
  });

  it('GET /api/upload/files matches UploadFilesResponse', async () => {
    fs.writeFileSync(path.join(dirs.monthly, 'sales_2026_01_january.csv'), 'data');
    const res = await request(app).get('/api/upload/files');
    expect(res.status).toBe(200);

    const body: UploadFilesResponse = res.body;
    expect(Array.isArray(body.files)).toBe(true);
    expect(body.files.length).toBe(1);
    assertKeysAndTypes(body.files[0], { name: 'string', size: 'number' });
  });
});

// --- Train contracts --------------------------------------------------------

describe('Contract: Train', () => {
  let app: express.Express;

  beforeEach(async () => {
    const { trainRouter } = await import('../src/routes/train');
    app = express();
    app.use(express.json());
    app.use('/api/train', trainRouter);
  });

  it('GET /api/train/status matches StatusResponse', async () => {
    const res = await request(app).get('/api/train/status');
    expect(res.status).toBe(200);

    const body: StatusResponse = res.body;
    expect(typeof body.running).toBe('boolean');
  });

  it('GET /api/train/results matches TrainResults when file exists', async () => {
    const resultsData = {
      MLP: { mse: 10, rmse_eur: 3.16, mae_eur: 2.5, original_profit: 5000, optimized_profit: 5400 },
      Transformer: { mse: 8, rmse_eur: 2.83, mae_eur: 2.0, original_profit: 5000, optimized_profit: 5600 },
    };
    fs.writeFileSync(path.join(dirs.results, 'training_results.json'), JSON.stringify(resultsData));

    const res = await request(app).get('/api/train/results');
    expect(res.status).toBe(200);

    const body: TrainResults = res.body;
    expect(typeof body).toBe('object');
    for (const [model, metrics] of Object.entries(body)) {
      expect(typeof model).toBe('string');
      if (metrics.mse !== undefined) expect(typeof metrics.mse).toBe('number');
      if (metrics.rmse_eur !== undefined) expect(typeof metrics.rmse_eur).toBe('number');
      if (metrics.mae_eur !== undefined) expect(typeof metrics.mae_eur).toBe('number');
    }
  });
});

// --- Evaluate contracts -----------------------------------------------------

describe('Contract: Evaluate', () => {
  let app: express.Express;

  beforeEach(async () => {
    const { evaluateRouter } = await import('../src/routes/evaluate');
    app = express();
    app.use(express.json());
    app.use('/api/evaluate', evaluateRouter);
  });

  it('GET /api/evaluate/status matches EvaluateStatusResponse', async () => {
    const res = await request(app).get('/api/evaluate/status');
    expect(res.status).toBe(200);

    const body: EvaluateStatusResponse = res.body;
    expect(typeof body.running).toBe('boolean');
    expect(Array.isArray(body.charts)).toBe(true);
    for (const chart of body.charts) {
      assertKeysAndTypes(chart, { name: 'string', exists: 'boolean', url: 'string' });
    }
  });
});

// --- Predict contracts ------------------------------------------------------

describe('Contract: Predict', () => {
  let app: express.Express;

  beforeEach(async () => {
    const { predictRouter } = await import('../src/routes/predict');
    app = express();
    app.use(express.json());
    app.use('/api/predict', predictRouter);
  });

  it('GET /api/predict/list matches PredictListResponse', async () => {
    fs.writeFileSync(path.join(dirs.results, 'optimized_2026_01_january.csv'), 'data');
    const res = await request(app).get('/api/predict/list');
    expect(res.status).toBe(200);

    const body: PredictListResponse = res.body;
    expect(Array.isArray(body.months)).toBe(true);
    for (const m of body.months) {
      expect(typeof m).toBe('string');
    }
  });

  it('GET /api/predict/status matches StatusResponse', async () => {
    const res = await request(app).get('/api/predict/status');
    expect(res.status).toBe(200);

    const body: StatusResponse = res.body;
    expect(typeof body.running).toBe('boolean');
  });
});

// --- Optimize contracts -----------------------------------------------------

describe('Contract: Optimize', () => {
  let app: express.Express;

  beforeEach(async () => {
    const { optimizeRouter } = await import('../src/routes/optimize');
    app = express();
    app.use(express.json());
    app.use('/api/optimize', optimizeRouter);
  });

  it('GET /api/optimize/status matches StatusResponse', async () => {
    const res = await request(app).get('/api/optimize/status');
    expect(res.status).toBe(200);

    const body: StatusResponse = res.body;
    expect(typeof body.running).toBe('boolean');
  });

  it('GET /api/optimize/default-month matches OptimizeDefaultMonthResponse', async () => {
    fs.writeFileSync(path.join(dirs.monthly, 'sales_2025_12_december.csv'), 'data');
    const res = await request(app).get('/api/optimize/default-month');
    expect(res.status).toBe(200);

    const body: OptimizeDefaultMonthResponse = res.body;
    expect(typeof body.month).toBe('string');
    expect(body.month).toMatch(/^\d{4}-\d{2}$/);
  });
});

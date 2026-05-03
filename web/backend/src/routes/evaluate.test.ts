import request from 'supertest';
import path from 'path';
import fs from 'fs';
import { createTmpDirs, cleanupTmpDirs, buildApp } from '../../tests/helpers';

jest.mock('../services/pythonRunner', () => ({
  isRunning: jest.fn(() => false),
  killProcess: jest.fn(),
  runPythonWithSSE: jest.fn((res: any) => { res.status(200).end(); }),
}));

import { evaluateRouter } from './evaluate';
import { isRunning, killProcess, runPythonWithSSE } from '../services/pythonRunner';

const mockIsRunning = isRunning as jest.MockedFunction<typeof isRunning>;

let dirs: ReturnType<typeof createTmpDirs>;
let app: ReturnType<typeof buildApp>;

beforeEach(() => {
  jest.clearAllMocks();
  dirs = createTmpDirs();
  app = buildApp(evaluateRouter, '/api/evaluate');
});

afterEach(() => cleanupTmpDirs(dirs.base));

describe('GET /api/evaluate/status', () => {
  it('reports which charts exist and running state', async () => {
    // Create two of the four expected charts
    fs.writeFileSync(path.join(dirs.results, 'mse_comparison.png'), '');
    fs.writeFileSync(path.join(dirs.results, 'profit_comparison.png'), '');

    const res = await request(app).get('/api/evaluate/status');
    expect(res.status).toBe(200);
    expect(res.body.charts).toHaveLength(4);

    const existing = res.body.charts.filter((c: any) => c.exists);
    expect(existing).toHaveLength(2);
    expect(existing.map((c: any) => c.name).sort()).toEqual([
      'mse_comparison.png',
      'profit_comparison.png',
    ]);

    const missing = res.body.charts.filter((c: any) => !c.exists);
    expect(missing).toHaveLength(2);

    expect(res.body.running).toBe(false);
  });

  it('reflects running state from pythonRunner', async () => {
    mockIsRunning.mockReturnValue(true);
    const res = await request(app).get('/api/evaluate/status');
    expect(res.body.running).toBe(true);
  });
});

describe('GET /api/evaluate/stream', () => {
  it('calls runPythonWithSSE with the evaluate script', async () => {
    await request(app).get('/api/evaluate/stream');
    expect(runPythonWithSSE).toHaveBeenCalledWith(
      expect.anything(), 'evaluate', '03_evaluate.py', [],
    );
  });
});

describe('POST /api/evaluate/stop', () => {
  it('stops the process and returns ok', async () => {
    const res = await request(app).post('/api/evaluate/stop');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ ok: true });
    expect(killProcess).toHaveBeenCalledWith('evaluate');
  });
});

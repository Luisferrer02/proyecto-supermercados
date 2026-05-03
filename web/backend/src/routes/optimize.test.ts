import request from 'supertest';
import path from 'path';
import fs from 'fs';
import { createTmpDirs, cleanupTmpDirs, buildApp } from '../../tests/helpers';

jest.mock('../services/pythonRunner', () => ({
  isRunning: jest.fn(() => false),
  killProcess: jest.fn(),
  runPythonChainWithSSE: jest.fn(async (res: any) => { res.status(200).end(); }),
}));

import { optimizeRouter } from './optimize';
import { isRunning, killProcess, runPythonChainWithSSE } from '../services/pythonRunner';

const mockIsRunning = isRunning as jest.MockedFunction<typeof isRunning>;
const mockRunChain = runPythonChainWithSSE as jest.MockedFunction<typeof runPythonChainWithSSE>;

let dirs: ReturnType<typeof createTmpDirs>;
let app: ReturnType<typeof buildApp>;

const BASELINE_CSV = [
  'name,price_numeric,profit_margin_percentage,estimated_monthly_sales,shelf_level,rack_id,Category',
  'Apple,2.00,20,100,3,R1,Fruit',
  'Banana,1.50,15,200,3,R1,Fruit',
].join('\n');

const OPTIMIZED_CSV = [
  'name,price_numeric,profit_margin_percentage,estimated_monthly_sales,shelf_level,rack_id,Category',
  'Apple,2.00,20,100,4,R1,Fruit',
  'Banana,1.50,15,200,4,R1,Fruit',
].join('\n');

beforeEach(() => {
  jest.clearAllMocks();
  dirs = createTmpDirs();
  app = buildApp(optimizeRouter, '/api/optimize');
});

afterEach(() => cleanupTmpDirs(dirs.base));

describe('GET /api/optimize/status', () => {
  it('returns running state', async () => {
    const res = await request(app).get('/api/optimize/status');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ running: false });
  });
});

describe('GET /api/optimize/default-month', () => {
  it('returns 2026-01 when no uploaded files', async () => {
    const res = await request(app).get('/api/optimize/default-month');
    expect(res.status).toBe(200);
    expect(res.body.month).toBe('2026-01');
  });

  it('returns the next month after newest uploaded CSV', async () => {
    fs.writeFileSync(path.join(dirs.monthly, 'sales_2026_03_march.csv'), 'data');
    const res = await request(app).get('/api/optimize/default-month');
    expect(res.body.month).toBe('2026-04');
  });

  it('wraps from December to January of next year', async () => {
    fs.writeFileSync(path.join(dirs.monthly, 'sales_2025_12_december.csv'), 'data');
    const res = await request(app).get('/api/optimize/default-month');
    expect(res.body.month).toBe('2026-01');
  });
});

describe('GET /api/optimize/run', () => {
  it('chains ingest then predict with default month', async () => {
    await request(app).get('/api/optimize/run');
    expect(mockRunChain).toHaveBeenCalledWith(
      expect.anything(),
      'optimize',
      expect.arrayContaining([
        expect.objectContaining({ name: 'ingest', script: '04_ingest.py' }),
        expect.objectContaining({ name: 'predict', script: '05_predict.py' }),
      ]),
    );
  });

  it('passes explicit month and dryRun to predict step', async () => {
    await request(app).get('/api/optimize/run').query({ month: '2026-06', dryRun: 'true' });

    const steps = mockRunChain.mock.calls[0][2];
    const predictStep = steps.find((s: any) => s.name === 'predict');
    expect(predictStep!.args).toContain('--month');
    expect(predictStep!.args).toContain('2026-06');
    expect(predictStep!.args).toContain('--dry-run');
  });
});

describe('POST /api/optimize/stop', () => {
  it('stops the process and returns ok', async () => {
    const res = await request(app).post('/api/optimize/stop');
    expect(res.status).toBe(200);
    expect(killProcess).toHaveBeenCalledWith('optimize');
  });
});

describe('GET /api/optimize/results', () => {
  it('returns 400 for invalid month format', async () => {
    const res = await request(app).get('/api/optimize/results').query({ month: 'bad' });
    expect(res.status).toBe(400);
  });

  it('returns 404 when optimized CSV does not exist', async () => {
    const res = await request(app).get('/api/optimize/results').query({ month: '2026-03' });
    expect(res.status).toBe(404);
  });

  it('computes KPIs and movements from baseline vs optimized CSVs', async () => {
    // Baseline in monthly dir
    fs.writeFileSync(path.join(dirs.monthly, 'sales_2026_02_february.csv'), BASELINE_CSV);
    // Optimized result
    fs.writeFileSync(path.join(dirs.results, 'optimized_2026_03_march.csv'), OPTIMIZED_CSV);

    const res = await request(app).get('/api/optimize/results').query({ month: '2026-03' });
    expect(res.status).toBe(200);

    // KPI checks
    expect(res.body.kpi).toBeDefined();
    expect(res.body.kpi.totalProducts).toBe(2);
    expect(res.body.kpi.productsMoved).toBe(2); // both moved from shelf 3 → 4
    expect(res.body.kpi.profitOptimized).toBeGreaterThan(res.body.kpi.profitOriginal);
    expect(res.body.kpi.profitLiftPct).toBeGreaterThan(0);

    // Movements
    expect(res.body.movements).toBeDefined();
    expect(res.body.movements.length).toBeGreaterThan(0);
    expect(res.body.movements[0]).toHaveProperty('from');
    expect(res.body.movements[0]).toHaveProperty('to');
    expect(res.body.movements[0]).toHaveProperty('count');

    // Racks
    expect(res.body.racks).toBeDefined();
    expect(res.body.racks[0].rack).toBe('R1');
    expect(res.body.racks[0].products).toBe(2);
  });

  it('includes forecast and explanations when files exist', async () => {
    fs.writeFileSync(path.join(dirs.monthly, 'sales_2026_02_february.csv'), BASELINE_CSV);
    fs.writeFileSync(path.join(dirs.results, 'optimized_2026_03_march.csv'), OPTIMIZED_CSV);
    fs.writeFileSync(
      path.join(dirs.results, 'forecast_2026_03.json'),
      JSON.stringify({ multipliers: { Fruit: 1.1 }, _source: 'llm' }),
    );
    fs.writeFileSync(
      path.join(dirs.results, 'explanations_2026_03.json'),
      JSON.stringify([{ product: 'Apple', reason: 'eye level' }]),
    );

    const res = await request(app).get('/api/optimize/results').query({ month: '2026-03' });
    expect(res.body.multipliers).toEqual({ Fruit: 1.1 });
    expect(res.body.forecastSource).toBe('llm');
    expect(res.body.explanations).toHaveLength(1);
  });

  it('returns null forecast/explanations when files are missing', async () => {
    fs.writeFileSync(path.join(dirs.monthly, 'sales_2026_02_february.csv'), BASELINE_CSV);
    fs.writeFileSync(path.join(dirs.results, 'optimized_2026_03_march.csv'), OPTIMIZED_CSV);

    const res = await request(app).get('/api/optimize/results').query({ month: '2026-03' });
    expect(res.body.multipliers).toEqual({});
    expect(res.body.forecastSource).toBeNull();
    expect(res.body.explanations).toBeNull();
  });
});

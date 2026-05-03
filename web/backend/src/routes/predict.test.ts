import request from 'supertest';
import path from 'path';
import fs from 'fs';
import { createTmpDirs, cleanupTmpDirs, buildApp } from '../../tests/helpers';

jest.mock('../services/pythonRunner', () => ({
  isRunning: jest.fn(() => false),
  killProcess: jest.fn(),
  runPythonWithSSE: jest.fn((res: any) => { res.status(200).end(); }),
}));

import { predictRouter } from './predict';
import { isRunning, killProcess, runPythonWithSSE } from '../services/pythonRunner';

const mockRunPython = runPythonWithSSE as jest.MockedFunction<typeof runPythonWithSSE>;

let dirs: ReturnType<typeof createTmpDirs>;
let app: ReturnType<typeof buildApp>;

const SAMPLE_CSV = [
  'name,price_numeric,profit_margin_percentage,estimated_monthly_sales,shelf_level,rack_id,Category',
  'Apple,2.00,20,100,4,R1,Fruit',
  'Banana,1.50,15,200,2,R1,Fruit',
].join('\n');

beforeEach(() => {
  jest.clearAllMocks();
  dirs = createTmpDirs();
  app = buildApp(predictRouter, '/api/predict');
});

afterEach(() => cleanupTmpDirs(dirs.base));

describe('GET /api/predict/status', () => {
  it('returns running state', async () => {
    const res = await request(app).get('/api/predict/status');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ running: false });
  });
});

describe('GET /api/predict/stream', () => {
  it('requires month query param', async () => {
    const res = await request(app).get('/api/predict/stream');
    expect(res.status).toBe(400);
    expect(res.body.error).toContain('month');
  });

  it('passes month, category, and dryRun args to python', async () => {
    await request(app)
      .get('/api/predict/stream')
      .query({ month: '2026-03', category: 'Fruit', dryRun: 'true' });

    expect(mockRunPython).toHaveBeenCalledWith(
      expect.anything(),
      'predict',
      '05_predict.py',
      ['--month', '2026-03', '--category', 'Fruit', '--dry-run'],
    );
  });

  it('passes only month when no optional params', async () => {
    await request(app).get('/api/predict/stream').query({ month: '2026-01' });
    expect(mockRunPython).toHaveBeenCalledWith(
      expect.anything(), 'predict', '05_predict.py', ['--month', '2026-01'],
    );
  });
});

describe('POST /api/predict/stop', () => {
  it('stops the process', async () => {
    const res = await request(app).post('/api/predict/stop');
    expect(res.status).toBe(200);
    expect(killProcess).toHaveBeenCalledWith('predict');
  });
});

describe('GET /api/predict/results', () => {
  it('returns 400 for missing month', async () => {
    const res = await request(app).get('/api/predict/results');
    expect(res.status).toBe(400);
  });

  it('returns 400 for invalid month format', async () => {
    const res = await request(app).get('/api/predict/results').query({ month: '2026-13' });
    expect(res.status).toBe(400);
  });

  it('returns 404 when CSV does not exist', async () => {
    const res = await request(app).get('/api/predict/results').query({ month: '2026-03' });
    expect(res.status).toBe(404);
  });

  it('returns products and rack summary from optimized CSV', async () => {
    fs.writeFileSync(path.join(dirs.results, 'optimized_2026_03_march.csv'), SAMPLE_CSV);

    const res = await request(app).get('/api/predict/results').query({ month: '2026-03' });
    expect(res.status).toBe(200);
    expect(res.body.products).toHaveLength(2);
    expect(res.body.rackSummary).toHaveProperty('R1');
    expect(res.body.rackSummary.R1.products).toBe(2);
    expect(res.body.rackSummary.R1.optimized).toBeGreaterThan(0);
    expect(res.body.forecast).toEqual({});
    expect(res.body.forecastSource).toBeNull();
    expect(res.body.explanations).toBeNull();
  });

  it('parses nested forecast format with multipliers and source', async () => {
    fs.writeFileSync(path.join(dirs.results, 'optimized_2026_03_march.csv'), SAMPLE_CSV);
    fs.writeFileSync(
      path.join(dirs.results, 'forecast_2026_03.json'),
      JSON.stringify({ multipliers: { Fruit: 1.1 }, _source: 'llm' }),
    );

    const res = await request(app).get('/api/predict/results').query({ month: '2026-03' });
    expect(res.body.forecast).toEqual({ Fruit: 1.1 });
    expect(res.body.forecastSource).toBe('llm');
  });

  it('parses flat forecast format', async () => {
    fs.writeFileSync(path.join(dirs.results, 'optimized_2026_03_march.csv'), SAMPLE_CSV);
    fs.writeFileSync(
      path.join(dirs.results, 'forecast_2026_03.json'),
      JSON.stringify({ Fruit: 0.9 }),
    );

    const res = await request(app).get('/api/predict/results').query({ month: '2026-03' });
    expect(res.body.forecast).toEqual({ Fruit: 0.9 });
    expect(res.body.forecastSource).toBeNull();
  });

  it('includes explanations when file exists', async () => {
    fs.writeFileSync(path.join(dirs.results, 'optimized_2026_03_march.csv'), SAMPLE_CSV);
    const explanations = [{ product: 'Apple', reason: 'moved to eye level' }];
    fs.writeFileSync(
      path.join(dirs.results, 'explanations_2026_03.json'),
      JSON.stringify(explanations),
    );

    const res = await request(app).get('/api/predict/results').query({ month: '2026-03' });
    expect(res.body.explanations).toEqual(explanations);
  });

  it('falls back to null when explanations JSON is malformed', async () => {
    fs.writeFileSync(path.join(dirs.results, 'optimized_2026_03_march.csv'), SAMPLE_CSV);
    fs.writeFileSync(path.join(dirs.results, 'explanations_2026_03.json'), '{ not valid json');

    const res = await request(app).get('/api/predict/results').query({ month: '2026-03' });
    expect(res.status).toBe(200);
    expect(res.body.explanations).toBeNull();
  });
});

describe('GET /api/predict/list', () => {
  it('returns empty list when no results', async () => {
    const res = await request(app).get('/api/predict/list');
    expect(res.status).toBe(200);
    expect(res.body.months).toEqual([]);
  });

  it('lists months from optimized CSV filenames', async () => {
    fs.writeFileSync(path.join(dirs.results, 'optimized_2026_01_january.csv'), '');
    fs.writeFileSync(path.join(dirs.results, 'optimized_2026_03_march.csv'), '');
    fs.writeFileSync(path.join(dirs.results, 'other_file.csv'), '');

    const res = await request(app).get('/api/predict/list');
    expect(res.body.months).toEqual(expect.arrayContaining(['2026-01', '2026-03']));
    expect(res.body.months).toHaveLength(2);
  });
});

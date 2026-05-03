import request from 'supertest';
import path from 'path';
import fs from 'fs';
import { createTmpDirs, cleanupTmpDirs, buildApp } from '../../tests/helpers';

jest.mock('../services/pythonRunner', () => ({
  isRunning: jest.fn(() => false),
  killProcess: jest.fn(),
  runPythonWithSSE: jest.fn((res: any) => { res.status(200).end(); }),
}));

import { trainRouter } from './train';
import { isRunning, killProcess, runPythonWithSSE } from '../services/pythonRunner';

const mockIsRunning = isRunning as jest.MockedFunction<typeof isRunning>;
const mockRunPython = runPythonWithSSE as jest.MockedFunction<typeof runPythonWithSSE>;

let dirs: ReturnType<typeof createTmpDirs>;
let app: ReturnType<typeof buildApp>;

beforeEach(() => {
  jest.clearAllMocks();
  dirs = createTmpDirs();
  app = buildApp(trainRouter, '/api/train');
});

afterEach(() => cleanupTmpDirs(dirs.base));

describe('GET /api/train/status', () => {
  it('returns running: false when idle', async () => {
    const res = await request(app).get('/api/train/status');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ running: false });
  });

  it('returns running: true when active', async () => {
    mockIsRunning.mockReturnValue(true);
    const res = await request(app).get('/api/train/status');
    expect(res.body).toEqual({ running: true });
  });
});

describe('GET /api/train/stream', () => {
  it('calls runPythonWithSSE with correct script', async () => {
    await request(app).get('/api/train/stream');
    expect(mockRunPython).toHaveBeenCalledWith(
      expect.anything(), 'train', '02_train_models.py', [],
    );
  });
});

describe('POST /api/train/stop', () => {
  it('kills the train process', async () => {
    const res = await request(app).post('/api/train/stop');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ ok: true });
    expect(killProcess).toHaveBeenCalledWith('train');
  });
});

describe('GET /api/train/results', () => {
  it('returns 404 when no results exist', async () => {
    const res = await request(app).get('/api/train/results');
    expect(res.status).toBe(404);
  });

  it('returns training results JSON', async () => {
    const data = { mse: 0.05, rmse: 0.22, epochs: 100 };
    fs.writeFileSync(
      path.join(dirs.results, 'training_results.json'),
      JSON.stringify(data),
    );
    const res = await request(app).get('/api/train/results');
    expect(res.status).toBe(200);
    expect(res.body).toEqual(data);
  });
});

import request from 'supertest';
import { createTmpDirs, cleanupTmpDirs, buildApp } from '../../tests/helpers';

jest.mock('../services/pythonRunner', () => ({
  isRunning: jest.fn(() => false),
  killProcess: jest.fn(),
  runPythonWithSSE: jest.fn((res: any) => { res.status(200).end(); }),
}));

import { ingestRouter } from './ingest';
import { isRunning, killProcess, runPythonWithSSE } from '../services/pythonRunner';

const mockIsRunning = isRunning as jest.MockedFunction<typeof isRunning>;

let dirs: ReturnType<typeof createTmpDirs>;
let app: ReturnType<typeof buildApp>;

beforeEach(() => {
  jest.clearAllMocks();
  dirs = createTmpDirs();
  app = buildApp(ingestRouter, '/api/ingest');
});

afterEach(() => cleanupTmpDirs(dirs.base));

describe('GET /api/ingest/status', () => {
  it('returns running: false when idle', async () => {
    const res = await request(app).get('/api/ingest/status');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ running: false });
  });

  it('returns running: true when active', async () => {
    mockIsRunning.mockReturnValue(true);
    const res = await request(app).get('/api/ingest/status');
    expect(res.body).toEqual({ running: true });
  });
});

describe('GET /api/ingest/stream', () => {
  it('calls runPythonWithSSE with ingest script and data path', async () => {
    await request(app).get('/api/ingest/stream');
    expect(runPythonWithSSE).toHaveBeenCalledWith(
      expect.anything(), 'ingest', '04_ingest.py', ['data/monthly'],
    );
  });
});

describe('POST /api/ingest/stop', () => {
  it('stops the process and returns ok', async () => {
    const res = await request(app).post('/api/ingest/stop');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ ok: true });
    expect(killProcess).toHaveBeenCalledWith('ingest');
  });
});

import request from 'supertest';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import { EventEmitter } from 'events';
import { createTmpDirs, cleanupTmpDirs, buildApp } from '../../tests/helpers';

// Mock only child_process — we don't want a real Python validator
jest.mock('child_process', () => ({
  spawn: jest.fn(() => {
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

import { uploadRouter } from './upload';

const mockSpawn = spawn as jest.MockedFunction<typeof spawn>;

let dirs: ReturnType<typeof createTmpDirs>;
let app: ReturnType<typeof buildApp>;

beforeEach(() => {
  jest.clearAllMocks();
  dirs = createTmpDirs();
  app = buildApp(uploadRouter, '/api/upload');
});

afterEach(() => cleanupTmpDirs(dirs.base));

describe('GET /api/upload/files', () => {
  it('returns empty list when no files exist', async () => {
    const res = await request(app).get('/api/upload/files');
    expect(res.status).toBe(200);
    expect(res.body.files).toEqual([]);
  });

  it('lists only sales_*.csv files, sorted', async () => {
    fs.writeFileSync(path.join(dirs.monthly, 'sales_2026_02_february.csv'), 'b');
    fs.writeFileSync(path.join(dirs.monthly, 'sales_2026_01_january.csv'), 'a');
    fs.writeFileSync(path.join(dirs.monthly, 'random.txt'), 'c');

    const res = await request(app).get('/api/upload/files');
    expect(res.status).toBe(200);
    expect(res.body.files).toHaveLength(2);
    expect(res.body.files[0].name).toBe('sales_2026_01_january.csv');
    expect(res.body.files[1].name).toBe('sales_2026_02_february.csv');
    expect(res.body.files[0]).toHaveProperty('size');
    expect(res.body.files[0]).toHaveProperty('modified');
  });
});

describe('POST /api/upload', () => {
  it('uploads a valid sales CSV and validates it', async () => {
    const csvContent = 'name,price\napple,1.5';
    const res = await request(app)
      .post('/api/upload')
      .attach('files', Buffer.from(csvContent), 'sales_2026_01_january.csv');

    expect(res.status).toBe(200);
    expect(res.body.uploaded).toHaveLength(1);
    expect(res.body.uploaded[0].name).toBe('sales_2026_01_january.csv');
    expect(res.body.uploaded[0].valid).toBe(true);
    expect(fs.existsSync(path.join(dirs.monthly, 'sales_2026_01_january.csv'))).toBe(true);
  });

  it('rejects files not matching sales_*.csv pattern', async () => {
    const res = await request(app)
      .post('/api/upload')
      .attach('files', Buffer.from('data'), 'bad_name.csv');

    expect(res.status).toBe(400);
    expect(res.body.error).toContain('sales_');
  });

  it('uploads multiple files at once', async () => {
    const res = await request(app)
      .post('/api/upload')
      .attach('files', Buffer.from('a'), 'sales_2026_01_january.csv')
      .attach('files', Buffer.from('b'), 'sales_2026_02_february.csv');

    expect(res.status).toBe(200);
    expect(res.body.uploaded).toHaveLength(2);
  });
});

describe('POST /api/upload/validate', () => {
  it('validates all CSVs in the monthly directory', async () => {
    fs.writeFileSync(path.join(dirs.monthly, 'sales_2026_01_january.csv'), 'name,price\na,1');
    const res = await request(app).post('/api/upload/validate');
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('valid');
    expect(res.body).toHaveProperty('report');
  });

  it('uses bash when a venv activate script exists', async () => {
    fs.mkdirSync(path.join(dirs.base, 'venv', 'bin'), { recursive: true });
    fs.writeFileSync(path.join(dirs.base, 'venv', 'bin', 'activate'), 'echo activated');
    fs.writeFileSync(path.join(dirs.monthly, 'sales_2026_01_january.csv'), 'name,price\na,1');

    const res = await request(app).post('/api/upload/validate');
    expect(res.status).toBe(200);
    expect(mockSpawn).toHaveBeenCalledWith(
      '/bin/bash',
      expect.arrayContaining(['-c', expect.stringContaining('python -m utils.csv_schema')]),
      expect.objectContaining({ cwd: dirs.base }),
    );
  });
});

describe('DELETE /api/upload/files/:name', () => {
  it('deletes an existing file', async () => {
    const fpath = path.join(dirs.monthly, 'sales_2026_01_january.csv');
    fs.writeFileSync(fpath, 'data');

    const res = await request(app).delete('/api/upload/files/sales_2026_01_january.csv');
    expect(res.status).toBe(200);
    expect(res.body.deleted).toBe('sales_2026_01_january.csv');
    expect(fs.existsSync(fpath)).toBe(false);
  });

  it('returns 404 for non-existent file', async () => {
    const res = await request(app).delete('/api/upload/files/sales_nope.csv');
    expect(res.status).toBe(404);
  });

  it('rejects path traversal with ../', async () => {
    const res = await request(app).delete('/api/upload/files/..%2Fetc%2Fpasswd');
    expect(res.status).toBe(400);
  });

  it('rejects names that do not match the sales_*.csv pattern', async () => {
    const badNames = ['not_a_sales.csv', 'sales_.csv..', '../secret'];
    for (const name of badNames) {
      const res = await request(app).delete(`/api/upload/files/${encodeURIComponent(name)}`);
      expect(res.status).toBe(400);
    }
  });
});

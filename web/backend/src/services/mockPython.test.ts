import { EventEmitter } from 'events';
import os from 'os';
import fs from 'fs';
import path from 'path';

function makeFakeRes() {
  const writes: string[] = [];
  const res: any = new EventEmitter();
  res.setHeader = jest.fn();
  res.flushHeaders = jest.fn();
  res.write = jest.fn((chunk: string) => {
    writes.push(chunk);
    return true;
  });
  res.end = jest.fn();
  res._writes = writes;
  return res;
}

function parseSSE(writes: string[]): { type: string; data: any }[] {
  return writes.map((w) => {
    const m = w.match(/^event: (.+)\ndata: (.+)\n\n$/);
    if (!m) throw new Error(`Malformed SSE: ${w}`);
    return { type: m[1], data: JSON.parse(m[2]) };
  });
}

async function flush() {
  await new Promise((r) => setImmediate(r));
  await new Promise((r) => setImmediate(r));
}

describe('mockPython service', () => {
  let tmpResultsDir: string;

  beforeEach(() => {
    jest.resetModules();
    tmpResultsDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mock-python-results-'));
    process.env.RESULTS_DIR = tmpResultsDir;
  });

  afterEach(() => {
    fs.rmSync(tmpResultsDir, { recursive: true, force: true });
  });

  it('runs the training fixture path', async () => {
    const { mockRunPythonWithSSE } = await import('./mockPython');
    const res = makeFakeRes();

    await mockRunPythonWithSSE(res, 'key', '02_train.py', []);
    await flush();

    const events = parseSSE(res._writes);
    expect(events.map((e) => e.type)).toContain('done');
    expect(fs.existsSync(path.join(tmpResultsDir, 'training_results.json'))).toBe(true);
  });

  it('runs the evaluation fixture path', async () => {
    const { mockRunPythonWithSSE } = await import('./mockPython');
    const res = makeFakeRes();

    await mockRunPythonWithSSE(res, 'key', '03_evaluate.py', []);
    await flush();

    for (const chart of [
      'mse_comparison.png',
      'profit_comparison.png',
      'rack_comparison.png',
      'alluvial_shelf_movement.png',
    ]) {
      expect(fs.existsSync(path.join(tmpResultsDir, chart))).toBe(true);
    }
  });

  it('runs the predict fixture path with explicit month', async () => {
    const { mockRunPythonWithSSE } = await import('./mockPython');
    const res = makeFakeRes();

    await mockRunPythonWithSSE(res, 'key', '05_predict.py', ['--month', '2026-03']);
    await flush();

    expect(fs.existsSync(path.join(tmpResultsDir, 'optimized_2026_03_march.csv'))).toBe(true);
    expect(fs.existsSync(path.join(tmpResultsDir, 'forecast_2026_03.json'))).toBe(true);
  });

  it('uses default month when none is provided', async () => {
    const { mockRunPythonWithSSE } = await import('./mockPython');
    const res = makeFakeRes();

    await mockRunPythonWithSSE(res, 'key', '05_predict.py', []);
    await flush();

    expect(fs.existsSync(path.join(tmpResultsDir, 'optimized_2025_12_december.csv'))).toBe(true);
    expect(fs.existsSync(path.join(tmpResultsDir, 'forecast_2025_12.json'))).toBe(true);
  });

  it('falls back to generic completion for other scripts', async () => {
    const { mockRunPythonWithSSE } = await import('./mockPython');
    const res = makeFakeRes();

    await mockRunPythonWithSSE(res, 'key', '01_unknown.py', []);
    await flush();

    const events = parseSSE(res._writes);
    expect(events.at(-1)).toEqual({
      type: 'done',
      data: { message: 'Process completed successfully.' },
    });
  });

  it('emits step events for chain predict runs', async () => {
    const { mockRunPythonChainWithSSE } = await import('./mockPython');
    const res = makeFakeRes();

    await mockRunPythonChainWithSSE(res, 'key', [
      { name: 'predict', script: '05_predict.py', args: ['--month', '2026-04'] },
    ]);
    await flush();

    const events = parseSSE(res._writes);
    expect(events[0]).toMatchObject({ type: 'step' });
    expect(events.at(-1)).toEqual({ type: 'done', data: { ok: true } });
    expect(fs.existsSync(path.join(tmpResultsDir, 'optimized_2026_04_april.csv'))).toBe(true);
  });
});
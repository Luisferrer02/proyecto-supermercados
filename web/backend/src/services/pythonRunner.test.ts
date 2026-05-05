import { EventEmitter } from 'events';

// ---------------------------------------------------------------------------
// Helpers — defined before jest.mock so the factory closure can reference them
// ---------------------------------------------------------------------------
function makeFakeProc() {
  const proc: any = new EventEmitter();
  proc.stdout = new EventEmitter();
  proc.stderr = new EventEmitter();
  proc.exitCode = null;
  proc.killed = false;
  proc.kill = jest.fn(() => {
    proc.killed = true;
    proc.exitCode = -1;
    process.nextTick(() => proc.emit('close', -1));
    return true;
  });
  return proc;
}

// ---------------------------------------------------------------------------
// Factory mocks
// ---------------------------------------------------------------------------
jest.mock('fs', () => ({
  ...jest.requireActual('fs'),
  existsSync: jest.fn(() => false),
}));

jest.mock('child_process', () => ({
  spawn: jest.fn(() => makeFakeProc()),
}));

import * as fs from 'fs';
import { spawn } from 'child_process';
import {
  isRunning,
  killProcess,
  runPythonWithSSE,
  runPythonChainWithSSE,
  resolveVenvBin,
} from './pythonRunner';

const mockExistsSync = fs.existsSync as jest.MockedFunction<typeof fs.existsSync>;
const mockSpawn = spawn as jest.MockedFunction<typeof spawn>;

// ---------------------------------------------------------------------------
// Fake Express Response
// ---------------------------------------------------------------------------
function makeFakeRes() {
  const writes: string[] = [];
  const res: any = new EventEmitter();
  res.setHeader = jest.fn();
  res.flushHeaders = jest.fn();
  res.write = jest.fn((chunk: string) => { writes.push(chunk); return true; });
  res.end = jest.fn();
  res._writes = writes;
  return res;
}

function parseSSE(writes: string[]): { type: string; data: any }[] {
  return writes.map((w) => {
    const m = w.match(/^event: (.+)\ndata: (.+)\n\n$/);
    if (!m) throw new Error('Malformed SSE: ' + JSON.stringify(w));
    return { type: m[1], data: JSON.parse(m[2]) };
  });
}

async function flush(n = 5) {
  for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r));
}

// Use a counter to guarantee unique process keys across tests
let keySeq = 0;
function uniqueKey(prefix: string) { return `${prefix}_${++keySeq}`; }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
beforeEach(() => {
  jest.clearAllMocks();
  mockExistsSync.mockReturnValue(false);
});

describe('resolveVenvBin', () => {
  it('returns empty string when no venv exists', () => {
    expect(resolveVenvBin()).toBe('');
  });

  it('returns the first existing venv bin path', () => {
    mockExistsSync.mockImplementation((p: any) =>
      String(p).includes('venv-new'),
    );
    expect(resolveVenvBin()).toContain('venv-new');
  });
});

describe('isRunning / killProcess', () => {
  it('returns false for unknown keys', () => {
    expect(isRunning('unknown')).toBe(false);
  });

  it('killProcess on unknown key does nothing', () => {
    expect(() => killProcess('unknown')).not.toThrow();
  });
});

describe('runPythonWithSSE', () => {
  it('forwards stdout as log events and emits done on exit 0', async () => {
    const proc = makeFakeProc();
    mockSpawn.mockReturnValueOnce(proc as any);
    const key = uniqueKey('sse');
    const res = makeFakeRes();

    runPythonWithSSE(res, key, 'foo.py', []);
    expect(isRunning(key)).toBe(true);

    proc.stdout.emit('data', Buffer.from('hello\nworld\n'));
    proc.exitCode = 0;
    proc.emit('close', 0);
    await flush();

    const events = parseSSE(res._writes);
    expect(events).toEqual([
      { type: 'log', data: { message: 'hello' } },
      { type: 'log', data: { message: 'world' } },
      { type: 'done', data: { message: 'Process completed successfully.' } },
    ]);
    expect(res.end).toHaveBeenCalled();
    expect(isRunning(key)).toBe(false);
  });

  it('reports non-zero exit code', async () => {
    const proc = makeFakeProc();
    mockSpawn.mockReturnValueOnce(proc as any);
    const key = uniqueKey('sse');
    const res = makeFakeRes();

    runPythonWithSSE(res, key, 'bar.py', []);
    proc.exitCode = 2;
    proc.emit('close', 2);
    await flush();

    const last = parseSSE(res._writes).at(-1);
    expect(last).toEqual({
      type: 'done',
      data: { message: 'Process exited with code 2.' },
    });
  });

  it('rejects duplicate runs on the same key', async () => {
    const proc = makeFakeProc();
    mockSpawn.mockReturnValueOnce(proc as any);
    const key = uniqueKey('sse');

    const res1 = makeFakeRes();
    runPythonWithSSE(res1, key, 'a.py', []);

    const res2 = makeFakeRes();
    runPythonWithSSE(res2, key, 'a.py', []);

    const events = parseSSE(res2._writes);
    expect(events[0].type).toBe('error');
    expect(events[0].data.message).toContain('already running');

    // cleanup
    proc.exitCode = 0;
    proc.emit('close', 0);
    await flush();
  });

  it('kills child process when client disconnects', () => {
    const proc = makeFakeProc();
    mockSpawn.mockReturnValueOnce(proc as any);
    const key = uniqueKey('sse');
    const res = makeFakeRes();

    runPythonWithSSE(res, key, 'x.py', []);
    res.emit('close');
    expect(proc.kill).toHaveBeenCalled();
  });

  it('emits error event when spawn fails', async () => {
    const proc = makeFakeProc();
    mockSpawn.mockReturnValueOnce(proc as any);
    const key = uniqueKey('sse');
    const res = makeFakeRes();

    runPythonWithSSE(res, key, 'fail.py', []);
    proc.emit('error', new Error('ENOENT'));
    await flush();

    const last = parseSSE(res._writes).at(-1);
    expect(last).toEqual({
      type: 'error',
      data: { message: 'Failed to start process: ENOENT' },
    });
    expect(res.end).toHaveBeenCalled();
  });

  it('uses venv python when a venv bin exists', async () => {
    const proc = makeFakeProc();
    mockExistsSync.mockImplementation((p: any) => {
      const value = String(p);
      return value.includes('venv') && value.includes('python');
    });
    mockSpawn.mockReturnValueOnce(proc as any);
    const key = uniqueKey('sse');
    const res = makeFakeRes();

    runPythonWithSSE(res, key, 'prog.py', ['--flag']);
    expect(mockSpawn).toHaveBeenCalledWith(
      expect.stringContaining('venv'),
      expect.arrayContaining(['-u', expect.stringContaining('prog.py'), '--flag']),
      expect.objectContaining({ cwd: expect.any(String) }),
    );

    proc.exitCode = 0;
    proc.emit('close', 0);
    await flush();
  });

  it('forwards stderr as log events', async () => {
    const proc = makeFakeProc();
    mockSpawn.mockReturnValueOnce(proc as any);
    const key = uniqueKey('sse');
    const res = makeFakeRes();

    runPythonWithSSE(res, key, 'prog.py', []);
    proc.stderr.emit('data', Buffer.from('warning\n'));
    proc.exitCode = 0;
    proc.emit('close', 0);
    await flush();

    expect(parseSSE(res._writes)[0]).toEqual({
      type: 'log',
      data: { message: 'warning' },
    });
  });
});

describe('runPythonChainWithSSE', () => {
  it('runs two steps sequentially and emits step + done', async () => {
    const proc1 = makeFakeProc();
    const proc2 = makeFakeProc();
    mockSpawn
      .mockReturnValueOnce(proc1 as any)
      .mockReturnValueOnce(proc2 as any);

    const key = uniqueKey('chain');
    const res = makeFakeRes();
    const promise = runPythonChainWithSSE(res, key, [
      { name: 'ingest', script: 'ingest.py', args: [] },
      { name: 'predict', script: 'predict.py', args: [] },
    ]);

    await flush(2);
    proc1.stdout.emit('data', Buffer.from('log1\n'));
    proc1.exitCode = 0;
    proc1.emit('close', 0);
    await flush(3);

    proc2.exitCode = 0;
    proc2.emit('close', 0);
    await promise;

    const events = parseSSE(res._writes);
    expect(events.map((e) => e.type)).toEqual(['step', 'log', 'step', 'done']);
    expect(events[0].data).toEqual({ name: 'ingest', index: 1, total: 2 });
    expect(events[3].data).toEqual({ ok: true });
  }, 10000);

  it('stops chain on first failing step', async () => {
    const proc1 = makeFakeProc();
    mockSpawn.mockReturnValueOnce(proc1 as any);

    const key = uniqueKey('chain');
    const res = makeFakeRes();
    const promise = runPythonChainWithSSE(res, key, [
      { name: 'ingest', script: 'ingest.py', args: [] },
      { name: 'predict', script: 'predict.py', args: [] },
    ]);

    await flush(2);
    proc1.exitCode = 3;
    proc1.emit('close', 3);
    await promise;

    expect(mockSpawn).toHaveBeenCalledTimes(1);
    expect(parseSSE(res._writes).at(-1)).toEqual({
      type: 'done',
      data: { ok: false, failedStep: 'ingest', code: 3 },
    });
  }, 10000);

  it('uses venv python for chain steps when a venv bin exists', async () => {
    const proc = makeFakeProc();
    mockExistsSync.mockImplementation((p: any) => {
      const value = String(p);
      return value.includes('venv') && value.includes('python');
    });
    mockSpawn.mockReturnValueOnce(proc as any);

    const key = uniqueKey('chain');
    const res = makeFakeRes();
    const promise = runPythonChainWithSSE(res, key, [
      { name: 'ingest', script: 'ingest.py', args: ['--month', '2026-03'] },
    ]);

    await flush(2);
    expect(mockSpawn).toHaveBeenCalledWith(
      expect.stringContaining('venv'),
      expect.arrayContaining(['-u', expect.stringContaining('ingest.py'), '--month', '2026-03']),
      expect.objectContaining({ cwd: expect.any(String) }),
    );

    proc.exitCode = 0;
    proc.emit('close', 0);
    await promise;
  }, 10000);

  it('rejects duplicate chain with same key', async () => {
    const proc1 = makeFakeProc();
    mockSpawn.mockReturnValueOnce(proc1 as any);

    const key = uniqueKey('chain');
    const res1 = makeFakeRes();
    const p1 = runPythonChainWithSSE(res1, key, [
      { name: 's', script: 's.py', args: [] },
    ]);
    await flush(2);

    const res2 = makeFakeRes();
    await runPythonChainWithSSE(res2, key, [
      { name: 's', script: 's.py', args: [] },
    ]);

    expect(parseSSE(res2._writes)[0].type).toBe('error');

    proc1.exitCode = 0;
    proc1.emit('close', 0);
    await p1;
  }, 10000);
});

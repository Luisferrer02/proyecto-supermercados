import { spawn, ChildProcess } from 'child_process';
import { Response } from 'express';
import path from 'path';
import fs from 'fs';
import { mockRunPythonWithSSE, mockRunPythonChainWithSSE } from './mockPython';

const MLOPS_DIR = process.env.MLOPS_DIR!;
const PYTHON = process.env.PYTHON_PATH || 'python3';

/**
 * Validate that an argument is safe for shell execution.
 * Rejects any string containing shell metacharacters.
 */
export function sanitizeArg(arg: string): string {
  if (/[`$\\!;&|<>(){}\n\r]/.test(arg)) {
    throw new Error(`Unsafe argument rejected: ${arg}`);
  }
  return arg;
}

// Pick the first venv bin directory that actually exists. Override via VENV_DIR.
export function resolveVenvBin(): string {
  const candidates = [
    process.env.VENV_DIR ? path.join(process.env.VENV_DIR, 'bin') : '',
    path.join(MLOPS_DIR, 'venv-new', 'bin'),
    path.join(MLOPS_DIR, 'venv', 'bin'),
  ].filter(Boolean);
  return candidates.find(p => fs.existsSync(path.join(p, 'python'))) ?? '';
}

/** Build env with venv's bin prepended to PATH (no shell needed). */
function buildEnv(): Record<string, string> {
  const env = { ...process.env, PYTHONUNBUFFERED: '1' } as Record<string, string>;
  const venvBin = resolveVenvBin();
  if (venvBin) {
    env.VIRTUAL_ENV = path.dirname(venvBin);
    env.PATH = `${venvBin}:${env.PATH || ''}`;
  }
  return env;
}

/** Resolve the python binary path. */
function resolvePython(): string {
  const venvBin = resolveVenvBin();
  if (venvBin) return path.join(venvBin, 'python');
  return PYTHON;
}

// Track running processes to prevent duplicate runs
const running: Record<string, ChildProcess> = {};

export function isRunning(key: string): boolean {
  const proc = running[key];
  return !!(proc && proc.exitCode === null);
}

export function killProcess(key: string): void {
  const proc = running[key];
  if (proc && proc.exitCode === null) {
    proc.kill();
    delete running[key];
  }
}

/**
 * Run a sequence of Python scripts over a single SSE connection.
 * Each step inherits the same process key (so only one chain runs at a
 * time). If any step exits non-zero, the chain stops and the client
 * receives a `done` event with the failed step.
 *
 * Steps are emitted as their own SSE events so the frontend can render
 * a stepper:  event: step   data: { "name": "ingest", "index": 1, "total": 2 }
 */
export async function runPythonChainWithSSE(
  res: Response,
  key: string,
  steps: { name: string; script: string; args: string[] }[],
): Promise<void> {
  if (process.env.MOCK_PYTHON === '1') {
    return mockRunPythonChainWithSSE(res, key, steps);
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  const sendEvent = (type: string, payload: object) => {
    res.write(`event: ${type}\ndata: ${JSON.stringify(payload)}\n\n`);
  };

  if (isRunning(key)) {
    sendEvent('error', { message: `Process '${key}' is already running.` });
    res.end();
    return;
  }

  let clientAlive = true;
  res.on('close', () => { clientAlive = false; });

  for (let i = 0; i < steps.length; i++) {
    if (!clientAlive) break;
    const step = steps[i];
    sendEvent('step', { name: step.name, index: i + 1, total: steps.length });

    const exitCode = await new Promise<number>((resolve) => {
      const scriptPath = path.join(MLOPS_DIR, step.script);
      const safeArgs = step.args.map(sanitizeArg);
      const pythonBin = resolvePython();
      const proc: ChildProcess = spawn(pythonBin, ['-u', scriptPath, ...safeArgs], {
        cwd: MLOPS_DIR,
        env: buildEnv(),
      });

      running[key] = proc;

      const forwardLog = (chunk: Buffer) => {
        chunk.toString().split('\n').filter(Boolean).forEach((line) => {
          sendEvent('log', { step: step.name, message: line });
        });
      };
      proc.stdout?.on('data', forwardLog);
      proc.stderr?.on('data', forwardLog);
      proc.on('close', (code) => {
        delete running[key];
        resolve(code ?? 1);
      });
      proc.on('error', (err) => {
        delete running[key];
        sendEvent('error', { step: step.name, message: String(err) });
        resolve(1);
      });
      res.on('close', () => {
        if (proc.exitCode === null) proc.kill();
      });
    });

    if (exitCode !== 0) {
      sendEvent('done', { ok: false, failedStep: step.name, code: exitCode });
      res.end();
      return;
    }
  }

  sendEvent('done', { ok: true });
  res.end();
}


export function runPythonWithSSE(
  res: Response,
  key: string,
  scriptName: string,
  args: string[]
): void {
  if (process.env.MOCK_PYTHON === '1') {
    mockRunPythonWithSSE(res, key, scriptName, args);
    return;
  }

  // SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  const sendEvent = (type: string, data: string) => {
    res.write(`event: ${type}\ndata: ${JSON.stringify({ message: data })}\n\n`);
  };

  if (isRunning(key)) {
    sendEvent('error', `Process '${key}' is already running. Wait for it to finish.`);
    res.end();
    return;
  }

  const scriptPath = path.join(MLOPS_DIR, scriptName);
  const safeArgs = args.map(sanitizeArg);
  const pythonBin = resolvePython();
  const proc: ChildProcess = spawn(pythonBin, ['-u', scriptPath, ...safeArgs], {
    cwd: MLOPS_DIR,
    env: buildEnv(),
  });

  running[key] = proc;

  proc.stdout?.on('data', (chunk: Buffer) => {
    chunk.toString().split('\n').filter(Boolean).forEach((line: string) => {
      sendEvent('log', line);
    });
  });

  proc.stderr?.on('data', (chunk: Buffer) => {
    chunk.toString().split('\n').filter(Boolean).forEach((line: string) => {
      // Many Python libraries write progress to stderr — treat as log unless it's a real error
      sendEvent('log', line);
    });
  });

  proc.on('close', (code) => {
    delete running[key];
    if (code === 0) {
      sendEvent('done', 'Process completed successfully.');
    } else {
      sendEvent('done', `Process exited with code ${code}.`);
    }
    res.end();
  });

  proc.on('error', (err) => {
    delete running[key];
    sendEvent('error', `Failed to start process: ${err.message}`);
    res.end();
  });

  // Kill process if client disconnects early
  res.on('close', () => {
    if (proc.exitCode === null) proc.kill();
    delete running[key];
  });
}

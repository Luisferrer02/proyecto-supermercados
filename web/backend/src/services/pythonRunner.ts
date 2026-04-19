import { spawn, ChildProcess } from 'child_process';
import { Response } from 'express';
import path from 'path';
import fs from 'fs';

const MLOPS_DIR = process.env.MLOPS_DIR!;
const PYTHON = process.env.PYTHON_PATH || 'python3';

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
      const venvActivate = path.join(MLOPS_DIR, 'venv', 'bin', 'activate');
      let proc: ChildProcess;
      if (fs.existsSync(venvActivate)) {
        const quotedArgs = [scriptPath, ...step.args]
          .map(a => `"${a.replace(/"/g, '\\"')}"`).join(' ');
        proc = spawn('/bin/bash',
          ['-c', `source "${venvActivate}" && python -u ${quotedArgs}`],
          { cwd: MLOPS_DIR, env: { ...process.env, PYTHONUNBUFFERED: '1' } });
      } else {
        proc = spawn(PYTHON, ['-u', scriptPath, ...step.args], {
          cwd: MLOPS_DIR,
          env: { ...process.env, PYTHONUNBUFFERED: '1' },
          shell: process.platform === 'win32',
        });
      }

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
  const venvActivate = path.join(MLOPS_DIR, 'venv', 'bin', 'activate');
  // When a POSIX-style venv exists (Mac/Linux dev setup) we use bash so the
  // child runs inside it. On Windows (no bash, no venv/bin/activate) fall
  // back to spawning python directly with the configured PYTHON_PATH.
  let proc: ChildProcess;
  if (fs.existsSync(venvActivate)) {
    const quotedArgs = [scriptPath, ...args].map(a => `"${a.replace(/"/g, '\\"')}"`).join(' ');
    const shellCmd = `source "${venvActivate}" && python -u ${quotedArgs}`;
    proc = spawn('/bin/bash', ['-c', shellCmd], {
      cwd: MLOPS_DIR,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });
  } else {
    proc = spawn(PYTHON, ['-u', scriptPath, ...args], {
      cwd: MLOPS_DIR,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      shell: process.platform === 'win32',
    });
  }

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

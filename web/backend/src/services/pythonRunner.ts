import { spawn, ChildProcess } from 'child_process';
import { Response } from 'express';
import path from 'path';

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
  const quotedArgs = [scriptPath, ...args].map(a => `"${a.replace(/"/g, '\\"')}"`).join(' ');
  const shellCmd = `source "${venvActivate}" && python -u ${quotedArgs}`;
  const proc: ChildProcess = spawn('/bin/bash', ['-c', shellCmd], {
    cwd: MLOPS_DIR,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
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

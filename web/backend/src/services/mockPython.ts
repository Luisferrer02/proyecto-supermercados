import { Response } from 'express';
import fs from 'fs';
import path from 'path';

const RESULTS_DIR = process.env.RESULTS_DIR || path.join(process.env.MLOPS_DIR || '.', 'results');
const FIXTURES_DIR = path.join(__dirname, 'fixtures');

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function copyFixture(name: string, destDir: string): void {
  const src = path.join(FIXTURES_DIR, name);
  if (fs.existsSync(src)) {
    fs.mkdirSync(destDir, { recursive: true });
    fs.copyFileSync(src, path.join(destDir, name));
  }
}

function resolveMonth(args: string[]): { year: string; month: string; monthName: string } {
  const monthArg = args.find((a) => /^\d{4}-\d{2}$/.test(a));
  if (monthArg) {
    const [year, month] = monthArg.split('-');
    const names = ['january','february','march','april','may','june','july','august','september','october','november','december'];
    return { year, month, monthName: names[parseInt(month) - 1] };
  }
  return { year: '2025', month: '12', monthName: 'december' };
}

export async function mockRunPythonWithSSE(
  res: Response,
  key: string,
  scriptName: string,
  args: string[],
): Promise<void> {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  const send = (type: string, data: string) => {
    res.write(`event: ${type}\ndata: ${JSON.stringify({ message: data })}\n\n`);
  };

  send('log', `[MOCK] Running ${scriptName}...`);
  await sleep(30);
  send('log', `[MOCK] Processing data...`);
  await sleep(30);

  // Copy relevant fixtures based on which script is running
  if (scriptName.includes('02_train')) {
    copyFixture('training_results.json', RESULTS_DIR);
    send('log', '[MOCK] Training complete.');
  } else if (scriptName.includes('03_evaluate')) {
    for (const chart of ['mse_comparison.png', 'profit_comparison.png', 'rack_comparison.png', 'alluvial_shelf_movement.png']) {
      copyFixture(chart, RESULTS_DIR);
    }
    send('log', '[MOCK] Charts generated.');
  } else if (scriptName.includes('05_predict')) {
    const { year, month, monthName } = resolveMonth(args);
    copyFixture('optimized_sample.csv', RESULTS_DIR);
    const dest = path.join(RESULTS_DIR, `optimized_${year}_${month}_${monthName}.csv`);
    const src = path.join(FIXTURES_DIR, 'optimized_sample.csv');
    if (fs.existsSync(src)) fs.copyFileSync(src, dest);

    copyFixture('forecast_sample.json', RESULTS_DIR);
    const fDest = path.join(RESULTS_DIR, `forecast_${year}_${month}.json`);
    const fSrc = path.join(FIXTURES_DIR, 'forecast_sample.json');
    if (fs.existsSync(fSrc)) fs.copyFileSync(fSrc, fDest);

    send('log', `[MOCK] Prediction for ${year}-${month} done.`);
  } else {
    send('log', '[MOCK] Done.');
  }

  await sleep(20);
  send('done', 'Process completed successfully.');
  res.end();
}

export async function mockRunPythonChainWithSSE(
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

  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    sendEvent('step', { name: step.name, index: i + 1, total: steps.length });
    await sleep(20);
    sendEvent('log', { step: step.name, message: `[MOCK] Running ${step.script}...` });
    await sleep(30);
    sendEvent('log', { step: step.name, message: `[MOCK] ${step.name} complete.` });

    // Copy fixtures for the predict step
    if (step.script.includes('05_predict')) {
      const { year, month, monthName } = resolveMonth(step.args);
      const src = path.join(FIXTURES_DIR, 'optimized_sample.csv');
      if (fs.existsSync(src)) {
        fs.mkdirSync(RESULTS_DIR, { recursive: true });
        fs.copyFileSync(src, path.join(RESULTS_DIR, `optimized_${year}_${month}_${monthName}.csv`));
      }
      const fSrc = path.join(FIXTURES_DIR, 'forecast_sample.json');
      if (fs.existsSync(fSrc)) {
        fs.copyFileSync(fSrc, path.join(RESULTS_DIR, `forecast_${year}_${month}.json`));
      }
    }
  }

  sendEvent('done', { ok: true });
  res.end();
}

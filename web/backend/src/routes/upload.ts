import { Router, Request, Response } from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';

const router = Router();

const PYTHON = process.env.PYTHON_PATH || 'python3';

// Runs utils/csv_schema.py on the given path and returns a structured report.
// Uses the venv python if available; works on mac/linux (bash) and gracefully
// degrades on windows where the activate script does not exist.
function findVenvActivate(mlopsDir: string): string {
  const candidates = [
    process.env.VENV_DIR ? path.join(process.env.VENV_DIR, 'bin', 'activate') : '',
    path.join(mlopsDir, 'venv-new', 'bin', 'activate'),
    path.join(mlopsDir, 'venv', 'bin', 'activate'),
  ].filter(Boolean);
  return candidates.find(p => fs.existsSync(p)) ?? '';
}

function runValidator(target: string): Promise<{ ok: boolean; output: string }> {
  return new Promise((resolve) => {
    const mlopsDir = process.env.MLOPS_DIR!;
    const venvActivate = findVenvActivate(mlopsDir);
    const useBash = !!venvActivate;

    let proc;
    if (useBash) {
      const cmd = `source "${venvActivate}" && python -m utils.csv_schema "${target}"`;
      proc = spawn('/bin/bash', ['-c', cmd], { cwd: mlopsDir });
    } else {
      proc = spawn(PYTHON, ['-m', 'utils.csv_schema', target], { cwd: mlopsDir });
    }

    let output = '';
    proc.stdout?.on('data', (c) => { output += c.toString(); });
    proc.stderr?.on('data', (c) => { output += c.toString(); });
    proc.on('close', (code) => resolve({ ok: code === 0, output }));
    proc.on('error', (err) => resolve({ ok: false, output: String(err) }));
  });
}

const monthlyDir = (): string => {
  const dir = path.join(process.env.MLOPS_DIR!, 'data', 'monthly');
  fs.mkdirSync(dir, { recursive: true });
  return dir;
};

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, monthlyDir()),
  filename: (_req, file, cb) => cb(null, file.originalname),
});

const upload = multer({
  storage,
  fileFilter: (_req, file, cb) => {
    if (file.originalname.match(/^sales_.*\.csv$/)) {
      cb(null, true);
    } else {
      cb(new Error('Only files named sales_*.csv are accepted'));
    }
  },
});

// POST /api/upload — upload one or more CSV files
router.post('/', (req: Request, res: Response) => {
  upload.array('files')(req, res, async (err) => {
    if (err) {
      res.status(400).json({ error: err.message });
      return;
    }
    const files = (req.files as Express.Multer.File[]) || [];

    // Validate each uploaded file's schema. Invalid files stay on disk so
    // the user can inspect them, but they are flagged in the response.
    const uploaded = await Promise.all(files.map(async (f) => {
      const { ok, output } = await runValidator(path.join(monthlyDir(), f.filename));
      return { name: f.filename, size: f.size, valid: ok, report: output };
    }));

    res.json({ uploaded });
  });
});

// POST /api/upload/validate — validate all CSVs currently in data/monthly/
router.post('/validate', async (_req: Request, res: Response) => {
  const { ok, output } = await runValidator(monthlyDir());
  res.json({ valid: ok, report: output });
});

// GET /api/upload/files — list files in data/monthly/
router.get('/files', (_req: Request, res: Response) => {
  const dir = monthlyDir();
  const files = fs.readdirSync(dir)
    .filter(f => f.match(/^sales_.*\.csv$/))
    .sort()
    .map(f => {
      const stat = fs.statSync(path.join(dir, f));
      return { name: f, size: stat.size, modified: stat.mtime };
    });
  res.json({ files });
});

// DELETE /api/upload/files/:name — delete a specific file
router.delete('/files/:name', (req: Request, res: Response) => {
  const name = req.params.name;
  // Reject any path separators / traversal segments / NUL bytes.
  // Filenames are validated again by matching the same pattern multer enforces.
  if (!/^sales_[A-Za-z0-9._-]+\.csv$/.test(name)) {
    res.status(400).json({ error: 'Invalid filename' });
    return;
  }
  const dir = monthlyDir();
  const filePath = path.resolve(dir, name);
  // Ensure resolved path is still inside dir (defense in depth).
  if (!filePath.startsWith(path.resolve(dir) + path.sep)) {
    res.status(400).json({ error: 'Invalid filename' });
    return;
  }
  if (!fs.existsSync(filePath)) {
    res.status(404).json({ error: 'File not found' });
    return;
  }
  fs.unlinkSync(filePath);
  res.json({ deleted: name });
});

export { router as uploadRouter };

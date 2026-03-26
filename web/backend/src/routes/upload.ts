import { Router, Request, Response } from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';

const router = Router();

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
router.post('/', upload.array('files'), (req: Request, res: Response) => {
  const files = req.files as Express.Multer.File[];
  res.json({ uploaded: files.map(f => ({ name: f.filename, size: f.size })) });
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
  const filePath = path.join(monthlyDir(), req.params.name);
  if (!fs.existsSync(filePath)) {
    res.status(404).json({ error: 'File not found' });
    return;
  }
  fs.unlinkSync(filePath);
  res.json({ deleted: req.params.name });
});

export { router as uploadRouter };

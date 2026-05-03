// Provide deterministic env vars for tests. Each test that needs to override
// a path can do so via process.env or a per-test temp directory before
// requiring the module under test.
import os from 'os';
import path from 'path';
import fs from 'fs';

if (!process.env.MLOPS_DIR) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'mlops-test-'));
  process.env.MLOPS_DIR = tmp;
}
if (!process.env.RESULTS_DIR) {
  process.env.RESULTS_DIR = path.join(process.env.MLOPS_DIR!, 'results');
}
process.env.PORT = process.env.PORT || '0';
process.env.NODE_ENV = 'test';

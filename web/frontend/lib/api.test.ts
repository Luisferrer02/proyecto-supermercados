import { api, BASE } from './api';

// Mock global fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

beforeEach(() => {
  mockFetch.mockReset();
  mockFetch.mockResolvedValue({ json: () => Promise.resolve({ ok: true }) });
});

describe('api', () => {
  // --- Upload ---
  it('uploadFiles sends POST with FormData', async () => {
    const fd = new FormData();
    await api.uploadFiles(fd);
    expect(mockFetch).toHaveBeenCalledWith(`${BASE}/api/upload`, {
      method: 'POST',
      body: fd,
    });
  });

  it('listFiles sends GET to /upload/files', async () => {
    await api.listFiles();
    expect(mockFetch).toHaveBeenCalledWith(`${BASE}/api/upload/files`);
  });

  it('deleteFile encodes filename and sends DELETE', async () => {
    await api.deleteFile('sales_2026_01_january.csv');
    expect(mockFetch).toHaveBeenCalledWith(
      `${BASE}/api/upload/files/sales_2026_01_january.csv`,
      { method: 'DELETE' },
    );
  });

  // --- Train ---
  it('trainStatus sends GET to /train/status', async () => {
    await api.trainStatus();
    expect(mockFetch).toHaveBeenCalledWith(`${BASE}/api/train/status`);
  });

  it('trainResults sends GET to /train/results', async () => {
    await api.trainResults();
    expect(mockFetch).toHaveBeenCalledWith(`${BASE}/api/train/results`);
  });

  it('trainStreamUrl returns the correct URL', () => {
    expect(api.trainStreamUrl()).toBe(`${BASE}/api/train/stream`);
  });

  it('trainStop sends POST to /train/stop', async () => {
    await api.trainStop();
    expect(mockFetch).toHaveBeenCalledWith(`${BASE}/api/train/stop`, { method: 'POST' });
  });

  // --- Evaluate ---
  it('evaluateStatus sends GET', async () => {
    await api.evaluateStatus();
    expect(mockFetch).toHaveBeenCalledWith(`${BASE}/api/evaluate/status`);
  });

  it('evaluateStop sends POST', async () => {
    await api.evaluateStop();
    expect(mockFetch).toHaveBeenCalledWith(`${BASE}/api/evaluate/stop`, { method: 'POST' });
  });

  // --- Predict ---
  it('predictStreamUrl builds correct URL with month', () => {
    const url = api.predictStreamUrl('2026-03');
    expect(url).toContain('month=2026-03');
    expect(url).not.toContain('category');
  });

  it('predictStreamUrl includes category and dryRun', () => {
    const url = api.predictStreamUrl('2026-03', 'Fruit', true);
    expect(url).toContain('month=2026-03');
    expect(url).toContain('category=Fruit');
    expect(url).toContain('dryRun=true');
  });

  it('predictResults sends GET with month query', async () => {
    await api.predictResults('2026-03');
    expect(mockFetch).toHaveBeenCalledWith(`${BASE}/api/predict/results?month=2026-03`);
  });

  it('predictList sends GET', async () => {
    await api.predictList();
    expect(mockFetch).toHaveBeenCalledWith(`${BASE}/api/predict/list`);
  });

  // --- Optimize ---
  it('optimizeDefaultMonth sends GET', async () => {
    await api.optimizeDefaultMonth();
    expect(mockFetch).toHaveBeenCalledWith(`${BASE}/api/optimize/default-month`);
  });

  it('optimizeStreamUrl includes month and optional dryRun', () => {
    const url = api.optimizeStreamUrl('2026-04', true);
    expect(url).toContain('month=2026-04');
    expect(url).toContain('dryRun=true');
  });

  it('optimizeResults sends GET with month', async () => {
    await api.optimizeResults('2026-04');
    expect(mockFetch).toHaveBeenCalledWith(`${BASE}/api/optimize/results?month=2026-04`);
  });

  // --- Static ---
  it('resultUrl returns the full static path', () => {
    expect(api.resultUrl('mse_comparison.png')).toBe(`${BASE}/results/mse_comparison.png`);
  });
});

/**
 * Fetch mock handler registry for page-level tests.
 * Instead of MSW (which has jsdom compatibility issues), we mock global.fetch
 * with a route-matching function that returns canned JSON responses.
 */

type Handler = { method: string; pattern: RegExp; response: () => unknown; status?: number };

const HANDLERS: Handler[] = [
  // Upload
  { method: 'GET', pattern: /\/api\/upload\/files$/, response: () => ({
    files: [
      { name: 'sales_2025_12_december.csv', size: 2048, modified: '2025-12-15T10:00:00Z' },
      { name: 'sales_2025_11_november.csv', size: 1900, modified: '2025-11-10T08:00:00Z' },
    ],
  })},
  { method: 'POST', pattern: /\/api\/upload$/, response: () => ({
    uploaded: [{ name: 'sales_2026_01_january.csv', size: 1024, valid: true, report: 'OK' }],
  })},
  { method: 'DELETE', pattern: /\/api\/upload\/files\//, response: () => ({ deleted: 'file.csv' }) },

  // Train
  { method: 'GET', pattern: /\/api\/train\/status$/, response: () => ({ running: false }) },
  { method: 'GET', pattern: /\/api\/train\/results$/, response: () => ({
    MLP: { mse: 12.5, rmse_eur: 3.54, mae_eur: 2.8, original_profit: 5000, optimized_profit: 5500 },
    Transformer: { mse: 8.2, rmse_eur: 2.86, mae_eur: 2.1, original_profit: 5000, optimized_profit: 5800 },
  })},
  { method: 'POST', pattern: /\/api\/train\/stop$/, response: () => ({ ok: true }) },

  // Evaluate
  { method: 'GET', pattern: /\/api\/evaluate\/status$/, response: () => ({
    charts: [
      { name: 'mse_comparison.png', exists: true, url: '/results/mse_comparison.png' },
      { name: 'profit_comparison.png', exists: true, url: '/results/profit_comparison.png' },
      { name: 'rack_comparison.png', exists: false, url: '/results/rack_comparison.png' },
      { name: 'alluvial_shelf_movement.png', exists: false, url: '/results/alluvial_shelf_movement.png' },
    ],
    running: false,
  })},
  { method: 'POST', pattern: /\/api\/evaluate\/stop$/, response: () => ({ ok: true }) },

  // Ingest
  { method: 'GET', pattern: /\/api\/ingest\/status$/, response: () => ({ running: false }) },
  { method: 'POST', pattern: /\/api\/ingest\/stop$/, response: () => ({ ok: true }) },

  // Predict
  { method: 'GET', pattern: /\/api\/predict\/list$/, response: () => ({ months: ['2025-12', '2025-11', '2025-10'] }) },
  { method: 'GET', pattern: /\/api\/predict\/results/, response: () => ({
    products: [
      { name: 'Manzana', Category: 'Fruta', price_numeric: '1.89', profit_margin_percentage: '32.5', estimated_monthly_sales: '180', product_width_cm: '12.0', rack_id: '0', shelf_level: '4' },
      { name: 'Leche', Category: 'Lácteos', price_numeric: '0.95', profit_margin_percentage: '18.0', estimated_monthly_sales: '320', product_width_cm: '8.5', rack_id: '1', shelf_level: '3' },
    ],
    forecast: { Fruta: 1.15, Lácteos: 0.95 },
    forecastSource: 'heuristic',
    rackSummary: { '0': { original: 800, optimized: 950, products: 2 }, '1': { original: 600, optimized: 680, products: 2 } },
    explanations: null,
  })},
  { method: 'POST', pattern: /\/api\/predict\/stop$/, response: () => ({ ok: true }) },

  // Optimize
  { method: 'GET', pattern: /\/api\/optimize\/status$/, response: () => ({ running: false }) },
  { method: 'GET', pattern: /\/api\/optimize\/default-month$/, response: () => ({ month: '2026-01' }) },
  { method: 'GET', pattern: /\/api\/optimize\/results/, response: () => ({
    kpi: { profitOriginal: 4500, profitOptimized: 5200, profitLiftEur: 700, profitLiftPct: 15.56, productsMoved: 4, totalProducts: 6, racksImproved: 2 },
    movements: [{ from: 1, to: 4, count: 2 }, { from: 2, to: 5, count: 2 }],
    racks: [
      { rack: '0', products: 2, original: 800, optimized: 950, lift: 150 },
      { rack: '1', products: 2, original: 600, optimized: 680, lift: 80 },
    ],
    multipliers: { Fruta: 1.15, Lácteos: 0.95, Chocolate: 1.35 },
    forecastSource: 'heuristic',
    explanations: null,
  })},
  { method: 'POST', pattern: /\/api\/optimize\/stop$/, response: () => ({ ok: true }) },
];

export function createFetchMock() {
  return jest.fn((url: string | URL | Request, init?: RequestInit) => {
    const urlStr = typeof url === 'string' ? url : url instanceof URL ? url.toString() : (url as Request).url;
    const method = (init?.method || 'GET').toUpperCase();

    const handler = HANDLERS.find(h => h.method === method && h.pattern.test(urlStr));
    if (handler) {
      return Promise.resolve({
        ok: true,
        status: handler.status || 200,
        json: () => Promise.resolve(handler.response()),
        text: () => Promise.resolve(JSON.stringify(handler.response())),
      } as Response);
    }

    return Promise.resolve({
      ok: false,
      status: 404,
      json: () => Promise.resolve({ error: 'Not found' }),
      text: () => Promise.resolve('Not found'),
    } as Response);
  });
}

export function setupFetchMock() {
  const mockFetch = createFetchMock();
  global.fetch = mockFetch as typeof fetch;
  return mockFetch;
}

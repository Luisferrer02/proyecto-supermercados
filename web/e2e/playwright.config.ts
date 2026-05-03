import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 30000,
  retries: 0,
  workers: 2,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: [
    {
      command: 'cd ../backend && MOCK_PYTHON=1 npx ts-node src/index.ts',
      port: 3001,
      reuseExistingServer: !process.env.CI,
      timeout: 15000,
      env: { MOCK_PYTHON: '1', PORT: '3001' },
    },
    {
      command: 'cd ../frontend && npx next dev --port 3000',
      port: 3000,
      reuseExistingServer: !process.env.CI,
      timeout: 30000,
    },
  ],
});

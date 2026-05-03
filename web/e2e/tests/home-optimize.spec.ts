import { test, expect } from '@playwright/test';
import path from 'path';

const CSV_PATH = path.resolve(__dirname, '../fixtures/sales_2025_12_december.csv');

test.describe('Home page optimization flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Upload a CSV so the optimize button becomes enabled
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(CSV_PATH);
    await expect(page.getByText('sales_2025_12_december.csv')).toBeVisible({ timeout: 10000 });
  });

  test('full optimize flow shows KPIs', async ({ page }) => {
    await page.getByRole('button', { name: /Optimizar/ }).click();
    await expect(page.getByText(/Optimizando tu supermercado/)).toBeVisible();

    // MOCK_PYTHON completes quickly — wait for results
    await expect(page.getByText(/Beneficio extra al mes/)).toBeVisible({ timeout: 20000 });
    await expect(page.getByText(/Estanterías con mayor mejora/)).toBeVisible();
  });

  test('can return to upload phase after results', async ({ page }) => {
    await page.getByRole('button', { name: /Optimizar/ }).click();
    await expect(page.getByText(/Beneficio extra al mes/)).toBeVisible({ timeout: 20000 });

    await page.getByText(/Optimizar otro mes/).click();
    await expect(page.getByRole('button', { name: /Optimizar/ })).toBeVisible();
  });
});

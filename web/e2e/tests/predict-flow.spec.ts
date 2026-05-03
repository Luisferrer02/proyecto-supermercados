import { test, expect } from '@playwright/test';
import path from 'path';

const CSV_PATH = path.resolve(__dirname, '../fixtures/sales_2025_12_december.csv');

test.describe('Predict flow', () => {
  test.beforeEach(async ({ page }) => {
    // Upload data first
    await page.goto('/upload');
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(CSV_PATH);
    await expect(page.getByText('sales_2025_12_december.csv')).toBeVisible({ timeout: 10000 });
  });

  test('run prediction and see results', async ({ page }) => {
    await page.goto('/predict');
    await expect(page.getByText('Predicción avanzada')).toBeVisible();

    await page.getByRole('button', { name: /Lanzar predicción/ }).click();
    await expect(page.getByRole('button', { name: /Ejecutando/ })).toBeDisabled();

    // MOCK_PYTHON completes quickly — wait for results
    await expect(page.getByText(/Detalle por estantería/)).toBeVisible({ timeout: 20000 });
  });

  test('dry run mode works', async ({ page }) => {
    await page.goto('/predict');

    // Enable dry run
    await page.getByText('Modo sin IA').click();
    await page.getByRole('button', { name: /Lanzar predicción/ }).click();

    await expect(page.getByText(/Detalle por estantería/)).toBeVisible({ timeout: 20000 });
  });

  test('cancel stops prediction', async ({ page }) => {
    await page.goto('/predict');
    await page.getByRole('button', { name: /Lanzar predicción/ }).click();
    await expect(page.getByRole('button', { name: /Cancelar/ })).toBeVisible();

    await page.getByRole('button', { name: /Cancelar/ }).click();
    await expect(page.getByRole('button', { name: /Lanzar predicción/ })).toBeVisible({ timeout: 5000 });
  });
});

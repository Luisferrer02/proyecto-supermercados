import { test, expect } from '@playwright/test';
import path from 'path';

const CSV_PATH = path.resolve(__dirname, '../fixtures/sales_2025_12_december.csv');

test.describe('Train and Evaluate flow', () => {
  test.beforeEach(async ({ page }) => {
    // Upload data first so training has something to work with
    await page.goto('/upload');
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(CSV_PATH);
    await expect(page.getByText('sales_2025_12_december.csv')).toBeVisible({ timeout: 10000 });
  });

  test('train models and see results chart', async ({ page }) => {
    await page.goto('/train');
    await expect(page.getByText('Comparación de modelos')).toBeVisible();

    await page.getByRole('button', { name: /Lanzar entrenamiento/ }).click();
    await expect(page.getByRole('button', { name: /Entrenando/ })).toBeDisabled();

    // MOCK_PYTHON finishes quickly
    await expect(page.getByRole('button', { name: /Lanzar entrenamiento/ })).toBeVisible({ timeout: 15000 });
  });

  test('evaluate generates charts', async ({ page }) => {
    await page.goto('/evaluate');
    await expect(page.getByText('Gráficas de evaluación')).toBeVisible();

    await page.getByRole('button', { name: /Regenerar gráficas/ }).click();
    await expect(page.getByRole('button', { name: /Generando/ })).toBeDisabled();

    // Wait for completion
    await expect(page.getByRole('button', { name: /Regenerar gráficas/ })).toBeVisible({ timeout: 15000 });
  });
});

import { test, expect } from '@playwright/test';
import path from 'path';

const CSV_PATH = path.resolve(__dirname, '../fixtures/sales_2025_12_december.csv');

test.describe('Upload management', () => {
  test('upload a CSV and see it in the list', async ({ page }) => {
    await page.goto('/upload');
    await expect(page.getByText('Subir archivos CSV')).toBeVisible();

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(CSV_PATH);

    await expect(page.getByText('sales_2025_12_december.csv')).toBeVisible({ timeout: 10000 });
  });

  test('delete an uploaded file', async ({ page }) => {
    await page.goto('/upload');
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(CSV_PATH);
    await expect(page.getByText('sales_2025_12_december.csv')).toBeVisible({ timeout: 10000 });

    await page.getByText('Eliminar').first().click();
    await expect(page.getByText('sales_2025_12_december.csv')).not.toBeVisible({ timeout: 5000 });
  });
});

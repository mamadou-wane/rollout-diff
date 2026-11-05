import { defineConfig } from '@playwright/test';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
export default defineConfig({
  testDir: './e2e', workers: 1, fullyParallel: false, timeout: 90_000,
  expect: { timeout: 30_000 },
  use: { browserName: 'chromium', actionTimeout: 15_000, navigationTimeout: 15_000, viewport: { width: 1440, height: 1000 }, trace: 'retain-on-failure' },
  outputDir: process.env.ROLLOUT_E2E_OUTPUT || join(tmpdir(), 'rollout-e2e-results'),
});

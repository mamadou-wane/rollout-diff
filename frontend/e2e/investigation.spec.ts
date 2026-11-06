import { test as base, expect, type Page } from '@playwright/test';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

type Session = { url: string; root: string };
const test = base.extend<{ session: Session; rootDump: boolean }>({
  rootDump: [false, { option: true }],
  session: async ({ browserName, rootDump }, use) => {
    const root = await mkdtemp(join(tmpdir(), `rollout-${browserName}-`));
    for (const [name, changed] of [['run-a', true], ['run-b', false]] as const) {
      await mkdir(join(root, name));
      for (let step = 1; step <= 24; step++) {
        const rows = [];
        for (let p = 0; p < 6; p++) for (let i = 0; i < 2; i++) {
          const present = changed && step > 15 && i === 0;
          rows.push({ step, sample_id: `${step}:${p}:${i}`, prompt_id: `p${p}`,
            input: `private prompt ${p}`, score: present ? 2 : 0,
            output: present ? '```python\ndef solve():\n    pass\n```\n<img src=x onerror="window.__injected=1">' : 'plain words.' });
        }
        await writeFile(join(root, name, `${step}.jsonl`), rows.map(x => JSON.stringify(x)).join('\n'));
      }
    }
    await writeFile(join(root, 'saved-replay.json'), JSON.stringify({
      replay_schema_version: 1, identity: { run: 'r', sample_id: '1:0:0', step: 1, prompt_id: 'p0' },
      source: { sha256: 'a'.repeat(64), response_sha256: createHash('sha256').update('plain words.').digest('hex') },
      historical: { status: 'unavailable', reason: 'not_attempted' },
      task: { status: 'unavailable', reason: 'not_attempted', determinate: false },
      mechanism_status: 'unresolved',
    }));
    await writeFile(join(root, 'bad.json'), '{"private_secret": broken}');
    const executable = process.env.ROLLOUT_EXECUTABLE || resolve('../.venv/bin/rollout');
    const env = { ...process.env }; delete env.PYTHONPATH;
    const server = spawn(executable, ['investigate', '--root', rootDump ? join(root, 'run-a') : root, '--port', '0', '--no-browser'], { env, cwd: root, stdio: ['ignore','pipe','pipe'] });
    let log = '';
    const url = await new Promise<string>((accept, reject) => {
      const timeout = setTimeout(() => reject(new Error('Local server did not start.')), 15_000);
      server.stdout.on('data', data => {
        log += String(data); const match = log.match(/http:\/\/127\.0\.0\.1:\d+\/#token=[\w-]+/);
        if (match) { clearTimeout(timeout); accept(match[0]); }
      });
      server.once('exit', code => { clearTimeout(timeout); reject(new Error(`Local server exited: ${code}`)); });
    });
    try { await use({ url, root }); }
    finally {
      server.kill('SIGINT');
      await new Promise<void>(accept => {
        if (server.exitCode !== null) return accept();
        server.once('exit', () => accept());
        setTimeout(() => { server.kill('SIGKILL'); accept(); }, 10_000).unref();
      });
      await rm(root, { recursive: true, force: true });
    }
  },
});
async function load(page: Page, directory: string, name: string) {
  await page.getByRole('button', { name: 'Load run', exact: true }).click();
  await page.getByRole('button', { name: `Select directory ${directory}`, exact: true }).click();
  await page.getByLabel('Run name', { exact: true }).fill(name);
  await page.getByLabel('Input format').selectOption('verl');
  await page.getByRole('button', { name: 'Load selected run', exact: true }).click();
  await expect(page.getByRole('button', { name: new RegExp(`^Run ${name}, (ready|failed)`) })).toBeVisible();
}

test('load, two-click citation, compare, and saved replay use the real local application', async ({ page, session }, testInfo) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto(session.url);
  await expect(page.getByRole('banner').getByText('Rollout Diff', { exact: true })).toBeVisible();
  await load(page, 'run-a', 'r');
  const event = page.getByRole('button', { name: /^Event 1:/ });
  await expect(event).toBeVisible();
  let clicks = 0;
  await event.click(); clicks++;
  await expect(page.getByRole('img', { name: /Feature mean and standard error/ })).toBeVisible();
  await page.getByRole('button', { name: /^Read response / }).first().click(); clicks++;
  const response = page.getByRole('dialog', { name: 'Original response' });
  await expect(response).toBeVisible();
  await expect(response.locator('pre.response').first()).toContainText('<img src=x');
  await expect(response.locator('img')).toHaveCount(0);
  expect(await page.evaluate(() => Object.hasOwn(window, '__injected'))).toBe(false);
  expect(clicks).toBe(2);
  await testInfo.attach('two-click-measurement', { body: JSON.stringify({ clicks, path: 'event row -> citation -> original response', realBackend: true }), contentType: 'application/json' });
  await response.getByRole('button', { name: 'Close' }).click();
  await page.getByRole('tab', { name: 'Compare', exact: true }).click();
  await page.getByLabel('A start').fill('1'); await page.getByLabel('A end').fill('15');
  await page.getByLabel('B start').fill('16'); await page.getByLabel('B end').fill('24');
  await page.getByRole('button', { name: 'Compare windows', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'What changed' })).toBeVisible();
  await page.getByLabel('Search comparison features').fill('def:solve');
  await page.getByRole('button', { name: 'def:solve', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Shared-prompt estimate' })).toBeVisible();
  await page.getByRole('button', { name: /^Read response / }).first().click();
  await expect(page.getByRole('dialog', { name: 'Original response' })).toBeVisible();
  await page.getByRole('dialog').getByRole('button', { name: 'Close' }).click();
  await page.getByRole('tab', { name: 'Saved evidence' }).click();
  await page.getByRole('button', { name: 'Open saved evidence', exact: true }).click();
  await page.getByRole('button', { name: 'Select saved-replay.json', exact: true }).click();
  await page.getByRole('button', { name: 'Read selected evidence', exact: true }).click();
  await expect(page.getByRole('rowheader', { name: 'Historical claim', exact: true })).toBeVisible();
  await expect(page.getByRole('rowheader', { name: 'Saved protected task result', exact: true })).toBeVisible();
  await expect(page.getByText('Mechanism unresolved.', { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('saved-evidence.png'), fullPage: true });
});

test('two dumps preserve independent sources and empty event results', async ({ page, session }) => {
  await page.goto(session.url); await load(page, 'run-a', 'r'); await load(page, 'run-b', 'control');
  await expect(page.getByText(/No .*events/i).first()).toBeVisible();
  await page.getByRole('tab', { name: 'Compare', exact: true }).click();
  await page.getByLabel('Comparison run').selectOption({ label: 'r' });
  for (const [label, value] of [['A start','1'],['A end','24'],['B start','1'],['B end','24']]) await page.getByLabel(label).fill(value);
  await page.getByRole('button', { name: 'Compare windows', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'What changed' })).toBeVisible();
  await page.getByLabel('Search comparison features').fill('def:solve');
  await page.getByRole('button', { name: 'def:solve', exact: true }).click();
  await page.getByRole('button', { name: /^Read response / }).first().click();
  await expect(page.getByRole('dialog').getByRole('region', { name: 'Response with the feature' })).toContainText('run r, side B');
});

test('mobile loading errors and missing authorization remain explicit', async ({ page, session }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(session.url.split('#')[0]);
  await expect(page.getByRole('alert')).toContainText('No local session token');
  await page.goto(session.url);
  await page.getByRole('tab', { name: 'Saved evidence' }).click();
  await page.getByRole('button', { name: 'Open saved evidence', exact: true }).click();
  await page.getByRole('button', { name: 'Select bad.json', exact: true }).click();
  await page.getByRole('button', { name: 'Read selected evidence', exact: true }).click();
  await expect(page.getByRole('dialog').getByRole('alert')).toContainText('invalid');
  await expect(page.getByRole('dialog')).not.toContainText('private_secret');
  await page.getByRole('dialog').getByRole('button', { name: 'Close' }).click();
  await load(page, 'run-a', 'r');
  await page.getByRole('button', { name: /^Event 1:/ }).click();
  await expect(page.getByRole('img', { name: /Feature mean and standard error/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('mobile-event.png'), fullPage: true });
  await page.getByRole('button', { name: /^Run r,/ }).click();
  const unload = await page.getByRole('button', { name: 'Unload run', exact: true }).boundingBox();
  expect(unload!.x).toBeGreaterThanOrEqual(0);
  expect(unload!.x + unload!.width).toBeLessThanOrEqual(390);
  await page.getByRole('button', { name: 'Unload run', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Load a rollout run to begin' })).toBeVisible();
});


test.describe('a dump used as the launch root', () => {
  test.use({ rootDump: true });
  test('loads the complete current directory', async ({ page, session }) => {
    await page.goto(session.url);
    await page.getByRole('button', { name: 'Load run', exact: true }).click();
    await page.getByRole('button', { name: 'Use this directory', exact: true }).click();
    await page.getByLabel('Run name', { exact: true }).fill('root-run');
    await page.getByLabel('Input format').selectOption('verl');
    await page.getByRole('button', { name: 'Load selected run', exact: true }).click();
    await expect(page.getByRole('button', { name: /^Event 1:/ })).toBeVisible();
    await page.getByRole('button', { name: /^Run root-run,/ }).click();
    await expect(page.getByRole('region', { name: 'Run summary' })).toContainText('288');
  });
});

test('reader pairing, paging, and timeline selection retain the displayed evidence', async ({ page, session }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(session.url);
  await load(page, 'run-a', 'r');
  await expect(page.getByRole('button', { name: /^Event 1:/ })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('region', { name: 'Investigation detail' })).toContainText('Sequential streak began');
  await expect(page.getByRole('region', { name: 'Evidence', exact: true })).toContainText('Retrospective boundary');
  await page.getByRole('button', { name: /^Read response / }).first().click();
  const reader = page.getByRole('dialog', { name: 'Original response' });
  await expect(reader.getByRole('region', { name: 'Matched response without the feature' })).toContainText('plain words.');
  const left = reader.getByRole('region', { name: 'Response with the feature' });
  const right = reader.getByRole('region', { name: 'Matched response without the feature' });
  const leftBox = await left.boundingBox(), rightBox = await right.boundingBox();
  expect(leftBox!.x + leftBox!.width).toBeLessThan(rightBox!.x);
  await expect(reader.locator('.reader-line.differs').first()).toBeVisible();
  await reader.getByLabel('Shade lines that differ').uncheck();
  await expect(reader.locator('.reader-line.differs')).toHaveCount(0);
  const first = await left.locator('.reader-identity').textContent();
  await reader.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(left.locator('.reader-identity')).not.toHaveText(first!);
  await reader.getByRole('button', { name: 'Previous', exact: true }).click();
  await expect(left.locator('.reader-identity')).toHaveText(first!);
  await page.screenshot({ path: testInfo.outputPath('paired-reader.png'), fullPage: true });
  await page.keyboard.press('Escape');
  await page.getByRole('tab', { name: 'Compare', exact: true }).click();
  await expect(page.locator('.compare-timeline svg')).toBeVisible();
  const timeline = (await page.locator('.compare-timeline svg').boundingBox())!;
  await page.getByRole('group', { name: 'Side to drag' }).getByRole('button', { name: 'A', exact: true }).click();
  await page.mouse.move(timeline.x + 1, timeline.y + timeline.height / 2);
  await page.mouse.down();
  await page.mouse.move(timeline.x + timeline.width * 14 / 23, timeline.y + timeline.height / 2, { steps: 5 });
  await page.mouse.up();
  await expect(page.getByLabel('A start')).toHaveValue('1');
  await expect(page.getByLabel('A end')).toHaveValue('15');
  await page.getByLabel('B start').fill('16');
  await page.getByLabel('B end').fill('24');
  await page.getByRole('button', { name: 'Compare windows', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'What changed' })).toBeVisible();
  await page.getByLabel('Search comparison features').fill('def:solve');
  const feature = page.getByRole('button', { name: 'def:solve', exact: true });
  await feature.click();
  await expect(page.getByRole('heading', { name: 'Shared-prompt estimate' })).toBeVisible();
  await feature.click();
  await expect(page.getByRole('heading', { name: 'Shared-prompt estimate' })).toHaveCount(0);
  await page.getByLabel('B start').fill('17');
  await expect(page.getByText('The windows changed. Results below are from the previous comparison.')).toBeVisible();
  await feature.click();
  await page.getByRole('button', { name: /^Read response / }).first().click();
  await expect(page.getByRole('dialog').getByRole('region', { name: 'Response with the feature' })).toContainText('run r, side B');
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Compare windows', exact: true }).click();
  await expect(page.getByText('The windows changed. Results below are from the previous comparison.')).toHaveCount(0);
  expect(errors).toEqual([]);
});

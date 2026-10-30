import { createRequire } from 'node:module'; const { chromium } = createRequire(import.meta.url)('playwright');
import { writeFileSync, mkdirSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const out = process.argv[2] || 'shots'; mkdirSync(out, { recursive: true });
const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox'] });
const p = await b.newPage({ viewport: { width: 1000, height: 1000 } });
p.on('console', (m) => { if (m.type() === 'error') console.log('console error:', m.text()); });
p.on('pageerror', (e) => console.log('pageerror:', e.message));
await p.goto('http://localhost:8765/index.html');
await p.waitForFunction('window.ready === true', null, { timeout: 120000 });
for (const [n, az, el] of [['front', 0, 0], ['three-quarter', 35, 10], ['side', 90, 0], ['back', 180, 0], ['top', 0, 60]]) {
  const d = await p.evaluate(([a, e]) => window.shot(a, e), [az, el]);
  writeFileSync(`${out}/${n}.png`, Buffer.from(d.split(',')[1], 'base64'));
  console.log('saved', n);
}
await b.close();

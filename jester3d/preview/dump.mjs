import { createRequire } from 'node:module'; const { chromium } = createRequire(import.meta.url)('playwright');
const b = await chromium.launch({ args: ['--use-gl=swiftshader','--enable-unsafe-swiftshader','--no-sandbox'] });
const p = await b.newPage(); await p.goto('http://localhost:8765/index.html');
await p.waitForFunction('window.ready===true',null,{timeout:120000});
const r = await p.evaluate(() => { const out=[]; window.__model.traverse(o=>{ if(!o.isMesh&&!o.isGroup) return; if(!o.isMesh) return; o.geometry.computeBoundingBox(); const b=o.geometry.boundingBox.clone().applyMatrix4(o.matrixWorld); const sz=b.getSize(new o.position.constructor()); out.push([o.name,sz.toArray().map(v=>+v.toFixed(2))]); }); return out; });
for (const x of r.filter(x=>/^(Centre head \(jester\)|Hand up L|Torso|Arm up L|Ruff collar|Curled shoe L|Harlequin bloomers|Centre head \(jester\) ear L|Skull charm|Playing cards)$/.test(x[0]))) console.log(JSON.stringify(x));
console.log(r.length,'nodes'); await b.close();

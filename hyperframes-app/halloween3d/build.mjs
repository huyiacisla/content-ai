// Đọc briefs/halloween-dance.json ⇒ ghi index.html + bundle stage/stage.ts → assets/stage.js
// Chạy: npm install && node build.mjs
import {readFileSync, writeFileSync} from 'node:fs';
import {build} from 'esbuild';

const brief = JSON.parse(readFileSync(new URL('../../briefs/halloween-dance.json', import.meta.url)));
const P = brief.style.palette;
const dur = brief.meta.format.durationSec;
const beat = brief.meta.beat.beatSec;
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
const SPRING = 'back.out(2.2)';

await build({
  entryPoints: [new URL('./stage/stage.ts', import.meta.url).pathname],
  bundle: true, format: 'esm', target: 'es2022', minify: true,
  outfile: new URL('./assets/stage.js', import.meta.url).pathname,
  nodePaths: [new URL('./node_modules', import.meta.url).pathname],
  logLevel: 'warning',
});

const bat = (id, x, y, s) => `
      <svg class="bat" id="${id}" style="left:${x}px;top:${y}px;width:${s}px" viewBox="0 0 64 28"><path fill="#0d0718" d="M32 10c2-4 4-5 6-3 2-4 8-6 14-5-3 2-4 5-3 8 4-1 8 0 11 3-5 0-9 2-11 6-3-2-7-2-10 1-2-2-4-3-7-3s-5 1-7 3c-3-3-7-3-10-1-2-4-6-6-11-6 3-3 7-4 11-3 1-3 0-6-3-8 6-1 12 1 14 5 2-2 4-1 6 3z"/></svg>`;
const pumpkin = (id, x, s) => `
      <svg class="pumpkin" id="${id}" style="left:${x}px;width:${s}px" viewBox="0 0 100 90"><rect x="46" y="2" width="9" height="16" rx="3" fill="#3a5a1c"/><ellipse cx="30" cy="52" rx="24" ry="34" fill="${P.pumpkin}"/><ellipse cx="70" cy="52" rx="24" ry="34" fill="${P.pumpkin}"/><ellipse cx="50" cy="52" rx="26" ry="36" fill="#f59a33"/><path d="M33 44l9-10 9 10zM55 44l9-10 9 10z" fill="#2b1206"/><path d="M30 62q20 16 40 0l-6 2-4-5-5 6-5-6-5 6-5-6-4 5z" fill="#2b1206"/></svg>`;

const overlays = brief.overlays.map((o) => `
      <div class="clip cap${o.accent ? ' accent' : ''}" id="${o.id}" data-start="${o.startSec}" data-duration="${o.endSec - o.startSec}" data-track-index="2">${esc(o.text)}</div>`).join('');

const tl = [];
for (const o of brief.overlays) tl.push(`tl.fromTo("#${o.id}", {scale: 0.5, y: -40}, {scale: 1, y: 0, duration: 0.45, ease: "${SPRING}"}, ${o.startSec});`);
// đầu mỗi ô nhịp: chữ đập nhẹ theo nhạc
for (const o of brief.overlays) for (let t = o.startSec + 2; t < o.endSec - 0.01; t += 4 * beat) {
  tl.push(`tl.to("#${o.id}", {scale: 1.12, duration: 0.08, ease: "power1.out"}, ${t});`);
  tl.push(`tl.to("#${o.id}", {scale: 1, duration: 0.22, ease: "power2.out"}, ${(t + 0.08).toFixed(2)});`);
}
// dơi bay: quỹ đạo sin, tuyến tính theo thời gian (seek-safe)
[['bat1', 1], ['bat2', -1], ['bat3', 1]].forEach(([id, d], i) => {
  tl.push(`tl.fromTo("#${id}", {x: ${d > 0 ? -260 : 260}}, {x: ${d > 0 ? 1300 : -1300}, duration: ${dur}, ease: "none"}, 0);`);
  tl.push(`tl.fromTo("#${id}", {y: 0}, {y: ${40 + i * 15}, duration: ${(1 + i * 0.25).toFixed(2)}, ease: "sine.inOut", repeat: ${Math.floor(dur / (1 + i * 0.25))}, yoyo: true}, 0);`);
});
tl.push(`tl.fromTo("#moon", {scale: 0.9}, {scale: 1, duration: ${dur}, ease: "none"}, 0);`);
tl.push(`tl.set(".pumpkin", {scaleY: 1}, 0);`);
for (let t = 0; t < dur; t += 2 * beat) {
  tl.push(`tl.to(".pumpkin", {scaleY: 0.9, duration: 0.1, ease: "power1.in"}, ${t});`);
  tl.push(`tl.to(".pumpkin", {scaleY: 1, duration: ${beat - 0.1}, ease: "${SPRING}"}, ${(t + 0.1).toFixed(2)});`);
}

const html = `<!doctype html>
<html lang="vi" data-resolution="portrait">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1080, height=1920" />
    <script src="assets/gsap.min.js"></script>
    <style>
      @font-face { font-family: "Quicksand"; font-weight: 700; font-display: block; src: url("assets/quicksand-vietnamese-700-normal.woff2") format("woff2"); unicode-range: U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB; }
      @font-face { font-family: "Quicksand"; font-weight: 700; font-display: block; src: url("assets/quicksand-latin-ext-700-normal.woff2") format("woff2"); unicode-range: U+0100-02BA,U+02BD-02C5,U+02C7-02CC,U+02CE-02D7,U+02DD-02FF,U+0304,U+0308,U+0329,U+1D00-1DBF,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20C0,U+2113,U+2C60-2C7F,U+A720-A7FF; }
      @font-face { font-family: "Quicksand"; font-weight: 700; font-display: block; src: url("assets/quicksand-latin-700-normal.woff2") format("woff2"); unicode-range: U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD; }
      * { margin: 0; padding: 0; box-sizing: border-box; }
      html, body { width: 1080px; height: 1920px; overflow: hidden; background: ${P.night}; }
      #root { position: relative; width: 100%; height: 100%; font-family: "Quicksand", sans-serif; font-weight: 700;
        background: linear-gradient(180deg, ${P.nightTop} 0%, ${P.night} 70%, #0d0718 100%); }
      #moon { position: absolute; right: 90px; top: 330px; width: 260px; height: 260px; border-radius: 50%; background: ${P.moon}; box-shadow: 0 0 120px 30px rgba(244,233,200,0.25); }
      #floor { position: absolute; left: 140px; top: 1500px; width: 800px; height: 160px; border-radius: 50%; background: radial-gradient(closest-side, rgba(240,138,28,0.45), rgba(240,138,28,0)); }
      .bat { position: absolute; z-index: 2; }
      .pumpkin { position: absolute; top: 1660px; z-index: 3; transform-origin: 50% 100%; }
      #stage { position: absolute; left: 0; top: 0; width: 1080px; height: 1920px; z-index: 4; }
      .cap { position: absolute; left: 0; top: 120px; width: 1080px; text-align: center; font-size: 104px; line-height: 1.1; color: ${P.ink}; z-index: 6; }
      .cap.accent { color: ${P.pumpkin}; }
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="${dur}" data-width="1080" data-height="1920">
      <div id="moon"></div>
      <div id="floor"></div>${bat('bat1', 60, 520, 140)}${bat('bat2', 700, 760, 110)}${bat('bat3', 200, 900, 90)}${pumpkin('pk1', 40, 220)}${pumpkin('pk2', 830, 200)}
      <canvas id="stage" class="clip" width="1080" height="1920" data-start="0" data-duration="${dur}" data-track-index="0"></canvas>${overlays}
      <audio id="music" src="${brief.meta.music.src}" data-start="0" data-duration="${dur}" data-volume="${brief.meta.music.volume}"></audio>
    </div>
    <script>window.__BRIEF = ${JSON.stringify(brief)};</script>
    <script type="module" src="assets/stage.js"></script>
    <script>
      const tl = gsap.timeline({ paused: true });
      ${tl.join('\n      ')}
      window.__timelines = window.__timelines || {};
      window.__timelines["main"] = tl;
      tl.seek(0);
    </script>
  </body>
</html>
`;
writeFileSync(new URL('./index.html', import.meta.url), html);
console.log(`index.html (halloween): ${brief.overlays.length} captions, ${dur}s`);

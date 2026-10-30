// Đọc briefs/homework-dialogue.json ⇒ ghi index.html (canvas Three.js + overlay 2D HTML).
// Chạy: node build3d.mjs
import {readFileSync, writeFileSync} from 'node:fs';

const brief = JSON.parse(readFileSync(new URL('../../briefs/homework-dialogue.json', import.meta.url)));
const {palette, type} = brief.style;
const dur = brief.meta.format.durationSec;
const SPRING = 'back.out(2.2)';
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
const bub = brief.stage3d.overlay2d.bubbles;

const lines = brief.scenes.flatMap((s) => s.lines);
const interactions = (brief.interactions || []).map((i) => ({
  type: i.type, startSec: i.startSec, endSec: i.endSec, approach: i.motion.approach.durationSec,
}));
const stage = {
  stage3d: brief.stage3d,
  lines: lines.map((l) => ({char: l.char, startSec: l.startSec, endSec: l.endSec})),
  interactions,
};

const sceneHtml = brief.scenes.map((s, i) => `
      <div id="scene-${s.id}" class="clip scene" data-start="${s.startSec}" data-duration="${s.endSec - s.startSec}" data-track-index="1">
        <div class="chip" id="chip-${s.id}">0${i + 1} · ${esc(s.label)}</div>
        <div class="title" id="title-${s.id}">${esc(s.title.text)}</div>
        <div class="bar${s.accent === 'heart' ? ' ink' : ''}" id="bar-${s.id}"></div>
      </div>`).join('');

const bubbleHtml = lines.map((l) => `
      <div class="clip bubble ${l.char}" id="${l.id}" data-start="${l.startSec}" data-duration="${l.endSec - l.startSec}" data-track-index="2"
        style="left:${bub[l.char].x}px;top:${bub[l.char].y}px">${esc(l.text)}</div>`).join('');

const tl = [];
for (const s of brief.scenes) {
  const t = s.startSec;
  tl.push(`tl.fromTo("#chip-${s.id}", {y: -40, scale: 0.7}, {y: 0, scale: 1, duration: 0.4, ease: "${SPRING}"}, ${t});`);
  tl.push(`tl.fromTo("#title-${s.id}", {y: 80, scale: 0.8}, {y: 0, scale: 1, duration: 0.5, ease: "${SPRING}"}, ${t});`);
  tl.push(`tl.fromTo("#bar-${s.id}", {scaleX: 0}, {scaleX: 1, duration: 0.5, ease: "${SPRING}"}, ${t + 0.1});`);
}
for (const l of lines) {
  tl.push(`tl.fromTo("#${l.id}", {scale: 0.6, y: 30}, {scale: 1, y: 0, duration: 0.4, ease: "${SPRING}"}, ${l.startSec});`);
}
// tim nổi lên ở cảnh ôm (điểm nhấn duy nhất của cảnh CTA)
const hug = (brief.interactions || []).find((i) => i.type === 'hug');
const heart = hug?.props.find((p) => p.kind === 'heart');
const heartHtml = heart ? `
      <div class="clip heart" id="heart" data-start="${heart.popAtSec}" data-duration="${hug.endSec - heart.popAtSec}" data-track-index="3" style="left:${heart.x - heart.size / 2}px;top:${heart.y - heart.size / 2 + 80}px;width:${heart.size}px;height:${heart.size}px">
        <svg viewBox="0 0 24 24" width="${heart.size}" height="${heart.size}"><path fill="var(--accent)" d="M12 21s-7.5-4.6-9.6-9.2C.9 8.4 2.9 4.5 6.6 4.5c2 0 3.5 1 5.4 3 1.9-2 3.4-3 5.4-3 3.7 0 5.7 3.9 4.2 7.3C19.5 16.4 12 21 12 21z"/></svg>
      </div>` : '';
if (heart) {
  tl.push(`tl.fromTo("#heart", {scale: 0}, {scale: 1, duration: 0.5, ease: "${SPRING}"}, ${heart.popAtSec});`);
  tl.push(`tl.fromTo("#heart", {y: 0}, {y: ${heart.floatPx}, duration: ${hug.endSec - heart.popAtSec}, ease: "none"}, ${heart.popAtSec});`);
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
      :root {
        --ground: ${palette.bg};
        --ink: ${palette.ink};
        --blue: ${palette.babybeat};
        --accent: ${palette.accent};
        --font-display: "${type.family}", sans-serif;
      }
      * { margin: 0; padding: 0; box-sizing: border-box; }
      html, body { width: 1080px; height: 1920px; overflow: hidden; background: var(--ground); }
      #root { position: relative; width: 100%; height: 100%; font-family: var(--font-display); font-weight: ${type.weight}; color: var(--ink); }
      #stage { position: absolute; left: 0; top: 0; width: 1080px; height: 1920px; z-index: 1; }
      .scene { position: absolute; inset: 0; z-index: 4; }
      .chip { position: absolute; left: 50px; top: 90px; padding: 14px 30px; border-radius: 40px; background: var(--blue); font-size: 40px; }
      .title { position: absolute; left: 50px; top: 170px; width: 980px; font-size: 104px; line-height: 1.1; }
      .bar.ink { background: var(--ink); }
      .bar { position: absolute; left: 50px; top: 300px; width: 260px; height: 16px; border-radius: 8px; background: var(--accent); transform-origin: left center; }
      .heart { position: absolute; z-index: 6; }
      .bubble { z-index: 5; position: absolute; width: 480px; padding: 36px 40px; border-radius: 48px; background: var(--blue); font-size: 62px; line-height: 1.2; }
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="${dur}" data-width="1080" data-height="1920">
      <canvas id="stage" class="clip" width="1080" height="1920" data-start="0" data-duration="${dur}" data-track-index="0"></canvas>${sceneHtml}${heartHtml}${bubbleHtml}
      <audio id="music" src="assets/music-120bpm.mp3" data-start="0" data-duration="${dur}" data-volume="0.5"></audio>
    </div>
    <script>window.__STAGE = ${JSON.stringify(stage)};</script>
    <script type="module" src="assets/stage3d.js"></script>
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
console.log('index.html (3D):', lines.length, 'lines,', brief.scenes.length, 'scenes,', dur + 's');

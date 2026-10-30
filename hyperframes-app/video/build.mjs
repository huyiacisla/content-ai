// Đọc briefs/homework-dialogue.json ⇒ ghi index.html (HyperFrames composition).
// Chạy: node build.mjs
import {readFileSync, writeFileSync} from 'node:fs';

const brief = JSON.parse(readFileSync(new URL('../../briefs/homework-dialogue.json', import.meta.url)));
const {palette, type} = brief.style;
const dur = brief.meta.format.durationSec;
const SPRING = 'back.out(2.2)'; // xấp xỉ lò xo giảm chấn (stiffness 180 / damping 18)
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');

const chars = {
  kai: {img: 'assets/kai.jpg', x: 50, y: 330, w: 470, h: 820, name: 'Kai', bubbleX: 550, bubbleY: 400},
  moc: {img: 'assets/moc.jpg', x: 560, y: 740, w: 470, h: 840, name: 'Mộc', bubbleX: 50, bubbleY: 1230},
};

const interactions = brief.interactions || [];
const interAt = (t) => interactions.find((i) => t >= i.startSec && t < i.endSec);

const sceneHtml = brief.scenes.map((s, i) => `
      <div id="scene-${s.id}" class="clip scene" data-start="${s.startSec}" data-duration="${s.endSec - s.startSec}" data-track-index="1">
        <div class="chip" id="chip-${s.id}">0${i + 1} · ${esc(s.label)}</div>
        <div class="title" id="title-${s.id}">${esc(s.title.text)}</div>
        <div class="bar${s.accent === 'heart' ? ' ink' : ''}" id="bar-${s.id}"></div>
      </div>`).join('');

const cardHtml = Object.entries(chars).map(([k, c]) => `
      <div class="card ${k}" id="card-${k}" style="left:${c.x}px;top:${c.y}px;width:${c.w}px;height:${c.h}px">
        <img src="${c.img}" alt="${c.name}" />
        <div class="name">${c.name}</div>
      </div>`).join('');

const lines = brief.scenes.flatMap((s) => s.lines);
const bubbleHtml = lines.map((l) => {
  const c = chars[l.char];
  const o = interAt(l.startSec)?.bubbles?.[l.char];
  const bx = o ? o.x : c.bubbleX;
  const by = o ? o.y : c.bubbleY;
  return `
      <div class="clip bubble ${l.char}" id="${l.id}" data-start="${l.startSec}" data-duration="${l.endSec - l.startSec}" data-track-index="2"
        style="left:${bx}px;top:${by}px">${esc(l.text)}</div>`;
}).join('');

// Props của hoạt cảnh: tim, nối tay, đường, skyline
const SKY = [220, 300, 180, 340, 260, 200, 320, 240];
const propsHtml = interactions.flatMap((it) => it.props.map((p) => {
  const id = `${it.id}-${p.id}`;
  const left = p.popAtSec ?? it.startSec;
  const d = it.endSec - left;
  if (p.kind === 'heart') {
    return `
      <div class="clip prop heart" id="${id}" data-start="${left}" data-duration="${d}" data-track-index="3" style="left:${p.x - p.size / 2}px;top:${p.y - p.size / 2}px;width:${p.size}px;height:${p.size}px">
        <svg viewBox="0 0 24 24" width="${p.size}" height="${p.size}"><path fill="var(--accent)" d="M12 21s-7.5-4.6-9.6-9.2C.9 8.4 2.9 4.5 6.6 4.5c2 0 3.5 1 5.4 3 1.9-2 3.4-3 5.4-3 3.7 0 5.7 3.9 4.2 7.3C19.500 16.400 12 21 12 21z"/></svg>
      </div>`;
  }
  if (p.kind === 'hand-link') {
    return `
      <div class="clip prop handlink" id="${id}" data-start="${left}" data-duration="${d}" data-track-index="3" style="left:${p.x - p.w / 2}px;top:${p.y - p.h / 2}px;width:${p.w}px;height:${p.h}px"></div>`;
  }
  if (p.kind === 'road') {
    return `
      <div class="clip prop road" id="${id}" data-start="${it.startSec}" data-duration="${it.endSec - it.startSec}" data-track-index="0" style="left:0;top:${p.y}px;width:1080px;height:${p.h}px">
        <div class="dashes" id="${id}-dashes"></div>
      </div>`;
  }
  if (p.kind === 'skyline') {
    const b = Array.from({length: 15}, (_, i) => `<i style="left:${i * 160}px;height:${SKY[i % SKY.length]}px;background:${i % 2 ? '#D6EEFB' : '#BFE3F7'}"></i>`).join('');
    return `
      <div class="clip prop skyline" id="${id}" data-start="${it.startSec}" data-duration="${it.endSec - it.startSec}" data-track-index="0" style="top:${p.y}px">
        <div class="blocks" id="${id}-blocks">${b}</div>
      </div>`;
  }
  return '';
})).join('');

const tl = [];
for (const s of brief.scenes) {
  const t = s.startSec;
  tl.push(`tl.fromTo("#chip-${s.id}", {y: -40, scale: 0.7}, {y: 0, scale: 1, duration: 0.4, ease: "${SPRING}"}, ${t});`);
  tl.push(`tl.fromTo("#title-${s.id}", {y: 80, scale: 0.8}, {y: 0, scale: 1, duration: 0.5, ease: "${SPRING}"}, ${t});`);
  tl.push(`tl.fromTo("#bar-${s.id}", {scaleX: 0}, {scaleX: 1, duration: 0.5, ease: "${SPRING}"}, ${t + 0.1});`);
}
for (const k of Object.keys(chars)) {
  tl.push(`tl.set("#card-${k}", {scale: 0.94, opacity: 0.55}, 0);`);
}
for (const l of lines) {
  tl.push(`tl.fromTo("#${l.id}", {scale: 0.6, y: 30}, {scale: 1, y: 0, duration: 0.4, ease: "${SPRING}"}, ${l.startSec});`);
  if (interAt(l.startSec)) continue; // trong hoạt cảnh: thẻ do pose điều khiển
  tl.push(`tl.set("#card-${l.char}", {scale: 1, opacity: 1}, ${l.startSec});`);
  tl.push(`tl.fromTo("#card-${l.char}", {y: 24}, {y: 0, duration: 0.4, ease: "${SPRING}"}, ${l.startSec});`);
  tl.push(`tl.set("#card-${l.char}", {scale: 0.94, opacity: 0.55}, ${l.endSec});`);
}

// Hoạt cảnh: pose (tâm cx,cy / scale / rotate) ⇒ x,y offset so với vị trí gốc của thẻ
const off = (k, pose) => ({
  x: Math.round(pose.cx - (chars[k].x + chars[k].w / 2)),
  y: Math.round(pose.cy - (chars[k].y + chars[k].h / 2)),
  scale: pose.scale,
  rotation: pose.rotate,
});
const STEP = brief.meta.beat.beatSec;
let prev = {kai: {x: 0, y: 0, scale: 0.94, rotation: 0, opacity: 0.55}, moc: {x: 0, y: 0, scale: 0.94, rotation: 0, opacity: 0.55}};
for (const it of interactions) {
  const dur0 = it.motion.approach.durationSec;
  const cur = {};
  for (const k of Object.keys(chars)) {
    const to = off(k, it.pose[k]);
    cur[k] = {...to, opacity: 1};
    const f = prev[k];
    tl.push(`tl.fromTo("#card-${k}", {x: ${f.x}, y: ${f.y}, scale: ${f.scale}, rotation: ${f.rotation}, opacity: ${f.opacity}}, {x: ${to.x}, y: ${to.y}, scale: ${to.scale}, rotation: ${to.rotation}, opacity: 1, duration: ${dur0}, ease: "${SPRING}"}, ${it.startSec});`);
    const sq = it.motion.squeeze;
    const times = !sq ? [] : sq.everyBeat
      ? Array.from({length: Math.floor((it.endSec - sq.fromSec) / STEP)}, (_, i) => sq.fromSec + i * STEP)
      : sq.atBeats.map((b) => it.startSec + b * STEP);
    for (const t of times) {
      tl.push(`tl.to("#card-${k}", {scale: ${+(to.scale * sq.scalePulse).toFixed(3)}, duration: 0.1, ease: "power2.out"}, ${t});`);
      tl.push(`tl.to("#card-${k}", {scale: ${to.scale}, duration: 0.4, ease: "${SPRING}"}, ${t + 0.1});`);
    }
    if (it.motion.step) {
      const st = it.motion.step;
      const n = Math.floor((it.endSec - (it.startSec + dur0)) / STEP);
      for (let i = 0; i < n; i++) {
        const t = it.startSec + dur0 + i * STEP;
        const sign = (st.alternateSign && k === 'moc' ? -1 : 1) * (i % 2 ? -1 : 1);
        tl.push(`tl.to("#card-${k}", {y: ${to.y - st.bobPx}, rotation: ${sign * st.swayDeg}, duration: 0.2, ease: "power1.out"}, ${t});`);
        tl.push(`tl.to("#card-${k}", {y: ${to.y}, rotation: 0, duration: 0.3, ease: "power1.in"}, ${t + 0.2});`);
      }
      tl.push(`tl.to("#card-${k}", {x: ${to.x + it.motion.driftPx}, duration: ${it.endSec - it.startSec - dur0}, ease: "none"}, ${it.startSec + dur0});`);
    }
  }
  prev = cur;
  for (const p of it.props) {
    const id = `${it.id}-${p.id}`;
    const left = p.popAtSec ?? it.startSec;
    if (p.kind === 'heart') {
      tl.push(`tl.fromTo("#${id}", {scale: 0}, {scale: 1, duration: 0.5, ease: "${SPRING}"}, ${left});`);
      tl.push(`tl.fromTo("#${id}", {y: 0}, {y: ${p.floatPx}, duration: ${it.endSec - left}, ease: "none"}, ${left});`);
    } else if (p.kind === 'hand-link') {
      tl.push(`tl.fromTo("#${id}", {scale: 0.3}, {scale: 1, duration: 0.4, ease: "${SPRING}"}, ${left});`);
    } else if (p.kind === 'road') {
      tl.push(`tl.fromTo("#${id}", {y: ${p.h}}, {y: 0, duration: 0.5, ease: "${SPRING}"}, ${it.startSec});`);
      tl.push(`tl.fromTo("#${id}-dashes", {x: 0}, {x: ${-p.dashes.scrollPxPerSec * (it.endSec - it.startSec)}, duration: ${it.endSec - it.startSec}, ease: "none"}, ${it.startSec});`);
    } else if (p.kind === 'skyline') {
      tl.push(`tl.fromTo("#${id}-blocks", {x: 0}, {x: ${-p.scrollPxPerSec * (it.endSec - it.startSec)}, duration: ${it.endSec - it.startSec}, ease: "none"}, ${it.startSec});`);
    }
  }
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
      .scene { position: absolute; inset: 0; }
      .chip { position: absolute; left: 50px; top: 90px; padding: 14px 30px; border-radius: 40px; background: var(--blue); font-size: 40px; }
      .title { position: absolute; left: 50px; top: 170px; width: 980px; font-size: 104px; line-height: 1.1; }
      .bar.ink { background: var(--ink); }
      .bar { position: absolute; left: 50px; top: 300px; width: 260px; height: 16px; border-radius: 8px; background: var(--accent); transform-origin: left center; }
      .card { z-index: 3; position: absolute; border-radius: 56px; overflow: hidden; border: 8px solid var(--blue); background: var(--blue); }
      .card img { width: 100%; height: 100%; object-fit: cover; object-position: top center; display: block; }
      .card.kai { z-index: 4; }
      .name { position: absolute; left: 24px; bottom: 24px; padding: 8px 26px; border-radius: 30px; background: var(--ground); font-size: 48px; }
      .prop { position: absolute; }
      .heart { z-index: 6; }
      .handlink { z-index: 6; border: 8px solid var(--ink); border-radius: 40px; background: #fff; background-image: repeating-linear-gradient(90deg, transparent 0 14px, var(--blue) 14px 22px); }
      .road { z-index: 2; background: var(--ink); overflow: hidden; }
      .dashes { position: absolute; left: 0; top: 123px; width: 2600px; height: 14px; background: repeating-linear-gradient(90deg, #fff 0 120px, transparent 120px 240px); }
      .skyline { z-index: 1; left: 0; width: 1080px; height: 380px; overflow: hidden; }
      .blocks { position: absolute; left: 0; top: 0; width: 2400px; height: 380px; }
      .blocks i { position: absolute; bottom: 0; width: 140px; border-radius: 16px 16px 0 0; }
      .bubble { z-index: 5; position: absolute; width: 480px; padding: 36px 40px; border-radius: 48px; background: var(--blue); font-size: 62px; line-height: 1.2; }
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="${dur}" data-width="1080" data-height="1920">${sceneHtml}${propsHtml}${cardHtml}${bubbleHtml}
      <audio id="music" src="assets/music-120bpm.mp3" data-start="0" data-duration="${dur}" data-volume="0.5"></audio>
    </div>
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
console.log('index.html:', lines.length, 'lines,', brief.scenes.length, 'scenes,', dur + 's');

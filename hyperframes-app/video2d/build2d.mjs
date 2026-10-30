// Đọc briefs/homework-dialogue.json (stage2d + interactions) ⇒ ghi index.html (nhân vật 2D tách nền).
// Chạy: node build2d.mjs
import {readFileSync, writeFileSync} from 'node:fs';

const brief = JSON.parse(readFileSync(new URL('../../briefs/homework-dialogue.json', import.meta.url)));
const st = brief.stage2d;
const {palette, type} = brief.style;
const dur = brief.meta.format.durationSec;
const BEAT = brief.meta.beat.beatSec;
const SPRING = 'back.out(2.2)';
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
const inters = brief.interactions;
const interAt = (t) => inters.find((i) => t >= i.startSec && t < i.endSec);
const pngSize = (f) => {
  const b = readFileSync(new URL(f, import.meta.url));
  return {w: b.readUInt32BE(16), h: b.readUInt32BE(20)};
};

// kích thước + vị trí gốc (tâm ngang, đáy)
const chars = {};
for (const k of ['kai', 'moc']) {
  const sz = pngSize(st.assets[k].replace(/^/, './'));
  const h = st.chars[k].heightPx;
  const w = Math.round((sz.w / sz.h) * h);
  chars[k] = {w, h, left: Math.round(st.poses.stand[k].cx - w / 2), top: st.chars[k].bottomY - h, z: st.chars[k].z};
}
const cx0 = (k) => chars[k].left + chars[k].w / 2;

const lines = brief.scenes.flatMap((s) => s.lines);

const sceneHtml = brief.scenes.map((s, i) => `
      <div id="scene-${s.id}" class="clip scene" data-start="${s.startSec}" data-duration="${s.endSec - s.startSec}" data-track-index="1">
        <div class="chip" id="chip-${s.id}">0${i + 1} · ${esc(s.label)}</div>
        <div class="title" id="title-${s.id}">${esc(s.title.text)}</div>
        <div class="bar${s.accent === 'heart' ? ' ink' : ''}" id="bar-${s.id}"></div>
      </div>`).join('');

const charHtml = Object.entries(chars).map(([k, c]) => `
      <img class="char ${k}" id="char-${k}" src="${st.assets[k]}" alt="${k}" style="left:${c.left}px;top:${c.top}px;width:${c.w}px;height:${c.h}px;z-index:${c.z}" />`).join('');

const bubbleHtml = lines.map((l) => `
      <div class="clip bubble ${l.char}" id="${l.id}" data-start="${l.startSec}" data-duration="${l.endSec - l.startSec}" data-track-index="2"
        style="left:${st.bubbles[l.char].x}px;top:${st.bubbles[l.char].y}px">${esc(l.text)}</div>`).join('');

const walk = inters.find((i) => i.type === 'walk');
const hug = inters.find((i) => i.type === 'hug');
const hold = inters.find((i) => i.type === 'hold-hands');
const SKY = [220, 300, 180, 340, 260, 200, 320, 240];
const sky = Array.from({length: 15}, (_, i) => `<i style="left:${i * 160}px;height:${SKY[i % SKY.length]}px;background:${st.ground.skyline.colors[i % 2]}"></i>`).join('');
const hl = st.props.handlink;
const hp = st.props.heart;
const propHtml = `
      <div class="clip ground" id="ground" data-start="0" data-duration="${walk.startSec}" data-track-index="0" style="top:${st.groundY}px"></div>
      <div class="clip skyline" id="skyline" data-start="${walk.startSec}" data-duration="${walk.endSec - walk.startSec}" data-track-index="0" style="top:${st.groundY - 380}px"><div class="blocks" id="blocks">${sky}</div></div>
      <div class="clip road" id="road" data-start="${walk.startSec}" data-duration="${walk.endSec - walk.startSec}" data-track-index="0" style="top:${st.groundY}px"><div class="dashes" id="dashes"></div></div>
      <div class="clip handlink" id="hl-hold" data-start="${hl['hold-hands'].popAtSec}" data-duration="${hold.endSec - hl['hold-hands'].popAtSec}" data-track-index="3" style="left:${hl['hold-hands'].x - hl['hold-hands'].w / 2}px;top:${hl['hold-hands'].y - hl['hold-hands'].h / 2}px;width:${hl['hold-hands'].w}px;height:${hl['hold-hands'].h}px"></div>
      <div class="clip heart" id="heart" data-start="${hp.popAtSec}" data-duration="${hug.endSec - hp.popAtSec}" data-track-index="3" style="left:${hp.x - hp.size / 2}px;top:${hp.y - hp.size / 2}px;width:${hp.size}px;height:${hp.size}px">
        <svg viewBox="0 0 24 24" width="${hp.size}" height="${hp.size}"><path fill="var(--accent)" d="M12 21s-7.5-4.6-9.6-9.2C.9 8.4 2.9 4.5 6.6 4.5c2 0 3.5 1 5.4 3 1.9-2 3.4-3 5.4-3 3.7 0 5.7 3.9 4.2 7.3C19.5 16.4 12 21 12 21z"/></svg>
      </div>`;

// ---- timeline ----
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

const poseOf = (type, k) => st.poses[type][k];
const stateOf = (type, k) => {
  const p = poseOf(type, k);
  return {x: Math.round(p.cx - cx0(k)), rot: p.rot, scale: p.scale};
};
const fmt = (o) => `x: ${o.x}, rotation: ${o.rot}, scale: ${o.scale}`;
const M = st.motion;

for (const k of Object.keys(chars)) {
  tl.push(`gsap.set("#char-${k}", {transformOrigin: "50% 100%"});`);
  // đứng yên ban đầu
  let cur = stateOf('stand', k);
  tl.push(`tl.set("#char-${k}", {${fmt(cur)}, y: 0}, 0);`);
  const seq = [...inters].sort((a, b) => a.startSec - b.startSec);
  // 1) hoạt cảnh: lò xo tới pose mới
  for (const it of seq) {
    const to = stateOf(it.type, k);
    tl.push(`tl.fromTo("#char-${k}", {${fmt(cur)}, y: 0}, {${fmt(to)}, y: 0, duration: ${it.motion.approach.durationSec}, ease: "${SPRING}"}, ${it.startSec});`);
    cur = to;
  }
  // 2) thoại: pop + wobble (đứng yên) quanh pose hiện tại
  for (const l of lines.filter((x) => x.char === k)) {
    const it = interAt(l.startSec);
    const base = it ? stateOf(it.type, k) : stateOf('stand', k);
    const idle = base.scale;
    if (!it) {
      tl.push(`tl.fromTo("#char-${k}", {scale: ${M.talk.popFrom}, y: 24}, {scale: ${M.talk.popTo}, y: 0, duration: 0.4, ease: "${SPRING}"}, ${l.startSec});`);
    }
    const n = Math.floor((l.endSec - l.startSec) * M.talk.wobbleHz);
    for (let i = 0; i < n; i++) {
      const a = (i % 2 ? -1 : 1) * M.talk.wobbleDeg;
      tl.push(`tl.fromTo("#char-${k}", {rotation: ${base.rot + (i % 2 ? M.talk.wobbleDeg : -M.talk.wobbleDeg)}}, {rotation: ${base.rot + a}, duration: ${1 / M.talk.wobbleHz}, ease: "sine.inOut"}, ${l.startSec + i / M.talk.wobbleHz});`);
    }
    if (!it) tl.push(`tl.set("#char-${k}", {scale: ${idle}, rotation: ${base.rot}}, ${l.endSec});`);
  }
  // 3) nhún ôm / nắm tay
  for (const [type, cfg, times] of [
    ['hug', M.hug, hug ? Array.from({length: Math.floor((hug.endSec - M.hug.squeezeFromSec) / BEAT)}, (_, i) => M.hug.squeezeFromSec + i * BEAT) : []],
    ['hold-hands', M['hold-hands'], [M['hold-hands'].squeezeAtSec]],
  ]) {
    const s0 = poseOf(type, k).scale;
    for (const t of times) {
      tl.push(`tl.fromTo("#char-${k}", {scale: ${+(s0 * cfg.squeezePulse).toFixed(3)}}, {scale: ${s0}, duration: 0.4, ease: "${SPRING}"}, ${t});`);
    }
  }
  // 4) đi bộ: nhún theo nhịp
  if (walk) {
    const w = stateOf('walk', k);
    const t0 = walk.startSec + walk.motion.approach.durationSec;
    const n = Math.floor((walk.endSec - t0) / BEAT);
    for (let i = 0; i < n; i++) {
      const t = t0 + i * BEAT;
      const sign = (M.walk.alternateSign && k === 'moc' ? -1 : 1) * (i % 2 ? -1 : 1);
      tl.push(`tl.fromTo("#char-${k}", {y: 0, rotation: 0}, {y: -${M.walk.bobPx}, rotation: ${sign * M.walk.swayDeg}, duration: 0.2, ease: "power1.out"}, ${t});`);
      tl.push(`tl.fromTo("#char-${k}", {y: -${M.walk.bobPx}, rotation: ${sign * M.walk.swayDeg}}, {y: 0, rotation: 0, duration: 0.3, ease: "power1.in"}, ${t + 0.2});`);
    }
  }
}
// prop
tl.push(`tl.fromTo("#hl-hold", {scale: 0.3}, {scale: 1, duration: 0.4, ease: "${SPRING}"}, ${hl['hold-hands'].popAtSec});`);
tl.push(`tl.fromTo("#heart", {scale: 0}, {scale: 1, duration: 0.5, ease: "${SPRING}"}, ${hp.popAtSec});`);
tl.push(`tl.fromTo("#heart", {y: 0}, {y: ${hp.floatPx}, duration: ${hug.endSec - hp.popAtSec}, ease: "none"}, ${hp.popAtSec});`);
const wd = walk.endSec - walk.startSec;
tl.push(`tl.fromTo("#road", {y: 260}, {y: 0, duration: 0.5, ease: "${SPRING}"}, ${walk.startSec});`);
tl.push(`tl.fromTo("#dashes", {x: 0}, {x: ${-st.ground.walkRoad.scrollPxPerSec * wd}, duration: ${wd}, ease: "none"}, ${walk.startSec});`);
tl.push(`tl.fromTo("#blocks", {x: 0}, {x: ${-st.ground.skyline.scrollPxPerSec * wd}, duration: ${wd}, ease: "none"}, ${walk.startSec});`);

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
      .scene { position: absolute; inset: 0; z-index: 8; }
      .chip { position: absolute; left: 50px; top: 90px; padding: 14px 30px; border-radius: 40px; background: var(--blue); font-size: 40px; }
      .title { position: absolute; left: 50px; top: 170px; width: 980px; font-size: 104px; line-height: 1.1; }
      .bar { position: absolute; left: 50px; top: 300px; width: 260px; height: 16px; border-radius: 8px; background: var(--accent); transform-origin: left center; }
      .bar.ink { background: var(--ink); }
      .char { position: absolute; display: block; }
      .ground { position: absolute; left: 0; width: 1080px; height: 220px; background: ${st.ground.color}; border-radius: 48px 48px 0 0; z-index: 6; }
      .road { position: absolute; left: 0; width: 1080px; height: 220px; background: var(--ink); overflow: hidden; z-index: 6; }
      .dashes { position: absolute; left: 0; top: 100px; width: 2800px; height: 14px; background: repeating-linear-gradient(90deg, #fff 0 120px, transparent 120px 240px); }
      .skyline { position: absolute; left: 0; width: 1080px; height: 380px; overflow: hidden; z-index: 1; }
      .blocks { position: absolute; left: 0; top: 0; width: 2400px; height: 380px; }
      .blocks i { position: absolute; bottom: 0; width: 140px; border-radius: 16px 16px 0 0; }
      .handlink { position: absolute; z-index: 7; border: 8px solid var(--ink); border-radius: 40px; background: #fff; background-image: repeating-linear-gradient(90deg, transparent 0 14px, var(--blue) 14px 22px); }
      .heart { position: absolute; z-index: 9; }
      .bubble { z-index: 9; position: absolute; width: 480px; padding: 36px 40px; border-radius: 48px; background: var(--blue); font-size: 62px; line-height: 1.2; }
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="${dur}" data-width="1080" data-height="1920">${sceneHtml}${propHtml}${charHtml}${bubbleHtml}
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
console.log('index.html (2D cutout):', lines.length, 'lines,', tl.length, 'tweens,', dur + 's');

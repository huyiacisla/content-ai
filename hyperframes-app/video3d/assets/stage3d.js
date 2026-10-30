// Sân khấu 3D toon (Three.js). Mọi thứ là hàm thuần của thời gian t ⇒ seek-safe.
// Dữ liệu: window.__STAGE = {stage3d, lines, interactions} (build3d.mjs nhúng từ briefs/homework-dialogue.json)
import * as THREE from './three.module.js';

const S = window.__STAGE;
const cfg = S.stage3d;
const [W, H] = cfg.render.canvas;
const INK = 0x1b2a41;
const V3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const lerp = (a, b, k) => a + (b - a) * k;
// lò xo giảm chấn 0→1 (có overshoot nhẹ)
const spring = (u) => (u <= 0 ? 0 : u >= 1.6 ? 1 : 1 - Math.exp(-9 * u) * Math.cos(13 * u));
const rnd = ((s) => () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296))(7);

const renderer = new THREE.WebGLRenderer({canvas: document.getElementById('stage'), antialias: true, alpha: true, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
renderer.setClearColor(0x000000, 0);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, W / H, 0.1, 100);
scene.add(new THREE.AmbientLight(0xffffff, cfg.render.lights.ambient));
const sun = new THREE.DirectionalLight(0xffffff, cfg.render.lights.sun.intensity);
sun.position.set(...cfg.render.lights.sun.pos);
scene.add(sun);

const grad = new THREE.DataTexture(new Uint8Array([110, 190, 255]), 3, 1, THREE.RedFormat);
grad.minFilter = grad.magFilter = THREE.NearestFilter;
grad.needsUpdate = true;
const toonMat = (c) => new THREE.MeshToonMaterial({color: c, gradientMap: grad});
const inkMat = new THREE.MeshBasicMaterial({color: INK, side: THREE.BackSide});

// mesh toon + viền (inverted hull)
function part(geo, color, outline = 1.06) {
  const g = new THREE.Group();
  g.add(new THREE.Mesh(geo, toonMat(color)));
  if (outline) {
    const o = new THREE.Mesh(geo, inkMat);
    o.scale.setScalar(outline);
    g.add(o);
  }
  return g;
}
const sph = (r, c, o) => part(new THREE.SphereGeometry(r, 24, 16), c, o);
const cyl = (rt, rb, h, c, o) => part(new THREE.CylinderGeometry(rt, rb, h, 24), c, o);
const cap = (r, l, c, o) => part(new THREE.CapsuleGeometry(r, l, 6, 16), c, o);
const at = (obj, x, y, z) => (obj.position.set(x, y, z), obj);
const sc = (obj, x, y, z) => (obj.scale.set(x, y, z), obj);

function makeArm(sleeve, skin) {
  const upper = cyl(0.15, 0.15, 1, sleeve, 1.18);
  const lower = cyl(0.14, 0.14, 1, sleeve, 1.18);
  const elbow = sph(0.16, sleeve, 1.15);
  const hand = sph(0.19, skin, 1.12);
  const g = new THREE.Group();
  g.add(upper, lower, elbow, hand);
  return {g, upper, lower, elbow, hand};
}

function makeChar(o) {
  const root = new THREE.Group();
  const body = new THREE.Group();
  root.add(body);
  const skin = o.skin;

  // chân + giày
  const legs = [-1, 1].map((s) => {
    const leg = new THREE.Group();
    leg.position.set(s * 0.27, 1.05, 0);
    leg.add(at(cap(0.18, 0.55, o.pants.leg), 0, -0.45, 0));
    if (o.pants.short) leg.add(at(cyl(0.3, 0.3, 0.36, o.pants.color, 1.05), 0, -0.2, 0));
    leg.add(at(sc(sph(0.3, o.shoes, 1.06), 1, 0.62, 1.45), 0, -0.86, 0.1));
    body.add(leg);
    return leg;
  });

  // thân
  const torso = new THREE.Group();
  body.add(torso);
  if (o.top.type === 'striped-sweater') {
    const [c1, c2, c3, c4] = o.top.bands;
    torso.add(at(sc(sph(0.62, c1, 1.05), 1.05, 0.85, 0.95), 0, 2.0, 0));
    torso.add(at(cyl(0.64, 0.66, 0.42, c1, 1.05), 0, 1.72, 0));
    torso.add(at(cyl(0.67, 0.66, 0.42, c2, 1.05), 0, 1.32, 0));
    torso.add(at(cyl(0.66, 0.64, 0.4, c3, 1.05), 0, 0.94, 0));
    torso.add(at(cyl(0.64, 0.62, 0.14, c4, 1.05), 0, 0.72, 0));
  } else {
    torso.add(at(sc(sph(0.6, o.top.color, 1.05), 1.0, 0.9, 0.9), 0, 2.0, 0));
    torso.add(at(cyl(0.6, 0.58, 1.2, o.top.color, 1.05), 0, 1.4, 0));
    torso.add(at(sc(sph(0.4, o.top.inner, 1.0), 0.9, 1.7, 0.4), 0, 1.45, 0.48));
  }
  body.add(at(cyl(0.16, 0.16, 0.3, skin, 1.08), 0, 2.35, 0));

  // đầu
  const headPivot = new THREE.Group();
  headPivot.position.set(0, 2.45, 0);
  body.add(headPivot);
  const head = new THREE.Group();
  head.position.set(0, 0.8, 0);
  headPivot.add(head);
  head.add(sc(sph(1.0, skin, 1.035), 1, 1.02, 0.96));
  [-1, 1].forEach((s) => head.add(at(sph(0.2, skin, 1.1), s * 1.0, -0.05, 0)));
  const eyes = [-1, 1].map((s) => {
    const e = new THREE.Group();
    e.position.set(s * 0.37, 0.1, 0.8);
    e.add(sc(sph(0.27, 0xffffff, 1.07), 1, 1.1, 0.6));
    e.add(at(sc(sph(0.17, o.eyes, 0), 1, 1, 0.5), 0, -0.02, 0.1));
    e.add(at(sc(sph(0.08, 0x1b2a41, 0), 1, 1, 0.5), 0, -0.02, 0.15));
    e.add(at(sph(0.045, 0xffffff, 0), 0.06, 0.06, 0.2));
    head.add(e);
    return e;
  });
  [-1, 1].forEach((s) => {
    const b = at(sc(cap(0.04, 0.22, o.hair.color, 0), 1, 1, 1), s * 0.38, 0.52, 0.86);
    b.rotation.z = Math.PI / 2 + s * -0.12;
    head.add(b);
    head.add(at(sc(sph(0.17, 0xf08a8a, 0), 1, 0.6, 0.3), s * 0.6, -0.2, 0.74));
  });
  head.add(at(sph(0.11, 0xe89a8a, 1.0), 0, -0.1, 0.96));
  const mouthOpen = at(sc(sph(0.16, 0x7a2f35, 0), 1, 0.1, 0.4), 0, -0.4, 0.9);
  head.add(mouthOpen);
  const smile = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.035, 8, 20, Math.PI), new THREE.MeshBasicMaterial({color: INK}));
  smile.position.set(0, -0.28, 0.93);
  smile.rotation.z = Math.PI;
  head.add(smile);

  // tóc
  const hairCol = o.hair.color;
  const capGeo = new THREE.SphereGeometry(1.0, 32, 16, 0, Math.PI * 2, 0, 1.25);
  const hairCap = part(capGeo, hairCol, 1.04);
  hairCap.scale.set(1.05, 1.04, 1.0);
  hairCap.rotation.x = -0.32;
  head.add(hairCap);
  if (o.hair.type === 'spiky') {
    for (let i = 0; i < o.hair.spikes; i++) {
      const a = (i / o.hair.spikes) * Math.PI * 2;
      const ph = 0.55 + (i % 3) * 0.2;
      const sp = part(new THREE.ConeGeometry(0.22, 0.85 + (i % 2) * 0.25, 10), hairCol, 1.12);
      sp.position.set(Math.sin(a) * 0.62, 0.78 + Math.cos(ph) * 0.3, Math.cos(a) * 0.45 - 0.15);
      sp.rotation.set(Math.cos(a) * 0.5 + 0.2, 0, -Math.sin(a) * 0.7);
      head.add(sp);
    }
  } else {
    for (let i = 0; i < o.hair.curls; i++) {
      const th = Math.acos(1 - rnd() * 0.45);
      const ph = rnd() * Math.PI * 2;
      const r = 0.97 + rnd() * 0.12;
      const c = sph(0.25 + rnd() * 0.08, hairCol, 1.12);
      c.position.set(Math.sin(th) * Math.sin(ph) * r * 1.02, Math.cos(th) * r - 0.1, Math.sin(th) * Math.cos(ph) * r * 0.85 - 0.3);
      head.add(c);
    }
  }

  // thú cưng: gấu mèo trên vai
  if (o.pet) {
    const pet = new THREE.Group();
    pet.position.set(-0.78, 2.45, 0.12);
    pet.add(sc(sph(0.34, 0xa8957f, 1.08), 1.1, 0.95, 0.95));
    [-1, 1].forEach((s) => {
      pet.add(at(sc(sph(0.1, 0x3b302b, 0), 1, 1.1, 0.5), s * 0.15, 0.04, 0.3));
      pet.add(at(sph(0.04, 0xffffff, 0), s * 0.15, 0.06, 0.35));
      const ear = part(new THREE.ConeGeometry(0.1, 0.18, 8), 0xa8957f, 1.1);
      ear.position.set(s * 0.22, 0.32, 0);
      pet.add(ear);
    });
    pet.add(at(sph(0.06, 0x2b2320, 0), 0, -0.04, 0.34));
    const tail = part(new THREE.TorusGeometry(0.22, 0.08, 8, 16, Math.PI * 1.4), 0x7b6a58, 1.1);
    tail.position.set(-0.3, -0.45, -0.1);
    tail.rotation.set(0, 0.6, 0.8);
    pet.add(tail);
    pet.add(at(sc(sph(0.28, 0xa8957f, 1.06), 1, 0.9, 0.9), 0.0, -0.32, 0));
    body.add(pet);
  }

  // tay
  const shoulder = {L: new THREE.Vector3(0.66, 2.0, 0), R: new THREE.Vector3(-0.66, 2.0, 0)};
  const arms = {L: makeArm(o.sleeve, skin), R: makeArm(o.sleeve, skin)};
  body.add(arms.L.g, arms.R.g);

  // bóng đổ
  const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.95, 24), new THREE.MeshBasicMaterial({color: 0x1b2a41, transparent: true, opacity: 0.16}));
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.012;
  root.add(shadow);

  scene.add(root);
  return {root, body, headPivot, head, legs, eyes, mouthOpen, arms, shoulder, torso};
}

const C = cfg.characters;
const chars = {
  kai: makeChar({skin: C.kai.skin, hair: {...C.kai.hair}, eyes: C.kai.eyes, top: C.kai.top, sleeve: C.kai.top.bands[0], pants: {leg: C.kai.pants, color: C.kai.pants, short: false}, shoes: C.kai.shoes, pet: true}),
  moc: makeChar({skin: C.moc.skin, hair: {...C.moc.hair}, eyes: C.moc.eyes, top: C.moc.top, sleeve: C.moc.top.color, pants: {leg: C.moc.skin, color: C.moc.pants, short: true}, shoes: C.moc.shoes, pet: false}),
};

// chế độ sprite: nhân vật = ảnh 2D cutout đặt trong sân khấu 3D (S.sprites)
const SP = S.sprites;
const sprites = {};
if (SP) {
  const loadImg = (src) => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });
  for (const who of ['kai', 'moc']) {
    const c = SP[who];
    const im = await loadImg(c.src);
    const cv = document.createElement('canvas');
    cv.width = im.width; cv.height = im.height;
    const g = cv.getContext('2d');
    g.drawImage(im, 0, 0);
    if (c.fadeBottom) { // ảnh cắt ngang đùi ⇒ mờ dần phần đáy thay vì cắt cứng
      const h = im.height * c.fadeBottom;
      const gr = g.createLinearGradient(0, im.height - h, 0, im.height);
      gr.addColorStop(0, 'rgba(0,0,0,0)'); gr.addColorStop(1, 'rgba(0,0,0,1)');
      g.globalCompositeOperation = 'destination-out';
      g.fillStyle = gr; g.fillRect(0, im.height - h, im.width, h);
    }
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    const w = c.height * im.width / im.height;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, c.height), new THREE.MeshBasicMaterial({map: tex, transparent: true, alphaTest: 0.02}));
    mesh.position.y = c.height / 2;
    const root = new THREE.Group();
    const lean = new THREE.Group(); // trục nghiêng đặt ở chân
    lean.add(mesh);
    root.add(lean);
    if (c.shadow) {
      const sh = new THREE.Mesh(new THREE.CircleGeometry(0.9, 32), new THREE.MeshBasicMaterial({color: 0x1b2a41, transparent: true, opacity: 0.12}));
      sh.rotation.x = -Math.PI / 2; sh.position.y = 0.01; sh.scale.set(1.2, 0.5, 1);
      root.add(sh);
    }
    scene.add(root);
    sprites[who] = {root, mesh: lean, c};
    if (chars[who]) chars[who].root.visible = false;
  }
}

// môi trường đi bộ: đường + vạch + skyline
const env = new THREE.Group();
const walkAngle = cfg.poses.walk.kai.rotY;
env.rotation.y = walkAngle - Math.PI / 2;
const road = new THREE.Mesh(new THREE.PlaneGeometry(60, 4.2), new THREE.MeshBasicMaterial({color: INK}));
road.rotation.x = -Math.PI / 2;
road.position.y = 0.004;
env.add(road);
const dashes = [];
for (let i = 0; i < 24; i++) {
  const d = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.02, 0.14), new THREE.MeshBasicMaterial({color: 0xffffff}));
  d.position.y = 0.012;
  env.add(d);
  dashes.push(d);
}
const blocks = [];
const sky = cfg.environment.skyline.colors;
for (let i = 0; i < 24; i++) {
  const h = 2.2 + rnd() * 3.6;
  const b = new THREE.Mesh(new THREE.BoxGeometry(1.8, h, 1.4), new THREE.MeshBasicMaterial({color: sky[i % 2] ? new THREE.Color(sky[i % 2]) : 0xd6eefb}));
  b.userData = {h};
  b.position.set(0, h / 2, -4.6);
  env.add(b);
  blocks.push(b);
}
scene.add(env);

const lines = S.lines;
const inters = S.interactions;
const poses = cfg.poses;
const M = cfg.motion;

const stateOf = (name, who) => ({...poses[name][who]});
function poseAt(t, who) {
  // chuỗi pose: stand → từng hoạt cảnh (lò xo trong approach)
  let A = stateOf('stand', who);
  let handsA = {};
  let curIdx = -1;
  inters.forEach((it, i) => { if (t >= it.startSec) curIdx = i; });
  if (curIdx < 0) return {st: A, handsA: {}, handsB: {}, p: 1, it: null, prevIt: null};
  for (let i = 0; i < curIdx; i++) {
    A = stateOf(inters[i].type, who);
    handsA = A.hands || {};
  }
  const it = inters[curIdx];
  const B = stateOf(it.type, who);
  const p = spring((t - it.startSec) / it.approach);
  const st = {x: lerp(A.x, B.x, p), z: lerp(A.z, B.z, p), rotY: lerp(A.rotY, B.rotY, p), tilt: lerp(A.tilt, B.tilt, p)};
  return {st, handsA, handsB: B.hands || {}, p, it, prevIt: curIdx > 0 ? inters[curIdx - 1] : null};
}

const tmpW = new THREE.Vector3();
function solveArm(arm, Sh, T, pole) {
  const L1 = 0.62, L2 = 0.62;
  const d = T.clone().sub(Sh);
  let len = d.length();
  const max = L1 + L2 - 0.03;
  if (len > max) { d.multiplyScalar(max / len); len = max; }
  len = Math.max(len, 0.25);
  const dir = d.clone().normalize();
  const a = (len * len + L1 * L1 - L2 * L2) / (2 * len);
  const h = Math.sqrt(Math.max(0, L1 * L1 - a * a));
  const perp = pole.clone().sub(dir.clone().multiplyScalar(pole.dot(dir))).normalize();
  const E = Sh.clone().add(dir.clone().multiplyScalar(a)).add(perp.multiplyScalar(h));
  const Tn = Sh.clone().add(d);
  const place = (m, P, Q) => {
    const v = Q.clone().sub(P);
    m.position.copy(P).addScaledVector(v, 0.5);
    m.scale.set(1, v.length(), 1);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), v.normalize());
  };
  place(arm.upper, Sh, E);
  place(arm.lower, E, Tn);
  arm.elbow.position.copy(E);
  arm.hand.position.copy(Tn);
}

function updateChar(who, t) {
  const ch = chars[who];
  const {st, handsA, handsB, p, it} = poseAt(t, who);
  const isWalk = it && it.type === 'walk';
  const walkT = isWalk ? t - it.startSec : 0;
  const walkAmt = isWalk ? clamp(p, 0, 1) : 0;
  const phase = (walkT / M.walk.cycleSec) * Math.PI * 2;
  const sgn = who === 'kai' ? 1 : -1;

  // nói
  const line = lines.find((l) => l.char === who && t >= l.startSec && t < l.endSec);
  const talking = !!line;
  const u = line ? t - line.startSec : 99;
  const pop = line ? Math.exp(-M.talk.popSpring.k * u) * Math.cos(M.talk.popSpring.hz * u) * M.talk.popSpring.amp : 0;

  const bob = isWalk ? Math.abs(Math.sin(phase)) * M.walk.bobUnits * walkAmt : 0;
  const breath = Math.sin(t * 2 + (who === 'kai' ? 0 : 1.3)) * M.idle.breathAmp;
  ch.root.position.set(st.x, bob, st.z);
  ch.root.rotation.y = st.rotY;
  ch.root.scale.set(1 + pop * 0.5, 1 + pop + breath, 1 + pop * 0.5);

  // ép thân khi ôm / nắm tay
  let squeeze = 0;
  if (it && it.type === 'hug' && t >= M.hug.squeezeFromSec) squeeze = Math.exp(-((t - M.hug.squeezeFromSec) % 0.5) * 10) * M.hug.squeezeAmp;
  if (it && it.type === 'hold-hands' && t >= M['hold-hands'].squeezeAtSec) squeeze = Math.exp(-(t - M['hold-hands'].squeezeAtSec) * 10) * 0.04;
  ch.body.scale.set(1 + squeeze, 1 - squeeze * 0.5, 1 + squeeze);

  // đầu
  ch.headPivot.rotation.z = -st.tilt;
  ch.headPivot.rotation.x = talking ? Math.sin(t * 10) * M.talk.nodRad : Math.sin(t * 1.7) * 0.02;
  ch.headPivot.rotation.y = isWalk ? 0 : -st.rotY * 0.25;
  const open = talking ? 0.25 + Math.abs(Math.sin(t * M.talk.mouthHz * Math.PI)) * 1.5 : 0.1;
  ch.mouthOpen.scale.y = open;
  const blink = (t + (who === 'kai' ? 0 : 1.1)) % M.idle.blinkEverySec < M.idle.blinkSec ? 0.12 : 1;
  ch.eyes.forEach((e) => (e.scale.y = blink));

  // chân
  ch.legs.forEach((leg, i) => {
    leg.rotation.x = isWalk ? Math.sin(phase + (i ? Math.PI : 0)) * M.walk.legSwingRad * walkAmt : 0;
  });

  // tay: IK tới mục tiêu thế giới hoặc tư thế buông
  ch.root.updateMatrixWorld(true);
  ['L', 'R'].forEach((side) => {
    const s = side === 'L' ? 1 : -1;
    const swing = isWalk ? Math.sin(phase + (side === 'L' ? Math.PI : 0)) * 0.35 * walkAmt : (talking && side === (who === 'kai' ? 'L' : 'R') ? 0.35 + Math.sin(t * 6) * 0.1 : 0);
    const hangLocal = new THREE.Vector3(s * 0.8, 0.95 + (talking ? swing * 0.5 : 0), 0.1 + swing);
    const hangW = ch.root.localToWorld(hangLocal.clone());
    const hasA = !!handsA[side], hasB = !!handsB[side];
    let target = hangW;
    const w = lerp(hasA ? 1 : 0, hasB ? 1 : 0, p);
    if (hasA || hasB) {
      const tA = hasA ? V3(handsA[side]) : hangW;
      const tB = hasB ? V3(handsB[side]) : hangW;
      const explicit = tA.clone().lerp(tB, hasA && hasB ? p : 0);
      const ex = hasA && hasB ? explicit : (hasB ? tB : tA);
      target = hangW.clone().lerp(ex, hasA && hasB ? 1 : w);
      if (isWalk && hasB) target.y += bob;
    }
    const ShW = ch.root.localToWorld(new THREE.Vector3(s * 0.66, 2.0, 0));
    const Tl = ch.root.worldToLocal(target.clone());
    const Sl = ch.root.worldToLocal(ShW.clone());
    solveArm(ch.arms[side], Sl, Tl, new THREE.Vector3(s * 0.5, -1, -0.25));
  });
}

function updateSprite(who, t) {
  const {root, mesh, c} = sprites[who];
  const {st, p, it} = poseAt(t, who);
  const isWalk = it && it.type === 'walk';
  const walkT = isWalk ? t - it.startSec : 0;
  const walkAmt = isWalk ? clamp(p, 0, 1) : 0;
  const phase = (walkT / M.walk.cycleSec) * Math.PI * 2;
  const line = lines.find((l) => l.char === who && t >= l.startSec && t < l.endSec);
  const u = line ? t - line.startSec : 99;
  const pop = line ? Math.exp(-M.talk.popSpring.k * u) * Math.cos(M.talk.popSpring.hz * u) * M.talk.popSpring.amp : 0;
  const bob = isWalk ? Math.abs(Math.sin(phase)) * M.walk.bobUnits * walkAmt : 0;
  const breath = Math.sin(t * 2 + (who === 'kai' ? 0 : 1.3)) * M.idle.breathAmp;
  root.position.set(st.x, bob, st.z + (who === 'kai' ? 0.05 : 0));
  root.rotation.y = st.rotY * 0.5; // xoay nhẹ ⇒ còn parallax 3D, ¬ lật mỏng
  mesh.rotation.z = -st.tilt * 0.6 + (isWalk ? Math.sin(phase) * 0.05 * walkAmt : 0);
  let squeeze = 0;
  if (it && it.type === 'hug' && t >= M.hug.squeezeFromSec) squeeze = Math.exp(-((t - M.hug.squeezeFromSec) % 0.5) * 10) * M.hug.squeezeAmp;
  if (it && it.type === 'hold-hands' && t >= M['hold-hands'].squeezeAtSec) squeeze = Math.exp(-(t - M['hold-hands'].squeezeAtSec) * 10) * 0.04;
  root.scale.set(1 + pop * 0.5 + squeeze, 1 + pop + breath - squeeze * 0.5, 1);
}

function updateEnv(t) {
  const f = cfg.environment;
  const on = t >= f.road.fromSec;
  env.visible = on;
  if (!on) return;
  const dt = t - f.road.fromSec;
  const spd = M.walk.roadSpeed;
  const L = 24 * 2.2;
  dashes.forEach((d, i) => { d.position.x = ((((i * 2.2 - dt * spd) % L) + L) % L) - L / 2; });
  const Lb = 24 * 2.4;
  blocks.forEach((b, i) => { b.position.x = ((((i * 2.4 - dt * spd * M.walk.skylineParallax) % Lb) + Lb) % Lb) - Lb / 2; });
}

function updateCamera(t) {
  let shot = cfg.shots[0];
  cfg.shots.forEach((s) => { if (t >= s.startSec) shot = s; });
  camera.position.set(...shot.pos);
  camera.fov = shot.fov;
  camera.updateProjectionMatrix();
  camera.lookAt(...shot.look);
}

function draw(t) {
  if (SP) { updateSprite('kai', t); updateSprite('moc', t); } else { updateChar('kai', t); updateChar('moc', t); }
  updateEnv(t);
  updateCamera(t);
  renderer.render(scene, camera);
}

window.addEventListener('hf-seek', (e) => draw(Math.max(0, Number(e.detail && e.detail.time) || 0)));
draw(window.__hfThreeTime || 0);
window.__stageDraw = draw;

// Sân khấu Halloween: model img2threejs (jester3d) + chiếu ảnh gốc + vũ đạo. Mọi thứ là hàm thuần của t ⇒ seek-safe.
// Brief: briefs/halloween-dance.json (build.mjs nhúng vào window.__BRIEF)
import * as THREE from 'three';
import { createMouseJesterModel } from '../../../jester3d/src/createObjectModel';

const B = (window as any).__BRIEF;
const PROJ = B.character.projection;
const BEAT = B.meta.beat.beatSec;
const W = B.meta.format.width, H = B.meta.format.height;

// factory tải map PBR theo URL tương đối ⇒ thay bằng ảnh 1px (ta dùng vật liệu chiếu ảnh, không cần chúng)
const BLANK = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==';
THREE.DefaultLoadingManager.setURLModifier((url) => (url.endsWith('.png') && !url.includes('jester-proj') ? BLANK : url));

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const smooth = (a: number, b: number, x: number) => { const u = clamp((x - a) / (b - a), 0, 1); return u * u * (3 - 2 * u); };
// cửa sổ mềm [a,b] (vào/ra 0.25s) ⇒ trộn động tác giữa các đoạn không giật
const win = (t: number, a: number, b: number, e = 0.25) => smooth(a - e, a + e, t) * (1 - smooth(b - e, b + e, t));
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;

const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('stage') as HTMLCanvasElement, antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
renderer.setClearColor(0x000000, 0);
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, W / H, 0.1, 100);

const model = createMouseJesterModel();
scene.add(model);
model.updateMatrixWorld(true);
const node = (name: string) => {
  const o = model.getObjectByName(name);
  if (!o) throw new Error(`missing node ${name}`);
  return o;
};

// ---- chiếu ảnh: UV = vị trí thế giới lúc đứng yên → pixel ảnh (front orthographic) ----
const SIZE = 1254;
function projectUV(mesh: THREE.Mesh) {
  const g = mesh.geometry.clone();
  const pos = g.attributes.position;
  const uv = new Float32Array(pos.count * 2);
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
    const px = v.x * 300 + 627, py = 1130 - v.y * 300;
    uv[i * 2] = px / SIZE;
    uv[i * 2 + 1] = 1 - py / SIZE;
  }
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  mesh.geometry = g;
}

// ---- khớp: nhóm pivot đặt ở vai/cổ/hông, attach() giữ nguyên vị trí thế giới ----
function pivotAt(x: number, y: number, z: number, children: string[]) {
  const p = new THREE.Group();
  p.position.set(x, y, z);
  model.add(p);
  for (const c of children) p.attach(node(c));
  return p;
}
const SH = B.character.rig.arms.shoulders;
const TIERS = ['up', 'mid', 'low'] as const;
const arms: Record<string, THREE.Group> = {};
for (const t of TIERS) {
  for (const side of ['L', 'R']) {
    const sx = side === 'L' ? 1 : -1; // trái của nhân vật = +X
    arms[`${t}-${side}`] = pivotAt(sx * SH[t][0], SH[t][1], 0, [`Arm ${t} ${side}__pivot`]);
  }
}
const HEADS = [
  {name: 'Left head (witch hat, wink)', x: -0.973},
  {name: 'Centre head (jester)', x: 0},
  {name: 'Right head (top hat, bored)', x: 0.993},
];
const heads = HEADS.map((h) => {
  const n = node(`${h.name}__pivot`);
  const w = n.getWorldPosition(new THREE.Vector3());
  return pivotAt(w.x, w.y - 0.4, w.z, [`${h.name}__pivot`]);
});
const legs = ['L', 'R'].map((side) => {
  const sx = side === 'L' ? 1 : -1;
  return pivotAt(sx * 0.33, 0.9, 0, [`Stocking ${side}__pivot`, `Curled shoe ${side}__pivot`]);
});
const tails = ['L', 'R'].map((side) => node(`Tail ${side}__pivot`));
const tailRest = tails.map((t) => t.rotation.z);

// vật liệu chiếu ảnh (unlit ⇒ giữ nguyên nét vẽ + màu tranh); nền trong suốt ⇒ viền cắt theo tranh
const tex = await new THREE.TextureLoader().loadAsync(PROJ.texture);
tex.colorSpace = THREE.SRGBColorSpace;
tex.anisotropy = 4;
const projMat = new THREE.MeshBasicMaterial({ map: tex, alphaTest: 0.5, side: THREE.DoubleSide });
model.updateMatrixWorld(true);
model.traverse((o) => {
  const m = o as THREE.Mesh;
  if (!m.isMesh) return;
  projectUV(m);
  m.material = projMat;
});
// gốc nhỏ ở (0,0,0) là neo của cả cây ⇒ chỉ ẩn vật liệu của nó, ¬ ẩn node (sẽ ẩn mọi phần con)
(node('Mouse Jester') as THREE.Mesh).material = new THREE.MeshBasicMaterial({ visible: false });

// ---- vũ đạo ----
const CH = Object.fromEntries(B.choreography.map((c: any) => [c.id, c]));
const seg = (id: string, t: number) => win(t, CH[id].startSec, CH[id].endSec);
const raise = (side: string, a: number) => (side === 'L' ? a : -a); // + = giơ lên/ra ngoài

function dance(t: number) {
  const beat = t / BEAT;
  const sB = Math.sin(Math.PI * beat);
  const wWave = seg('arm-wave', t), wBob = seg('head-bob', t), wSpin = seg('spin-roof', t), wFin = seg('finale', t);
  const steps = clamp(wWave + wBob + wSpin, 0, 1);

  // thân: nhún mỗi nhịp, nhảy bật lúc finale
  const jumpU = clamp((t - CH.finale.startSec) / 0.6, 0, 1);
  const jump = t >= CH.finale.startSec ? Math.sin(Math.PI * jumpU) * 0.55 : 0;
  const deep = wSpin * (Math.floor(beat) % 2 === 0 ? 1.6 : 1);
  model.position.y = Math.abs(sB) * 0.09 * (1 + deep * 0.5) + jump;
  model.scale.set(1 + (1 - Math.abs(sB)) * 0.015, 1 - (1 - Math.abs(sB)) * 0.025 * (1 + deep), 1);
  model.rotation.y = wSpin * Math.sin((2 * Math.PI * beat) / 4) * THREE.MathUtils.degToRad(25);
  model.rotation.z = wWave * Math.sin(Math.PI * beat) * 0.04;

  // tay
  TIERS.forEach((tier, i) => {
    for (const side of ['L', 'R']) {
      const ph = i * (Math.PI / 2) + (side === 'R' ? Math.PI : 0);
      let a = Math.sin(Math.PI * beat - ph) * 0.35 * wWave; // sóng từ trên xuống
      if (tier === 'mid') a += wBob * (-0.55 + Math.abs(Math.sin(Math.PI * beat)) * 0.45); // vỗ tay
      else a += wBob * Math.sin(Math.PI * beat * 0.5 + i) * 0.15;
      if (tier === 'up') a += wSpin * (0.5 + Math.abs(Math.sin(2 * Math.PI * beat)) * 0.25); // đẩy trần
      else a += wSpin * Math.sin(Math.PI * beat + i) * 0.2;
      a += wFin * smooth(CH.finale.startSec, CH.finale.startSec + 0.4, t) * [0.55, 0.4, 0.3][i]; // xòe tay tạo dáng
      a += (1 - steps - wFin) * Math.sin(t * 2 + i) * 0.04; // thở
      arms[`${tier}-${side}`].rotation.z = raise(side, a);
    }
  });

  // đầu: canon trái → giữa → phải
  heads.forEach((h, i) => {
    const ph = beat * Math.PI - (i * Math.PI) / 3;
    h.rotation.z = wBob * Math.sin(ph) * 0.28 + (wWave + wSpin) * Math.sin(Math.PI * beat + i) * 0.06;
    h.rotation.x = wBob * Math.max(0, Math.sin(ph * 2)) * 0.18;
    h.position.y = (h.userData.y0 ??= h.position.y) + Math.abs(Math.sin(ph)) * 0.05 * wBob;
  });

  // chân: nhấc xen kẽ
  legs[0].position.y = (legs[0].userData.y0 ??= legs[0].position.y) + Math.max(0, sB) * 0.16 * steps;
  legs[1].position.y = (legs[1].userData.y0 ??= legs[1].position.y) + Math.max(0, -sB) * 0.16 * steps;

  // đuôi
  tails.forEach((tl, i) => {
    const amp = 0.12 + 0.25 * wBob;
    tl.rotation.z = tailRest[i] + Math.sin(Math.PI * beat + i * Math.PI) * amp;
  });
}

// ---- camera ----
function cameraAt(t: number) {
  let yaw = 0, dist = 7.6;
  for (const c of B.camera) {
    if (t < c.startSec || t > c.endSec) continue;
    const u = (t - c.startSec) / (c.endSec - c.startSec);
    if (c.id === 'c1' || c.id === 'c4') { dist = lerp(c.dist[0], c.dist[1], smooth(0, 1, u)); yaw = 0; }
    else if (c.id === 'c2') { yaw = Math.sin((2 * Math.PI * (t / BEAT)) / 8) * c.yawDeg[1]; dist = c.dist[0]; }
    else { yaw = Math.sin(Math.PI * 2 * u) * c.yawDeg[1]; dist = lerp(c.dist[0], c.dist[1], u); }
  }
  const r = THREE.MathUtils.degToRad(yaw);
  camera.position.set(Math.sin(r) * dist, 1.9, Math.cos(r) * dist);
  camera.lookAt(0, 2.15, 0);
}

function draw(t: number) {
  dance(t);
  cameraAt(t);
  renderer.render(scene, camera);
}

const w = window as any;
w.__stageDraw = draw;
window.addEventListener('hf-seek', (e: any) => draw(Math.max(0, Number(e.detail && e.detail.time) || 0)));
draw(w.__hfThreeTime || 0);

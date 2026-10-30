// Sân khấu Halloween: puppet 2.5D — nhân vật cắt từ chính tranh (tools/cut_parts.py) thành các tấm có khớp,
// xếp lớp theo chiều sâu trong Three.js. Mọi thứ là hàm thuần của t ⇒ seek-safe.
// Brief: briefs/halloween-dance.json (window.__BRIEF), phần cắt: assets/parts.json (window.__PARTS)
import * as THREE from 'three';

const B = (window as any).__BRIEF;
const P = (window as any).__PARTS;
const BEAT = B.meta.beat.beatSec;
const W = B.meta.format.width, H = B.meta.format.height;

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

// px tranh → đơn vị thế giới (đáy giày ở y≈0)
const [OX, OY] = P.origin, K = P.pxPerUnit;
const px2w = (px: number, py: number) => new THREE.Vector3((px - OX) / K, (OY - py) / K, 0);

// mỗi phần = 1 tấm alpha đặt đúng chỗ trong tranh, gắn vào nhóm pivot ở khớp của nó
const loader = new THREE.TextureLoader();
async function part(id: string, parent: THREE.Object3D) {
  const p = P.parts[id];
  const tex = await loader.loadAsync(p.file);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const [x0, y0, x1, y1] = p.bbox;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry((x1 - x0) / K, (y1 - y0) / K),
    new THREE.MeshBasicMaterial({ map: tex, alphaToCoverage: true, side: THREE.DoubleSide }), // viền mịn nhờ MSAA
  );
  const pivot = px2w(p.pivot[0], p.pivot[1]);
  const g = new THREE.Group();
  // vị trí pivot tính theo hệ của cha (cha đặt ở pivot của nó)
  const parentPivot: THREE.Vector3 = parent.userData.pivot ?? new THREE.Vector3();
  g.position.copy(pivot).sub(parentPivot).setZ(p.z);
  g.userData.pivot = pivot;
  mesh.position.copy(px2w((x0 + x1) / 2, (y0 + y1) / 2).sub(pivot));
  g.add(mesh);
  parent.add(g);
  return g;
}

const root = new THREE.Group(); // nhún/nhảy/xoay toàn thân
scene.add(root);
const body = await part('body', root); // pivot ở hông ⇒ nghiêng thân
const heads = await Promise.all(['head-w', 'head-c', 'head-t'].map((id) => part(id, body)));
const TIERS = ['up', 'mid', 'low'] as const;
const arms: Record<string, THREE.Group> = {};
for (const t of TIERS) for (const s of ['L', 'R']) arms[`${t}-${s}`] = await part(`${t}-${s}`, body);
const tails = await Promise.all(['tail-L', 'tail-R'].map((id) => part(id, body)));
const legs = await Promise.all(['leg-L', 'leg-R'].map((id) => part(id, root)));
const star = await part('star', root);
const starRest = star.position.clone();

// ---- vũ đạo ----
const CH = Object.fromEntries(B.choreography.map((c: any) => [c.id, c]));
const seg = (id: string, t: number) => win(t, CH[id].startSec, CH[id].endSec);
const raise = (side: string, a: number) => (side === 'L' ? a : -a); // + = giơ lên/ra ngoài (trái nhân vật = +X)

function dance(t: number) {
  const beat = t / BEAT;
  const sB = Math.sin(Math.PI * beat);
  const wWave = seg('arm-wave', t), wBob = seg('head-bob', t), wSpin = seg('spin-roof', t), wFin = seg('finale', t);
  const steps = clamp(wWave + wBob + wSpin, 0, 1);

  // toàn thân: nhún mỗi nhịp, nhảy bật lúc finale, xoay nhẹ (≤ maxYawDeg để tấm phẳng không lộ mỏng)
  const jumpU = clamp((t - CH.finale.startSec) / 0.6, 0, 1);
  const jump = t >= CH.finale.startSec ? Math.sin(Math.PI * jumpU) * 0.5 : 0;
  const deep = wSpin * (Math.floor(beat) % 2 === 0 ? 1.6 : 1);
  root.position.y = Math.abs(sB) * 0.09 * (1 + deep * 0.5) + jump;
  root.scale.set(1 + (1 - Math.abs(sB)) * 0.015, 1 - (1 - Math.abs(sB)) * 0.025 * (1 + deep), 1);
  root.rotation.y = wSpin * Math.sin((2 * Math.PI * beat) / 4) * THREE.MathUtils.degToRad(B.style.maxYawDeg ?? 10);
  // thân lắc hông
  body.rotation.z = wWave * Math.sin(Math.PI * beat) * 0.05 + wSpin * Math.sin((2 * Math.PI * beat) / 4) * 0.07 + wBob * Math.sin(Math.PI * beat) * 0.03;

  // tay
  TIERS.forEach((tier, i) => {
    for (const side of ['L', 'R']) {
      const ph = i * (Math.PI / 2) + (side === 'R' ? Math.PI : 0);
      let a = Math.sin(Math.PI * beat - ph) * 0.3 * wWave; // sóng từ trên xuống
      if (tier === 'mid') a += wBob * (-0.25 + Math.abs(Math.sin(Math.PI * beat)) * 0.3); // vỗ tay
      else a += wBob * Math.sin(Math.PI * beat * 0.5 + i) * 0.12;
      if (tier === 'up') a += wSpin * (0.4 + Math.abs(Math.sin(2 * Math.PI * beat)) * 0.25); // đẩy trần
      else a += wSpin * Math.sin(Math.PI * beat + i) * 0.15;
      a += wFin * smooth(CH.finale.startSec, CH.finale.startSec + 0.4, t) * [0.5, 0.35, 0.25][i]; // xòe tay tạo dáng
      a += Math.sin(t * 2 + i) * 0.03; // thở
      arms[`${tier}-${side}`].rotation.z = raise(side, a);
    }
  });

  // đầu: canon trái → giữa → phải, nghiêng quanh cổ
  heads.forEach((h, i) => {
    const ph = beat * Math.PI - (i * Math.PI) / 3;
    h.rotation.z = wBob * Math.sin(ph) * 0.22 + (wWave + wSpin) * Math.sin(Math.PI * beat + i) * 0.05 + wFin * Math.sin(Math.PI * beat) * 0.04;
    h.position.y = (h.userData.y0 ??= h.position.y) + Math.abs(Math.sin(ph)) * 0.05 * wBob;
  });

  // chân: nhấc xen kẽ + mũi giày hất nhẹ
  legs.forEach((l, i) => {
    const lift = Math.max(0, i === 0 ? sB : -sB) * steps;
    l.position.y = (l.userData.y0 ??= l.position.y) + lift * 0.16;
    l.rotation.z = (i === 0 ? 1 : -1) * lift * 0.12;
  });

  // đuôi quẫy quanh gốc
  tails.forEach((tl, i) => {
    const amp = 0.08 + 0.18 * wBob;
    tl.rotation.z = Math.sin(Math.PI * beat + i * Math.PI) * amp;
  });

  // ngôi sao lơ lửng
  star.position.set(starRest.x, starRest.y + Math.sin(Math.PI * beat) * 0.08, starRest.z);
  star.rotation.z = Math.sin(Math.PI * beat * 0.5) * 0.35;
}

// ---- camera ----
function cameraAt(t: number) {
  let yaw = 0, dist = 13;
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

import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { BokehPass } from 'three/examples/jsm/postprocessing/BokehPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export type ProceduralModelOptions = {
  wireframe?: boolean;
  castShadow?: boolean;
  receiveShadow?: boolean;
  textureSize?: number;
  textureAnisotropy?: number;
  qualityPriority?: 'reference-fidelity' | 'balanced';
};

export type ProceduralModelRuntime = {
  nodes: Record<string, THREE.Object3D>;
  meshes: Record<string, THREE.Mesh>;
  sockets: Record<string, THREE.Object3D>;
  colliders: Record<string, unknown>;
  destructionGroups: Record<string, THREE.Object3D[]>;
};

type SculptMaterialSpec = Record<string, any>;

// THREE.CapsuleGeometry duplicates every UV-seam vertex (measured: 194 boundary
// edges on the default radius/segments below) -- same benign pattern as box/
// cylinder/sphere/torus, all of which weld cleanly to 0 given a CORRECT weld.
// (A naive vertex-only mergeVertices() reports 64 'non-manifold' edges here, but
// that is a counting artifact, not a real defect: it double-counts a handful of
// near-pole triangles that become degenerate once two of their three corners
// coincide -- confirmed by replicating subdivideCatmullClark's own degenerate-
// triangle-aware vertex identity, which finds a perfectly ordinary 2-manifold.)
// A capsule is the primary shape for skinned limbs/torso (PLAN_1.5), and skinning
// weight computation is O(vertices x bones), so fewer, guaranteed-simple vertices
// is worth having regardless -- authored as a deterministic, closed-by-
// construction mesh instead: shared pole vertices, and
// the radial index taken `% radialSegments` so the seam is never a duplicate
// vertex in the first place, rather than something to weld away afterward.
// Adapted from forge/stage5_rig/emit_rig.py's buildWatertightCapsule (verified
// there: 0 boundary edges, 0 non-manifold edges, deterministic across repeated
// runs) -- ported here rather than imported because this factory and the rig
// emitter are separate generated-output surfaces with no shared runtime module;
// see forge/tests/test_primitive_watertightness.py for the measured proof, and
// coordinate with the rig owner before changing either copy independently.
function buildWatertightCapsule(
  radius: number,
  cylLength: number,
  capSegments: number,
  radialSegments: number,
  heightSegments: number,
): THREE.BufferGeometry {
  const positions: number[] = [];
  const indices: number[] = [];
  const uvs: number[] = [];
  const halfCyl = cylLength / 2;
  const totalSpan = 2 * (Math.PI / 2 * radius) + Math.max(0, cylLength);
  const vOf = (fromBottom: number) => (totalSpan > 0 ? fromBottom / totalSpan : 0);

  const bottomPoleIndex = positions.length / 3;
  positions.push(0, -halfCyl - radius, 0);
  uvs.push(0.5, vOf(0));

  const ringStarts: number[] = [];
  const ringV: number[] = [];
  for (let ring = 1; ring <= capSegments; ring += 1) {
    const phi = (Math.PI / 2) * (ring / capSegments);
    const y = -halfCyl - radius * Math.cos(phi);
    const r = radius * Math.sin(phi);
    const start = positions.length / 3;
    ringStarts.push(start);
    ringV.push(vOf(radius * phi));
    for (let radial = 0; radial < radialSegments; radial += 1) {
      const theta = (radial / radialSegments) * Math.PI * 2;
      positions.push(r * Math.cos(theta), y, r * Math.sin(theta));
      uvs.push(radial / radialSegments, vOf(radius * phi));
    }
  }

  const cylinderRingStarts: number[] = [];
  if (cylLength > 0) {
    for (let step = 1; step <= heightSegments; step += 1) {
      const y = -halfCyl + (cylLength * step) / heightSegments;
      const start = positions.length / 3;
      cylinderRingStarts.push(start);
      const v = vOf(radius * (Math.PI / 2) + halfCyl + y);
      for (let radial = 0; radial < radialSegments; radial += 1) {
        const theta = (radial / radialSegments) * Math.PI * 2;
        positions.push(radius * Math.cos(theta), y, radius * Math.sin(theta));
        uvs.push(radial / radialSegments, v);
      }
    }
  }

  const topRingStarts: number[] = [];
  for (let ring = capSegments - 1; ring >= 1; ring -= 1) {
    const phi = (Math.PI / 2) * (ring / capSegments);
    const y = halfCyl + radius * Math.cos(phi);
    const r = radius * Math.sin(phi);
    const start = positions.length / 3;
    topRingStarts.push(start);
    const v = vOf(radius * (Math.PI / 2) + Math.max(0, cylLength) + radius * (Math.PI / 2 - phi));
    for (let radial = 0; radial < radialSegments; radial += 1) {
      const theta = (radial / radialSegments) * Math.PI * 2;
      positions.push(r * Math.cos(theta), y, r * Math.sin(theta));
      uvs.push(radial / radialSegments, v);
    }
  }

  const topPoleIndex = positions.length / 3;
  positions.push(0, halfCyl + radius, 0);
  uvs.push(0.5, vOf(totalSpan));

  const firstBottomRing = ringStarts[0];
  for (let radial = 0; radial < radialSegments; radial += 1) {
    const next = (radial + 1) % radialSegments;
    indices.push(bottomPoleIndex, firstBottomRing + radial, firstBottomRing + next);
  }

  const allRings = [...ringStarts, ...cylinderRingStarts, ...topRingStarts];
  for (let i = 0; i < allRings.length - 1; i += 1) {
    const a = allRings[i];
    const b = allRings[i + 1];
    for (let radial = 0; radial < radialSegments; radial += 1) {
      const next = (radial + 1) % radialSegments;
      indices.push(a + radial, a + next, b + next);
      indices.push(a + radial, b + next, b + radial);
    }
  }

  const lastRing = allRings[allRings.length - 1];
  for (let radial = 0; radial < radialSegments; radial += 1) {
    const next = (radial + 1) % radialSegments;
    indices.push(topPoleIndex, lastRing + next, lastRing + radial);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function buildTubeGeometry(
  path: { points: [number, number, number][]; radius?: number; radialSegments?: number; closed?: boolean },
): THREE.TubeGeometry {
  const vectors = path.points.map(([x, y, z]) => new THREE.Vector3(x, y, z));
  const curve = new THREE.CatmullRomCurve3(vectors, path.closed ?? false);
  const tubularSegments = Math.max(8, path.points.length * 6);
  return new THREE.TubeGeometry(curve, tubularSegments, path.radius ?? 0.05, path.radialSegments ?? 8, path.closed ?? false);
}

function hashString(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function readLayerNumber(value: unknown, keys: string[], fallback: number): number {
  if (typeof value === 'number') return value;
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    for (const key of keys) {
      if (typeof record[key] === 'number') return record[key] as number;
    }
  }
  return fallback;
}

function hexToRgb(hex: string): [number, number, number] {
  const normalized = /^#[0-9a-f]{3}$/i.test(hex)
    ? '#' + hex.slice(1).split('').map((part) => part + part).join('')
    : hex;
  const value = /^#[0-9a-f]{6}$/i.test(normalized) ? Number.parseInt(normalized.slice(1), 16) : 0x8a7a5f;
  return [clampAlbedoChannel((value >> 16) & 255), clampAlbedoChannel((value >> 8) & 255), clampAlbedoChannel(value & 255)];
}

function materialPalette(spec: SculptMaterialSpec): string[] {
  const palette = spec.colorVariation?.palette;
  if (Array.isArray(palette) && palette.length > 0) return palette.filter((value) => typeof value === 'string');
  const secondary = spec.albedo?.secondary;
  const colors = [spec.baseColor ?? spec.color ?? spec.albedo?.dominant, ...(Array.isArray(secondary) ? secondary : [])];
  return colors.filter((value): value is string => typeof value === 'string' && value.startsWith('#'));
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function clampAlbedoChannel(value: number): number {
  return Math.max(30, Math.min(240, Math.round(value)));
}

function clampPbrF0(value: number): number {
  return Math.max(0.02, Math.min(1, value));
}

function clampPbrIor(value: number): number {
  return Math.max(1, Math.min(2.5, value));
}

function clampPbrMetalness(value: number): number {
  return value >= 0.5 ? 1 : 0;
}

function clampedAlbedoColor(spec: SculptMaterialSpec): THREE.Color {
  const source = typeof spec.baseColor === 'string' ? spec.baseColor : '#8A7A5F';
  // setStyle with an explicit SRGBColorSpace, NOT the numeric constructor.
  //
  // `new THREE.Color(r, g, b)` treats its arguments as LINEAR working-space components,
  // while an authored `baseColor` hex is sRGB. Feeding one to the other skipped the
  // transfer function and lifted every dark albedo: #2e2a28, authored as a near-black
  // vinyl, rendered at roughly sRGB 0.46 — a mid grey. The error is largest exactly where
  // it matters most, because the transfer curve is steepest near black.
  return new THREE.Color().setStyle(source, THREE.SRGBColorSpace);
}

function smoothCurve(value: number): number {
  return value * value * (3 - 2 * value);
}

function periodicHash(x: number, y: number, seed: number, periodX: number, periodY: number): number {
  const wrappedX = ((x % periodX) + periodX) % periodX;
  const wrappedY = ((y % periodY) + periodY) % periodY;
  let value = Math.imul(wrappedX + seed * 17, 374761393) ^ Math.imul(wrappedY + seed * 31, 668265263);
  value = Math.imul(value ^ (value >>> 13), 1274126177);
  return ((value ^ (value >>> 16)) >>> 0) / 4294967295;
}

function periodicValueNoise(u: number, v: number, seed: number, periodX: number, periodY: number): number {
  const x = u * periodX;
  const y = v * periodY;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = smoothCurve(x - x0);
  const ty = smoothCurve(y - y0);
  const a = periodicHash(x0, y0, seed, periodX, periodY);
  const b = periodicHash(x0 + 1, y0, seed, periodX, periodY);
  const c = periodicHash(x0, y0 + 1, seed, periodX, periodY);
  const d = periodicHash(x0 + 1, y0 + 1, seed, periodX, periodY);
  return THREE.MathUtils.lerp(THREE.MathUtils.lerp(a, b, tx), THREE.MathUtils.lerp(c, d, tx), ty);
}

type SurfaceBand = {
  frequency: number;
  amplitude: number;
  stretchX: number;
  stretchY: number;
  ridge: boolean;
};

function surfaceBands(spec: SculptMaterialSpec): SurfaceBand[] {
  const source = Array.isArray(spec.surfaceFrequencyBands) ? spec.surfaceFrequencyBands : [];
  const parsed = source.flatMap((item: unknown) => {
    if (!item || typeof item !== 'object') return [];
    const band = item as Record<string, unknown>;
    const frequency = typeof band.frequency === 'number' ? band.frequency : 0;
    const amplitude = typeof band.amplitude === 'number' ? band.amplitude : 0;
    if (frequency <= 0 || amplitude <= 0) return [];
    const stretch = Array.isArray(band.stretch) ? band.stretch : [1, 1];
    const description = `${String(band.pattern ?? '')} ${String(band.role ?? '')}`.toLowerCase();
    return [{
      frequency,
      amplitude,
      stretchX: typeof stretch[0] === 'number' ? Math.max(0.1, stretch[0]) : 1,
      stretchY: typeof stretch[1] === 'number' ? Math.max(0.1, stretch[1]) : 1,
      ridge: /(ridge|groove|grain|fiber|striated|crack)/.test(description),
    }];
  });
  return parsed.length > 0 ? parsed : [
    { frequency: 2, amplitude: 0.42, stretchX: 1, stretchY: 1, ridge: false },
    { frequency: 12, amplitude: 0.22, stretchX: 1, stretchY: 1, ridge: false },
    { frequency: 56, amplitude: 0.08, stretchX: 1, stretchY: 1, ridge: false },
  ];
}

function sampleSurface(u: number, v: number, bands: SurfaceBand[], seed: number): number {
  let value = 0;
  let weight = 0;
  for (let index = 0; index < bands.length; index += 1) {
    const band = bands[index];
    const periodX = Math.max(1, Math.round(band.frequency * band.stretchX));
    const periodY = Math.max(1, Math.round(band.frequency * band.stretchY));
    let sample = periodicValueNoise(u, v, seed + index * 1013, periodX, periodY);
    if (band.ridge) sample = 1 - Math.abs(sample * 2 - 1);
    value += sample * band.amplitude;
    weight += band.amplitude;
  }
  return weight > 0 ? clamp01(value / weight) : 0.5;
}

function mixPalette(colors: [number, number, number][], value: number): [number, number, number] {
  if (colors.length === 1) return colors[0];
  const scaled = clamp01(value) * (colors.length - 1);
  const index = Math.min(colors.length - 2, Math.floor(scaled));
  const mix = scaled - index;
  const a = colors[index];
  const b = colors[index + 1];
  return [
    Math.round(THREE.MathUtils.lerp(a[0], b[0], mix)),
    Math.round(THREE.MathUtils.lerp(a[1], b[1], mix)),
    Math.round(THREE.MathUtils.lerp(a[2], b[2], mix)),
  ];
}

type ColorGradientStop = { offset: number; color: string };
type ColorGradientSpec = {
  type: 'linear' | 'radial';
  axis: [number, number];
  stops: ColorGradientStop[];
};

function parseRgba(value: string): [number, number, number] {
  const match = /rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/.exec(value);
  if (!match) return [138, 122, 95];
  return [clampAlbedoChannel(Number(match[1])), clampAlbedoChannel(Number(match[2])), clampAlbedoChannel(Number(match[3]))];
}

// Analytical per-pixel gradient sample. The extraction schema's colorGradient carries
// exact rgba(...) stop colors (see extract_part_color_recipe.py), so this samples the
// same trend directly in JS math rather than round-tripping through a Canvas 2D
// createLinearGradient/createRadialGradient object — same visual result, and it composes
// directly with the existing noise/height-correlated colorVariation blend below.
function sampleColorGradient(gradient: ColorGradientSpec, u: number, v: number): [number, number, number] {
  const stops = gradient.stops.length >= 2 ? gradient.stops : [{ offset: 0, color: 'rgba(138,122,95,1)' }, { offset: 1, color: 'rgba(138,122,95,1)' }];
  let t: number;
  if (gradient.type === 'radial') {
    const [cx, cy] = gradient.axis;
    const dx = u - cx;
    const dy = v - cy;
    const maxRadius = Math.max(0.001, Math.hypot(Math.max(cx, 1 - cx), Math.max(cy, 1 - cy)));
    t = clamp01(Math.hypot(dx, dy) / maxRadius);
  } else {
    const [ax, ay] = gradient.axis;
    const projection = (u - 0.5) * ax + (v - 0.5) * ay;
    const maxProjection = 0.5 * (Math.abs(ax) + Math.abs(ay)) || 0.5;
    t = clamp01(projection / maxProjection + 0.5);
  }
  const scaled = t * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.max(0, Math.floor(scaled)));
  const mix = scaled - index;
  const a = parseRgba(stops[index].color);
  const b = parseRgba(stops[index + 1].color);
  return [
    THREE.MathUtils.lerp(a[0], b[0], mix),
    THREE.MathUtils.lerp(a[1], b[1], mix),
    THREE.MathUtils.lerp(a[2], b[2], mix),
  ];
}

function writePixel(data: Uint8ClampedArray, offset: number, red: number, green: number, blue: number): void {
  data[offset] = Math.max(0, Math.min(255, Math.round(red)));
  data[offset + 1] = Math.max(0, Math.min(255, Math.round(green)));
  data[offset + 2] = Math.max(0, Math.min(255, Math.round(blue)));
  data[offset + 3] = 255;
}

function makeCanvas(size: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  return canvas;
}

function createMapTexture(
  canvas: HTMLCanvasElement,
  colorSpace: THREE.ColorSpace,
  spec: SculptMaterialSpec,
  options: ProceduralModelOptions,
): THREE.CanvasTexture {
  const texture = new THREE.CanvasTexture(canvas);
  const projection = spec.textureProjection && typeof spec.textureProjection === 'object' ? spec.textureProjection : {};
  const repeat = Array.isArray(projection.repeat) ? projection.repeat : [2, 2];
  texture.colorSpace = colorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(
    typeof repeat[0] === 'number' ? repeat[0] : 2,
    typeof repeat[1] === 'number' ? repeat[1] : 2,
  );
  texture.anisotropy = Math.max(1, Math.round(options.textureAnisotropy ?? projection.anisotropy ?? 8));
  texture.needsUpdate = true;
  return texture;
}

type ProceduralTextureSet = {
  albedo: THREE.Texture;
  roughness: THREE.Texture;
  height: THREE.Texture;
  normal: THREE.Texture;
  ao: THREE.Texture;
  source: 'reference-pixel-extraction' | 'procedural';
};

function referenceMapUrl(spec: SculptMaterialSpec, channel: string): string | null {
  const reference = spec.referencePbr;
  if (!reference || typeof reference !== 'object') return null;
  if (reference.usable === false) return null;
  const confidence = typeof reference.confidence === 'number'
    ? reference.confidence
    : (typeof reference.estimatedFidelity === 'number' ? reference.estimatedFidelity : 0);
  const threshold = typeof reference.targetThreshold === 'number' ? reference.targetThreshold : 0.7;
  if (confidence < threshold) return null;
  const maps = reference.maps;
  if (!maps || typeof maps !== 'object') return null;
  const map = (maps as Record<string, unknown>)[channel];
  if (!map || typeof map !== 'object') return null;
  const record = map as Record<string, unknown>;
  const url = typeof record.url === 'string' && record.url.trim() ? record.url : record.path;
  return typeof url === 'string' && url.trim() ? url : null;
}

function createLoadedMapTexture(
  url: string,
  colorSpace: THREE.ColorSpace,
  spec: SculptMaterialSpec,
  options: ProceduralModelOptions,
): THREE.Texture {
  const texture = new THREE.TextureLoader().load(url);
  const projection = spec.textureProjection && typeof spec.textureProjection === 'object' ? spec.textureProjection : {};
  const repeat = Array.isArray(projection.repeat) ? projection.repeat : [1, 1];
  texture.colorSpace = colorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(
    typeof repeat[0] === 'number' ? repeat[0] : 1,
    typeof repeat[1] === 'number' ? repeat[1] : 1,
  );
  texture.anisotropy = Math.max(1, Math.round(options.textureAnisotropy ?? projection.anisotropy ?? 8));
  texture.needsUpdate = true;
  return texture;
}

function makeReferenceTextureSet(spec: SculptMaterialSpec, options: ProceduralModelOptions): ProceduralTextureSet | null {
  const albedo = referenceMapUrl(spec, 'albedo');
  const roughness = referenceMapUrl(spec, 'roughness');
  const height = referenceMapUrl(spec, 'height');
  const normal = referenceMapUrl(spec, 'normal');
  const ao = referenceMapUrl(spec, 'ao');
  if (!albedo || !roughness || !height || !normal || !ao) return null;
  return {
    albedo: createLoadedMapTexture(albedo, THREE.SRGBColorSpace, spec, options),
    roughness: createLoadedMapTexture(roughness, THREE.NoColorSpace, spec, options),
    height: createLoadedMapTexture(height, THREE.NoColorSpace, spec, options),
    normal: createLoadedMapTexture(normal, THREE.NoColorSpace, spec, options),
    ao: createLoadedMapTexture(ao, THREE.NoColorSpace, spec, options),
    source: 'reference-pixel-extraction',
  };
}

function makeProceduralTextureSet(
  id: string,
  spec: SculptMaterialSpec,
  options: ProceduralModelOptions,
): ProceduralTextureSet | null {
  if (typeof document === 'undefined') return null;
  const qualityFirst = (options.qualityPriority ?? 'reference-fidelity') === 'reference-fidelity';
  const requested = options.textureSize ?? spec.textureResolution;
  const requestedSize = typeof requested === 'number' && Number.isFinite(requested)
    ? requested
    : (qualityFirst ? 1024 : 512);
  const size = Math.max(256, Math.min(2048, 2 ** Math.round(Math.log2(requestedSize))));
  const canvases = {
    albedo: makeCanvas(size),
    roughness: makeCanvas(size),
    height: makeCanvas(size),
    normal: makeCanvas(size),
    ao: makeCanvas(size),
  };
  const contexts = {
    albedo: canvases.albedo.getContext('2d'),
    roughness: canvases.roughness.getContext('2d'),
    height: canvases.height.getContext('2d'),
    normal: canvases.normal.getContext('2d'),
    ao: canvases.ao.getContext('2d'),
  };
  if (!contexts.albedo || !contexts.roughness || !contexts.height || !contexts.normal || !contexts.ao) return null;
  const images = {
    albedo: contexts.albedo.createImageData(size, size),
    roughness: contexts.roughness.createImageData(size, size),
    height: contexts.height.createImageData(size, size),
    normal: contexts.normal.createImageData(size, size),
    ao: contexts.ao.createImageData(size, size),
  };
  const seed = hashString(id);
  const bands = surfaceBands(spec);
  const heightField = new Float32Array(size * size);
  const roughnessField = new Float32Array(size * size);
  const palette = materialPalette(spec);
  const fallback = typeof spec.baseColor === 'string' ? spec.baseColor : '#8A7A5F';
  const colors = (palette.length >= 2 ? palette : [fallback, '#6E614B', '#A08F70']).map(hexToRgb);
  const baseRoughness = clamp01(readLayerNumber(spec.roughness, ['base'], 0.76));
  const roughnessVariation = clamp01(readLayerNumber(spec.roughness, ['variation'], 0.18));
  const colorAmplitude = clamp01(readLayerNumber(spec.colorVariation, ['amplitude', 'variation'], 0.18));
  const heightCorrelation = clamp01(readLayerNumber(spec.colorVariation, ['heightCorrelation'], 0.3));
  const colorGradient: ColorGradientSpec | undefined = spec.colorGradient;
  for (let y = 0; y < size; y += 1) {
    const v = y / size;
    for (let x = 0; x < size; x += 1) {
      const u = x / size;
      const index = y * size + x;
      const height = sampleSurface(u, v, bands, seed + 101);
      const roughNoise = sampleSurface(u, v, bands, seed + 7001);
      const colorNoise = sampleSurface(u, v, bands, seed + 15013);
      heightField[index] = height;
      roughnessField[index] = clamp01(baseRoughness + (roughNoise - 0.5) * roughnessVariation * 2);
      let color: [number, number, number];
      if (colorGradient) {
        // Evidence-derived spatial gradient (Plan 1.3 Workstream C) takes priority
        // over the noise-based palette blend below — it is a measured trend, not a guess.
        color = sampleColorGradient(colorGradient, u, v);
      } else {
        const paletteValue = clamp01(
          0.5 + (colorNoise - 0.5) * colorAmplitude * 2 + (height - 0.5) * heightCorrelation
        );
        color = mixPalette(colors, paletteValue);
      }
      writePixel(images.albedo.data, index * 4, color[0], color[1], color[2]);
    }
  }
  const normalStrength = Math.max(0.05, readLayerNumber(spec.normal, ['strength', 'amplitude'], 0.35));
  const aoStrength = clamp01(readLayerNumber(spec.ambientOcclusion, ['cavityStrength', 'strength'], 0.35));
  for (let y = 0; y < size; y += 1) {
    const up = ((y - 1 + size) % size) * size;
    const down = ((y + 1) % size) * size;
    for (let x = 0; x < size; x += 1) {
      const left = (x - 1 + size) % size;
      const right = (x + 1) % size;
      const index = y * size + x;
      const center = heightField[index];
      const dx = (heightField[y * size + right] - heightField[y * size + left]) * normalStrength * 6;
      const dy = (heightField[down + x] - heightField[up + x]) * normalStrength * 6;
      const inverseLength = 1 / Math.sqrt(dx * dx + dy * dy + 1);
      const normalX = -dx * inverseLength;
      const normalY = -dy * inverseLength;
      const normalZ = inverseLength;
      const neighborAverage = (
        heightField[y * size + left] + heightField[y * size + right]
        + heightField[up + x] + heightField[down + x]
      ) * 0.25;
      const cavity = Math.max(0, neighborAverage - center);
      const ao = clamp01(1 - aoStrength * (cavity * 12 + (1 - center) * 0.16));
      const offset = index * 4;
      const heightByte = center * 255;
      const roughnessByte = roughnessField[index] * 255;
      writePixel(images.height.data, offset, heightByte, heightByte, heightByte);
      writePixel(images.roughness.data, offset, roughnessByte, roughnessByte, roughnessByte);
      writePixel(
        images.normal.data, offset,
        (normalX * 0.5 + 0.5) * 255,
        (normalY * 0.5 + 0.5) * 255,
        (normalZ * 0.5 + 0.5) * 255,
      );
      writePixel(images.ao.data, offset, ao * 255, ao * 255, ao * 255);
    }
  }
  contexts.albedo.putImageData(images.albedo, 0, 0);
  contexts.roughness.putImageData(images.roughness, 0, 0);
  contexts.height.putImageData(images.height, 0, 0);
  contexts.normal.putImageData(images.normal, 0, 0);
  contexts.ao.putImageData(images.ao, 0, 0);
  return {
    albedo: createMapTexture(canvases.albedo, THREE.SRGBColorSpace, spec, options),
    roughness: createMapTexture(canvases.roughness, THREE.NoColorSpace, spec, options),
    height: createMapTexture(canvases.height, THREE.NoColorSpace, spec, options),
    normal: createMapTexture(canvases.normal, THREE.NoColorSpace, spec, options),
    ao: createMapTexture(canvases.ao, THREE.NoColorSpace, spec, options),
    source: 'procedural',
  };
}

function createSculptMaterial(id: string, spec: SculptMaterialSpec, options: ProceduralModelOptions, denseComponent = false): THREE.MeshPhysicalMaterial {
  // A material that declares -- with evidence -- that its subject carries no texture
  // detail gets NO texture set. Synthesising one anyway is not a harmless default: the
  // branch below then forces color to white and roughness to 1 and reads both from the
  // generated maps, so the authored albedo and the reference-derived roughness are both
  // discarded, and the model gains mottling the reference does not have. Measured on the
  // tuxedo cat, whose black fur rendered as speckled grey-and-white from a palette that
  // only ever described two flat regions.
  const textureless = (spec.textureless as { declared?: boolean } | undefined)?.declared === true;
  const textures = textureless
    ? null
    : makeReferenceTextureSet(spec, options) ?? makeProceduralTextureSet(id, spec, options);
  const material = new THREE.MeshPhysicalMaterial({
    color: textures ? 0xffffff : clampedAlbedoColor(spec),
    roughness: textures ? 1 : clamp01(readLayerNumber(spec.roughness, ['base'], 0.76)),
    metalness: clampPbrMetalness(readLayerNumber(spec.metalness, ['base'], 0.0)),
    clearcoat: clamp01(readLayerNumber(spec.clearcoat, ['base', 'amount'], 0)),
    clearcoatRoughness: clamp01(readLayerNumber(spec.clearcoatRoughness, ['base'], 0.25)),
    transmission: clamp01(readLayerNumber(spec.transmission, ['base', 'amount'], 0)),
    ior: clampPbrIor(readLayerNumber(spec.ior, ['base', 'value'], 1.5)),
    thickness: Math.max(0, readLayerNumber(spec.thickness, ['base', 'amount'], 0)),
    attenuationDistance: Math.max(0.001, readLayerNumber(spec.attenuationDistance, ['base', 'value'], Infinity)),
    attenuationColor: new THREE.Color(typeof spec.attenuationColor === 'string' ? spec.attenuationColor : '#ffffff'),
    sheen: clamp01(readLayerNumber(spec.sheen, ['base', 'amount'], 0)),
    sheenColor: new THREE.Color(typeof spec.sheenColor === 'string' ? spec.sheenColor : '#ffffff'),
    sheenRoughness: clamp01(readLayerNumber(spec.sheenRoughness, ['base'], 1.0)),
    iridescence: clamp01(readLayerNumber(spec.iridescence, ['base', 'amount'], 0)),
    iridescenceIOR: clampPbrIor(readLayerNumber(spec.iridescenceIOR, ['base', 'value'], 1.3)),
    anisotropy: clamp01(readLayerNumber(spec.anisotropy, ['base', 'amount'], 0)),
    anisotropyRotation: readLayerNumber(spec.anisotropy, ['rotation'], 0),
    specularIntensity: clampPbrF0(readLayerNumber(spec.specularF0 ?? spec.f0 ?? spec.specularIntensity, ['base', 'value'], 1.0)),
    specularColor: new THREE.Color(typeof spec.specularColor === 'string' ? spec.specularColor : '#ffffff'),
    emissive: new THREE.Color(typeof spec.emissive === 'string' ? spec.emissive : '#000000'),
    emissiveIntensity: Math.max(0, readLayerNumber(spec.emissiveIntensity, ['base'], 1.0)),
    opacity: clamp01(readLayerNumber(spec.opacity, ['base'], 1)),
    transparent: readLayerNumber(spec.transmission, ['base', 'amount'], 0) > 0 || readLayerNumber(spec.opacity, ['base'], 1) < 1,
    alphaTest: Math.max(0, readLayerNumber(spec.alpha, ['cutoff', 'alphaTest'], 0)),
    wireframe: options.wireframe ?? false,
    side: spec.doubleSided === true ? THREE.DoubleSide : THREE.FrontSide,
    flatShading: spec.flatShading === true,
  });
  if (textures) {
    material.map = textures.albedo;
    material.roughnessMap = textures.roughness;
    material.normalMap = textures.normal;
    material.normalScale.setScalar(Math.max(0.05, readLayerNumber(spec.normal, ['strength', 'amplitude'], 0.35)));
    material.aoMap = textures.ao;
    material.aoMap.channel = 0;
    material.aoMapIntensity = readLayerNumber(spec.ambientOcclusion, ['cavityStrength', 'strength'], 0.35);
    const denseMesh = denseComponent || spec.denseMesh === true || spec.geometryDensity === 'dense' || spec.topologyClass === 'dense';
    const bumpScale = Math.max(0, readLayerNumber(spec.bump, ['amplitude', 'strength'], 0));
    const effectiveBumpScale = denseMesh ? Math.max(0.05, bumpScale) : bumpScale;
    if (effectiveBumpScale > 0) {
      material.bumpMap = textures.height;
      material.bumpScale = effectiveBumpScale;
    }
    const displacementScale = Math.max(0, readLayerNumber(spec.displacement, ['amplitude', 'strength'], 0));
    const effectiveDisplacementScale = denseMesh ? Math.max(0.005, displacementScale) : displacementScale;
    if (effectiveDisplacementScale > 0) {
      material.displacementMap = textures.height;
      material.displacementScale = effectiveDisplacementScale;
      material.displacementBias = -effectiveDisplacementScale * 0.5;
    }
  }
  material.envMapIntensity = readLayerNumber(spec, ['envMapIntensity'], 0.8);
  material.userData.sculptMaterial = spec;
  material.userData.proceduralMapsIndependent = true;
  material.userData.pbrConstraints = { albedoRange: [30, 240], binaryMetalness: true, f0Range: [0.02, 1], iorRange: [1, 2.5] };
  material.userData.pbrTextureSource = textures?.source ?? 'flat-fallback';
  material.userData.referencePbr = spec.referencePbr ?? null;
  material.userData.referenceMaterialId = spec.referenceMaterialId ?? spec.materialReference?.profileId ?? null;
  material.userData.materialEvidence = spec.materialEvidence ?? null;
  material.userData.validationViews = spec.materialReference?.validationViews ?? [];
  material.needsUpdate = true;
  return material;
}

type AttachmentEndpoint = {
  start: THREE.Vector3;
  midpoint: THREE.Vector3;
  quaternion: THREE.Quaternion;
  length: number;
  baseRadius: number;
  endRadius: number;
};

function readVector3(value: unknown, fallback: [number, number, number]): THREE.Vector3 {
  if (Array.isArray(value) && value.length === 3 && value.every((item) => typeof item === 'number')) {
    return new THREE.Vector3(value[0], value[1], value[2]);
  }
  return new THREE.Vector3(fallback[0], fallback[1], fallback[2]);
}

function readNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function makeAttachmentEndpoint(attachment: unknown): AttachmentEndpoint | null {
  if (!attachment || typeof attachment !== 'object') return null;
  const record = attachment as Record<string, unknown>;
  const start = readVector3(record.localStart, [0, 0, 0]);
  const end = readVector3(record.localEnd, [0, 1, 0]);
  const delta = end.clone().sub(start);
  const length = delta.length();
  if (length <= 0.0001) return null;
  const direction = delta.clone().normalize();
  const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  const baseRadius = Math.max(0.005, readNumber(record.baseRadius, 0.06));
  const endRadius = Math.max(0.003, readNumber(record.endRadius, baseRadius * 0.55));
  return {
    start,
    midpoint: delta.multiplyScalar(0.5),
    quaternion,
    length,
    baseRadius,
    endRadius,
  };
}

// Generated from ObjectSculptSpec target: Mouse Jester
// Sculpt build pass: blockout
// This factory is intentionally pass-gated. Finish browser screenshot review before unlocking deeper passes.
export function createMouseJesterModel(options: ProceduralModelOptions = {}): THREE.Group {
  const root = new THREE.Group();
  root.name = "Mouse Jester";
  root.userData.reconstructionEvidence = {"itemFamily": null, "subtype": null, "componentAdapter": null, "route": null, "exactnessTier": null, "referenceCamera": {"solved": false, "fovDegrees": 38, "aspect": 1.0, "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}, "positionHint": [0, 1.85, 6.5], "note": "For likeness work, solve the reference camera (forge/stage1_intake/solve_camera_pose.py) so the review render aligns with the photo and the reference can be projected. Confirm by overlay review."}, "approximationNotes": []};
  root.userData.materialPipeline = {};
  root.userData.materialReferenceRegistry = null;

  const materialMap: Record<string, THREE.Material> = {};
  materialMap["fur-black"] = createSculptMaterial(
    "fur-black",
    {"id": "fur-black", "name": "Black cartoon fur", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR approximation", "baseColor": "#16161c", "color": "#16161c", "albedo": {"dominant": "#24221E", "secondary": ["#13130F", "#282521", "#1E1D19"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights.", "map": {"path": "/home/user/content-ai/jester3d/material-evidence/fur-black/fur-black_albedo.png", "url": "fur-black_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}}, "colorVariation": {"palette": ["#24221E", "#13130F", "#282521", "#1E1D19", "#191814"], "pattern": "reference-derived pixel palette", "amplitude": 0.08, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.308, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.175, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.069, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.68, "variation": 0.1, "map": "independent-procedural-field", "localResponse": "slightly lower roughness on fold highlights, higher in creases"}, "metalness": 0.0, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.163, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/fur-black/fur-black_normal.png", "url": "fur-black_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "heightSource": {"path": "/home/user/content-ai/jester3d/material-evidence/fur-black/fur-black_height.png", "url": "fur-black_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.01, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/fur-black/fur-black_height.png", "url": "fur-black_height.png", "channel": "height", "source": "reference-pixel-extraction"}}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"strength": 0.5, "source": "cavity-bias at folds and under hats/ruff"}, "wear": {"amount": 0.05, "note": "clean ink-black cartoon fills; faint cel-shade only"}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "reference-pbr-pixel-evidence", "type": "material-map-evidence", "evidenceRefs": ["full-object"], "channels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "notes": "Use generated maps as material evidence, then refine after browser screenshot comparison."}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "referencePbr": {"version": "1.0", "sourceImage": "/home/user/content-ai/jester3d/crops/fur-black.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.727, "estimatedFidelity": 0.727, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "maps": {"albedo": {"path": "/home/user/content-ai/jester3d/material-evidence/fur-black/fur-black_albedo.png", "url": "fur-black_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/home/user/content-ai/jester3d/material-evidence/fur-black/fur-black_roughness.png", "url": "fur-black_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/home/user/content-ai/jester3d/material-evidence/fur-black/fur-black_height.png", "url": "fur-black_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/home/user/content-ai/jester3d/material-evidence/fur-black/fur-black_normal.png", "url": "fur-black_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/home/user/content-ai/jester3d/material-evidence/fur-black/fur-black_ao.png", "url": "fur-black_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "diagnostics": {"sourceWidth": 160, "sourceHeight": 160, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 160, "height": 160}, "mask": {"backgroundColor": "#232321", "backgroundNoise": 5.385, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.1042}, "mapStats": {"valueRange": 0.08, "heightP90Gradient": 0.00591, "roughnessBase": 0.68, "roughnessVariation": 0.05, "normalStrength": 0.163, "blurRadius": 21}, "palette": ["#24221E", "#13130F", "#282521", "#1E1D19", "#191814"]}, "warnings": ["single-image inverse rendering cannot prove true physical PBR; confidence is capped", "low value range weakens height/roughness inference", "low high-frequency detail weakens normal/roughness inference"], "albedoNote": "albedo map flattened to the dominant cel-shaded fill colour (source is flat toon colour; crop texture would inject neighbouring colours)"}},
    options
  );
  materialMap["face-cream"] = createSculptMaterial(
    "face-cream",
    {"id": "face-cream", "name": "face-cream", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR approximation", "baseColor": "#efe6d0", "color": "#efe6d0", "albedo": {"dominant": "#F1DFC7", "secondary": ["#EEDBC2", "#F3E3CA", "#E5CFB7"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights.", "map": {"path": "/home/user/content-ai/jester3d/material-evidence/face-cream/face-cream_albedo.png", "url": "face-cream_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}}, "colorVariation": {"palette": ["#F1DFC7", "#EEDBC2", "#F3E3CA", "#E5CFB7", "#4B4038"], "pattern": "reference-derived pixel palette", "amplitude": 0.08, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.308, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.216, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.093, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.685, "variation": 0.1, "map": "independent-procedural-field", "localResponse": "slightly lower roughness on fold highlights, higher in creases"}, "metalness": 0.0, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.175, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/face-cream/face-cream_normal.png", "url": "face-cream_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "heightSource": {"path": "/home/user/content-ai/jester3d/material-evidence/face-cream/face-cream_height.png", "url": "face-cream_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.01, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/face-cream/face-cream_height.png", "url": "face-cream_height.png", "channel": "height", "source": "reference-pixel-extraction"}}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"strength": 0.5, "source": "cavity-bias at folds and under hats/ruff"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "reference-pbr-pixel-evidence", "type": "material-map-evidence", "evidenceRefs": ["full-object"], "channels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "notes": "Use generated maps as material evidence, then refine after browser screenshot comparison."}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "referencePbr": {"version": "1.0", "sourceImage": "/home/user/content-ai/jester3d/crops/face-cream.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.802, "estimatedFidelity": 0.802, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "maps": {"albedo": {"path": "/home/user/content-ai/jester3d/material-evidence/face-cream/face-cream_albedo.png", "url": "face-cream_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/home/user/content-ai/jester3d/material-evidence/face-cream/face-cream_roughness.png", "url": "face-cream_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/home/user/content-ai/jester3d/material-evidence/face-cream/face-cream_height.png", "url": "face-cream_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/home/user/content-ai/jester3d/material-evidence/face-cream/face-cream_normal.png", "url": "face-cream_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/home/user/content-ai/jester3d/material-evidence/face-cream/face-cream_ao.png", "url": "face-cream_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "diagnostics": {"sourceWidth": 160, "sourceHeight": 160, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 160, "height": 160}, "mask": {"backgroundColor": "#F1E1C8", "backgroundNoise": 23.388, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.6067}, "mapStats": {"valueRange": 0.08, "heightP90Gradient": 0.01573, "roughnessBase": 0.685, "roughnessVariation": 0.05, "normalStrength": 0.175, "blurRadius": 21}, "palette": ["#F1DFC7", "#EEDBC2", "#F3E3CA", "#E5CFB7", "#4B4038"]}, "warnings": ["single-image inverse rendering cannot prove true physical PBR; confidence is capped", "low value range weakens height/roughness inference"], "albedoNote": "albedo map flattened to the dominant cel-shaded fill colour (source is flat toon colour; crop texture would inject neighbouring colours)"}},
    options
  );
  materialMap["cloth-teal"] = createSculptMaterial(
    "cloth-teal",
    {"id": "cloth-teal", "name": "cloth-teal", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR approximation", "baseColor": "#1f8a86", "color": "#1f8a86", "albedo": {"dominant": "#469892", "secondary": ["#274A4F", "#408D8A", "#3A8180"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights.", "map": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-teal/cloth-teal_albedo.png", "url": "cloth-teal_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}}, "colorVariation": {"palette": ["#469892", "#274A4F", "#408D8A", "#3A8180", "#2F6B6C"], "pattern": "reference-derived pixel palette", "amplitude": 0.119, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.379, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.215, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.092, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.686, "variation": 0.1, "map": "independent-procedural-field", "localResponse": "slightly lower roughness on fold highlights, higher in creases"}, "metalness": 0.0, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.174, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-teal/cloth-teal_normal.png", "url": "cloth-teal_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "heightSource": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-teal/cloth-teal_height.png", "url": "cloth-teal_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.01, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-teal/cloth-teal_height.png", "url": "cloth-teal_height.png", "channel": "height", "source": "reference-pixel-extraction"}}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"strength": 0.5, "source": "cavity-bias at folds and under hats/ruff"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "diamond-seams", "region": "bloomers harlequin diamonds", "note": "darker teal diamond panels with purple stripe seams"}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "referencePbr": {"version": "1.0", "sourceImage": "/home/user/content-ai/jester3d/crops/cloth-teal.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.749, "estimatedFidelity": 0.749, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "maps": {"albedo": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-teal/cloth-teal_albedo.png", "url": "cloth-teal_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-teal/cloth-teal_roughness.png", "url": "cloth-teal_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-teal/cloth-teal_height.png", "url": "cloth-teal_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-teal/cloth-teal_normal.png", "url": "cloth-teal_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-teal/cloth-teal_ao.png", "url": "cloth-teal_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "diagnostics": {"sourceWidth": 216, "sourceHeight": 216, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 216, "height": 216}, "mask": {"backgroundColor": "#3C494E", "backgroundNoise": 67.654, "transparentPixelFraction": 0.0, "foregroundCoverage": 1.0}, "mapStats": {"valueRange": 0.2826, "heightP90Gradient": 0.01551, "roughnessBase": 0.686, "roughnessVariation": 0.05, "normalStrength": 0.174, "blurRadius": 21}, "palette": ["#469892", "#274A4F", "#408D8A", "#3A8180", "#2F6B6C"]}, "warnings": ["image is not clearly isolated from background; using most pixels as material evidence", "object/background separation is weak", "single-image inverse rendering cannot prove true physical PBR; confidence is capped"], "albedoNote": "albedo map flattened to the dominant cel-shaded fill colour (source is flat toon colour; crop texture would inject neighbouring colours)"}},
    options
  );
  materialMap["cloth-orange"] = createSculptMaterial(
    "cloth-orange",
    {"id": "cloth-orange", "name": "cloth-orange", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR approximation", "baseColor": "#f08a1c", "color": "#f08a1c", "albedo": {"dominant": "#D06E26", "secondary": ["#D16F1D", "#2F4F4F", "#EA8B31"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights.", "map": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-orange/cloth-orange_albedo.png", "url": "cloth-orange_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}}, "colorVariation": {"palette": ["#D06E26", "#D16F1D", "#2F4F4F", "#EA8B31", "#A3693F"], "pattern": "reference-derived pixel palette", "amplitude": 0.127, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.386, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.234, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.103, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.683, "variation": 0.1, "map": "independent-procedural-field", "localResponse": "slightly lower roughness on fold highlights, higher in creases"}, "metalness": 0.0, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.18, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-orange/cloth-orange_normal.png", "url": "cloth-orange_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "heightSource": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-orange/cloth-orange_height.png", "url": "cloth-orange_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.01, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-orange/cloth-orange_height.png", "url": "cloth-orange_height.png", "channel": "height", "source": "reference-pixel-extraction"}}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"strength": 0.5, "source": "cavity-bias at folds and under hats/ruff"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "reference-pbr-pixel-evidence", "type": "material-map-evidence", "evidenceRefs": ["full-object"], "channels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "notes": "Use generated maps as material evidence, then refine after browser screenshot comparison."}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "referencePbr": {"version": "1.0", "sourceImage": "/home/user/content-ai/jester3d/crops/cloth-orange.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.785, "estimatedFidelity": 0.785, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "maps": {"albedo": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-orange/cloth-orange_albedo.png", "url": "cloth-orange_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-orange/cloth-orange_roughness.png", "url": "cloth-orange_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-orange/cloth-orange_height.png", "url": "cloth-orange_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-orange/cloth-orange_normal.png", "url": "cloth-orange_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-orange/cloth-orange_ao.png", "url": "cloth-orange_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "diagnostics": {"sourceWidth": 216, "sourceHeight": 216, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 216, "height": 216}, "mask": {"backgroundColor": "#CE6E6A", "backgroundNoise": 158.155, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.9731}, "mapStats": {"valueRange": 0.3015, "heightP90Gradient": 0.02005, "roughnessBase": 0.683, "roughnessVariation": 0.05, "normalStrength": 0.18, "blurRadius": 21}, "palette": ["#D06E26", "#D16F1D", "#2F4F4F", "#EA8B31", "#A3693F"]}, "warnings": ["image is not clearly isolated from background; using most pixels as material evidence", "object/background separation is weak", "single-image inverse rendering cannot prove true physical PBR; confidence is capped"], "albedoNote": "albedo map flattened to the dominant cel-shaded fill colour (source is flat toon colour; crop texture would inject neighbouring colours)"}},
    options
  );
  materialMap["cloth-purple"] = createSculptMaterial(
    "cloth-purple",
    {"id": "cloth-purple", "name": "cloth-purple", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR approximation", "baseColor": "#6a3fa0", "color": "#6a3fa0", "albedo": {"dominant": "#7B568E", "secondary": ["#583E68", "#382534", "#9F6351"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights.", "map": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-purple/cloth-purple_albedo.png", "url": "cloth-purple_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}}, "colorVariation": {"palette": ["#7B568E", "#583E68", "#382534", "#9F6351", "#BF842C"], "pattern": "reference-derived pixel palette", "amplitude": 0.133, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.39, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.233, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.103, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.688, "variation": 0.1, "map": "independent-procedural-field", "localResponse": "slightly lower roughness on fold highlights, higher in creases"}, "metalness": 0.0, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.18, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-purple/cloth-purple_normal.png", "url": "cloth-purple_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "heightSource": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-purple/cloth-purple_height.png", "url": "cloth-purple_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.01, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-purple/cloth-purple_height.png", "url": "cloth-purple_height.png", "channel": "height", "source": "reference-pixel-extraction"}}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"strength": 0.5, "source": "cavity-bias at folds and under hats/ruff"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "star-moon-print", "region": "purple skirt/hat", "note": "orange star + crescent print motifs"}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "referencePbr": {"version": "1.0", "sourceImage": "/home/user/content-ai/jester3d/crops/cloth-purple.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.788, "estimatedFidelity": 0.788, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "maps": {"albedo": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-purple/cloth-purple_albedo.png", "url": "cloth-purple_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-purple/cloth-purple_roughness.png", "url": "cloth-purple_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-purple/cloth-purple_height.png", "url": "cloth-purple_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-purple/cloth-purple_normal.png", "url": "cloth-purple_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/home/user/content-ai/jester3d/material-evidence/cloth-purple/cloth-purple_ao.png", "url": "cloth-purple_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "diagnostics": {"sourceWidth": 216, "sourceHeight": 216, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 216, "height": 216}, "mask": {"backgroundColor": "#5D4171", "backgroundNoise": 103.542, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.9908}, "mapStats": {"valueRange": 0.3157, "heightP90Gradient": 0.01985, "roughnessBase": 0.688, "roughnessVariation": 0.05, "normalStrength": 0.18, "blurRadius": 21}, "palette": ["#7B568E", "#583E68", "#382534", "#9F6351", "#BF842C"]}, "warnings": ["image is not clearly isolated from background; using most pixels as material evidence", "object/background separation is weak", "single-image inverse rendering cannot prove true physical PBR; confidence is capped"], "albedoNote": "albedo map flattened to the dominant cel-shaded fill colour (source is flat toon colour; crop texture would inject neighbouring colours)"}},
    options
  );
  materialMap["gold-metal"] = createSculptMaterial(
    "gold-metal",
    {"id": "gold-metal", "name": "gold-metal", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR approximation", "baseColor": "#d9a520", "color": "#d9a520", "albedo": {"dominant": "#D99E41", "secondary": ["#9F671F", "#B27C37", "#49290D"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights.", "map": {"path": "/home/user/content-ai/jester3d/material-evidence/gold-metal/gold-metal_albedo.png", "url": "gold-metal_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}}, "colorVariation": {"palette": ["#D99E41", "#9F671F", "#B27C37", "#49290D", "#E1B66F"], "pattern": "reference-derived pixel palette", "amplitude": 0.248, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.486, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.176, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.07, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.684, "variation": 0.1, "map": "independent-procedural-field", "localResponse": "slightly lower roughness on fold highlights, higher in creases"}, "metalness": 1.0, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.163, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/gold-metal/gold-metal_normal.png", "url": "gold-metal_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "heightSource": {"path": "/home/user/content-ai/jester3d/material-evidence/gold-metal/gold-metal_height.png", "url": "gold-metal_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.01, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/gold-metal/gold-metal_height.png", "url": "gold-metal_height.png", "channel": "height", "source": "reference-pixel-extraction"}}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"strength": 0.5, "source": "cavity-bias at folds and under hats/ruff"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "bell-slot", "region": "bell lower slit", "roughness": 0.2, "note": "dark slit line on every bell"}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "referencePbr": {"version": "1.0", "sourceImage": "/home/user/content-ai/jester3d/crops/gold-metal.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.725, "estimatedFidelity": 0.725, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "maps": {"albedo": {"path": "/home/user/content-ai/jester3d/material-evidence/gold-metal/gold-metal_albedo.png", "url": "gold-metal_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/home/user/content-ai/jester3d/material-evidence/gold-metal/gold-metal_roughness.png", "url": "gold-metal_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/home/user/content-ai/jester3d/material-evidence/gold-metal/gold-metal_height.png", "url": "gold-metal_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/home/user/content-ai/jester3d/material-evidence/gold-metal/gold-metal_normal.png", "url": "gold-metal_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/home/user/content-ai/jester3d/material-evidence/gold-metal/gold-metal_ao.png", "url": "gold-metal_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "diagnostics": {"sourceWidth": 128, "sourceHeight": 128, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 128, "height": 128}, "mask": {"backgroundColor": "#E7D7C2", "backgroundNoise": 151.344, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.9756}, "mapStats": {"valueRange": 0.5897, "heightP90Gradient": 0.00618, "roughnessBase": 0.684, "roughnessVariation": 0.05, "normalStrength": 0.163, "blurRadius": 21}, "palette": ["#D99E41", "#9F671F", "#B27C37", "#49290D", "#E1B66F"]}, "warnings": ["image is not clearly isolated from background; using most pixels as material evidence", "object/background separation is weak", "single-image inverse rendering cannot prove true physical PBR; confidence is capped", "low high-frequency detail weakens normal/roughness inference"], "albedoNote": "albedo map flattened to the dominant cel-shaded fill colour (source is flat toon colour; crop texture would inject neighbouring colours)"}},
    options
  );
  materialMap["bone"] = createSculptMaterial(
    "bone",
    {"id": "bone", "name": "bone", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR approximation", "baseColor": "#e8dcc0", "color": "#e8dcc0", "albedo": {"dominant": "#E5D0B5", "secondary": ["#EEDAC2", "#271D18", "#C4AF98"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights.", "map": {"path": "/home/user/content-ai/jester3d/material-evidence/bone/bone_albedo.png", "url": "bone_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}}, "colorVariation": {"palette": ["#E5D0B5", "#EEDAC2", "#271D18", "#C4AF98", "#79695C"], "pattern": "reference-derived pixel palette", "amplitude": 0.333, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.52, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.25, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.112, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.687, "variation": 0.1, "map": "independent-procedural-field", "localResponse": "slightly lower roughness on fold highlights, higher in creases"}, "metalness": 0.0, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.184, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/bone/bone_normal.png", "url": "bone_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "heightSource": {"path": "/home/user/content-ai/jester3d/material-evidence/bone/bone_height.png", "url": "bone_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.011, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/bone/bone_height.png", "url": "bone_height.png", "channel": "height", "source": "reference-pixel-extraction"}}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"strength": 0.5, "source": "cavity-bias at folds and under hats/ruff"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "reference-pbr-pixel-evidence", "type": "material-map-evidence", "evidenceRefs": ["full-object"], "channels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "notes": "Use generated maps as material evidence, then refine after browser screenshot comparison."}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "referencePbr": {"version": "1.0", "sourceImage": "/home/user/content-ai/jester3d/crops/bone.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.86, "estimatedFidelity": 0.86, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "maps": {"albedo": {"path": "/home/user/content-ai/jester3d/material-evidence/bone/bone_albedo.png", "url": "bone_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/home/user/content-ai/jester3d/material-evidence/bone/bone_roughness.png", "url": "bone_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/home/user/content-ai/jester3d/material-evidence/bone/bone_height.png", "url": "bone_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/home/user/content-ai/jester3d/material-evidence/bone/bone_normal.png", "url": "bone_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/home/user/content-ai/jester3d/material-evidence/bone/bone_ao.png", "url": "bone_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "diagnostics": {"sourceWidth": 160, "sourceHeight": 160, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 160, "height": 160}, "mask": {"backgroundColor": "#DEC5AB", "backgroundNoise": 213.216, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.8625}, "mapStats": {"valueRange": 0.7937, "heightP90Gradient": 0.02377, "roughnessBase": 0.687, "roughnessVariation": 0.05, "normalStrength": 0.184, "blurRadius": 21}, "palette": ["#E5D0B5", "#EEDAC2", "#271D18", "#C4AF98", "#79695C"]}, "warnings": ["single-image inverse rendering cannot prove true physical PBR; confidence is capped"], "albedoNote": "albedo map flattened to the dominant cel-shaded fill colour (source is flat toon colour; crop texture would inject neighbouring colours)"}},
    options
  );
  materialMap["tongue-pink"] = createSculptMaterial(
    "tongue-pink",
    {"id": "tongue-pink", "name": "tongue-pink", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR approximation", "baseColor": "#e58a9a", "color": "#e58a9a", "albedo": {"dominant": "#DB8673", "secondary": ["#D37D6B", "#E1907C", "#A96052"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights.", "map": {"path": "/home/user/content-ai/jester3d/material-evidence/tongue-pink/tongue-pink_albedo.png", "url": "tongue-pink_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}}, "colorVariation": {"palette": ["#DB8673", "#D37D6B", "#E1907C", "#A96052", "#632E20"], "pattern": "reference-derived pixel palette", "amplitude": 0.131, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.389, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.196, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.081, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.684, "variation": 0.1, "map": "independent-procedural-field", "localResponse": "slightly lower roughness on fold highlights, higher in creases"}, "metalness": 0.0, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.169, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/tongue-pink/tongue-pink_normal.png", "url": "tongue-pink_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "heightSource": {"path": "/home/user/content-ai/jester3d/material-evidence/tongue-pink/tongue-pink_height.png", "url": "tongue-pink_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.01, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/tongue-pink/tongue-pink_height.png", "url": "tongue-pink_height.png", "channel": "height", "source": "reference-pixel-extraction"}}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"strength": 0.5, "source": "cavity-bias at folds and under hats/ruff"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "reference-pbr-pixel-evidence", "type": "material-map-evidence", "evidenceRefs": ["full-object"], "channels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "notes": "Use generated maps as material evidence, then refine after browser screenshot comparison."}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "referencePbr": {"version": "1.0", "sourceImage": "/home/user/content-ai/jester3d/crops/tongue-pink.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.718, "estimatedFidelity": 0.718, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "maps": {"albedo": {"path": "/home/user/content-ai/jester3d/material-evidence/tongue-pink/tongue-pink_albedo.png", "url": "tongue-pink_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/home/user/content-ai/jester3d/material-evidence/tongue-pink/tongue-pink_roughness.png", "url": "tongue-pink_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/home/user/content-ai/jester3d/material-evidence/tongue-pink/tongue-pink_height.png", "url": "tongue-pink_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/home/user/content-ai/jester3d/material-evidence/tongue-pink/tongue-pink_normal.png", "url": "tongue-pink_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/home/user/content-ai/jester3d/material-evidence/tongue-pink/tongue-pink_ao.png", "url": "tongue-pink_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "diagnostics": {"sourceWidth": 160, "sourceHeight": 160, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 160, "height": 160}, "mask": {"backgroundColor": "#8B4B3B", "backgroundNoise": 93.941, "transparentPixelFraction": 0.0, "foregroundCoverage": 1.0}, "mapStats": {"valueRange": 0.3128, "heightP90Gradient": 0.01088, "roughnessBase": 0.684, "roughnessVariation": 0.05, "normalStrength": 0.169, "blurRadius": 21}, "palette": ["#DB8673", "#D37D6B", "#E1907C", "#A96052", "#632E20"]}, "warnings": ["image is not clearly isolated from background; using most pixels as material evidence", "object/background separation is weak", "single-image inverse rendering cannot prove true physical PBR; confidence is capped"], "albedoNote": "albedo map flattened to the dominant cel-shaded fill colour (source is flat toon colour; crop texture would inject neighbouring colours)"}},
    options
  );
  materialMap["shoe-satin"] = createSculptMaterial(
    "shoe-satin",
    {"id": "shoe-satin", "name": "shoe-satin", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR approximation", "baseColor": "#1f8a86", "color": "#1f8a86", "albedo": {"dominant": "#316E74", "secondary": ["#397B7D", "#2C5457", "#122B2E"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights.", "map": {"path": "/home/user/content-ai/jester3d/material-evidence/shoe-satin/shoe-satin_albedo.png", "url": "shoe-satin_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}}, "colorVariation": {"palette": ["#316E74", "#397B7D", "#2C5457", "#122B2E", "#BD9549"], "pattern": "reference-derived pixel palette", "amplitude": 0.117, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.377, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.244, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.109, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.686, "variation": 0.1, "map": "independent-procedural-field", "localResponse": "slightly lower roughness on fold highlights, higher in creases"}, "metalness": 0.05, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.183, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/shoe-satin/shoe-satin_normal.png", "url": "shoe-satin_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "heightSource": {"path": "/home/user/content-ai/jester3d/material-evidence/shoe-satin/shoe-satin_height.png", "url": "shoe-satin_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.01, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/shoe-satin/shoe-satin_height.png", "url": "shoe-satin_height.png", "channel": "height", "source": "reference-pixel-extraction"}}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"strength": 0.5, "source": "cavity-bias at folds and under hats/ruff"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "reference-pbr-pixel-evidence", "type": "material-map-evidence", "evidenceRefs": ["full-object"], "channels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "notes": "Use generated maps as material evidence, then refine after browser screenshot comparison."}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "referencePbr": {"version": "1.0", "sourceImage": "/home/user/content-ai/jester3d/crops/shoe-satin.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.779, "estimatedFidelity": 0.779, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "maps": {"albedo": {"path": "/home/user/content-ai/jester3d/material-evidence/shoe-satin/shoe-satin_albedo.png", "url": "shoe-satin_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/home/user/content-ai/jester3d/material-evidence/shoe-satin/shoe-satin_roughness.png", "url": "shoe-satin_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/home/user/content-ai/jester3d/material-evidence/shoe-satin/shoe-satin_height.png", "url": "shoe-satin_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/home/user/content-ai/jester3d/material-evidence/shoe-satin/shoe-satin_normal.png", "url": "shoe-satin_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/home/user/content-ai/jester3d/material-evidence/shoe-satin/shoe-satin_ao.png", "url": "shoe-satin_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "diagnostics": {"sourceWidth": 216, "sourceHeight": 216, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 216, "height": 216}, "mask": {"backgroundColor": "#C39363", "backgroundNoise": 165.415, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.9607}, "mapStats": {"valueRange": 0.2776, "heightP90Gradient": 0.02246, "roughnessBase": 0.686, "roughnessVariation": 0.05, "normalStrength": 0.183, "blurRadius": 21}, "palette": ["#316E74", "#397B7D", "#2C5457", "#122B2E", "#BD9549"]}, "warnings": ["image is not clearly isolated from background; using most pixels as material evidence", "object/background separation is weak", "single-image inverse rendering cannot prove true physical PBR; confidence is capped"], "albedoNote": "albedo map flattened to the dominant cel-shaded fill colour (source is flat toon colour; crop texture would inject neighbouring colours)"}},
    options
  );
  materialMap["ink-white"] = createSculptMaterial(
    "ink-white",
    {"id": "ink-white", "name": "ink-white", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR approximation", "baseColor": "#ffffff", "color": "#ffffff", "albedo": {"dominant": "#383231", "secondary": ["#EEDDC4", "#EAD9C1", "#302A28"], "samplingNotes": "Reference-derived from foreground pixels; de-lit to reduce baked shadows/highlights.", "map": {"path": "/home/user/content-ai/jester3d/material-evidence/ink-white/ink-white_albedo.png", "url": "ink-white_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}}, "colorVariation": {"palette": ["#383231", "#EEDDC4", "#EAD9C1", "#302A28", "#83786D"], "pattern": "reference-derived pixel palette", "amplitude": 0.299, "heightCorrelation": 0.42}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.52, "role": "reference-derived broad albedo and height breakup"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.25, "role": "reference-derived cracks, ridges, pores, grain, or leaf clusters"}, {"id": "micro", "frequency": 72.0, "amplitude": 0.112, "role": "reference-derived micro highlight breakup under grazing light"}], "roughness": {"base": 0.682, "variation": 0.1, "map": "independent-procedural-field", "localResponse": "slightly lower roughness on fold highlights, higher in creases"}, "metalness": 0.0, "normal": {"pattern": "reference-derived height-gradient normal map", "strength": 0.184, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/ink-white/ink-white_normal.png", "url": "ink-white_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "heightSource": {"path": "/home/user/content-ai/jester3d/material-evidence/ink-white/ink-white_height.png", "url": "ink-white_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "space": "tangent"}, "bump": {"pattern": "reference-derived height field", "amplitude": 0.011, "map": {"path": "/home/user/content-ai/jester3d/material-evidence/ink-white/ink-white_height.png", "url": "ink-white_height.png", "channel": "height", "source": "reference-pixel-extraction"}}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"strength": 0.5, "source": "cavity-bias at folds and under hats/ruff"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "reference-pbr-pixel-evidence", "type": "material-map-evidence", "evidenceRefs": ["full-object"], "channels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "notes": "Use generated maps as material evidence, then refine after browser screenshot comparison."}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there.", "Reference-derived maps are estimates from image pixels; verify with neutral, grazing, and reference-matched renders.", "Do not treat baked image shadows as final albedo; rerun extraction with a tighter material crop if highlights/shadows pollute the maps."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "referencePbr": {"version": "1.0", "sourceImage": "/home/user/content-ai/jester3d/crops/ink-white.png", "extractor": "stage1_intake/extract_pbr_evidence.py", "method": "single-image pixel evidence with de-lighting estimate; not photogrammetry", "usable": true, "verdict": "pass", "confidence": 0.86, "estimatedFidelity": 0.86, "targetThreshold": 0.7, "hardLimit": "A single image cannot uniquely recover true albedo/roughness/normal/AO; maps are reference-derived estimates.", "maps": {"albedo": {"path": "/home/user/content-ai/jester3d/material-evidence/ink-white/ink-white_albedo.png", "url": "ink-white_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/home/user/content-ai/jester3d/material-evidence/ink-white/ink-white_roughness.png", "url": "ink-white_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/home/user/content-ai/jester3d/material-evidence/ink-white/ink-white_height.png", "url": "ink-white_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/home/user/content-ai/jester3d/material-evidence/ink-white/ink-white_normal.png", "url": "ink-white_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/home/user/content-ai/jester3d/material-evidence/ink-white/ink-white_ao.png", "url": "ink-white_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "diagnostics": {"sourceWidth": 112, "sourceHeight": 144, "mapSize": 1024, "cropBBoxPixels": {"x": 0, "y": 0, "width": 112, "height": 144}, "mask": {"backgroundColor": "#EDDFC3", "backgroundNoise": 46.433, "transparentPixelFraction": 0.0, "foregroundCoverage": 0.8412}, "mapStats": {"valueRange": 0.7108, "heightP90Gradient": 0.02391, "roughnessBase": 0.682, "roughnessVariation": 0.05, "normalStrength": 0.184, "blurRadius": 21}, "palette": ["#383231", "#EEDDC4", "#EAD9C1", "#302A28", "#83786D"]}, "warnings": ["single-image inverse rendering cannot prove true physical PBR; confidence is capped"], "albedoNote": "albedo map flattened to the dominant cel-shaded fill colour (source is flat toon colour; crop texture would inject neighbouring colours)"}},
    options
  );

  const nodes: Record<string, THREE.Object3D> = { root };
  const meshes: Record<string, THREE.Mesh> = {};
  const sockets: Record<string, THREE.Object3D> = {};
  const colliders: Record<string, unknown> = {};
  const destructionGroups: Record<string, THREE.Object3D[]> = {};

  const endpoint_root_0 = makeAttachmentEndpoint(null);
  const node_root_0 = new THREE.Group();
  node_root_0.name = "Mouse Jester__pivot";
  node_root_0.scale.set(1, 1, 1);
  if (endpoint_root_0) {
    node_root_0.position.copy(endpoint_root_0.start);
    node_root_0.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_root_0.position.set(0.0, 0.0, 0.0);
    node_root_0.rotation.set(0.0, 0.0, 0.0);
  }
  node_root_0.userData.sculptComponent = {"id": "root", "name": "Mouse Jester", "level": "macro", "role": "root", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": null, "attachment": null, "dimensions": {"width": 0.1, "height": 0.1, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "root", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_root_0.userData.actionProfile = {"animationRole": "root", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["root"] ?? root).add(node_root_0);
  nodes["root"] = node_root_0;
  const mesh_root_0Geometry = endpoint_root_0
    ? new THREE.CylinderGeometry(endpoint_root_0.endRadius, endpoint_root_0.baseRadius, endpoint_root_0.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_root_0) {
    mesh_root_0Geometry.scale(0.1, 0.1, 0.1);
  }
  const mesh_root_0 = new THREE.Mesh(
    mesh_root_0Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_root_0.name = "Mouse Jester";
  if (endpoint_root_0) {
    mesh_root_0.position.copy(endpoint_root_0.midpoint);
    mesh_root_0.quaternion.copy(endpoint_root_0.quaternion);
  }
  mesh_root_0.castShadow = options.castShadow ?? true;
  mesh_root_0.receiveShadow = options.receiveShadow ?? true;
  mesh_root_0.userData.sculptComponent = {"id": "root", "name": "Mouse Jester", "level": "macro", "role": "root", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": null, "attachment": null, "dimensions": {"width": 0.1, "height": 0.1, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "root", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_root_0.add(mesh_root_0);
  meshes["root"] = mesh_root_0;
  colliders["root"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_root_0);

  const endpoint_torso_1 = makeAttachmentEndpoint(null);
  const node_torso_1 = new THREE.Group();
  node_torso_1.name = "Torso__pivot";
  node_torso_1.scale.set(1, 1, 1);
  if (endpoint_torso_1) {
    node_torso_1.position.copy(endpoint_torso_1.start);
    node_torso_1.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_torso_1.position.set(0.0, 2.25, 0.0);
    node_torso_1.rotation.set(-0.0, 0.0, -0.0);
  }
  node_torso_1.userData.sculptComponent = {"id": "torso", "name": "Torso", "level": "macro", "role": "body", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.8, "height": 1.2, "depth": 0.6, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 2.25, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "body", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_torso_1.userData.actionProfile = {"animationRole": "body", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["root"] ?? root).add(node_torso_1);
  nodes["torso"] = node_torso_1;
  const mesh_torso_1Geometry = endpoint_torso_1
    ? new THREE.CylinderGeometry(endpoint_torso_1.endRadius, endpoint_torso_1.baseRadius, endpoint_torso_1.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_torso_1) {
    mesh_torso_1Geometry.scale(0.8, 1.2, 0.6);
  }
  const mesh_torso_1 = new THREE.Mesh(
    mesh_torso_1Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_torso_1.name = "Torso";
  if (endpoint_torso_1) {
    mesh_torso_1.position.copy(endpoint_torso_1.midpoint);
    mesh_torso_1.quaternion.copy(endpoint_torso_1.quaternion);
  }
  mesh_torso_1.castShadow = options.castShadow ?? true;
  mesh_torso_1.receiveShadow = options.receiveShadow ?? true;
  mesh_torso_1.userData.sculptComponent = {"id": "torso", "name": "Torso", "level": "macro", "role": "body", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.8, "height": 1.2, "depth": 0.6, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 2.25, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "body", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_torso_1.add(mesh_torso_1);
  meshes["torso"] = mesh_torso_1;
  colliders["torso"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_torso_1);

  const endpoint_ruff_2 = makeAttachmentEndpoint(null);
  const node_ruff_2 = new THREE.Group();
  node_ruff_2.name = "Ruff collar__pivot";
  node_ruff_2.scale.set(1, 1, 1);
  if (endpoint_ruff_2) {
    node_ruff_2.position.copy(endpoint_ruff_2.start);
    node_ruff_2.rotation.set(1.5708, 0.0, -0.0);
  } else {
    node_ruff_2.position.set(0.0, 0.25, 0.02);
    node_ruff_2.rotation.set(1.5708, 0.0, -0.0);
  }
  node_ruff_2.userData.sculptComponent = {"id": "ruff", "name": "Ruff collar", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 0.3, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.25, 0.02], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-teal", "materialLayers": ["cloth-teal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "ruff-scallops", "kind": "ridge", "note": "scalloped orange-edged ruff points"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_ruff_2.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_ruff_2);
  nodes["ruff"] = node_ruff_2;
  const mesh_ruff_2Geometry = endpoint_ruff_2
    ? new THREE.CylinderGeometry(endpoint_ruff_2.endRadius, endpoint_ruff_2.baseRadius, endpoint_ruff_2.length, 32, 12)
    : new THREE.TorusGeometry(0.45, 0.08, 24, 96);
  if (!endpoint_ruff_2) {
    mesh_ruff_2Geometry.scale(1.0, 1.0, 0.3);
  }
  const mesh_ruff_2 = new THREE.Mesh(
    mesh_ruff_2Geometry,
    materialMap["cloth-teal"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_ruff_2.name = "Ruff collar";
  if (endpoint_ruff_2) {
    mesh_ruff_2.position.copy(endpoint_ruff_2.midpoint);
    mesh_ruff_2.quaternion.copy(endpoint_ruff_2.quaternion);
  }
  mesh_ruff_2.castShadow = options.castShadow ?? true;
  mesh_ruff_2.receiveShadow = options.receiveShadow ?? true;
  mesh_ruff_2.userData.sculptComponent = {"id": "ruff", "name": "Ruff collar", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 0.3, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.25, 0.02], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-teal", "materialLayers": ["cloth-teal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "ruff-scallops", "kind": "ridge", "note": "scalloped orange-edged ruff points"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_ruff_2.add(mesh_ruff_2);
  meshes["ruff"] = mesh_ruff_2;
  colliders["ruff"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_ruff_2);

  const endpoint_ruff_trim_3 = makeAttachmentEndpoint(null);
  const node_ruff_trim_3 = new THREE.Group();
  node_ruff_trim_3.name = "Ruff orange trim__pivot";
  node_ruff_trim_3.scale.set(1, 1, 1);
  if (endpoint_ruff_trim_3) {
    node_ruff_trim_3.position.copy(endpoint_ruff_trim_3.start);
    node_ruff_trim_3.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_ruff_trim_3.position.set(0.0, 0.04, 0.0);
    node_ruff_trim_3.rotation.set(-0.0, 0.0, -0.0);
  }
  node_ruff_trim_3.userData.sculptComponent = {"id": "ruff-trim", "name": "Ruff orange trim", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "ruff", "attachment": {"parentSocket": "ruff-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 1.1, "height": 1.1, "depth": 0.12, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.04, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_ruff_trim_3.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["ruff"] ?? root).add(node_ruff_trim_3);
  nodes["ruff-trim"] = node_ruff_trim_3;
  const mesh_ruff_trim_3Geometry = endpoint_ruff_trim_3
    ? new THREE.CylinderGeometry(endpoint_ruff_trim_3.endRadius, endpoint_ruff_trim_3.baseRadius, endpoint_ruff_trim_3.length, 32, 12)
    : new THREE.TorusGeometry(0.45, 0.08, 24, 96);
  if (!endpoint_ruff_trim_3) {
    mesh_ruff_trim_3Geometry.scale(1.1, 1.1, 0.12);
  }
  const mesh_ruff_trim_3 = new THREE.Mesh(
    mesh_ruff_trim_3Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_ruff_trim_3.name = "Ruff orange trim";
  if (endpoint_ruff_trim_3) {
    mesh_ruff_trim_3.position.copy(endpoint_ruff_trim_3.midpoint);
    mesh_ruff_trim_3.quaternion.copy(endpoint_ruff_trim_3.quaternion);
  }
  mesh_ruff_trim_3.castShadow = options.castShadow ?? true;
  mesh_ruff_trim_3.receiveShadow = options.receiveShadow ?? true;
  mesh_ruff_trim_3.userData.sculptComponent = {"id": "ruff-trim", "name": "Ruff orange trim", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "ruff", "attachment": {"parentSocket": "ruff-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 1.1, "height": 1.1, "depth": 0.12, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.04, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_ruff_trim_3.add(mesh_ruff_trim_3);
  meshes["ruff-trim"] = mesh_ruff_trim_3;
  colliders["ruff-trim"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_ruff_trim_3);

  const attachment_bowtie_a_4 = {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_bowtie_a_4 = makeAttachmentEndpoint(attachment_bowtie_a_4);
  const node_bowtie_a_4 = new THREE.Group();
  node_bowtie_a_4.name = "Bow-tie wing A__pivot";
  node_bowtie_a_4.scale.set(1, 1, 1);
  if (endpoint_bowtie_a_4) {
    node_bowtie_a_4.position.copy(endpoint_bowtie_a_4.start);
    node_bowtie_a_4.rotation.set(-0.0, 0.0, 1.5708);
  } else {
    node_bowtie_a_4.position.set(-0.22, -0.28, 0.32);
    node_bowtie_a_4.rotation.set(-0.0, 0.0, 1.5708);
  }
  node_bowtie_a_4.userData.sculptComponent = {"id": "bowtie-a", "name": "Bow-tie wing A", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cone", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.38, "depth": 0.3, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.22, -0.28, 0.32], "rotation": [-0.0, 0.0, 1.5708]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_bowtie_a_4.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_bowtie_a_4);
  nodes["bowtie-a"] = node_bowtie_a_4;
  const mesh_bowtie_a_4Geometry = endpoint_bowtie_a_4
    ? new THREE.CylinderGeometry(endpoint_bowtie_a_4.endRadius, endpoint_bowtie_a_4.baseRadius, endpoint_bowtie_a_4.length, 32, 12)
    : new THREE.ConeGeometry(0.5, 1, 48, 1);
  if (!endpoint_bowtie_a_4) {
    mesh_bowtie_a_4Geometry.scale(0.3, 0.38, 0.3);
  }
  const mesh_bowtie_a_4 = new THREE.Mesh(
    mesh_bowtie_a_4Geometry,
    materialMap["cloth-purple"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_bowtie_a_4.name = "Bow-tie wing A";
  if (endpoint_bowtie_a_4) {
    mesh_bowtie_a_4.position.copy(endpoint_bowtie_a_4.midpoint);
    mesh_bowtie_a_4.quaternion.copy(endpoint_bowtie_a_4.quaternion);
  }
  mesh_bowtie_a_4.castShadow = options.castShadow ?? true;
  mesh_bowtie_a_4.receiveShadow = options.receiveShadow ?? true;
  mesh_bowtie_a_4.userData.sculptComponent = {"id": "bowtie-a", "name": "Bow-tie wing A", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cone", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.38, "depth": 0.3, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.22, -0.28, 0.32], "rotation": [-0.0, 0.0, 1.5708]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_bowtie_a_4.add(mesh_bowtie_a_4);
  meshes["bowtie-a"] = mesh_bowtie_a_4;
  colliders["bowtie-a"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_bowtie_a_4);

  const attachment_bowtie_b_5 = {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_bowtie_b_5 = makeAttachmentEndpoint(attachment_bowtie_b_5);
  const node_bowtie_b_5 = new THREE.Group();
  node_bowtie_b_5.name = "Bow-tie wing B__pivot";
  node_bowtie_b_5.scale.set(1, 1, 1);
  if (endpoint_bowtie_b_5) {
    node_bowtie_b_5.position.copy(endpoint_bowtie_b_5.start);
    node_bowtie_b_5.rotation.set(-0.0, 0.0, -1.5708);
  } else {
    node_bowtie_b_5.position.set(0.22, -0.28, 0.32);
    node_bowtie_b_5.rotation.set(-0.0, 0.0, -1.5708);
  }
  node_bowtie_b_5.userData.sculptComponent = {"id": "bowtie-b", "name": "Bow-tie wing B", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cone", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.38, "depth": 0.3, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.22, -0.28, 0.32], "rotation": [-0.0, 0.0, -1.5708]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_bowtie_b_5.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_bowtie_b_5);
  nodes["bowtie-b"] = node_bowtie_b_5;
  const mesh_bowtie_b_5Geometry = endpoint_bowtie_b_5
    ? new THREE.CylinderGeometry(endpoint_bowtie_b_5.endRadius, endpoint_bowtie_b_5.baseRadius, endpoint_bowtie_b_5.length, 32, 12)
    : new THREE.ConeGeometry(0.5, 1, 48, 1);
  if (!endpoint_bowtie_b_5) {
    mesh_bowtie_b_5Geometry.scale(0.3, 0.38, 0.3);
  }
  const mesh_bowtie_b_5 = new THREE.Mesh(
    mesh_bowtie_b_5Geometry,
    materialMap["cloth-purple"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_bowtie_b_5.name = "Bow-tie wing B";
  if (endpoint_bowtie_b_5) {
    mesh_bowtie_b_5.position.copy(endpoint_bowtie_b_5.midpoint);
    mesh_bowtie_b_5.quaternion.copy(endpoint_bowtie_b_5.quaternion);
  }
  mesh_bowtie_b_5.castShadow = options.castShadow ?? true;
  mesh_bowtie_b_5.receiveShadow = options.receiveShadow ?? true;
  mesh_bowtie_b_5.userData.sculptComponent = {"id": "bowtie-b", "name": "Bow-tie wing B", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cone", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.38, "depth": 0.3, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.22, -0.28, 0.32], "rotation": [-0.0, 0.0, -1.5708]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_bowtie_b_5.add(mesh_bowtie_b_5);
  meshes["bowtie-b"] = mesh_bowtie_b_5;
  colliders["bowtie-b"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_bowtie_b_5);

  const endpoint_skull_6 = makeAttachmentEndpoint(null);
  const node_skull_6 = new THREE.Group();
  node_skull_6.name = "Skull charm__pivot";
  node_skull_6.scale.set(1, 1, 1);
  if (endpoint_skull_6) {
    node_skull_6.position.copy(endpoint_skull_6.start);
    node_skull_6.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_skull_6.position.set(0.0, -0.28, 0.38);
    node_skull_6.rotation.set(-0.0, 0.0, -0.0);
  }
  node_skull_6.userData.sculptComponent = {"id": "skull", "name": "Skull charm", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.18, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.28, 0.38], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "bone", "materialLayers": ["bone"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "skull-eyes", "kind": "hole", "note": "two dark eye sockets + nose triangle on bone skull"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(232, 220, 192, 1.0)", "secondaryAlbedo": "rgba(232, 220, 192, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_skull_6.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_skull_6);
  nodes["skull"] = node_skull_6;
  const mesh_skull_6Geometry = endpoint_skull_6
    ? new THREE.CylinderGeometry(endpoint_skull_6.endRadius, endpoint_skull_6.baseRadius, endpoint_skull_6.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_skull_6) {
    mesh_skull_6Geometry.scale(0.2, 0.2, 0.18);
  }
  const mesh_skull_6 = new THREE.Mesh(
    mesh_skull_6Geometry,
    materialMap["bone"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_skull_6.name = "Skull charm";
  if (endpoint_skull_6) {
    mesh_skull_6.position.copy(endpoint_skull_6.midpoint);
    mesh_skull_6.quaternion.copy(endpoint_skull_6.quaternion);
  }
  mesh_skull_6.castShadow = options.castShadow ?? true;
  mesh_skull_6.receiveShadow = options.receiveShadow ?? true;
  mesh_skull_6.userData.sculptComponent = {"id": "skull", "name": "Skull charm", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.18, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.28, 0.38], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "bone", "materialLayers": ["bone"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "skull-eyes", "kind": "hole", "note": "two dark eye sockets + nose triangle on bone skull"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(232, 220, 192, 1.0)", "secondaryAlbedo": "rgba(232, 220, 192, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_skull_6.add(mesh_skull_6);
  meshes["skull"] = mesh_skull_6;
  colliders["skull"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_skull_6);

  const endpoint_bloomers_7 = makeAttachmentEndpoint(null);
  const node_bloomers_7 = new THREE.Group();
  node_bloomers_7.name = "Harlequin bloomers__pivot";
  node_bloomers_7.scale.set(1, 1, 1);
  if (endpoint_bloomers_7) {
    node_bloomers_7.position.copy(endpoint_bloomers_7.start);
    node_bloomers_7.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_bloomers_7.position.set(0.0, 1.3, 0.0);
    node_bloomers_7.rotation.set(-0.0, 0.0, -0.0);
  }
  node_bloomers_7.userData.sculptComponent = {"id": "bloomers", "name": "Harlequin bloomers", "level": "macro", "role": "body", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 1.5, "height": 1.25, "depth": 0.95, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 1.3, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "body", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-teal", "materialLayers": ["cloth-teal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "harlequin-diamonds", "kind": "linework", "note": "teal diamond panels edged with purple/orange stripes"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_bloomers_7.userData.actionProfile = {"animationRole": "body", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["root"] ?? root).add(node_bloomers_7);
  nodes["bloomers"] = node_bloomers_7;
  const mesh_bloomers_7Geometry = endpoint_bloomers_7
    ? new THREE.CylinderGeometry(endpoint_bloomers_7.endRadius, endpoint_bloomers_7.baseRadius, endpoint_bloomers_7.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_bloomers_7) {
    mesh_bloomers_7Geometry.scale(1.5, 1.25, 0.95);
  }
  const mesh_bloomers_7 = new THREE.Mesh(
    mesh_bloomers_7Geometry,
    materialMap["cloth-teal"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_bloomers_7.name = "Harlequin bloomers";
  if (endpoint_bloomers_7) {
    mesh_bloomers_7.position.copy(endpoint_bloomers_7.midpoint);
    mesh_bloomers_7.quaternion.copy(endpoint_bloomers_7.quaternion);
  }
  mesh_bloomers_7.castShadow = options.castShadow ?? true;
  mesh_bloomers_7.receiveShadow = options.receiveShadow ?? true;
  mesh_bloomers_7.userData.sculptComponent = {"id": "bloomers", "name": "Harlequin bloomers", "level": "macro", "role": "body", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 1.5, "height": 1.25, "depth": 0.95, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 1.3, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "body", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-teal", "materialLayers": ["cloth-teal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "harlequin-diamonds", "kind": "linework", "note": "teal diamond panels edged with purple/orange stripes"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_bloomers_7.add(mesh_bloomers_7);
  meshes["bloomers"] = mesh_bloomers_7;
  colliders["bloomers"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_bloomers_7);

  const attachment_belt_band_8 = {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_belt_band_8 = makeAttachmentEndpoint(attachment_belt_band_8);
  const node_belt_band_8 = new THREE.Group();
  node_belt_band_8.name = "Waist band__pivot";
  node_belt_band_8.scale.set(1, 1, 1);
  if (endpoint_belt_band_8) {
    node_belt_band_8.position.copy(endpoint_belt_band_8.start);
    node_belt_band_8.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_belt_band_8.position.set(0.0, 0.55, 0.0);
    node_belt_band_8.rotation.set(-0.0, 0.0, -0.0);
  }
  node_belt_band_8.userData.sculptComponent = {"id": "belt-band", "name": "Waist band", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "bloomers", "attachment": {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.9, "height": 0.12, "depth": 0.6, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.55, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_belt_band_8.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["bloomers"] ?? root).add(node_belt_band_8);
  nodes["belt-band"] = node_belt_band_8;
  const mesh_belt_band_8Geometry = endpoint_belt_band_8
    ? new THREE.CylinderGeometry(endpoint_belt_band_8.endRadius, endpoint_belt_band_8.baseRadius, endpoint_belt_band_8.length, 32, 12)
    : new THREE.CylinderGeometry(0.5, 0.5, 1, 48, 16);
  if (!endpoint_belt_band_8) {
    mesh_belt_band_8Geometry.scale(0.9, 0.12, 0.6);
  }
  const mesh_belt_band_8 = new THREE.Mesh(
    mesh_belt_band_8Geometry,
    materialMap["cloth-purple"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_belt_band_8.name = "Waist band";
  if (endpoint_belt_band_8) {
    mesh_belt_band_8.position.copy(endpoint_belt_band_8.midpoint);
    mesh_belt_band_8.quaternion.copy(endpoint_belt_band_8.quaternion);
  }
  mesh_belt_band_8.castShadow = options.castShadow ?? true;
  mesh_belt_band_8.receiveShadow = options.receiveShadow ?? true;
  mesh_belt_band_8.userData.sculptComponent = {"id": "belt-band", "name": "Waist band", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "bloomers", "attachment": {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.9, "height": 0.12, "depth": 0.6, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.55, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_belt_band_8.add(mesh_belt_band_8);
  meshes["belt-band"] = mesh_belt_band_8;
  colliders["belt-band"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_belt_band_8);

  const endpoint_suspender_a_9 = makeAttachmentEndpoint(null);
  const node_suspender_a_9 = new THREE.Group();
  node_suspender_a_9.name = "Suspender A__pivot";
  node_suspender_a_9.scale.set(1, 1, 1);
  if (endpoint_suspender_a_9) {
    node_suspender_a_9.position.copy(endpoint_suspender_a_9.start);
    node_suspender_a_9.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_suspender_a_9.position.set(-0.18, -0.1, 0.34);
    node_suspender_a_9.rotation.set(-0.0, 0.0, -0.0);
  }
  node_suspender_a_9.userData.sculptComponent = {"id": "suspender-a", "name": "Suspender A", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "box", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.7, "depth": 0.04, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.18, -0.1, 0.34], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_suspender_a_9.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_suspender_a_9);
  nodes["suspender-a"] = node_suspender_a_9;
  const mesh_suspender_a_9Geometry = endpoint_suspender_a_9
    ? new THREE.CylinderGeometry(endpoint_suspender_a_9.endRadius, endpoint_suspender_a_9.baseRadius, endpoint_suspender_a_9.length, 32, 12)
    : new THREE.BoxGeometry(1, 1, 1, 12, 12, 12);
  if (!endpoint_suspender_a_9) {
    mesh_suspender_a_9Geometry.scale(0.07, 0.7, 0.04);
  }
  const mesh_suspender_a_9 = new THREE.Mesh(
    mesh_suspender_a_9Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_suspender_a_9.name = "Suspender A";
  if (endpoint_suspender_a_9) {
    mesh_suspender_a_9.position.copy(endpoint_suspender_a_9.midpoint);
    mesh_suspender_a_9.quaternion.copy(endpoint_suspender_a_9.quaternion);
  }
  mesh_suspender_a_9.castShadow = options.castShadow ?? true;
  mesh_suspender_a_9.receiveShadow = options.receiveShadow ?? true;
  mesh_suspender_a_9.userData.sculptComponent = {"id": "suspender-a", "name": "Suspender A", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "box", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.7, "depth": 0.04, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.18, -0.1, 0.34], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_suspender_a_9.add(mesh_suspender_a_9);
  meshes["suspender-a"] = mesh_suspender_a_9;
  colliders["suspender-a"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_suspender_a_9);

  const endpoint_suspender_b_10 = makeAttachmentEndpoint(null);
  const node_suspender_b_10 = new THREE.Group();
  node_suspender_b_10.name = "Suspender B__pivot";
  node_suspender_b_10.scale.set(1, 1, 1);
  if (endpoint_suspender_b_10) {
    node_suspender_b_10.position.copy(endpoint_suspender_b_10.start);
    node_suspender_b_10.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_suspender_b_10.position.set(0.18, -0.1, 0.34);
    node_suspender_b_10.rotation.set(-0.0, 0.0, -0.0);
  }
  node_suspender_b_10.userData.sculptComponent = {"id": "suspender-b", "name": "Suspender B", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "box", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.7, "depth": 0.04, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.18, -0.1, 0.34], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_suspender_b_10.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_suspender_b_10);
  nodes["suspender-b"] = node_suspender_b_10;
  const mesh_suspender_b_10Geometry = endpoint_suspender_b_10
    ? new THREE.CylinderGeometry(endpoint_suspender_b_10.endRadius, endpoint_suspender_b_10.baseRadius, endpoint_suspender_b_10.length, 32, 12)
    : new THREE.BoxGeometry(1, 1, 1, 12, 12, 12);
  if (!endpoint_suspender_b_10) {
    mesh_suspender_b_10Geometry.scale(0.07, 0.7, 0.04);
  }
  const mesh_suspender_b_10 = new THREE.Mesh(
    mesh_suspender_b_10Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_suspender_b_10.name = "Suspender B";
  if (endpoint_suspender_b_10) {
    mesh_suspender_b_10.position.copy(endpoint_suspender_b_10.midpoint);
    mesh_suspender_b_10.quaternion.copy(endpoint_suspender_b_10.quaternion);
  }
  mesh_suspender_b_10.castShadow = options.castShadow ?? true;
  mesh_suspender_b_10.receiveShadow = options.receiveShadow ?? true;
  mesh_suspender_b_10.userData.sculptComponent = {"id": "suspender-b", "name": "Suspender B", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "box", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.7, "depth": 0.04, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.18, -0.1, 0.34], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_suspender_b_10.add(mesh_suspender_b_10);
  meshes["suspender-b"] = mesh_suspender_b_10;
  colliders["suspender-b"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_suspender_b_10);

  const attachment_xbutton_R_11 = {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_xbutton_R_11 = makeAttachmentEndpoint(attachment_xbutton_R_11);
  const node_xbutton_R_11 = new THREE.Group();
  node_xbutton_R_11.name = "X button R__pivot";
  node_xbutton_R_11.scale.set(1, 1, 1);
  if (endpoint_xbutton_R_11) {
    node_xbutton_R_11.position.copy(endpoint_xbutton_R_11.start);
    node_xbutton_R_11.rotation.set(1.5708, 0.0, -0.0);
  } else {
    node_xbutton_R_11.position.set(-0.18, 0.48, 0.43);
    node_xbutton_R_11.rotation.set(1.5708, 0.0, -0.0);
  }
  node_xbutton_R_11.userData.sculptComponent = {"id": "xbutton-R", "name": "X button R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "bloomers", "attachment": {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.14, "height": 0.03, "depth": 0.14, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.18, 0.48, 0.43], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_xbutton_R_11.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["bloomers"] ?? root).add(node_xbutton_R_11);
  nodes["xbutton-R"] = node_xbutton_R_11;
  const mesh_xbutton_R_11Geometry = endpoint_xbutton_R_11
    ? new THREE.CylinderGeometry(endpoint_xbutton_R_11.endRadius, endpoint_xbutton_R_11.baseRadius, endpoint_xbutton_R_11.length, 32, 12)
    : new THREE.CylinderGeometry(0.5, 0.5, 1, 48, 16);
  if (!endpoint_xbutton_R_11) {
    mesh_xbutton_R_11Geometry.scale(0.14, 0.03, 0.14);
  }
  const mesh_xbutton_R_11 = new THREE.Mesh(
    mesh_xbutton_R_11Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_xbutton_R_11.name = "X button R";
  if (endpoint_xbutton_R_11) {
    mesh_xbutton_R_11.position.copy(endpoint_xbutton_R_11.midpoint);
    mesh_xbutton_R_11.quaternion.copy(endpoint_xbutton_R_11.quaternion);
  }
  mesh_xbutton_R_11.castShadow = options.castShadow ?? true;
  mesh_xbutton_R_11.receiveShadow = options.receiveShadow ?? true;
  mesh_xbutton_R_11.userData.sculptComponent = {"id": "xbutton-R", "name": "X button R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "bloomers", "attachment": {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.14, "height": 0.03, "depth": 0.14, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.18, 0.48, 0.43], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_xbutton_R_11.add(mesh_xbutton_R_11);
  meshes["xbutton-R"] = mesh_xbutton_R_11;
  colliders["xbutton-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_xbutton_R_11);

  const endpoint_diamond_R_12 = makeAttachmentEndpoint(null);
  const node_diamond_R_12 = new THREE.Group();
  node_diamond_R_12.name = "Diamond print R__pivot";
  node_diamond_R_12.scale.set(1, 1, 1);
  if (endpoint_diamond_R_12) {
    node_diamond_R_12.position.copy(endpoint_diamond_R_12.start);
    node_diamond_R_12.rotation.set(-0.0, 0.0, 0.7854);
  } else {
    node_diamond_R_12.position.set(-0.35, -0.3, 0.48);
    node_diamond_R_12.rotation.set(-0.0, 0.0, 0.7854);
  }
  node_diamond_R_12.userData.sculptComponent = {"id": "diamond-R", "name": "Diamond print R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "box", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "bloomers", "attachment": {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.3, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.35, -0.3, 0.48], "rotation": [-0.0, 0.0, 0.7854]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_diamond_R_12.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["bloomers"] ?? root).add(node_diamond_R_12);
  nodes["diamond-R"] = node_diamond_R_12;
  const mesh_diamond_R_12Geometry = endpoint_diamond_R_12
    ? new THREE.CylinderGeometry(endpoint_diamond_R_12.endRadius, endpoint_diamond_R_12.baseRadius, endpoint_diamond_R_12.length, 32, 12)
    : new THREE.BoxGeometry(1, 1, 1, 12, 12, 12);
  if (!endpoint_diamond_R_12) {
    mesh_diamond_R_12Geometry.scale(0.3, 0.3, 0.03);
  }
  const mesh_diamond_R_12 = new THREE.Mesh(
    mesh_diamond_R_12Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_diamond_R_12.name = "Diamond print R";
  if (endpoint_diamond_R_12) {
    mesh_diamond_R_12.position.copy(endpoint_diamond_R_12.midpoint);
    mesh_diamond_R_12.quaternion.copy(endpoint_diamond_R_12.quaternion);
  }
  mesh_diamond_R_12.castShadow = options.castShadow ?? true;
  mesh_diamond_R_12.receiveShadow = options.receiveShadow ?? true;
  mesh_diamond_R_12.userData.sculptComponent = {"id": "diamond-R", "name": "Diamond print R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "box", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "bloomers", "attachment": {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.3, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.35, -0.3, 0.48], "rotation": [-0.0, 0.0, 0.7854]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_diamond_R_12.add(mesh_diamond_R_12);
  meshes["diamond-R"] = mesh_diamond_R_12;
  colliders["diamond-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_diamond_R_12);

  const attachment_leg_R_13 = {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_leg_R_13 = makeAttachmentEndpoint(attachment_leg_R_13);
  const node_leg_R_13 = new THREE.Group();
  node_leg_R_13.name = "Stocking R__pivot";
  node_leg_R_13.scale.set(1, 1, 1);
  if (endpoint_leg_R_13) {
    node_leg_R_13.position.copy(endpoint_leg_R_13.start);
    node_leg_R_13.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_leg_R_13.position.set(-0.45, 0.72, 0.0);
    node_leg_R_13.rotation.set(-0.0, 0.0, -0.0);
  }
  node_leg_R_13.userData.sculptComponent = {"id": "leg-R", "name": "Stocking R", "level": "meso", "role": "leg", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.28571428571428575, "height": 0.35714285714285715, "depth": 0.28571428571428575, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.45, 0.72, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "leg", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_leg_R_13.userData.actionProfile = {"animationRole": "leg", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["root"] ?? root).add(node_leg_R_13);
  nodes["leg-R"] = node_leg_R_13;
  const mesh_leg_R_13Geometry = endpoint_leg_R_13
    ? new THREE.CylinderGeometry(endpoint_leg_R_13.endRadius, endpoint_leg_R_13.baseRadius, endpoint_leg_R_13.length, 32, 12)
    : buildWatertightCapsule(0.35, 0.7, 16, 32, 1);
  if (!endpoint_leg_R_13) {
    mesh_leg_R_13Geometry.scale(0.28571428571428575, 0.35714285714285715, 0.28571428571428575);
  }
  const mesh_leg_R_13 = new THREE.Mesh(
    mesh_leg_R_13Geometry,
    materialMap["cloth-purple"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_leg_R_13.name = "Stocking R";
  if (endpoint_leg_R_13) {
    mesh_leg_R_13.position.copy(endpoint_leg_R_13.midpoint);
    mesh_leg_R_13.quaternion.copy(endpoint_leg_R_13.quaternion);
  }
  mesh_leg_R_13.castShadow = options.castShadow ?? true;
  mesh_leg_R_13.receiveShadow = options.receiveShadow ?? true;
  mesh_leg_R_13.userData.sculptComponent = {"id": "leg-R", "name": "Stocking R", "level": "meso", "role": "leg", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.28571428571428575, "height": 0.35714285714285715, "depth": 0.28571428571428575, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.45, 0.72, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "leg", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_leg_R_13.add(mesh_leg_R_13);
  meshes["leg-R"] = mesh_leg_R_13;
  colliders["leg-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_leg_R_13);

  const endpoint_shoe_R_14 = makeAttachmentEndpoint(null);
  const node_shoe_R_14 = new THREE.Group();
  node_shoe_R_14.name = "Curled shoe R__pivot";
  node_shoe_R_14.scale.set(1, 1, 1);
  if (endpoint_shoe_R_14) {
    node_shoe_R_14.position.copy(endpoint_shoe_R_14.start);
    node_shoe_R_14.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_shoe_R_14.position.set(-0.62, 0.24, 0.12);
    node_shoe_R_14.rotation.set(-0.0, 0.0, -0.0);
  }
  node_shoe_R_14.userData.sculptComponent = {"id": "shoe-R", "name": "Curled shoe R", "level": "macro", "role": "foot", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.95, "height": 0.5, "depth": 0.75, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.62, 0.24, 0.12], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "foot", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "shoe-satin", "materialLayers": ["shoe-satin"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_shoe_R_14.userData.actionProfile = {"animationRole": "foot", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["root"] ?? root).add(node_shoe_R_14);
  nodes["shoe-R"] = node_shoe_R_14;
  const mesh_shoe_R_14Geometry = endpoint_shoe_R_14
    ? new THREE.CylinderGeometry(endpoint_shoe_R_14.endRadius, endpoint_shoe_R_14.baseRadius, endpoint_shoe_R_14.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_shoe_R_14) {
    mesh_shoe_R_14Geometry.scale(0.95, 0.5, 0.75);
  }
  const mesh_shoe_R_14 = new THREE.Mesh(
    mesh_shoe_R_14Geometry,
    materialMap["shoe-satin"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_shoe_R_14.name = "Curled shoe R";
  if (endpoint_shoe_R_14) {
    mesh_shoe_R_14.position.copy(endpoint_shoe_R_14.midpoint);
    mesh_shoe_R_14.quaternion.copy(endpoint_shoe_R_14.quaternion);
  }
  mesh_shoe_R_14.castShadow = options.castShadow ?? true;
  mesh_shoe_R_14.receiveShadow = options.receiveShadow ?? true;
  mesh_shoe_R_14.userData.sculptComponent = {"id": "shoe-R", "name": "Curled shoe R", "level": "macro", "role": "foot", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.95, "height": 0.5, "depth": 0.75, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.62, 0.24, 0.12], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "foot", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "shoe-satin", "materialLayers": ["shoe-satin"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_shoe_R_14.add(mesh_shoe_R_14);
  meshes["shoe-R"] = mesh_shoe_R_14;
  colliders["shoe-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_shoe_R_14);

  const attachment_shoe_toe_R_15 = {"parentSocket": "shoe-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_shoe_toe_R_15 = makeAttachmentEndpoint(attachment_shoe_toe_R_15);
  const node_shoe_toe_R_15 = new THREE.Group();
  node_shoe_toe_R_15.name = "Shoe toe curl R__pivot";
  node_shoe_toe_R_15.scale.set(1, 1, 1);
  if (endpoint_shoe_toe_R_15) {
    node_shoe_toe_R_15.position.copy(endpoint_shoe_toe_R_15.start);
    node_shoe_toe_R_15.rotation.set(-0.0, 0.0, 1.122);
  } else {
    node_shoe_toe_R_15.position.set(-0.13, 0.06, 0.1);
    node_shoe_toe_R_15.rotation.set(-0.0, 0.0, 1.122);
  }
  node_shoe_toe_R_15.userData.sculptComponent = {"id": "shoe-toe-R", "name": "Shoe toe curl R", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cone", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "shoe-R", "attachment": {"parentSocket": "shoe-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.55, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.13, 0.06, 0.1], "rotation": [-0.0, 0.0, 1.122]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_shoe_toe_R_15.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["shoe-R"] ?? root).add(node_shoe_toe_R_15);
  nodes["shoe-toe-R"] = node_shoe_toe_R_15;
  const mesh_shoe_toe_R_15Geometry = endpoint_shoe_toe_R_15
    ? new THREE.CylinderGeometry(endpoint_shoe_toe_R_15.endRadius, endpoint_shoe_toe_R_15.baseRadius, endpoint_shoe_toe_R_15.length, 32, 12)
    : new THREE.ConeGeometry(0.5, 1, 48, 1);
  if (!endpoint_shoe_toe_R_15) {
    mesh_shoe_toe_R_15Geometry.scale(0.2, 0.55, 0.2);
  }
  const mesh_shoe_toe_R_15 = new THREE.Mesh(
    mesh_shoe_toe_R_15Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_shoe_toe_R_15.name = "Shoe toe curl R";
  if (endpoint_shoe_toe_R_15) {
    mesh_shoe_toe_R_15.position.copy(endpoint_shoe_toe_R_15.midpoint);
    mesh_shoe_toe_R_15.quaternion.copy(endpoint_shoe_toe_R_15.quaternion);
  }
  mesh_shoe_toe_R_15.castShadow = options.castShadow ?? true;
  mesh_shoe_toe_R_15.receiveShadow = options.receiveShadow ?? true;
  mesh_shoe_toe_R_15.userData.sculptComponent = {"id": "shoe-toe-R", "name": "Shoe toe curl R", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cone", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "shoe-R", "attachment": {"parentSocket": "shoe-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.55, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.13, 0.06, 0.1], "rotation": [-0.0, 0.0, 1.122]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_shoe_toe_R_15.add(mesh_shoe_toe_R_15);
  meshes["shoe-toe-R"] = mesh_shoe_toe_R_15;
  colliders["shoe-toe-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_shoe_toe_R_15);

  const endpoint_shoe_bell_R_16 = makeAttachmentEndpoint(null);
  const node_shoe_bell_R_16 = new THREE.Group();
  node_shoe_bell_R_16.name = "Shoe bell R__pivot";
  node_shoe_bell_R_16.scale.set(1, 1, 1);
  if (endpoint_shoe_bell_R_16) {
    node_shoe_bell_R_16.position.copy(endpoint_shoe_bell_R_16.start);
    node_shoe_bell_R_16.rotation.set(-0.0, 0.0, -1.122);
  } else {
    node_shoe_bell_R_16.position.set(0.0083, 0.2593, 0.0);
    node_shoe_bell_R_16.rotation.set(-0.0, 0.0, -1.122);
  }
  node_shoe_bell_R_16.userData.sculptComponent = {"id": "shoe-bell-R", "name": "Shoe bell R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "shoe-toe-R", "attachment": {"parentSocket": "shoe-toe-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.12, "height": 0.12, "depth": 0.12, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0083, 0.2593, 0.0], "rotation": [-0.0, 0.0, -1.122]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "gold-metal", "materialLayers": ["gold-metal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(217, 165, 32, 1.0)", "secondaryAlbedo": "rgba(217, 165, 32, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_shoe_bell_R_16.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["shoe-toe-R"] ?? root).add(node_shoe_bell_R_16);
  nodes["shoe-bell-R"] = node_shoe_bell_R_16;
  const mesh_shoe_bell_R_16Geometry = endpoint_shoe_bell_R_16
    ? new THREE.CylinderGeometry(endpoint_shoe_bell_R_16.endRadius, endpoint_shoe_bell_R_16.baseRadius, endpoint_shoe_bell_R_16.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_shoe_bell_R_16) {
    mesh_shoe_bell_R_16Geometry.scale(0.12, 0.12, 0.12);
  }
  const mesh_shoe_bell_R_16 = new THREE.Mesh(
    mesh_shoe_bell_R_16Geometry,
    materialMap["gold-metal"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_shoe_bell_R_16.name = "Shoe bell R";
  if (endpoint_shoe_bell_R_16) {
    mesh_shoe_bell_R_16.position.copy(endpoint_shoe_bell_R_16.midpoint);
    mesh_shoe_bell_R_16.quaternion.copy(endpoint_shoe_bell_R_16.quaternion);
  }
  mesh_shoe_bell_R_16.castShadow = options.castShadow ?? true;
  mesh_shoe_bell_R_16.receiveShadow = options.receiveShadow ?? true;
  mesh_shoe_bell_R_16.userData.sculptComponent = {"id": "shoe-bell-R", "name": "Shoe bell R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "shoe-toe-R", "attachment": {"parentSocket": "shoe-toe-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.12, "height": 0.12, "depth": 0.12, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0083, 0.2593, 0.0], "rotation": [-0.0, 0.0, -1.122]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "gold-metal", "materialLayers": ["gold-metal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(217, 165, 32, 1.0)", "secondaryAlbedo": "rgba(217, 165, 32, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_shoe_bell_R_16.add(mesh_shoe_bell_R_16);
  meshes["shoe-bell-R"] = mesh_shoe_bell_R_16;
  colliders["shoe-bell-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_shoe_bell_R_16);

  const attachment_xbutton_L_17 = {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_xbutton_L_17 = makeAttachmentEndpoint(attachment_xbutton_L_17);
  const node_xbutton_L_17 = new THREE.Group();
  node_xbutton_L_17.name = "X button L__pivot";
  node_xbutton_L_17.scale.set(1, 1, 1);
  if (endpoint_xbutton_L_17) {
    node_xbutton_L_17.position.copy(endpoint_xbutton_L_17.start);
    node_xbutton_L_17.rotation.set(1.5708, 0.0, -0.0);
  } else {
    node_xbutton_L_17.position.set(0.18, 0.48, 0.43);
    node_xbutton_L_17.rotation.set(1.5708, 0.0, -0.0);
  }
  node_xbutton_L_17.userData.sculptComponent = {"id": "xbutton-L", "name": "X button L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "bloomers", "attachment": {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.14, "height": 0.03, "depth": 0.14, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.18, 0.48, 0.43], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "x-stitch", "kind": "fastener", "note": "X-shaped stitching across orange buttons"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_xbutton_L_17.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["bloomers"] ?? root).add(node_xbutton_L_17);
  nodes["xbutton-L"] = node_xbutton_L_17;
  const mesh_xbutton_L_17Geometry = endpoint_xbutton_L_17
    ? new THREE.CylinderGeometry(endpoint_xbutton_L_17.endRadius, endpoint_xbutton_L_17.baseRadius, endpoint_xbutton_L_17.length, 32, 12)
    : new THREE.CylinderGeometry(0.5, 0.5, 1, 48, 16);
  if (!endpoint_xbutton_L_17) {
    mesh_xbutton_L_17Geometry.scale(0.14, 0.03, 0.14);
  }
  const mesh_xbutton_L_17 = new THREE.Mesh(
    mesh_xbutton_L_17Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_xbutton_L_17.name = "X button L";
  if (endpoint_xbutton_L_17) {
    mesh_xbutton_L_17.position.copy(endpoint_xbutton_L_17.midpoint);
    mesh_xbutton_L_17.quaternion.copy(endpoint_xbutton_L_17.quaternion);
  }
  mesh_xbutton_L_17.castShadow = options.castShadow ?? true;
  mesh_xbutton_L_17.receiveShadow = options.receiveShadow ?? true;
  mesh_xbutton_L_17.userData.sculptComponent = {"id": "xbutton-L", "name": "X button L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "bloomers", "attachment": {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.14, "height": 0.03, "depth": 0.14, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.18, 0.48, 0.43], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "x-stitch", "kind": "fastener", "note": "X-shaped stitching across orange buttons"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_xbutton_L_17.add(mesh_xbutton_L_17);
  meshes["xbutton-L"] = mesh_xbutton_L_17;
  colliders["xbutton-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_xbutton_L_17);

  const endpoint_diamond_L_18 = makeAttachmentEndpoint(null);
  const node_diamond_L_18 = new THREE.Group();
  node_diamond_L_18.name = "Diamond print L__pivot";
  node_diamond_L_18.scale.set(1, 1, 1);
  if (endpoint_diamond_L_18) {
    node_diamond_L_18.position.copy(endpoint_diamond_L_18.start);
    node_diamond_L_18.rotation.set(-0.0, 0.0, 0.7854);
  } else {
    node_diamond_L_18.position.set(0.35, -0.3, 0.48);
    node_diamond_L_18.rotation.set(-0.0, 0.0, 0.7854);
  }
  node_diamond_L_18.userData.sculptComponent = {"id": "diamond-L", "name": "Diamond print L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "box", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "bloomers", "attachment": {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.3, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.35, -0.3, 0.48], "rotation": [-0.0, 0.0, 0.7854]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_diamond_L_18.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["bloomers"] ?? root).add(node_diamond_L_18);
  nodes["diamond-L"] = node_diamond_L_18;
  const mesh_diamond_L_18Geometry = endpoint_diamond_L_18
    ? new THREE.CylinderGeometry(endpoint_diamond_L_18.endRadius, endpoint_diamond_L_18.baseRadius, endpoint_diamond_L_18.length, 32, 12)
    : new THREE.BoxGeometry(1, 1, 1, 12, 12, 12);
  if (!endpoint_diamond_L_18) {
    mesh_diamond_L_18Geometry.scale(0.3, 0.3, 0.03);
  }
  const mesh_diamond_L_18 = new THREE.Mesh(
    mesh_diamond_L_18Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_diamond_L_18.name = "Diamond print L";
  if (endpoint_diamond_L_18) {
    mesh_diamond_L_18.position.copy(endpoint_diamond_L_18.midpoint);
    mesh_diamond_L_18.quaternion.copy(endpoint_diamond_L_18.quaternion);
  }
  mesh_diamond_L_18.castShadow = options.castShadow ?? true;
  mesh_diamond_L_18.receiveShadow = options.receiveShadow ?? true;
  mesh_diamond_L_18.userData.sculptComponent = {"id": "diamond-L", "name": "Diamond print L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "box", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "bloomers", "attachment": {"parentSocket": "bloomers-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.3, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.35, -0.3, 0.48], "rotation": [-0.0, 0.0, 0.7854]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_diamond_L_18.add(mesh_diamond_L_18);
  meshes["diamond-L"] = mesh_diamond_L_18;
  colliders["diamond-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_diamond_L_18);

  const attachment_leg_L_19 = {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_leg_L_19 = makeAttachmentEndpoint(attachment_leg_L_19);
  const node_leg_L_19 = new THREE.Group();
  node_leg_L_19.name = "Stocking L__pivot";
  node_leg_L_19.scale.set(1, 1, 1);
  if (endpoint_leg_L_19) {
    node_leg_L_19.position.copy(endpoint_leg_L_19.start);
    node_leg_L_19.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_leg_L_19.position.set(0.45, 0.72, 0.0);
    node_leg_L_19.rotation.set(-0.0, 0.0, -0.0);
  }
  node_leg_L_19.userData.sculptComponent = {"id": "leg-L", "name": "Stocking L", "level": "meso", "role": "leg", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.28571428571428575, "height": 0.35714285714285715, "depth": 0.28571428571428575, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.45, 0.72, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "leg", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_leg_L_19.userData.actionProfile = {"animationRole": "leg", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["root"] ?? root).add(node_leg_L_19);
  nodes["leg-L"] = node_leg_L_19;
  const mesh_leg_L_19Geometry = endpoint_leg_L_19
    ? new THREE.CylinderGeometry(endpoint_leg_L_19.endRadius, endpoint_leg_L_19.baseRadius, endpoint_leg_L_19.length, 32, 12)
    : buildWatertightCapsule(0.35, 0.7, 16, 32, 1);
  if (!endpoint_leg_L_19) {
    mesh_leg_L_19Geometry.scale(0.28571428571428575, 0.35714285714285715, 0.28571428571428575);
  }
  const mesh_leg_L_19 = new THREE.Mesh(
    mesh_leg_L_19Geometry,
    materialMap["cloth-purple"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_leg_L_19.name = "Stocking L";
  if (endpoint_leg_L_19) {
    mesh_leg_L_19.position.copy(endpoint_leg_L_19.midpoint);
    mesh_leg_L_19.quaternion.copy(endpoint_leg_L_19.quaternion);
  }
  mesh_leg_L_19.castShadow = options.castShadow ?? true;
  mesh_leg_L_19.receiveShadow = options.receiveShadow ?? true;
  mesh_leg_L_19.userData.sculptComponent = {"id": "leg-L", "name": "Stocking L", "level": "meso", "role": "leg", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.28571428571428575, "height": 0.35714285714285715, "depth": 0.28571428571428575, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.45, 0.72, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "leg", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_leg_L_19.add(mesh_leg_L_19);
  meshes["leg-L"] = mesh_leg_L_19;
  colliders["leg-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_leg_L_19);

  const endpoint_shoe_L_20 = makeAttachmentEndpoint(null);
  const node_shoe_L_20 = new THREE.Group();
  node_shoe_L_20.name = "Curled shoe L__pivot";
  node_shoe_L_20.scale.set(1, 1, 1);
  if (endpoint_shoe_L_20) {
    node_shoe_L_20.position.copy(endpoint_shoe_L_20.start);
    node_shoe_L_20.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_shoe_L_20.position.set(0.62, 0.24, 0.12);
    node_shoe_L_20.rotation.set(-0.0, 0.0, -0.0);
  }
  node_shoe_L_20.userData.sculptComponent = {"id": "shoe-L", "name": "Curled shoe L", "level": "macro", "role": "foot", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.95, "height": 0.5, "depth": 0.75, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.62, 0.24, 0.12], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "foot", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "shoe-satin", "materialLayers": ["shoe-satin"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_shoe_L_20.userData.actionProfile = {"animationRole": "foot", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["root"] ?? root).add(node_shoe_L_20);
  nodes["shoe-L"] = node_shoe_L_20;
  const mesh_shoe_L_20Geometry = endpoint_shoe_L_20
    ? new THREE.CylinderGeometry(endpoint_shoe_L_20.endRadius, endpoint_shoe_L_20.baseRadius, endpoint_shoe_L_20.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_shoe_L_20) {
    mesh_shoe_L_20Geometry.scale(0.95, 0.5, 0.75);
  }
  const mesh_shoe_L_20 = new THREE.Mesh(
    mesh_shoe_L_20Geometry,
    materialMap["shoe-satin"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_shoe_L_20.name = "Curled shoe L";
  if (endpoint_shoe_L_20) {
    mesh_shoe_L_20.position.copy(endpoint_shoe_L_20.midpoint);
    mesh_shoe_L_20.quaternion.copy(endpoint_shoe_L_20.quaternion);
  }
  mesh_shoe_L_20.castShadow = options.castShadow ?? true;
  mesh_shoe_L_20.receiveShadow = options.receiveShadow ?? true;
  mesh_shoe_L_20.userData.sculptComponent = {"id": "shoe-L", "name": "Curled shoe L", "level": "macro", "role": "foot", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.95, "height": 0.5, "depth": 0.75, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.62, 0.24, 0.12], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "foot", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "shoe-satin", "materialLayers": ["shoe-satin"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_shoe_L_20.add(mesh_shoe_L_20);
  meshes["shoe-L"] = mesh_shoe_L_20;
  colliders["shoe-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_shoe_L_20);

  const attachment_shoe_toe_L_21 = {"parentSocket": "shoe-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_shoe_toe_L_21 = makeAttachmentEndpoint(attachment_shoe_toe_L_21);
  const node_shoe_toe_L_21 = new THREE.Group();
  node_shoe_toe_L_21.name = "Shoe toe curl L__pivot";
  node_shoe_toe_L_21.scale.set(1, 1, 1);
  if (endpoint_shoe_toe_L_21) {
    node_shoe_toe_L_21.position.copy(endpoint_shoe_toe_L_21.start);
    node_shoe_toe_L_21.rotation.set(-0.0, 0.0, -1.122);
  } else {
    node_shoe_toe_L_21.position.set(0.13, 0.06, 0.1);
    node_shoe_toe_L_21.rotation.set(-0.0, 0.0, -1.122);
  }
  node_shoe_toe_L_21.userData.sculptComponent = {"id": "shoe-toe-L", "name": "Shoe toe curl L", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cone", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "shoe-L", "attachment": {"parentSocket": "shoe-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.55, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.13, 0.06, 0.1], "rotation": [-0.0, 0.0, -1.122]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_shoe_toe_L_21.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["shoe-L"] ?? root).add(node_shoe_toe_L_21);
  nodes["shoe-toe-L"] = node_shoe_toe_L_21;
  const mesh_shoe_toe_L_21Geometry = endpoint_shoe_toe_L_21
    ? new THREE.CylinderGeometry(endpoint_shoe_toe_L_21.endRadius, endpoint_shoe_toe_L_21.baseRadius, endpoint_shoe_toe_L_21.length, 32, 12)
    : new THREE.ConeGeometry(0.5, 1, 48, 1);
  if (!endpoint_shoe_toe_L_21) {
    mesh_shoe_toe_L_21Geometry.scale(0.2, 0.55, 0.2);
  }
  const mesh_shoe_toe_L_21 = new THREE.Mesh(
    mesh_shoe_toe_L_21Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_shoe_toe_L_21.name = "Shoe toe curl L";
  if (endpoint_shoe_toe_L_21) {
    mesh_shoe_toe_L_21.position.copy(endpoint_shoe_toe_L_21.midpoint);
    mesh_shoe_toe_L_21.quaternion.copy(endpoint_shoe_toe_L_21.quaternion);
  }
  mesh_shoe_toe_L_21.castShadow = options.castShadow ?? true;
  mesh_shoe_toe_L_21.receiveShadow = options.receiveShadow ?? true;
  mesh_shoe_toe_L_21.userData.sculptComponent = {"id": "shoe-toe-L", "name": "Shoe toe curl L", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cone", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "shoe-L", "attachment": {"parentSocket": "shoe-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.55, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.13, 0.06, 0.1], "rotation": [-0.0, 0.0, -1.122]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_shoe_toe_L_21.add(mesh_shoe_toe_L_21);
  meshes["shoe-toe-L"] = mesh_shoe_toe_L_21;
  colliders["shoe-toe-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_shoe_toe_L_21);

  const endpoint_shoe_bell_L_22 = makeAttachmentEndpoint(null);
  const node_shoe_bell_L_22 = new THREE.Group();
  node_shoe_bell_L_22.name = "Shoe bell L__pivot";
  node_shoe_bell_L_22.scale.set(1, 1, 1);
  if (endpoint_shoe_bell_L_22) {
    node_shoe_bell_L_22.position.copy(endpoint_shoe_bell_L_22.start);
    node_shoe_bell_L_22.rotation.set(-0.0, 0.0, 1.122);
  } else {
    node_shoe_bell_L_22.position.set(-0.0083, 0.2593, 0.0);
    node_shoe_bell_L_22.rotation.set(-0.0, 0.0, 1.122);
  }
  node_shoe_bell_L_22.userData.sculptComponent = {"id": "shoe-bell-L", "name": "Shoe bell L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "shoe-toe-L", "attachment": {"parentSocket": "shoe-toe-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.12, "height": 0.12, "depth": 0.12, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0083, 0.2593, 0.0], "rotation": [-0.0, 0.0, 1.122]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "gold-metal", "materialLayers": ["gold-metal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "bell-slit", "kind": "groove", "note": "slit across gold bell"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(217, 165, 32, 1.0)", "secondaryAlbedo": "rgba(217, 165, 32, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_shoe_bell_L_22.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["shoe-toe-L"] ?? root).add(node_shoe_bell_L_22);
  nodes["shoe-bell-L"] = node_shoe_bell_L_22;
  const mesh_shoe_bell_L_22Geometry = endpoint_shoe_bell_L_22
    ? new THREE.CylinderGeometry(endpoint_shoe_bell_L_22.endRadius, endpoint_shoe_bell_L_22.baseRadius, endpoint_shoe_bell_L_22.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_shoe_bell_L_22) {
    mesh_shoe_bell_L_22Geometry.scale(0.12, 0.12, 0.12);
  }
  const mesh_shoe_bell_L_22 = new THREE.Mesh(
    mesh_shoe_bell_L_22Geometry,
    materialMap["gold-metal"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_shoe_bell_L_22.name = "Shoe bell L";
  if (endpoint_shoe_bell_L_22) {
    mesh_shoe_bell_L_22.position.copy(endpoint_shoe_bell_L_22.midpoint);
    mesh_shoe_bell_L_22.quaternion.copy(endpoint_shoe_bell_L_22.quaternion);
  }
  mesh_shoe_bell_L_22.castShadow = options.castShadow ?? true;
  mesh_shoe_bell_L_22.receiveShadow = options.receiveShadow ?? true;
  mesh_shoe_bell_L_22.userData.sculptComponent = {"id": "shoe-bell-L", "name": "Shoe bell L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "shoe-toe-L", "attachment": {"parentSocket": "shoe-toe-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.12, "height": 0.12, "depth": 0.12, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0083, 0.2593, 0.0], "rotation": [-0.0, 0.0, 1.122]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "gold-metal", "materialLayers": ["gold-metal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "bell-slit", "kind": "groove", "note": "slit across gold bell"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(217, 165, 32, 1.0)", "secondaryAlbedo": "rgba(217, 165, 32, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_shoe_bell_L_22.add(mesh_shoe_bell_L_22);
  meshes["shoe-bell-L"] = mesh_shoe_bell_L_22;
  colliders["shoe-bell-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_shoe_bell_L_22);

  const endpoint_head_c_23 = makeAttachmentEndpoint(null);
  const node_head_c_23 = new THREE.Group();
  node_head_c_23.name = "Centre head (jester)__pivot";
  node_head_c_23.scale.set(1, 1, 1);
  if (endpoint_head_c_23) {
    node_head_c_23.position.copy(endpoint_head_c_23.start);
    node_head_c_23.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_c_23.position.set(0.0, 0.8, 0.12);
    node_head_c_23.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_c_23.userData.sculptComponent = {"id": "head-c", "name": "Centre head (jester)", "level": "macro", "role": "head", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 1.0, "height": 0.9, "depth": 0.85, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.8, 0.12], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "head", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "face-cream", "materialLayers": ["face-cream"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "star-pupils", "kind": "decal", "note": "orange star-shaped pupils both eyes"}, {"id": "centre-fangs", "kind": "contour", "note": "two small fangs in open grin"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(239, 230, 208, 1.0)", "secondaryAlbedo": "rgba(239, 230, 208, 1.0)", "materialClass": "skin", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_23.userData.actionProfile = {"animationRole": "head", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_head_c_23);
  nodes["head-c"] = node_head_c_23;
  const mesh_head_c_23Geometry = endpoint_head_c_23
    ? new THREE.CylinderGeometry(endpoint_head_c_23.endRadius, endpoint_head_c_23.baseRadius, endpoint_head_c_23.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_c_23) {
    mesh_head_c_23Geometry.scale(1.0, 0.9, 0.85);
  }
  const mesh_head_c_23 = new THREE.Mesh(
    mesh_head_c_23Geometry,
    materialMap["face-cream"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_c_23.name = "Centre head (jester)";
  if (endpoint_head_c_23) {
    mesh_head_c_23.position.copy(endpoint_head_c_23.midpoint);
    mesh_head_c_23.quaternion.copy(endpoint_head_c_23.quaternion);
  }
  mesh_head_c_23.castShadow = options.castShadow ?? true;
  mesh_head_c_23.receiveShadow = options.receiveShadow ?? true;
  mesh_head_c_23.userData.sculptComponent = {"id": "head-c", "name": "Centre head (jester)", "level": "macro", "role": "head", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 1.0, "height": 0.9, "depth": 0.85, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.8, 0.12], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "head", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "face-cream", "materialLayers": ["face-cream"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "star-pupils", "kind": "decal", "note": "orange star-shaped pupils both eyes"}, {"id": "centre-fangs", "kind": "contour", "note": "two small fangs in open grin"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(239, 230, 208, 1.0)", "secondaryAlbedo": "rgba(239, 230, 208, 1.0)", "materialClass": "skin", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_23.add(mesh_head_c_23);
  meshes["head-c"] = mesh_head_c_23;
  colliders["head-c"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_c_23);

  const attachment_head_c_ear_R_24 = {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_head_c_ear_R_24 = makeAttachmentEndpoint(attachment_head_c_ear_R_24);
  const node_head_c_ear_R_24 = new THREE.Group();
  node_head_c_ear_R_24.name = "Centre head (jester) ear R__pivot";
  node_head_c_ear_R_24.scale.set(1, 1, 1);
  if (endpoint_head_c_ear_R_24) {
    node_head_c_ear_R_24.position.copy(endpoint_head_c_ear_R_24.start);
    node_head_c_ear_R_24.rotation.set(1.5708, 0.0, -0.0);
  } else {
    node_head_c_ear_R_24.position.set(-0.5, 0.32, -0.17);
    node_head_c_ear_R_24.rotation.set(1.5708, 0.0, -0.0);
  }
  node_head_c_ear_R_24.userData.sculptComponent = {"id": "head-c-ear-R", "name": "Centre head (jester) ear R", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.42, "height": 0.06, "depth": 0.42, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.5, 0.32, -0.17], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_ear_R_24.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-c"] ?? root).add(node_head_c_ear_R_24);
  nodes["head-c-ear-R"] = node_head_c_ear_R_24;
  const mesh_head_c_ear_R_24Geometry = endpoint_head_c_ear_R_24
    ? new THREE.CylinderGeometry(endpoint_head_c_ear_R_24.endRadius, endpoint_head_c_ear_R_24.baseRadius, endpoint_head_c_ear_R_24.length, 32, 12)
    : new THREE.CylinderGeometry(0.5, 0.5, 1, 48, 16);
  if (!endpoint_head_c_ear_R_24) {
    mesh_head_c_ear_R_24Geometry.scale(0.42, 0.06, 0.42);
  }
  const mesh_head_c_ear_R_24 = new THREE.Mesh(
    mesh_head_c_ear_R_24Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_c_ear_R_24.name = "Centre head (jester) ear R";
  if (endpoint_head_c_ear_R_24) {
    mesh_head_c_ear_R_24.position.copy(endpoint_head_c_ear_R_24.midpoint);
    mesh_head_c_ear_R_24.quaternion.copy(endpoint_head_c_ear_R_24.quaternion);
  }
  mesh_head_c_ear_R_24.castShadow = options.castShadow ?? true;
  mesh_head_c_ear_R_24.receiveShadow = options.receiveShadow ?? true;
  mesh_head_c_ear_R_24.userData.sculptComponent = {"id": "head-c-ear-R", "name": "Centre head (jester) ear R", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.42, "height": 0.06, "depth": 0.42, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.5, 0.32, -0.17], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_ear_R_24.add(mesh_head_c_ear_R_24);
  meshes["head-c-ear-R"] = mesh_head_c_ear_R_24;
  colliders["head-c-ear-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_c_ear_R_24);

  const endpoint_head_c_eye_R_25 = makeAttachmentEndpoint(null);
  const node_head_c_eye_R_25 = new THREE.Group();
  node_head_c_eye_R_25.name = "Centre head (jester) eye R__pivot";
  node_head_c_eye_R_25.scale.set(1, 1, 1);
  if (endpoint_head_c_eye_R_25) {
    node_head_c_eye_R_25.position.copy(endpoint_head_c_eye_R_25.start);
    node_head_c_eye_R_25.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_c_eye_R_25.position.set(-0.17, 0.08, 0.28);
    node_head_c_eye_R_25.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_c_eye_R_25.userData.sculptComponent = {"id": "head-c-eye-R", "name": "Centre head (jester) eye R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.11, "height": 0.13, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.17, 0.08, 0.28], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_eye_R_25.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-c"] ?? root).add(node_head_c_eye_R_25);
  nodes["head-c-eye-R"] = node_head_c_eye_R_25;
  const mesh_head_c_eye_R_25Geometry = endpoint_head_c_eye_R_25
    ? new THREE.CylinderGeometry(endpoint_head_c_eye_R_25.endRadius, endpoint_head_c_eye_R_25.baseRadius, endpoint_head_c_eye_R_25.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_c_eye_R_25) {
    mesh_head_c_eye_R_25Geometry.scale(0.11, 0.13, 0.05);
  }
  const mesh_head_c_eye_R_25 = new THREE.Mesh(
    mesh_head_c_eye_R_25Geometry,
    materialMap["ink-white"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_c_eye_R_25.name = "Centre head (jester) eye R";
  if (endpoint_head_c_eye_R_25) {
    mesh_head_c_eye_R_25.position.copy(endpoint_head_c_eye_R_25.midpoint);
    mesh_head_c_eye_R_25.quaternion.copy(endpoint_head_c_eye_R_25.quaternion);
  }
  mesh_head_c_eye_R_25.castShadow = options.castShadow ?? true;
  mesh_head_c_eye_R_25.receiveShadow = options.receiveShadow ?? true;
  mesh_head_c_eye_R_25.userData.sculptComponent = {"id": "head-c-eye-R", "name": "Centre head (jester) eye R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.11, "height": 0.13, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.17, 0.08, 0.28], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_eye_R_25.add(mesh_head_c_eye_R_25);
  meshes["head-c-eye-R"] = mesh_head_c_eye_R_25;
  colliders["head-c-eye-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_c_eye_R_25);

  const endpoint_head_c_pupil_R_26 = makeAttachmentEndpoint(null);
  const node_head_c_pupil_R_26 = new THREE.Group();
  node_head_c_pupil_R_26.name = "Centre head (jester) star pupil R__pivot";
  node_head_c_pupil_R_26.scale.set(1, 1, 1);
  if (endpoint_head_c_pupil_R_26) {
    node_head_c_pupil_R_26.position.copy(endpoint_head_c_pupil_R_26.start);
    node_head_c_pupil_R_26.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_c_pupil_R_26.position.set(0.0, 0.0, 0.04);
    node_head_c_pupil_R_26.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_c_pupil_R_26.userData.sculptComponent = {"id": "head-c-pupil-R", "name": "Centre head (jester) star pupil R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c-eye-R", "attachment": {"parentSocket": "head-c-eye-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.08, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.04], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_pupil_R_26.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-c-eye-R"] ?? root).add(node_head_c_pupil_R_26);
  nodes["head-c-pupil-R"] = node_head_c_pupil_R_26;
  const mesh_head_c_pupil_R_26Geometry = endpoint_head_c_pupil_R_26
    ? new THREE.CylinderGeometry(endpoint_head_c_pupil_R_26.endRadius, endpoint_head_c_pupil_R_26.baseRadius, endpoint_head_c_pupil_R_26.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_c_pupil_R_26) {
    mesh_head_c_pupil_R_26Geometry.scale(0.07, 0.08, 0.03);
  }
  const mesh_head_c_pupil_R_26 = new THREE.Mesh(
    mesh_head_c_pupil_R_26Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_c_pupil_R_26.name = "Centre head (jester) star pupil R";
  if (endpoint_head_c_pupil_R_26) {
    mesh_head_c_pupil_R_26.position.copy(endpoint_head_c_pupil_R_26.midpoint);
    mesh_head_c_pupil_R_26.quaternion.copy(endpoint_head_c_pupil_R_26.quaternion);
  }
  mesh_head_c_pupil_R_26.castShadow = options.castShadow ?? true;
  mesh_head_c_pupil_R_26.receiveShadow = options.receiveShadow ?? true;
  mesh_head_c_pupil_R_26.userData.sculptComponent = {"id": "head-c-pupil-R", "name": "Centre head (jester) star pupil R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c-eye-R", "attachment": {"parentSocket": "head-c-eye-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.08, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.04], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_pupil_R_26.add(mesh_head_c_pupil_R_26);
  meshes["head-c-pupil-R"] = mesh_head_c_pupil_R_26;
  colliders["head-c-pupil-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_c_pupil_R_26);

  const attachment_head_c_ear_L_27 = {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_head_c_ear_L_27 = makeAttachmentEndpoint(attachment_head_c_ear_L_27);
  const node_head_c_ear_L_27 = new THREE.Group();
  node_head_c_ear_L_27.name = "Centre head (jester) ear L__pivot";
  node_head_c_ear_L_27.scale.set(1, 1, 1);
  if (endpoint_head_c_ear_L_27) {
    node_head_c_ear_L_27.position.copy(endpoint_head_c_ear_L_27.start);
    node_head_c_ear_L_27.rotation.set(1.5708, 0.0, -0.0);
  } else {
    node_head_c_ear_L_27.position.set(0.5, 0.32, -0.17);
    node_head_c_ear_L_27.rotation.set(1.5708, 0.0, -0.0);
  }
  node_head_c_ear_L_27.userData.sculptComponent = {"id": "head-c-ear-L", "name": "Centre head (jester) ear L", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.42, "height": 0.06, "depth": 0.42, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.5, 0.32, -0.17], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_ear_L_27.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-c"] ?? root).add(node_head_c_ear_L_27);
  nodes["head-c-ear-L"] = node_head_c_ear_L_27;
  const mesh_head_c_ear_L_27Geometry = endpoint_head_c_ear_L_27
    ? new THREE.CylinderGeometry(endpoint_head_c_ear_L_27.endRadius, endpoint_head_c_ear_L_27.baseRadius, endpoint_head_c_ear_L_27.length, 32, 12)
    : new THREE.CylinderGeometry(0.5, 0.5, 1, 48, 16);
  if (!endpoint_head_c_ear_L_27) {
    mesh_head_c_ear_L_27Geometry.scale(0.42, 0.06, 0.42);
  }
  const mesh_head_c_ear_L_27 = new THREE.Mesh(
    mesh_head_c_ear_L_27Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_c_ear_L_27.name = "Centre head (jester) ear L";
  if (endpoint_head_c_ear_L_27) {
    mesh_head_c_ear_L_27.position.copy(endpoint_head_c_ear_L_27.midpoint);
    mesh_head_c_ear_L_27.quaternion.copy(endpoint_head_c_ear_L_27.quaternion);
  }
  mesh_head_c_ear_L_27.castShadow = options.castShadow ?? true;
  mesh_head_c_ear_L_27.receiveShadow = options.receiveShadow ?? true;
  mesh_head_c_ear_L_27.userData.sculptComponent = {"id": "head-c-ear-L", "name": "Centre head (jester) ear L", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.42, "height": 0.06, "depth": 0.42, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.5, 0.32, -0.17], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_ear_L_27.add(mesh_head_c_ear_L_27);
  meshes["head-c-ear-L"] = mesh_head_c_ear_L_27;
  colliders["head-c-ear-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_c_ear_L_27);

  const endpoint_head_c_eye_L_28 = makeAttachmentEndpoint(null);
  const node_head_c_eye_L_28 = new THREE.Group();
  node_head_c_eye_L_28.name = "Centre head (jester) eye L__pivot";
  node_head_c_eye_L_28.scale.set(1, 1, 1);
  if (endpoint_head_c_eye_L_28) {
    node_head_c_eye_L_28.position.copy(endpoint_head_c_eye_L_28.start);
    node_head_c_eye_L_28.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_c_eye_L_28.position.set(0.17, 0.08, 0.28);
    node_head_c_eye_L_28.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_c_eye_L_28.userData.sculptComponent = {"id": "head-c-eye-L", "name": "Centre head (jester) eye L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.11, "height": 0.13, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.17, 0.08, 0.28], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_eye_L_28.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-c"] ?? root).add(node_head_c_eye_L_28);
  nodes["head-c-eye-L"] = node_head_c_eye_L_28;
  const mesh_head_c_eye_L_28Geometry = endpoint_head_c_eye_L_28
    ? new THREE.CylinderGeometry(endpoint_head_c_eye_L_28.endRadius, endpoint_head_c_eye_L_28.baseRadius, endpoint_head_c_eye_L_28.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_c_eye_L_28) {
    mesh_head_c_eye_L_28Geometry.scale(0.11, 0.13, 0.05);
  }
  const mesh_head_c_eye_L_28 = new THREE.Mesh(
    mesh_head_c_eye_L_28Geometry,
    materialMap["ink-white"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_c_eye_L_28.name = "Centre head (jester) eye L";
  if (endpoint_head_c_eye_L_28) {
    mesh_head_c_eye_L_28.position.copy(endpoint_head_c_eye_L_28.midpoint);
    mesh_head_c_eye_L_28.quaternion.copy(endpoint_head_c_eye_L_28.quaternion);
  }
  mesh_head_c_eye_L_28.castShadow = options.castShadow ?? true;
  mesh_head_c_eye_L_28.receiveShadow = options.receiveShadow ?? true;
  mesh_head_c_eye_L_28.userData.sculptComponent = {"id": "head-c-eye-L", "name": "Centre head (jester) eye L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.11, "height": 0.13, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.17, 0.08, 0.28], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_eye_L_28.add(mesh_head_c_eye_L_28);
  meshes["head-c-eye-L"] = mesh_head_c_eye_L_28;
  colliders["head-c-eye-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_c_eye_L_28);

  const endpoint_head_c_pupil_L_29 = makeAttachmentEndpoint(null);
  const node_head_c_pupil_L_29 = new THREE.Group();
  node_head_c_pupil_L_29.name = "Centre head (jester) star pupil L__pivot";
  node_head_c_pupil_L_29.scale.set(1, 1, 1);
  if (endpoint_head_c_pupil_L_29) {
    node_head_c_pupil_L_29.position.copy(endpoint_head_c_pupil_L_29.start);
    node_head_c_pupil_L_29.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_c_pupil_L_29.position.set(0.0, 0.0, 0.04);
    node_head_c_pupil_L_29.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_c_pupil_L_29.userData.sculptComponent = {"id": "head-c-pupil-L", "name": "Centre head (jester) star pupil L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c-eye-L", "attachment": {"parentSocket": "head-c-eye-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.08, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.04], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_pupil_L_29.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-c-eye-L"] ?? root).add(node_head_c_pupil_L_29);
  nodes["head-c-pupil-L"] = node_head_c_pupil_L_29;
  const mesh_head_c_pupil_L_29Geometry = endpoint_head_c_pupil_L_29
    ? new THREE.CylinderGeometry(endpoint_head_c_pupil_L_29.endRadius, endpoint_head_c_pupil_L_29.baseRadius, endpoint_head_c_pupil_L_29.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_c_pupil_L_29) {
    mesh_head_c_pupil_L_29Geometry.scale(0.07, 0.08, 0.03);
  }
  const mesh_head_c_pupil_L_29 = new THREE.Mesh(
    mesh_head_c_pupil_L_29Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_c_pupil_L_29.name = "Centre head (jester) star pupil L";
  if (endpoint_head_c_pupil_L_29) {
    mesh_head_c_pupil_L_29.position.copy(endpoint_head_c_pupil_L_29.midpoint);
    mesh_head_c_pupil_L_29.quaternion.copy(endpoint_head_c_pupil_L_29.quaternion);
  }
  mesh_head_c_pupil_L_29.castShadow = options.castShadow ?? true;
  mesh_head_c_pupil_L_29.receiveShadow = options.receiveShadow ?? true;
  mesh_head_c_pupil_L_29.userData.sculptComponent = {"id": "head-c-pupil-L", "name": "Centre head (jester) star pupil L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c-eye-L", "attachment": {"parentSocket": "head-c-eye-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.08, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.04], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_pupil_L_29.add(mesh_head_c_pupil_L_29);
  meshes["head-c-pupil-L"] = mesh_head_c_pupil_L_29;
  colliders["head-c-pupil-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_c_pupil_L_29);

  const endpoint_head_c_nose_30 = makeAttachmentEndpoint(null);
  const node_head_c_nose_30 = new THREE.Group();
  node_head_c_nose_30.name = "Centre head (jester) nose__pivot";
  node_head_c_nose_30.scale.set(1, 1, 1);
  if (endpoint_head_c_nose_30) {
    node_head_c_nose_30.position.copy(endpoint_head_c_nose_30.start);
    node_head_c_nose_30.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_c_nose_30.position.set(0.0, -0.1, 0.32);
    node_head_c_nose_30.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_c_nose_30.userData.sculptComponent = {"id": "head-c-nose", "name": "Centre head (jester) nose", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.1, "height": 0.07, "depth": 0.06, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.1, 0.32], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_nose_30.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-c"] ?? root).add(node_head_c_nose_30);
  nodes["head-c-nose"] = node_head_c_nose_30;
  const mesh_head_c_nose_30Geometry = endpoint_head_c_nose_30
    ? new THREE.CylinderGeometry(endpoint_head_c_nose_30.endRadius, endpoint_head_c_nose_30.baseRadius, endpoint_head_c_nose_30.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_c_nose_30) {
    mesh_head_c_nose_30Geometry.scale(0.1, 0.07, 0.06);
  }
  const mesh_head_c_nose_30 = new THREE.Mesh(
    mesh_head_c_nose_30Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_c_nose_30.name = "Centre head (jester) nose";
  if (endpoint_head_c_nose_30) {
    mesh_head_c_nose_30.position.copy(endpoint_head_c_nose_30.midpoint);
    mesh_head_c_nose_30.quaternion.copy(endpoint_head_c_nose_30.quaternion);
  }
  mesh_head_c_nose_30.castShadow = options.castShadow ?? true;
  mesh_head_c_nose_30.receiveShadow = options.receiveShadow ?? true;
  mesh_head_c_nose_30.userData.sculptComponent = {"id": "head-c-nose", "name": "Centre head (jester) nose", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.1, "height": 0.07, "depth": 0.06, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.1, 0.32], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_nose_30.add(mesh_head_c_nose_30);
  meshes["head-c-nose"] = mesh_head_c_nose_30;
  colliders["head-c-nose"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_c_nose_30);

  const endpoint_head_c_mouth_31 = makeAttachmentEndpoint(null);
  const node_head_c_mouth_31 = new THREE.Group();
  node_head_c_mouth_31.name = "Centre head (jester) mouth__pivot";
  node_head_c_mouth_31.scale.set(1, 1, 1);
  if (endpoint_head_c_mouth_31) {
    node_head_c_mouth_31.position.copy(endpoint_head_c_mouth_31.start);
    node_head_c_mouth_31.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_c_mouth_31.position.set(0.0, -0.28, 0.26);
    node_head_c_mouth_31.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_c_mouth_31.userData.sculptComponent = {"id": "head-c-mouth", "name": "Centre head (jester) mouth", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.16, "depth": 0.08, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.28, 0.26], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_mouth_31.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-c"] ?? root).add(node_head_c_mouth_31);
  nodes["head-c-mouth"] = node_head_c_mouth_31;
  const mesh_head_c_mouth_31Geometry = endpoint_head_c_mouth_31
    ? new THREE.CylinderGeometry(endpoint_head_c_mouth_31.endRadius, endpoint_head_c_mouth_31.baseRadius, endpoint_head_c_mouth_31.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_c_mouth_31) {
    mesh_head_c_mouth_31Geometry.scale(0.3, 0.16, 0.08);
  }
  const mesh_head_c_mouth_31 = new THREE.Mesh(
    mesh_head_c_mouth_31Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_c_mouth_31.name = "Centre head (jester) mouth";
  if (endpoint_head_c_mouth_31) {
    mesh_head_c_mouth_31.position.copy(endpoint_head_c_mouth_31.midpoint);
    mesh_head_c_mouth_31.quaternion.copy(endpoint_head_c_mouth_31.quaternion);
  }
  mesh_head_c_mouth_31.castShadow = options.castShadow ?? true;
  mesh_head_c_mouth_31.receiveShadow = options.receiveShadow ?? true;
  mesh_head_c_mouth_31.userData.sculptComponent = {"id": "head-c-mouth", "name": "Centre head (jester) mouth", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.16, "depth": 0.08, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.28, 0.26], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_c_mouth_31.add(mesh_head_c_mouth_31);
  meshes["head-c-mouth"] = mesh_head_c_mouth_31;
  colliders["head-c-mouth"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_c_mouth_31);

  const endpoint_head_w_32 = makeAttachmentEndpoint(null);
  const node_head_w_32 = new THREE.Group();
  node_head_w_32.name = "Left head (witch hat, wink)__pivot";
  node_head_w_32.scale.set(1, 1, 1);
  if (endpoint_head_w_32) {
    node_head_w_32.position.copy(endpoint_head_w_32.start);
    node_head_w_32.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_w_32.position.set(-0.973, 0.5, 0.12);
    node_head_w_32.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_w_32.userData.sculptComponent = {"id": "head-w", "name": "Left head (witch hat, wink)", "level": "macro", "role": "head", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.95, "height": 0.85, "depth": 0.8, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.973, 0.5, 0.12], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "head", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "face-cream", "materialLayers": ["face-cream"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "wink", "kind": "contour", "note": "left eye closed in a wink arc with star-mark cheek"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(239, 230, 208, 1.0)", "secondaryAlbedo": "rgba(239, 230, 208, 1.0)", "materialClass": "skin", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_32.userData.actionProfile = {"animationRole": "head", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_head_w_32);
  nodes["head-w"] = node_head_w_32;
  const mesh_head_w_32Geometry = endpoint_head_w_32
    ? new THREE.CylinderGeometry(endpoint_head_w_32.endRadius, endpoint_head_w_32.baseRadius, endpoint_head_w_32.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_w_32) {
    mesh_head_w_32Geometry.scale(0.95, 0.85, 0.8);
  }
  const mesh_head_w_32 = new THREE.Mesh(
    mesh_head_w_32Geometry,
    materialMap["face-cream"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_w_32.name = "Left head (witch hat, wink)";
  if (endpoint_head_w_32) {
    mesh_head_w_32.position.copy(endpoint_head_w_32.midpoint);
    mesh_head_w_32.quaternion.copy(endpoint_head_w_32.quaternion);
  }
  mesh_head_w_32.castShadow = options.castShadow ?? true;
  mesh_head_w_32.receiveShadow = options.receiveShadow ?? true;
  mesh_head_w_32.userData.sculptComponent = {"id": "head-w", "name": "Left head (witch hat, wink)", "level": "macro", "role": "head", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.95, "height": 0.85, "depth": 0.8, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.973, 0.5, 0.12], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "head", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "face-cream", "materialLayers": ["face-cream"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "wink", "kind": "contour", "note": "left eye closed in a wink arc with star-mark cheek"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(239, 230, 208, 1.0)", "secondaryAlbedo": "rgba(239, 230, 208, 1.0)", "materialClass": "skin", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_32.add(mesh_head_w_32);
  meshes["head-w"] = mesh_head_w_32;
  colliders["head-w"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_w_32);

  const attachment_head_w_ear_R_33 = {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_head_w_ear_R_33 = makeAttachmentEndpoint(attachment_head_w_ear_R_33);
  const node_head_w_ear_R_33 = new THREE.Group();
  node_head_w_ear_R_33.name = "Left head (witch hat, wink) ear R__pivot";
  node_head_w_ear_R_33.scale.set(1, 1, 1);
  if (endpoint_head_w_ear_R_33) {
    node_head_w_ear_R_33.position.copy(endpoint_head_w_ear_R_33.start);
    node_head_w_ear_R_33.rotation.set(1.5708, 0.0, -0.0);
  } else {
    node_head_w_ear_R_33.position.set(-0.5, 0.32, -0.17);
    node_head_w_ear_R_33.rotation.set(1.5708, 0.0, -0.0);
  }
  node_head_w_ear_R_33.userData.sculptComponent = {"id": "head-w-ear-R", "name": "Left head (witch hat, wink) ear R", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.42, "height": 0.06, "depth": 0.42, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.5, 0.32, -0.17], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_ear_R_33.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-w"] ?? root).add(node_head_w_ear_R_33);
  nodes["head-w-ear-R"] = node_head_w_ear_R_33;
  const mesh_head_w_ear_R_33Geometry = endpoint_head_w_ear_R_33
    ? new THREE.CylinderGeometry(endpoint_head_w_ear_R_33.endRadius, endpoint_head_w_ear_R_33.baseRadius, endpoint_head_w_ear_R_33.length, 32, 12)
    : new THREE.CylinderGeometry(0.5, 0.5, 1, 48, 16);
  if (!endpoint_head_w_ear_R_33) {
    mesh_head_w_ear_R_33Geometry.scale(0.42, 0.06, 0.42);
  }
  const mesh_head_w_ear_R_33 = new THREE.Mesh(
    mesh_head_w_ear_R_33Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_w_ear_R_33.name = "Left head (witch hat, wink) ear R";
  if (endpoint_head_w_ear_R_33) {
    mesh_head_w_ear_R_33.position.copy(endpoint_head_w_ear_R_33.midpoint);
    mesh_head_w_ear_R_33.quaternion.copy(endpoint_head_w_ear_R_33.quaternion);
  }
  mesh_head_w_ear_R_33.castShadow = options.castShadow ?? true;
  mesh_head_w_ear_R_33.receiveShadow = options.receiveShadow ?? true;
  mesh_head_w_ear_R_33.userData.sculptComponent = {"id": "head-w-ear-R", "name": "Left head (witch hat, wink) ear R", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.42, "height": 0.06, "depth": 0.42, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.5, 0.32, -0.17], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_ear_R_33.add(mesh_head_w_ear_R_33);
  meshes["head-w-ear-R"] = mesh_head_w_ear_R_33;
  colliders["head-w-ear-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_w_ear_R_33);

  const endpoint_head_w_eye_R_34 = makeAttachmentEndpoint(null);
  const node_head_w_eye_R_34 = new THREE.Group();
  node_head_w_eye_R_34.name = "Left head (witch hat, wink) eye R__pivot";
  node_head_w_eye_R_34.scale.set(1, 1, 1);
  if (endpoint_head_w_eye_R_34) {
    node_head_w_eye_R_34.position.copy(endpoint_head_w_eye_R_34.start);
    node_head_w_eye_R_34.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_w_eye_R_34.position.set(-0.17, 0.08, 0.28);
    node_head_w_eye_R_34.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_w_eye_R_34.userData.sculptComponent = {"id": "head-w-eye-R", "name": "Left head (witch hat, wink) eye R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.11, "height": 0.13, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.17, 0.08, 0.28], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_eye_R_34.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-w"] ?? root).add(node_head_w_eye_R_34);
  nodes["head-w-eye-R"] = node_head_w_eye_R_34;
  const mesh_head_w_eye_R_34Geometry = endpoint_head_w_eye_R_34
    ? new THREE.CylinderGeometry(endpoint_head_w_eye_R_34.endRadius, endpoint_head_w_eye_R_34.baseRadius, endpoint_head_w_eye_R_34.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_w_eye_R_34) {
    mesh_head_w_eye_R_34Geometry.scale(0.11, 0.13, 0.05);
  }
  const mesh_head_w_eye_R_34 = new THREE.Mesh(
    mesh_head_w_eye_R_34Geometry,
    materialMap["ink-white"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_w_eye_R_34.name = "Left head (witch hat, wink) eye R";
  if (endpoint_head_w_eye_R_34) {
    mesh_head_w_eye_R_34.position.copy(endpoint_head_w_eye_R_34.midpoint);
    mesh_head_w_eye_R_34.quaternion.copy(endpoint_head_w_eye_R_34.quaternion);
  }
  mesh_head_w_eye_R_34.castShadow = options.castShadow ?? true;
  mesh_head_w_eye_R_34.receiveShadow = options.receiveShadow ?? true;
  mesh_head_w_eye_R_34.userData.sculptComponent = {"id": "head-w-eye-R", "name": "Left head (witch hat, wink) eye R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.11, "height": 0.13, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.17, 0.08, 0.28], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_eye_R_34.add(mesh_head_w_eye_R_34);
  meshes["head-w-eye-R"] = mesh_head_w_eye_R_34;
  colliders["head-w-eye-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_w_eye_R_34);

  const endpoint_head_w_pupil_R_35 = makeAttachmentEndpoint(null);
  const node_head_w_pupil_R_35 = new THREE.Group();
  node_head_w_pupil_R_35.name = "Left head (witch hat, wink) star pupil R__pivot";
  node_head_w_pupil_R_35.scale.set(1, 1, 1);
  if (endpoint_head_w_pupil_R_35) {
    node_head_w_pupil_R_35.position.copy(endpoint_head_w_pupil_R_35.start);
    node_head_w_pupil_R_35.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_w_pupil_R_35.position.set(0.0, 0.0, 0.04);
    node_head_w_pupil_R_35.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_w_pupil_R_35.userData.sculptComponent = {"id": "head-w-pupil-R", "name": "Left head (witch hat, wink) star pupil R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w-eye-R", "attachment": {"parentSocket": "head-w-eye-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.08, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.04], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_pupil_R_35.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-w-eye-R"] ?? root).add(node_head_w_pupil_R_35);
  nodes["head-w-pupil-R"] = node_head_w_pupil_R_35;
  const mesh_head_w_pupil_R_35Geometry = endpoint_head_w_pupil_R_35
    ? new THREE.CylinderGeometry(endpoint_head_w_pupil_R_35.endRadius, endpoint_head_w_pupil_R_35.baseRadius, endpoint_head_w_pupil_R_35.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_w_pupil_R_35) {
    mesh_head_w_pupil_R_35Geometry.scale(0.07, 0.08, 0.03);
  }
  const mesh_head_w_pupil_R_35 = new THREE.Mesh(
    mesh_head_w_pupil_R_35Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_w_pupil_R_35.name = "Left head (witch hat, wink) star pupil R";
  if (endpoint_head_w_pupil_R_35) {
    mesh_head_w_pupil_R_35.position.copy(endpoint_head_w_pupil_R_35.midpoint);
    mesh_head_w_pupil_R_35.quaternion.copy(endpoint_head_w_pupil_R_35.quaternion);
  }
  mesh_head_w_pupil_R_35.castShadow = options.castShadow ?? true;
  mesh_head_w_pupil_R_35.receiveShadow = options.receiveShadow ?? true;
  mesh_head_w_pupil_R_35.userData.sculptComponent = {"id": "head-w-pupil-R", "name": "Left head (witch hat, wink) star pupil R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w-eye-R", "attachment": {"parentSocket": "head-w-eye-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.08, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.04], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_pupil_R_35.add(mesh_head_w_pupil_R_35);
  meshes["head-w-pupil-R"] = mesh_head_w_pupil_R_35;
  colliders["head-w-pupil-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_w_pupil_R_35);

  const attachment_head_w_ear_L_36 = {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_head_w_ear_L_36 = makeAttachmentEndpoint(attachment_head_w_ear_L_36);
  const node_head_w_ear_L_36 = new THREE.Group();
  node_head_w_ear_L_36.name = "Left head (witch hat, wink) ear L__pivot";
  node_head_w_ear_L_36.scale.set(1, 1, 1);
  if (endpoint_head_w_ear_L_36) {
    node_head_w_ear_L_36.position.copy(endpoint_head_w_ear_L_36.start);
    node_head_w_ear_L_36.rotation.set(1.5708, 0.0, -0.0);
  } else {
    node_head_w_ear_L_36.position.set(0.5, 0.32, -0.17);
    node_head_w_ear_L_36.rotation.set(1.5708, 0.0, -0.0);
  }
  node_head_w_ear_L_36.userData.sculptComponent = {"id": "head-w-ear-L", "name": "Left head (witch hat, wink) ear L", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.42, "height": 0.06, "depth": 0.42, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.5, 0.32, -0.17], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_ear_L_36.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-w"] ?? root).add(node_head_w_ear_L_36);
  nodes["head-w-ear-L"] = node_head_w_ear_L_36;
  const mesh_head_w_ear_L_36Geometry = endpoint_head_w_ear_L_36
    ? new THREE.CylinderGeometry(endpoint_head_w_ear_L_36.endRadius, endpoint_head_w_ear_L_36.baseRadius, endpoint_head_w_ear_L_36.length, 32, 12)
    : new THREE.CylinderGeometry(0.5, 0.5, 1, 48, 16);
  if (!endpoint_head_w_ear_L_36) {
    mesh_head_w_ear_L_36Geometry.scale(0.42, 0.06, 0.42);
  }
  const mesh_head_w_ear_L_36 = new THREE.Mesh(
    mesh_head_w_ear_L_36Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_w_ear_L_36.name = "Left head (witch hat, wink) ear L";
  if (endpoint_head_w_ear_L_36) {
    mesh_head_w_ear_L_36.position.copy(endpoint_head_w_ear_L_36.midpoint);
    mesh_head_w_ear_L_36.quaternion.copy(endpoint_head_w_ear_L_36.quaternion);
  }
  mesh_head_w_ear_L_36.castShadow = options.castShadow ?? true;
  mesh_head_w_ear_L_36.receiveShadow = options.receiveShadow ?? true;
  mesh_head_w_ear_L_36.userData.sculptComponent = {"id": "head-w-ear-L", "name": "Left head (witch hat, wink) ear L", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.42, "height": 0.06, "depth": 0.42, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.5, 0.32, -0.17], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_ear_L_36.add(mesh_head_w_ear_L_36);
  meshes["head-w-ear-L"] = mesh_head_w_ear_L_36;
  colliders["head-w-ear-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_w_ear_L_36);

  const endpoint_head_w_eye_L_37 = makeAttachmentEndpoint(null);
  const node_head_w_eye_L_37 = new THREE.Group();
  node_head_w_eye_L_37.name = "Left head (witch hat, wink) eye L__pivot";
  node_head_w_eye_L_37.scale.set(1, 1, 1);
  if (endpoint_head_w_eye_L_37) {
    node_head_w_eye_L_37.position.copy(endpoint_head_w_eye_L_37.start);
    node_head_w_eye_L_37.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_w_eye_L_37.position.set(0.17, 0.08, 0.28);
    node_head_w_eye_L_37.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_w_eye_L_37.userData.sculptComponent = {"id": "head-w-eye-L", "name": "Left head (witch hat, wink) eye L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.11, "height": 0.13, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.17, 0.08, 0.28], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_eye_L_37.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-w"] ?? root).add(node_head_w_eye_L_37);
  nodes["head-w-eye-L"] = node_head_w_eye_L_37;
  const mesh_head_w_eye_L_37Geometry = endpoint_head_w_eye_L_37
    ? new THREE.CylinderGeometry(endpoint_head_w_eye_L_37.endRadius, endpoint_head_w_eye_L_37.baseRadius, endpoint_head_w_eye_L_37.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_w_eye_L_37) {
    mesh_head_w_eye_L_37Geometry.scale(0.11, 0.13, 0.05);
  }
  const mesh_head_w_eye_L_37 = new THREE.Mesh(
    mesh_head_w_eye_L_37Geometry,
    materialMap["ink-white"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_w_eye_L_37.name = "Left head (witch hat, wink) eye L";
  if (endpoint_head_w_eye_L_37) {
    mesh_head_w_eye_L_37.position.copy(endpoint_head_w_eye_L_37.midpoint);
    mesh_head_w_eye_L_37.quaternion.copy(endpoint_head_w_eye_L_37.quaternion);
  }
  mesh_head_w_eye_L_37.castShadow = options.castShadow ?? true;
  mesh_head_w_eye_L_37.receiveShadow = options.receiveShadow ?? true;
  mesh_head_w_eye_L_37.userData.sculptComponent = {"id": "head-w-eye-L", "name": "Left head (witch hat, wink) eye L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.11, "height": 0.13, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.17, 0.08, 0.28], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_eye_L_37.add(mesh_head_w_eye_L_37);
  meshes["head-w-eye-L"] = mesh_head_w_eye_L_37;
  colliders["head-w-eye-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_w_eye_L_37);

  const endpoint_head_w_pupil_L_38 = makeAttachmentEndpoint(null);
  const node_head_w_pupil_L_38 = new THREE.Group();
  node_head_w_pupil_L_38.name = "Left head (witch hat, wink) star pupil L__pivot";
  node_head_w_pupil_L_38.scale.set(1, 1, 1);
  if (endpoint_head_w_pupil_L_38) {
    node_head_w_pupil_L_38.position.copy(endpoint_head_w_pupil_L_38.start);
    node_head_w_pupil_L_38.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_w_pupil_L_38.position.set(0.0, 0.0, 0.04);
    node_head_w_pupil_L_38.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_w_pupil_L_38.userData.sculptComponent = {"id": "head-w-pupil-L", "name": "Left head (witch hat, wink) star pupil L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w-eye-L", "attachment": {"parentSocket": "head-w-eye-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.08, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.04], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_pupil_L_38.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-w-eye-L"] ?? root).add(node_head_w_pupil_L_38);
  nodes["head-w-pupil-L"] = node_head_w_pupil_L_38;
  const mesh_head_w_pupil_L_38Geometry = endpoint_head_w_pupil_L_38
    ? new THREE.CylinderGeometry(endpoint_head_w_pupil_L_38.endRadius, endpoint_head_w_pupil_L_38.baseRadius, endpoint_head_w_pupil_L_38.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_w_pupil_L_38) {
    mesh_head_w_pupil_L_38Geometry.scale(0.07, 0.08, 0.03);
  }
  const mesh_head_w_pupil_L_38 = new THREE.Mesh(
    mesh_head_w_pupil_L_38Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_w_pupil_L_38.name = "Left head (witch hat, wink) star pupil L";
  if (endpoint_head_w_pupil_L_38) {
    mesh_head_w_pupil_L_38.position.copy(endpoint_head_w_pupil_L_38.midpoint);
    mesh_head_w_pupil_L_38.quaternion.copy(endpoint_head_w_pupil_L_38.quaternion);
  }
  mesh_head_w_pupil_L_38.castShadow = options.castShadow ?? true;
  mesh_head_w_pupil_L_38.receiveShadow = options.receiveShadow ?? true;
  mesh_head_w_pupil_L_38.userData.sculptComponent = {"id": "head-w-pupil-L", "name": "Left head (witch hat, wink) star pupil L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w-eye-L", "attachment": {"parentSocket": "head-w-eye-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.08, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.04], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_pupil_L_38.add(mesh_head_w_pupil_L_38);
  meshes["head-w-pupil-L"] = mesh_head_w_pupil_L_38;
  colliders["head-w-pupil-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_w_pupil_L_38);

  const endpoint_head_w_nose_39 = makeAttachmentEndpoint(null);
  const node_head_w_nose_39 = new THREE.Group();
  node_head_w_nose_39.name = "Left head (witch hat, wink) nose__pivot";
  node_head_w_nose_39.scale.set(1, 1, 1);
  if (endpoint_head_w_nose_39) {
    node_head_w_nose_39.position.copy(endpoint_head_w_nose_39.start);
    node_head_w_nose_39.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_w_nose_39.position.set(0.0, -0.1, 0.32);
    node_head_w_nose_39.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_w_nose_39.userData.sculptComponent = {"id": "head-w-nose", "name": "Left head (witch hat, wink) nose", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.1, "height": 0.07, "depth": 0.06, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.1, 0.32], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_nose_39.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-w"] ?? root).add(node_head_w_nose_39);
  nodes["head-w-nose"] = node_head_w_nose_39;
  const mesh_head_w_nose_39Geometry = endpoint_head_w_nose_39
    ? new THREE.CylinderGeometry(endpoint_head_w_nose_39.endRadius, endpoint_head_w_nose_39.baseRadius, endpoint_head_w_nose_39.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_w_nose_39) {
    mesh_head_w_nose_39Geometry.scale(0.1, 0.07, 0.06);
  }
  const mesh_head_w_nose_39 = new THREE.Mesh(
    mesh_head_w_nose_39Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_w_nose_39.name = "Left head (witch hat, wink) nose";
  if (endpoint_head_w_nose_39) {
    mesh_head_w_nose_39.position.copy(endpoint_head_w_nose_39.midpoint);
    mesh_head_w_nose_39.quaternion.copy(endpoint_head_w_nose_39.quaternion);
  }
  mesh_head_w_nose_39.castShadow = options.castShadow ?? true;
  mesh_head_w_nose_39.receiveShadow = options.receiveShadow ?? true;
  mesh_head_w_nose_39.userData.sculptComponent = {"id": "head-w-nose", "name": "Left head (witch hat, wink) nose", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.1, "height": 0.07, "depth": 0.06, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.1, 0.32], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_nose_39.add(mesh_head_w_nose_39);
  meshes["head-w-nose"] = mesh_head_w_nose_39;
  colliders["head-w-nose"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_w_nose_39);

  const endpoint_head_w_mouth_40 = makeAttachmentEndpoint(null);
  const node_head_w_mouth_40 = new THREE.Group();
  node_head_w_mouth_40.name = "Left head (witch hat, wink) mouth__pivot";
  node_head_w_mouth_40.scale.set(1, 1, 1);
  if (endpoint_head_w_mouth_40) {
    node_head_w_mouth_40.position.copy(endpoint_head_w_mouth_40.start);
    node_head_w_mouth_40.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_w_mouth_40.position.set(0.0, -0.28, 0.26);
    node_head_w_mouth_40.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_w_mouth_40.userData.sculptComponent = {"id": "head-w-mouth", "name": "Left head (witch hat, wink) mouth", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.16, "depth": 0.08, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.28, 0.26], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_mouth_40.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-w"] ?? root).add(node_head_w_mouth_40);
  nodes["head-w-mouth"] = node_head_w_mouth_40;
  const mesh_head_w_mouth_40Geometry = endpoint_head_w_mouth_40
    ? new THREE.CylinderGeometry(endpoint_head_w_mouth_40.endRadius, endpoint_head_w_mouth_40.baseRadius, endpoint_head_w_mouth_40.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_w_mouth_40) {
    mesh_head_w_mouth_40Geometry.scale(0.3, 0.16, 0.08);
  }
  const mesh_head_w_mouth_40 = new THREE.Mesh(
    mesh_head_w_mouth_40Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_w_mouth_40.name = "Left head (witch hat, wink) mouth";
  if (endpoint_head_w_mouth_40) {
    mesh_head_w_mouth_40.position.copy(endpoint_head_w_mouth_40.midpoint);
    mesh_head_w_mouth_40.quaternion.copy(endpoint_head_w_mouth_40.quaternion);
  }
  mesh_head_w_mouth_40.castShadow = options.castShadow ?? true;
  mesh_head_w_mouth_40.receiveShadow = options.receiveShadow ?? true;
  mesh_head_w_mouth_40.userData.sculptComponent = {"id": "head-w-mouth", "name": "Left head (witch hat, wink) mouth", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.16, "depth": 0.08, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.28, 0.26], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_mouth_40.add(mesh_head_w_mouth_40);
  meshes["head-w-mouth"] = mesh_head_w_mouth_40;
  colliders["head-w-mouth"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_w_mouth_40);

  const endpoint_head_t_41 = makeAttachmentEndpoint(null);
  const node_head_t_41 = new THREE.Group();
  node_head_t_41.name = "Right head (top hat, bored)__pivot";
  node_head_t_41.scale.set(1, 1, 1);
  if (endpoint_head_t_41) {
    node_head_t_41.position.copy(endpoint_head_t_41.start);
    node_head_t_41.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_t_41.position.set(0.993, 0.5, 0.12);
    node_head_t_41.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_t_41.userData.sculptComponent = {"id": "head-t", "name": "Right head (top hat, bored)", "level": "macro", "role": "head", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.95, "height": 0.85, "depth": 0.8, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.993, 0.5, 0.12], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "head", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "face-cream", "materialLayers": ["face-cream"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "half-lid", "kind": "contour", "note": "right head half-lidded bored eye + fang"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(239, 230, 208, 1.0)", "secondaryAlbedo": "rgba(239, 230, 208, 1.0)", "materialClass": "skin", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_41.userData.actionProfile = {"animationRole": "head", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_head_t_41);
  nodes["head-t"] = node_head_t_41;
  const mesh_head_t_41Geometry = endpoint_head_t_41
    ? new THREE.CylinderGeometry(endpoint_head_t_41.endRadius, endpoint_head_t_41.baseRadius, endpoint_head_t_41.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_t_41) {
    mesh_head_t_41Geometry.scale(0.95, 0.85, 0.8);
  }
  const mesh_head_t_41 = new THREE.Mesh(
    mesh_head_t_41Geometry,
    materialMap["face-cream"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_t_41.name = "Right head (top hat, bored)";
  if (endpoint_head_t_41) {
    mesh_head_t_41.position.copy(endpoint_head_t_41.midpoint);
    mesh_head_t_41.quaternion.copy(endpoint_head_t_41.quaternion);
  }
  mesh_head_t_41.castShadow = options.castShadow ?? true;
  mesh_head_t_41.receiveShadow = options.receiveShadow ?? true;
  mesh_head_t_41.userData.sculptComponent = {"id": "head-t", "name": "Right head (top hat, bored)", "level": "macro", "role": "head", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.95, "height": 0.85, "depth": 0.8, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.993, 0.5, 0.12], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "head", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "face-cream", "materialLayers": ["face-cream"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "half-lid", "kind": "contour", "note": "right head half-lidded bored eye + fang"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(239, 230, 208, 1.0)", "secondaryAlbedo": "rgba(239, 230, 208, 1.0)", "materialClass": "skin", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_41.add(mesh_head_t_41);
  meshes["head-t"] = mesh_head_t_41;
  colliders["head-t"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_t_41);

  const attachment_head_t_ear_R_42 = {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_head_t_ear_R_42 = makeAttachmentEndpoint(attachment_head_t_ear_R_42);
  const node_head_t_ear_R_42 = new THREE.Group();
  node_head_t_ear_R_42.name = "Right head (top hat, bored) ear R__pivot";
  node_head_t_ear_R_42.scale.set(1, 1, 1);
  if (endpoint_head_t_ear_R_42) {
    node_head_t_ear_R_42.position.copy(endpoint_head_t_ear_R_42.start);
    node_head_t_ear_R_42.rotation.set(1.5708, 0.0, -0.0);
  } else {
    node_head_t_ear_R_42.position.set(-0.5, 0.32, -0.17);
    node_head_t_ear_R_42.rotation.set(1.5708, 0.0, -0.0);
  }
  node_head_t_ear_R_42.userData.sculptComponent = {"id": "head-t-ear-R", "name": "Right head (top hat, bored) ear R", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.42, "height": 0.06, "depth": 0.42, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.5, 0.32, -0.17], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_ear_R_42.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-t"] ?? root).add(node_head_t_ear_R_42);
  nodes["head-t-ear-R"] = node_head_t_ear_R_42;
  const mesh_head_t_ear_R_42Geometry = endpoint_head_t_ear_R_42
    ? new THREE.CylinderGeometry(endpoint_head_t_ear_R_42.endRadius, endpoint_head_t_ear_R_42.baseRadius, endpoint_head_t_ear_R_42.length, 32, 12)
    : new THREE.CylinderGeometry(0.5, 0.5, 1, 48, 16);
  if (!endpoint_head_t_ear_R_42) {
    mesh_head_t_ear_R_42Geometry.scale(0.42, 0.06, 0.42);
  }
  const mesh_head_t_ear_R_42 = new THREE.Mesh(
    mesh_head_t_ear_R_42Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_t_ear_R_42.name = "Right head (top hat, bored) ear R";
  if (endpoint_head_t_ear_R_42) {
    mesh_head_t_ear_R_42.position.copy(endpoint_head_t_ear_R_42.midpoint);
    mesh_head_t_ear_R_42.quaternion.copy(endpoint_head_t_ear_R_42.quaternion);
  }
  mesh_head_t_ear_R_42.castShadow = options.castShadow ?? true;
  mesh_head_t_ear_R_42.receiveShadow = options.receiveShadow ?? true;
  mesh_head_t_ear_R_42.userData.sculptComponent = {"id": "head-t-ear-R", "name": "Right head (top hat, bored) ear R", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.42, "height": 0.06, "depth": 0.42, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.5, 0.32, -0.17], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_ear_R_42.add(mesh_head_t_ear_R_42);
  meshes["head-t-ear-R"] = mesh_head_t_ear_R_42;
  colliders["head-t-ear-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_t_ear_R_42);

  const endpoint_head_t_eye_R_43 = makeAttachmentEndpoint(null);
  const node_head_t_eye_R_43 = new THREE.Group();
  node_head_t_eye_R_43.name = "Right head (top hat, bored) eye R__pivot";
  node_head_t_eye_R_43.scale.set(1, 1, 1);
  if (endpoint_head_t_eye_R_43) {
    node_head_t_eye_R_43.position.copy(endpoint_head_t_eye_R_43.start);
    node_head_t_eye_R_43.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_t_eye_R_43.position.set(-0.17, 0.08, 0.28);
    node_head_t_eye_R_43.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_t_eye_R_43.userData.sculptComponent = {"id": "head-t-eye-R", "name": "Right head (top hat, bored) eye R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.11, "height": 0.13, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.17, 0.08, 0.28], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_eye_R_43.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-t"] ?? root).add(node_head_t_eye_R_43);
  nodes["head-t-eye-R"] = node_head_t_eye_R_43;
  const mesh_head_t_eye_R_43Geometry = endpoint_head_t_eye_R_43
    ? new THREE.CylinderGeometry(endpoint_head_t_eye_R_43.endRadius, endpoint_head_t_eye_R_43.baseRadius, endpoint_head_t_eye_R_43.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_t_eye_R_43) {
    mesh_head_t_eye_R_43Geometry.scale(0.11, 0.13, 0.05);
  }
  const mesh_head_t_eye_R_43 = new THREE.Mesh(
    mesh_head_t_eye_R_43Geometry,
    materialMap["ink-white"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_t_eye_R_43.name = "Right head (top hat, bored) eye R";
  if (endpoint_head_t_eye_R_43) {
    mesh_head_t_eye_R_43.position.copy(endpoint_head_t_eye_R_43.midpoint);
    mesh_head_t_eye_R_43.quaternion.copy(endpoint_head_t_eye_R_43.quaternion);
  }
  mesh_head_t_eye_R_43.castShadow = options.castShadow ?? true;
  mesh_head_t_eye_R_43.receiveShadow = options.receiveShadow ?? true;
  mesh_head_t_eye_R_43.userData.sculptComponent = {"id": "head-t-eye-R", "name": "Right head (top hat, bored) eye R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.11, "height": 0.13, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.17, 0.08, 0.28], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_eye_R_43.add(mesh_head_t_eye_R_43);
  meshes["head-t-eye-R"] = mesh_head_t_eye_R_43;
  colliders["head-t-eye-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_t_eye_R_43);

  const endpoint_head_t_pupil_R_44 = makeAttachmentEndpoint(null);
  const node_head_t_pupil_R_44 = new THREE.Group();
  node_head_t_pupil_R_44.name = "Right head (top hat, bored) star pupil R__pivot";
  node_head_t_pupil_R_44.scale.set(1, 1, 1);
  if (endpoint_head_t_pupil_R_44) {
    node_head_t_pupil_R_44.position.copy(endpoint_head_t_pupil_R_44.start);
    node_head_t_pupil_R_44.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_t_pupil_R_44.position.set(0.0, 0.0, 0.04);
    node_head_t_pupil_R_44.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_t_pupil_R_44.userData.sculptComponent = {"id": "head-t-pupil-R", "name": "Right head (top hat, bored) star pupil R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t-eye-R", "attachment": {"parentSocket": "head-t-eye-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.08, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.04], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_pupil_R_44.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-t-eye-R"] ?? root).add(node_head_t_pupil_R_44);
  nodes["head-t-pupil-R"] = node_head_t_pupil_R_44;
  const mesh_head_t_pupil_R_44Geometry = endpoint_head_t_pupil_R_44
    ? new THREE.CylinderGeometry(endpoint_head_t_pupil_R_44.endRadius, endpoint_head_t_pupil_R_44.baseRadius, endpoint_head_t_pupil_R_44.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_t_pupil_R_44) {
    mesh_head_t_pupil_R_44Geometry.scale(0.07, 0.08, 0.03);
  }
  const mesh_head_t_pupil_R_44 = new THREE.Mesh(
    mesh_head_t_pupil_R_44Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_t_pupil_R_44.name = "Right head (top hat, bored) star pupil R";
  if (endpoint_head_t_pupil_R_44) {
    mesh_head_t_pupil_R_44.position.copy(endpoint_head_t_pupil_R_44.midpoint);
    mesh_head_t_pupil_R_44.quaternion.copy(endpoint_head_t_pupil_R_44.quaternion);
  }
  mesh_head_t_pupil_R_44.castShadow = options.castShadow ?? true;
  mesh_head_t_pupil_R_44.receiveShadow = options.receiveShadow ?? true;
  mesh_head_t_pupil_R_44.userData.sculptComponent = {"id": "head-t-pupil-R", "name": "Right head (top hat, bored) star pupil R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t-eye-R", "attachment": {"parentSocket": "head-t-eye-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.08, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.04], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_pupil_R_44.add(mesh_head_t_pupil_R_44);
  meshes["head-t-pupil-R"] = mesh_head_t_pupil_R_44;
  colliders["head-t-pupil-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_t_pupil_R_44);

  const attachment_head_t_ear_L_45 = {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_head_t_ear_L_45 = makeAttachmentEndpoint(attachment_head_t_ear_L_45);
  const node_head_t_ear_L_45 = new THREE.Group();
  node_head_t_ear_L_45.name = "Right head (top hat, bored) ear L__pivot";
  node_head_t_ear_L_45.scale.set(1, 1, 1);
  if (endpoint_head_t_ear_L_45) {
    node_head_t_ear_L_45.position.copy(endpoint_head_t_ear_L_45.start);
    node_head_t_ear_L_45.rotation.set(1.5708, 0.0, -0.0);
  } else {
    node_head_t_ear_L_45.position.set(0.5, 0.32, -0.17);
    node_head_t_ear_L_45.rotation.set(1.5708, 0.0, -0.0);
  }
  node_head_t_ear_L_45.userData.sculptComponent = {"id": "head-t-ear-L", "name": "Right head (top hat, bored) ear L", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.42, "height": 0.06, "depth": 0.42, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.5, 0.32, -0.17], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_ear_L_45.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-t"] ?? root).add(node_head_t_ear_L_45);
  nodes["head-t-ear-L"] = node_head_t_ear_L_45;
  const mesh_head_t_ear_L_45Geometry = endpoint_head_t_ear_L_45
    ? new THREE.CylinderGeometry(endpoint_head_t_ear_L_45.endRadius, endpoint_head_t_ear_L_45.baseRadius, endpoint_head_t_ear_L_45.length, 32, 12)
    : new THREE.CylinderGeometry(0.5, 0.5, 1, 48, 16);
  if (!endpoint_head_t_ear_L_45) {
    mesh_head_t_ear_L_45Geometry.scale(0.42, 0.06, 0.42);
  }
  const mesh_head_t_ear_L_45 = new THREE.Mesh(
    mesh_head_t_ear_L_45Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_t_ear_L_45.name = "Right head (top hat, bored) ear L";
  if (endpoint_head_t_ear_L_45) {
    mesh_head_t_ear_L_45.position.copy(endpoint_head_t_ear_L_45.midpoint);
    mesh_head_t_ear_L_45.quaternion.copy(endpoint_head_t_ear_L_45.quaternion);
  }
  mesh_head_t_ear_L_45.castShadow = options.castShadow ?? true;
  mesh_head_t_ear_L_45.receiveShadow = options.receiveShadow ?? true;
  mesh_head_t_ear_L_45.userData.sculptComponent = {"id": "head-t-ear-L", "name": "Right head (top hat, bored) ear L", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.42, "height": 0.06, "depth": 0.42, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.5, 0.32, -0.17], "rotation": [1.5708, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_ear_L_45.add(mesh_head_t_ear_L_45);
  meshes["head-t-ear-L"] = mesh_head_t_ear_L_45;
  colliders["head-t-ear-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_t_ear_L_45);

  const endpoint_head_t_eye_L_46 = makeAttachmentEndpoint(null);
  const node_head_t_eye_L_46 = new THREE.Group();
  node_head_t_eye_L_46.name = "Right head (top hat, bored) eye L__pivot";
  node_head_t_eye_L_46.scale.set(1, 1, 1);
  if (endpoint_head_t_eye_L_46) {
    node_head_t_eye_L_46.position.copy(endpoint_head_t_eye_L_46.start);
    node_head_t_eye_L_46.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_t_eye_L_46.position.set(0.17, 0.08, 0.28);
    node_head_t_eye_L_46.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_t_eye_L_46.userData.sculptComponent = {"id": "head-t-eye-L", "name": "Right head (top hat, bored) eye L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.11, "height": 0.13, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.17, 0.08, 0.28], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_eye_L_46.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-t"] ?? root).add(node_head_t_eye_L_46);
  nodes["head-t-eye-L"] = node_head_t_eye_L_46;
  const mesh_head_t_eye_L_46Geometry = endpoint_head_t_eye_L_46
    ? new THREE.CylinderGeometry(endpoint_head_t_eye_L_46.endRadius, endpoint_head_t_eye_L_46.baseRadius, endpoint_head_t_eye_L_46.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_t_eye_L_46) {
    mesh_head_t_eye_L_46Geometry.scale(0.11, 0.13, 0.05);
  }
  const mesh_head_t_eye_L_46 = new THREE.Mesh(
    mesh_head_t_eye_L_46Geometry,
    materialMap["ink-white"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_t_eye_L_46.name = "Right head (top hat, bored) eye L";
  if (endpoint_head_t_eye_L_46) {
    mesh_head_t_eye_L_46.position.copy(endpoint_head_t_eye_L_46.midpoint);
    mesh_head_t_eye_L_46.quaternion.copy(endpoint_head_t_eye_L_46.quaternion);
  }
  mesh_head_t_eye_L_46.castShadow = options.castShadow ?? true;
  mesh_head_t_eye_L_46.receiveShadow = options.receiveShadow ?? true;
  mesh_head_t_eye_L_46.userData.sculptComponent = {"id": "head-t-eye-L", "name": "Right head (top hat, bored) eye L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.11, "height": 0.13, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.17, 0.08, 0.28], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_eye_L_46.add(mesh_head_t_eye_L_46);
  meshes["head-t-eye-L"] = mesh_head_t_eye_L_46;
  colliders["head-t-eye-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_t_eye_L_46);

  const endpoint_head_t_pupil_L_47 = makeAttachmentEndpoint(null);
  const node_head_t_pupil_L_47 = new THREE.Group();
  node_head_t_pupil_L_47.name = "Right head (top hat, bored) star pupil L__pivot";
  node_head_t_pupil_L_47.scale.set(1, 1, 1);
  if (endpoint_head_t_pupil_L_47) {
    node_head_t_pupil_L_47.position.copy(endpoint_head_t_pupil_L_47.start);
    node_head_t_pupil_L_47.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_t_pupil_L_47.position.set(0.0, 0.0, 0.04);
    node_head_t_pupil_L_47.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_t_pupil_L_47.userData.sculptComponent = {"id": "head-t-pupil-L", "name": "Right head (top hat, bored) star pupil L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t-eye-L", "attachment": {"parentSocket": "head-t-eye-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.08, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.04], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_pupil_L_47.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-t-eye-L"] ?? root).add(node_head_t_pupil_L_47);
  nodes["head-t-pupil-L"] = node_head_t_pupil_L_47;
  const mesh_head_t_pupil_L_47Geometry = endpoint_head_t_pupil_L_47
    ? new THREE.CylinderGeometry(endpoint_head_t_pupil_L_47.endRadius, endpoint_head_t_pupil_L_47.baseRadius, endpoint_head_t_pupil_L_47.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_t_pupil_L_47) {
    mesh_head_t_pupil_L_47Geometry.scale(0.07, 0.08, 0.03);
  }
  const mesh_head_t_pupil_L_47 = new THREE.Mesh(
    mesh_head_t_pupil_L_47Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_t_pupil_L_47.name = "Right head (top hat, bored) star pupil L";
  if (endpoint_head_t_pupil_L_47) {
    mesh_head_t_pupil_L_47.position.copy(endpoint_head_t_pupil_L_47.midpoint);
    mesh_head_t_pupil_L_47.quaternion.copy(endpoint_head_t_pupil_L_47.quaternion);
  }
  mesh_head_t_pupil_L_47.castShadow = options.castShadow ?? true;
  mesh_head_t_pupil_L_47.receiveShadow = options.receiveShadow ?? true;
  mesh_head_t_pupil_L_47.userData.sculptComponent = {"id": "head-t-pupil-L", "name": "Right head (top hat, bored) star pupil L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t-eye-L", "attachment": {"parentSocket": "head-t-eye-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.07, "height": 0.08, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.04], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_pupil_L_47.add(mesh_head_t_pupil_L_47);
  meshes["head-t-pupil-L"] = mesh_head_t_pupil_L_47;
  colliders["head-t-pupil-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_t_pupil_L_47);

  const endpoint_head_t_nose_48 = makeAttachmentEndpoint(null);
  const node_head_t_nose_48 = new THREE.Group();
  node_head_t_nose_48.name = "Right head (top hat, bored) nose__pivot";
  node_head_t_nose_48.scale.set(1, 1, 1);
  if (endpoint_head_t_nose_48) {
    node_head_t_nose_48.position.copy(endpoint_head_t_nose_48.start);
    node_head_t_nose_48.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_t_nose_48.position.set(0.0, -0.1, 0.32);
    node_head_t_nose_48.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_t_nose_48.userData.sculptComponent = {"id": "head-t-nose", "name": "Right head (top hat, bored) nose", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.1, "height": 0.07, "depth": 0.06, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.1, 0.32], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_nose_48.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-t"] ?? root).add(node_head_t_nose_48);
  nodes["head-t-nose"] = node_head_t_nose_48;
  const mesh_head_t_nose_48Geometry = endpoint_head_t_nose_48
    ? new THREE.CylinderGeometry(endpoint_head_t_nose_48.endRadius, endpoint_head_t_nose_48.baseRadius, endpoint_head_t_nose_48.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_t_nose_48) {
    mesh_head_t_nose_48Geometry.scale(0.1, 0.07, 0.06);
  }
  const mesh_head_t_nose_48 = new THREE.Mesh(
    mesh_head_t_nose_48Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_t_nose_48.name = "Right head (top hat, bored) nose";
  if (endpoint_head_t_nose_48) {
    mesh_head_t_nose_48.position.copy(endpoint_head_t_nose_48.midpoint);
    mesh_head_t_nose_48.quaternion.copy(endpoint_head_t_nose_48.quaternion);
  }
  mesh_head_t_nose_48.castShadow = options.castShadow ?? true;
  mesh_head_t_nose_48.receiveShadow = options.receiveShadow ?? true;
  mesh_head_t_nose_48.userData.sculptComponent = {"id": "head-t-nose", "name": "Right head (top hat, bored) nose", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.1, "height": 0.07, "depth": 0.06, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.1, 0.32], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_nose_48.add(mesh_head_t_nose_48);
  meshes["head-t-nose"] = mesh_head_t_nose_48;
  colliders["head-t-nose"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_t_nose_48);

  const endpoint_head_t_mouth_49 = makeAttachmentEndpoint(null);
  const node_head_t_mouth_49 = new THREE.Group();
  node_head_t_mouth_49.name = "Right head (top hat, bored) mouth__pivot";
  node_head_t_mouth_49.scale.set(1, 1, 1);
  if (endpoint_head_t_mouth_49) {
    node_head_t_mouth_49.position.copy(endpoint_head_t_mouth_49.start);
    node_head_t_mouth_49.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_t_mouth_49.position.set(0.0, -0.28, 0.26);
    node_head_t_mouth_49.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_t_mouth_49.userData.sculptComponent = {"id": "head-t-mouth", "name": "Right head (top hat, bored) mouth", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.16, "depth": 0.08, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.28, 0.26], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_mouth_49.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-t"] ?? root).add(node_head_t_mouth_49);
  nodes["head-t-mouth"] = node_head_t_mouth_49;
  const mesh_head_t_mouth_49Geometry = endpoint_head_t_mouth_49
    ? new THREE.CylinderGeometry(endpoint_head_t_mouth_49.endRadius, endpoint_head_t_mouth_49.baseRadius, endpoint_head_t_mouth_49.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_t_mouth_49) {
    mesh_head_t_mouth_49Geometry.scale(0.3, 0.16, 0.08);
  }
  const mesh_head_t_mouth_49 = new THREE.Mesh(
    mesh_head_t_mouth_49Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_t_mouth_49.name = "Right head (top hat, bored) mouth";
  if (endpoint_head_t_mouth_49) {
    mesh_head_t_mouth_49.position.copy(endpoint_head_t_mouth_49.midpoint);
    mesh_head_t_mouth_49.quaternion.copy(endpoint_head_t_mouth_49.quaternion);
  }
  mesh_head_t_mouth_49.castShadow = options.castShadow ?? true;
  mesh_head_t_mouth_49.receiveShadow = options.receiveShadow ?? true;
  mesh_head_t_mouth_49.userData.sculptComponent = {"id": "head-t-mouth", "name": "Right head (top hat, bored) mouth", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.3, "height": 0.16, "depth": 0.08, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, -0.28, 0.26], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_t_mouth_49.add(mesh_head_t_mouth_49);
  meshes["head-t-mouth"] = mesh_head_t_mouth_49;
  colliders["head-t-mouth"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_t_mouth_49);

  const endpoint_head_w_tongue_50 = makeAttachmentEndpoint(null);
  const node_head_w_tongue_50 = new THREE.Group();
  node_head_w_tongue_50.name = "Left head tongue__pivot";
  node_head_w_tongue_50.scale.set(1, 1, 1);
  if (endpoint_head_w_tongue_50) {
    node_head_w_tongue_50.position.copy(endpoint_head_w_tongue_50.start);
    node_head_w_tongue_50.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_head_w_tongue_50.position.set(-0.027, -0.2, 0.33);
    node_head_w_tongue_50.rotation.set(-0.0, 0.0, -0.0);
  }
  node_head_w_tongue_50.userData.sculptComponent = {"id": "head-w-tongue", "name": "Left head tongue", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.12, "height": 0.2, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.027, -0.2, 0.33], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "tongue-pink", "materialLayers": ["tongue-pink"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(229, 138, 154, 1.0)", "secondaryAlbedo": "rgba(229, 138, 154, 1.0)", "materialClass": "skin", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_tongue_50.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-w"] ?? root).add(node_head_w_tongue_50);
  nodes["head-w-tongue"] = node_head_w_tongue_50;
  const mesh_head_w_tongue_50Geometry = endpoint_head_w_tongue_50
    ? new THREE.CylinderGeometry(endpoint_head_w_tongue_50.endRadius, endpoint_head_w_tongue_50.baseRadius, endpoint_head_w_tongue_50.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_w_tongue_50) {
    mesh_head_w_tongue_50Geometry.scale(0.12, 0.2, 0.05);
  }
  const mesh_head_w_tongue_50 = new THREE.Mesh(
    mesh_head_w_tongue_50Geometry,
    materialMap["tongue-pink"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_w_tongue_50.name = "Left head tongue";
  if (endpoint_head_w_tongue_50) {
    mesh_head_w_tongue_50.position.copy(endpoint_head_w_tongue_50.midpoint);
    mesh_head_w_tongue_50.quaternion.copy(endpoint_head_w_tongue_50.quaternion);
  }
  mesh_head_w_tongue_50.castShadow = options.castShadow ?? true;
  mesh_head_w_tongue_50.receiveShadow = options.receiveShadow ?? true;
  mesh_head_w_tongue_50.userData.sculptComponent = {"id": "head-w-tongue", "name": "Left head tongue", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "ellipsoid", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.12, "height": 0.2, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.027, -0.2, 0.33], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "tongue-pink", "materialLayers": ["tongue-pink"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(229, 138, 154, 1.0)", "secondaryAlbedo": "rgba(229, 138, 154, 1.0)", "materialClass": "skin", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_head_w_tongue_50.add(mesh_head_w_tongue_50);
  meshes["head-w-tongue"] = mesh_head_w_tongue_50;
  colliders["head-w-tongue"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_head_w_tongue_50);

  const attachment_hat_crown_51 = {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_hat_crown_51 = makeAttachmentEndpoint(attachment_hat_crown_51);
  const node_hat_crown_51 = new THREE.Group();
  node_hat_crown_51.name = "Jester crown__pivot";
  node_hat_crown_51.scale.set(1, 1, 1);
  if (endpoint_hat_crown_51) {
    node_hat_crown_51.position.copy(endpoint_hat_crown_51.start);
    node_hat_crown_51.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_hat_crown_51.position.set(0.0, 0.517, -0.02);
    node_hat_crown_51.rotation.set(-0.0, 0.0, -0.0);
  }
  node_hat_crown_51.userData.sculptComponent = {"id": "hat-crown", "name": "Jester crown", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cone", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.5, "height": 0.45, "depth": 0.4, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.517, -0.02], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hat_crown_51.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-c"] ?? root).add(node_hat_crown_51);
  nodes["hat-crown"] = node_hat_crown_51;
  const mesh_hat_crown_51Geometry = endpoint_hat_crown_51
    ? new THREE.CylinderGeometry(endpoint_hat_crown_51.endRadius, endpoint_hat_crown_51.baseRadius, endpoint_hat_crown_51.length, 32, 12)
    : new THREE.ConeGeometry(0.5, 1, 48, 1);
  if (!endpoint_hat_crown_51) {
    mesh_hat_crown_51Geometry.scale(0.5, 0.45, 0.4);
  }
  const mesh_hat_crown_51 = new THREE.Mesh(
    mesh_hat_crown_51Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_hat_crown_51.name = "Jester crown";
  if (endpoint_hat_crown_51) {
    mesh_hat_crown_51.position.copy(endpoint_hat_crown_51.midpoint);
    mesh_hat_crown_51.quaternion.copy(endpoint_hat_crown_51.quaternion);
  }
  mesh_hat_crown_51.castShadow = options.castShadow ?? true;
  mesh_hat_crown_51.receiveShadow = options.receiveShadow ?? true;
  mesh_hat_crown_51.userData.sculptComponent = {"id": "hat-crown", "name": "Jester crown", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cone", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-c", "attachment": {"parentSocket": "head-c-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.5, "height": 0.45, "depth": 0.4, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.517, -0.02], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hat_crown_51.add(mesh_hat_crown_51);
  meshes["hat-crown"] = mesh_hat_crown_51;
  colliders["hat-crown"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_hat_crown_51);

  const attachment_hat_witch_52 = {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_hat_witch_52 = makeAttachmentEndpoint(attachment_hat_witch_52);
  const node_hat_witch_52 = new THREE.Group();
  node_hat_witch_52.name = "Witch/jingle hat__pivot";
  node_hat_witch_52.scale.set(1, 1, 1);
  if (endpoint_hat_witch_52) {
    node_hat_witch_52.position.copy(endpoint_hat_witch_52.start);
    node_hat_witch_52.rotation.set(-0.0, 0.0, 0.5);
  } else {
    node_hat_witch_52.position.set(0.0, 0.617, -0.07);
    node_hat_witch_52.rotation.set(-0.0, 0.0, 0.5);
  }
  node_hat_witch_52.userData.sculptComponent = {"id": "hat-witch", "name": "Witch/jingle hat", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cone", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.55, "height": 0.6, "depth": 0.5, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.617, -0.07], "rotation": [-0.0, 0.0, 0.5]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hat_witch_52.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-w"] ?? root).add(node_hat_witch_52);
  nodes["hat-witch"] = node_hat_witch_52;
  const mesh_hat_witch_52Geometry = endpoint_hat_witch_52
    ? new THREE.CylinderGeometry(endpoint_hat_witch_52.endRadius, endpoint_hat_witch_52.baseRadius, endpoint_hat_witch_52.length, 32, 12)
    : new THREE.ConeGeometry(0.5, 1, 48, 1);
  if (!endpoint_hat_witch_52) {
    mesh_hat_witch_52Geometry.scale(0.55, 0.6, 0.5);
  }
  const mesh_hat_witch_52 = new THREE.Mesh(
    mesh_hat_witch_52Geometry,
    materialMap["cloth-purple"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_hat_witch_52.name = "Witch/jingle hat";
  if (endpoint_hat_witch_52) {
    mesh_hat_witch_52.position.copy(endpoint_hat_witch_52.midpoint);
    mesh_hat_witch_52.quaternion.copy(endpoint_hat_witch_52.quaternion);
  }
  mesh_hat_witch_52.castShadow = options.castShadow ?? true;
  mesh_hat_witch_52.receiveShadow = options.receiveShadow ?? true;
  mesh_hat_witch_52.userData.sculptComponent = {"id": "hat-witch", "name": "Witch/jingle hat", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cone", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-w", "attachment": {"parentSocket": "head-w-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.55, "height": 0.6, "depth": 0.5, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.617, -0.07], "rotation": [-0.0, 0.0, 0.5]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hat_witch_52.add(mesh_hat_witch_52);
  meshes["hat-witch"] = mesh_hat_witch_52;
  colliders["hat-witch"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_hat_witch_52);

  const attachment_hat_top_53 = {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_hat_top_53 = makeAttachmentEndpoint(attachment_hat_top_53);
  const node_hat_top_53 = new THREE.Group();
  node_hat_top_53.name = "Top hat__pivot";
  node_hat_top_53.scale.set(1, 1, 1);
  if (endpoint_hat_top_53) {
    node_hat_top_53.position.copy(endpoint_hat_top_53.start);
    node_hat_top_53.rotation.set(-0.0, 0.0, -0.1);
  } else {
    node_hat_top_53.position.set(0.017, 0.517, -0.07);
    node_hat_top_53.rotation.set(-0.0, 0.0, -0.1);
  }
  node_hat_top_53.userData.sculptComponent = {"id": "hat-top", "name": "Top hat", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4, "height": 0.3, "depth": 0.4, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.017, 0.517, -0.07], "rotation": [-0.0, 0.0, -0.1]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-teal", "materialLayers": ["cloth-teal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "spade-card", "kind": "decal", "note": "spade playing-card badge on top-hat band"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hat_top_53.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["head-t"] ?? root).add(node_hat_top_53);
  nodes["hat-top"] = node_hat_top_53;
  const mesh_hat_top_53Geometry = endpoint_hat_top_53
    ? new THREE.CylinderGeometry(endpoint_hat_top_53.endRadius, endpoint_hat_top_53.baseRadius, endpoint_hat_top_53.length, 32, 12)
    : new THREE.CylinderGeometry(0.5, 0.5, 1, 48, 16);
  if (!endpoint_hat_top_53) {
    mesh_hat_top_53Geometry.scale(0.4, 0.3, 0.4);
  }
  const mesh_hat_top_53 = new THREE.Mesh(
    mesh_hat_top_53Geometry,
    materialMap["cloth-teal"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_hat_top_53.name = "Top hat";
  if (endpoint_hat_top_53) {
    mesh_hat_top_53.position.copy(endpoint_hat_top_53.midpoint);
    mesh_hat_top_53.quaternion.copy(endpoint_hat_top_53.quaternion);
  }
  mesh_hat_top_53.castShadow = options.castShadow ?? true;
  mesh_hat_top_53.receiveShadow = options.receiveShadow ?? true;
  mesh_hat_top_53.userData.sculptComponent = {"id": "hat-top", "name": "Top hat", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head-t", "attachment": {"parentSocket": "head-t-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4, "height": 0.3, "depth": 0.4, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.017, 0.517, -0.07], "rotation": [-0.0, 0.0, -0.1]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-teal", "materialLayers": ["cloth-teal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "spade-card", "kind": "decal", "note": "spade playing-card badge on top-hat band"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hat_top_53.add(mesh_hat_top_53);
  meshes["hat-top"] = mesh_hat_top_53;
  colliders["hat-top"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_hat_top_53);

  const attachment_hat_top_brim_54 = {"parentSocket": "hat-top-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_hat_top_brim_54 = makeAttachmentEndpoint(attachment_hat_top_brim_54);
  const node_hat_top_brim_54 = new THREE.Group();
  node_hat_top_brim_54.name = "Top-hat brim__pivot";
  node_hat_top_brim_54.scale.set(1, 1, 1);
  if (endpoint_hat_top_brim_54) {
    node_hat_top_brim_54.position.copy(endpoint_hat_top_brim_54.start);
    node_hat_top_brim_54.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_hat_top_brim_54.position.set(0.014, -0.1393, 0.0);
    node_hat_top_brim_54.rotation.set(-0.0, 0.0, -0.0);
  }
  node_hat_top_brim_54.userData.sculptComponent = {"id": "hat-top-brim", "name": "Top-hat brim", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "hat-top", "attachment": {"parentSocket": "hat-top-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.55, "height": 0.04, "depth": 0.55, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.014, -0.1393, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-teal", "materialLayers": ["cloth-teal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hat_top_brim_54.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["hat-top"] ?? root).add(node_hat_top_brim_54);
  nodes["hat-top-brim"] = node_hat_top_brim_54;
  const mesh_hat_top_brim_54Geometry = endpoint_hat_top_brim_54
    ? new THREE.CylinderGeometry(endpoint_hat_top_brim_54.endRadius, endpoint_hat_top_brim_54.baseRadius, endpoint_hat_top_brim_54.length, 32, 12)
    : new THREE.CylinderGeometry(0.5, 0.5, 1, 48, 16);
  if (!endpoint_hat_top_brim_54) {
    mesh_hat_top_brim_54Geometry.scale(0.55, 0.04, 0.55);
  }
  const mesh_hat_top_brim_54 = new THREE.Mesh(
    mesh_hat_top_brim_54Geometry,
    materialMap["cloth-teal"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_hat_top_brim_54.name = "Top-hat brim";
  if (endpoint_hat_top_brim_54) {
    mesh_hat_top_brim_54.position.copy(endpoint_hat_top_brim_54.midpoint);
    mesh_hat_top_brim_54.quaternion.copy(endpoint_hat_top_brim_54.quaternion);
  }
  mesh_hat_top_brim_54.castShadow = options.castShadow ?? true;
  mesh_hat_top_brim_54.receiveShadow = options.receiveShadow ?? true;
  mesh_hat_top_brim_54.userData.sculptComponent = {"id": "hat-top-brim", "name": "Top-hat brim", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "hat-top", "attachment": {"parentSocket": "hat-top-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.55, "height": 0.04, "depth": 0.55, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.014, -0.1393, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-teal", "materialLayers": ["cloth-teal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hat_top_brim_54.add(mesh_hat_top_brim_54);
  meshes["hat-top-brim"] = mesh_hat_top_brim_54;
  colliders["hat-top-brim"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_hat_top_brim_54);

  const endpoint_bell_crown_55 = makeAttachmentEndpoint(null);
  const node_bell_crown_55 = new THREE.Group();
  node_bell_crown_55.name = "bell-crown__pivot";
  node_bell_crown_55.scale.set(1, 1, 1);
  if (endpoint_bell_crown_55) {
    node_bell_crown_55.position.copy(endpoint_bell_crown_55.start);
    node_bell_crown_55.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_bell_crown_55.position.set(0.0, 0.15, 0.0);
    node_bell_crown_55.rotation.set(-0.0, 0.0, -0.0);
  }
  node_bell_crown_55.userData.sculptComponent = {"id": "bell-crown", "name": "bell-crown", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "hat-crown", "attachment": {"parentSocket": "hat-crown-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.1, "height": 0.1, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.15, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "gold-metal", "materialLayers": ["gold-metal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(217, 165, 32, 1.0)", "secondaryAlbedo": "rgba(217, 165, 32, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_bell_crown_55.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["hat-crown"] ?? root).add(node_bell_crown_55);
  nodes["bell-crown"] = node_bell_crown_55;
  const mesh_bell_crown_55Geometry = endpoint_bell_crown_55
    ? new THREE.CylinderGeometry(endpoint_bell_crown_55.endRadius, endpoint_bell_crown_55.baseRadius, endpoint_bell_crown_55.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_bell_crown_55) {
    mesh_bell_crown_55Geometry.scale(0.1, 0.1, 0.1);
  }
  const mesh_bell_crown_55 = new THREE.Mesh(
    mesh_bell_crown_55Geometry,
    materialMap["gold-metal"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_bell_crown_55.name = "bell-crown";
  if (endpoint_bell_crown_55) {
    mesh_bell_crown_55.position.copy(endpoint_bell_crown_55.midpoint);
    mesh_bell_crown_55.quaternion.copy(endpoint_bell_crown_55.quaternion);
  }
  mesh_bell_crown_55.castShadow = options.castShadow ?? true;
  mesh_bell_crown_55.receiveShadow = options.receiveShadow ?? true;
  mesh_bell_crown_55.userData.sculptComponent = {"id": "bell-crown", "name": "bell-crown", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "hat-crown", "attachment": {"parentSocket": "hat-crown-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.1, "height": 0.1, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.15, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "gold-metal", "materialLayers": ["gold-metal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(217, 165, 32, 1.0)", "secondaryAlbedo": "rgba(217, 165, 32, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_bell_crown_55.add(mesh_bell_crown_55);
  meshes["bell-crown"] = mesh_bell_crown_55;
  colliders["bell-crown"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_bell_crown_55);

  const endpoint_bell_witch_56 = makeAttachmentEndpoint(null);
  const node_bell_witch_56 = new THREE.Group();
  node_bell_witch_56.name = "bell-witch__pivot";
  node_bell_witch_56.scale.set(1, 1, 1);
  if (endpoint_bell_witch_56) {
    node_bell_witch_56.position.copy(endpoint_bell_witch_56.start);
    node_bell_witch_56.rotation.set(-0.0, 0.0, -0.5);
  } else {
    node_bell_witch_56.position.set(-0.1515, 0.1398, 0.05);
    node_bell_witch_56.rotation.set(-0.0, 0.0, -0.5);
  }
  node_bell_witch_56.userData.sculptComponent = {"id": "bell-witch", "name": "bell-witch", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "hat-witch", "attachment": {"parentSocket": "hat-witch-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.1, "height": 0.1, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.1515, 0.1398, 0.05], "rotation": [-0.0, 0.0, -0.5]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "gold-metal", "materialLayers": ["gold-metal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(217, 165, 32, 1.0)", "secondaryAlbedo": "rgba(217, 165, 32, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_bell_witch_56.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["hat-witch"] ?? root).add(node_bell_witch_56);
  nodes["bell-witch"] = node_bell_witch_56;
  const mesh_bell_witch_56Geometry = endpoint_bell_witch_56
    ? new THREE.CylinderGeometry(endpoint_bell_witch_56.endRadius, endpoint_bell_witch_56.baseRadius, endpoint_bell_witch_56.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_bell_witch_56) {
    mesh_bell_witch_56Geometry.scale(0.1, 0.1, 0.1);
  }
  const mesh_bell_witch_56 = new THREE.Mesh(
    mesh_bell_witch_56Geometry,
    materialMap["gold-metal"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_bell_witch_56.name = "bell-witch";
  if (endpoint_bell_witch_56) {
    mesh_bell_witch_56.position.copy(endpoint_bell_witch_56.midpoint);
    mesh_bell_witch_56.quaternion.copy(endpoint_bell_witch_56.quaternion);
  }
  mesh_bell_witch_56.castShadow = options.castShadow ?? true;
  mesh_bell_witch_56.receiveShadow = options.receiveShadow ?? true;
  mesh_bell_witch_56.userData.sculptComponent = {"id": "bell-witch", "name": "bell-witch", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "hat-witch", "attachment": {"parentSocket": "hat-witch-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.1, "height": 0.1, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.1515, 0.1398, 0.05], "rotation": [-0.0, 0.0, -0.5]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "gold-metal", "materialLayers": ["gold-metal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(217, 165, 32, 1.0)", "secondaryAlbedo": "rgba(217, 165, 32, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_bell_witch_56.add(mesh_bell_witch_56);
  meshes["bell-witch"] = mesh_bell_witch_56;
  colliders["bell-witch"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_bell_witch_56);

  const attachment_arm_up_R_57 = {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_arm_up_R_57 = makeAttachmentEndpoint(attachment_arm_up_R_57);
  const node_arm_up_R_57 = new THREE.Group();
  node_arm_up_R_57.name = "Arm up R__pivot";
  node_arm_up_R_57.scale.set(1, 1, 1);
  if (endpoint_arm_up_R_57) {
    node_arm_up_R_57.position.copy(endpoint_arm_up_R_57.start);
    node_arm_up_R_57.rotation.set(-0.0, 0.0, 1.49);
  } else {
    node_arm_up_R_57.position.set(-0.974, 0.133, 0.0);
    node_arm_up_R_57.rotation.set(-0.0, 0.0, 1.49);
  }
  node_arm_up_R_57.userData.sculptComponent = {"id": "arm-up-R", "name": "Arm up R", "level": "meso", "role": "arm", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4285714285714286, "height": 0.8836060739252256, "depth": 0.4285714285714286, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.974, 0.133, 0.0], "rotation": [-0.0, 0.0, 1.49]}, "actionProfile": {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_arm_up_R_57.userData.actionProfile = {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_arm_up_R_57);
  nodes["arm-up-R"] = node_arm_up_R_57;
  const mesh_arm_up_R_57Geometry = endpoint_arm_up_R_57
    ? new THREE.CylinderGeometry(endpoint_arm_up_R_57.endRadius, endpoint_arm_up_R_57.baseRadius, endpoint_arm_up_R_57.length, 32, 12)
    : buildWatertightCapsule(0.35, 0.7, 16, 32, 1);
  if (!endpoint_arm_up_R_57) {
    mesh_arm_up_R_57Geometry.scale(0.4285714285714286, 0.8836060739252256, 0.4285714285714286);
  }
  const mesh_arm_up_R_57 = new THREE.Mesh(
    mesh_arm_up_R_57Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_arm_up_R_57.name = "Arm up R";
  if (endpoint_arm_up_R_57) {
    mesh_arm_up_R_57.position.copy(endpoint_arm_up_R_57.midpoint);
    mesh_arm_up_R_57.quaternion.copy(endpoint_arm_up_R_57.quaternion);
  }
  mesh_arm_up_R_57.castShadow = options.castShadow ?? true;
  mesh_arm_up_R_57.receiveShadow = options.receiveShadow ?? true;
  mesh_arm_up_R_57.userData.sculptComponent = {"id": "arm-up-R", "name": "Arm up R", "level": "meso", "role": "arm", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4285714285714286, "height": 0.8836060739252256, "depth": 0.4285714285714286, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.974, 0.133, 0.0], "rotation": [-0.0, 0.0, 1.49]}, "actionProfile": {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_arm_up_R_57.add(mesh_arm_up_R_57);
  meshes["arm-up-R"] = mesh_arm_up_R_57;
  colliders["arm-up-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_arm_up_R_57);

  const endpoint_hand_up_R_58 = makeAttachmentEndpoint(null);
  const node_hand_up_R_58 = new THREE.Group();
  node_hand_up_R_58.name = "Hand up R__pivot";
  node_hand_up_R_58.scale.set(1, 1, 1);
  if (endpoint_hand_up_R_58) {
    node_hand_up_R_58.position.copy(endpoint_hand_up_R_58.start);
    node_hand_up_R_58.rotation.set(-0.0, 0.0, -1.49);
  } else {
    node_hand_up_R_58.position.set(0.0001, 0.618, 0.0);
    node_hand_up_R_58.rotation.set(-0.0, 0.0, -1.49);
  }
  node_hand_up_R_58.userData.sculptComponent = {"id": "hand-up-R", "name": "Hand up R", "level": "meso", "role": "hand", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-up-R", "attachment": {"parentSocket": "arm-up-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.38, "height": 0.38, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0001, 0.618, 0.0], "rotation": [-0.0, 0.0, -1.49]}, "actionProfile": {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hand_up_R_58.userData.actionProfile = {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["arm-up-R"] ?? root).add(node_hand_up_R_58);
  nodes["hand-up-R"] = node_hand_up_R_58;
  const mesh_hand_up_R_58Geometry = endpoint_hand_up_R_58
    ? new THREE.CylinderGeometry(endpoint_hand_up_R_58.endRadius, endpoint_hand_up_R_58.baseRadius, endpoint_hand_up_R_58.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_hand_up_R_58) {
    mesh_hand_up_R_58Geometry.scale(0.38, 0.38, 0.2);
  }
  const mesh_hand_up_R_58 = new THREE.Mesh(
    mesh_hand_up_R_58Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_hand_up_R_58.name = "Hand up R";
  if (endpoint_hand_up_R_58) {
    mesh_hand_up_R_58.position.copy(endpoint_hand_up_R_58.midpoint);
    mesh_hand_up_R_58.quaternion.copy(endpoint_hand_up_R_58.quaternion);
  }
  mesh_hand_up_R_58.castShadow = options.castShadow ?? true;
  mesh_hand_up_R_58.receiveShadow = options.receiveShadow ?? true;
  mesh_hand_up_R_58.userData.sculptComponent = {"id": "hand-up-R", "name": "Hand up R", "level": "meso", "role": "hand", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-up-R", "attachment": {"parentSocket": "arm-up-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.38, "height": 0.38, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0001, 0.618, 0.0], "rotation": [-0.0, 0.0, -1.49]}, "actionProfile": {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hand_up_R_58.add(mesh_hand_up_R_58);
  meshes["hand-up-R"] = mesh_hand_up_R_58;
  colliders["hand-up-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_hand_up_R_58);

  const endpoint_cuff_up_R_59 = makeAttachmentEndpoint(null);
  const node_cuff_up_R_59 = new THREE.Group();
  node_cuff_up_R_59.name = "Cuff scrunchie up R__pivot";
  node_cuff_up_R_59.scale.set(1, 1, 1);
  if (endpoint_cuff_up_R_59) {
    node_cuff_up_R_59.position.copy(endpoint_cuff_up_R_59.start);
    node_cuff_up_R_59.rotation.set(-0.0, 0.0, 0.0);
  } else {
    node_cuff_up_R_59.position.set(-0.0004, 0.4384, 0.0);
    node_cuff_up_R_59.rotation.set(-0.0, 0.0, 0.0);
  }
  node_cuff_up_R_59.userData.sculptComponent = {"id": "cuff-up-R", "name": "Cuff scrunchie up R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-up-R", "attachment": {"parentSocket": "arm-up-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0004, 0.4384, 0.0], "rotation": [-0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_cuff_up_R_59.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["arm-up-R"] ?? root).add(node_cuff_up_R_59);
  nodes["cuff-up-R"] = node_cuff_up_R_59;
  const mesh_cuff_up_R_59Geometry = endpoint_cuff_up_R_59
    ? new THREE.CylinderGeometry(endpoint_cuff_up_R_59.endRadius, endpoint_cuff_up_R_59.baseRadius, endpoint_cuff_up_R_59.length, 32, 12)
    : new THREE.TorusGeometry(0.45, 0.08, 24, 96);
  if (!endpoint_cuff_up_R_59) {
    mesh_cuff_up_R_59Geometry.scale(0.2, 0.2, 0.1);
  }
  const mesh_cuff_up_R_59 = new THREE.Mesh(
    mesh_cuff_up_R_59Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_cuff_up_R_59.name = "Cuff scrunchie up R";
  if (endpoint_cuff_up_R_59) {
    mesh_cuff_up_R_59.position.copy(endpoint_cuff_up_R_59.midpoint);
    mesh_cuff_up_R_59.quaternion.copy(endpoint_cuff_up_R_59.quaternion);
  }
  mesh_cuff_up_R_59.castShadow = options.castShadow ?? true;
  mesh_cuff_up_R_59.receiveShadow = options.receiveShadow ?? true;
  mesh_cuff_up_R_59.userData.sculptComponent = {"id": "cuff-up-R", "name": "Cuff scrunchie up R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-up-R", "attachment": {"parentSocket": "arm-up-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0004, 0.4384, 0.0], "rotation": [-0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_cuff_up_R_59.add(mesh_cuff_up_R_59);
  meshes["cuff-up-R"] = mesh_cuff_up_R_59;
  colliders["cuff-up-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_cuff_up_R_59);

  const attachment_arm_mid_R_60 = {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_arm_mid_R_60 = makeAttachmentEndpoint(attachment_arm_mid_R_60);
  const node_arm_mid_R_60 = new THREE.Group();
  node_arm_mid_R_60.name = "Arm mid R__pivot";
  node_arm_mid_R_60.scale.set(1, 1, 1);
  if (endpoint_arm_mid_R_60) {
    node_arm_mid_R_60.position.copy(endpoint_arm_mid_R_60.start);
    node_arm_mid_R_60.rotation.set(-0.0, 0.0, 2.073);
  } else {
    node_arm_mid_R_60.position.set(-0.815, -0.283, 0.0);
    node_arm_mid_R_60.rotation.set(-0.0, 0.0, 2.073);
  }
  node_arm_mid_R_60.userData.sculptComponent = {"id": "arm-mid-R", "name": "Arm mid R", "level": "meso", "role": "arm", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4285714285714286, "height": 0.6927426267660852, "depth": 0.4285714285714286, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.815, -0.283, 0.0], "rotation": [-0.0, 0.0, 2.073]}, "actionProfile": {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_arm_mid_R_60.userData.actionProfile = {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_arm_mid_R_60);
  nodes["arm-mid-R"] = node_arm_mid_R_60;
  const mesh_arm_mid_R_60Geometry = endpoint_arm_mid_R_60
    ? new THREE.CylinderGeometry(endpoint_arm_mid_R_60.endRadius, endpoint_arm_mid_R_60.baseRadius, endpoint_arm_mid_R_60.length, 32, 12)
    : buildWatertightCapsule(0.35, 0.7, 16, 32, 1);
  if (!endpoint_arm_mid_R_60) {
    mesh_arm_mid_R_60Geometry.scale(0.4285714285714286, 0.6927426267660852, 0.4285714285714286);
  }
  const mesh_arm_mid_R_60 = new THREE.Mesh(
    mesh_arm_mid_R_60Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_arm_mid_R_60.name = "Arm mid R";
  if (endpoint_arm_mid_R_60) {
    mesh_arm_mid_R_60.position.copy(endpoint_arm_mid_R_60.midpoint);
    mesh_arm_mid_R_60.quaternion.copy(endpoint_arm_mid_R_60.quaternion);
  }
  mesh_arm_mid_R_60.castShadow = options.castShadow ?? true;
  mesh_arm_mid_R_60.receiveShadow = options.receiveShadow ?? true;
  mesh_arm_mid_R_60.userData.sculptComponent = {"id": "arm-mid-R", "name": "Arm mid R", "level": "meso", "role": "arm", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4285714285714286, "height": 0.6927426267660852, "depth": 0.4285714285714286, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.815, -0.283, 0.0], "rotation": [-0.0, 0.0, 2.073]}, "actionProfile": {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_arm_mid_R_60.add(mesh_arm_mid_R_60);
  meshes["arm-mid-R"] = mesh_arm_mid_R_60;
  colliders["arm-mid-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_arm_mid_R_60);

  const endpoint_hand_mid_R_61 = makeAttachmentEndpoint(null);
  const node_hand_mid_R_61 = new THREE.Group();
  node_hand_mid_R_61.name = "Hand mid R__pivot";
  node_hand_mid_R_61.scale.set(1, 1, 1);
  if (endpoint_hand_mid_R_61) {
    node_hand_mid_R_61.position.copy(endpoint_hand_mid_R_61.start);
    node_hand_mid_R_61.rotation.set(-0.0, 0.0, -2.073);
  } else {
    node_hand_mid_R_61.position.set(-0.0005, 0.4852, 0.0);
    node_hand_mid_R_61.rotation.set(-0.0, 0.0, -2.073);
  }
  node_hand_mid_R_61.userData.sculptComponent = {"id": "hand-mid-R", "name": "Hand mid R", "level": "meso", "role": "hand", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-mid-R", "attachment": {"parentSocket": "arm-mid-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.38, "height": 0.38, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0005, 0.4852, 0.0], "rotation": [-0.0, 0.0, -2.073]}, "actionProfile": {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hand_mid_R_61.userData.actionProfile = {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["arm-mid-R"] ?? root).add(node_hand_mid_R_61);
  nodes["hand-mid-R"] = node_hand_mid_R_61;
  const mesh_hand_mid_R_61Geometry = endpoint_hand_mid_R_61
    ? new THREE.CylinderGeometry(endpoint_hand_mid_R_61.endRadius, endpoint_hand_mid_R_61.baseRadius, endpoint_hand_mid_R_61.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_hand_mid_R_61) {
    mesh_hand_mid_R_61Geometry.scale(0.38, 0.38, 0.2);
  }
  const mesh_hand_mid_R_61 = new THREE.Mesh(
    mesh_hand_mid_R_61Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_hand_mid_R_61.name = "Hand mid R";
  if (endpoint_hand_mid_R_61) {
    mesh_hand_mid_R_61.position.copy(endpoint_hand_mid_R_61.midpoint);
    mesh_hand_mid_R_61.quaternion.copy(endpoint_hand_mid_R_61.quaternion);
  }
  mesh_hand_mid_R_61.castShadow = options.castShadow ?? true;
  mesh_hand_mid_R_61.receiveShadow = options.receiveShadow ?? true;
  mesh_hand_mid_R_61.userData.sculptComponent = {"id": "hand-mid-R", "name": "Hand mid R", "level": "meso", "role": "hand", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-mid-R", "attachment": {"parentSocket": "arm-mid-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.38, "height": 0.38, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0005, 0.4852, 0.0], "rotation": [-0.0, 0.0, -2.073]}, "actionProfile": {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hand_mid_R_61.add(mesh_hand_mid_R_61);
  meshes["hand-mid-R"] = mesh_hand_mid_R_61;
  colliders["hand-mid-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_hand_mid_R_61);

  const endpoint_cuff_mid_R_62 = makeAttachmentEndpoint(null);
  const node_cuff_mid_R_62 = new THREE.Group();
  node_cuff_mid_R_62.name = "Cuff scrunchie mid R__pivot";
  node_cuff_mid_R_62.scale.set(1, 1, 1);
  if (endpoint_cuff_mid_R_62) {
    node_cuff_mid_R_62.position.copy(endpoint_cuff_mid_R_62.start);
    node_cuff_mid_R_62.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_cuff_mid_R_62.position.set(-0.0003, 0.3048, 0.0);
    node_cuff_mid_R_62.rotation.set(-0.0, 0.0, -0.0);
  }
  node_cuff_mid_R_62.userData.sculptComponent = {"id": "cuff-mid-R", "name": "Cuff scrunchie mid R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-mid-R", "attachment": {"parentSocket": "arm-mid-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0003, 0.3048, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-teal", "materialLayers": ["cloth-teal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_cuff_mid_R_62.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["arm-mid-R"] ?? root).add(node_cuff_mid_R_62);
  nodes["cuff-mid-R"] = node_cuff_mid_R_62;
  const mesh_cuff_mid_R_62Geometry = endpoint_cuff_mid_R_62
    ? new THREE.CylinderGeometry(endpoint_cuff_mid_R_62.endRadius, endpoint_cuff_mid_R_62.baseRadius, endpoint_cuff_mid_R_62.length, 32, 12)
    : new THREE.TorusGeometry(0.45, 0.08, 24, 96);
  if (!endpoint_cuff_mid_R_62) {
    mesh_cuff_mid_R_62Geometry.scale(0.2, 0.2, 0.1);
  }
  const mesh_cuff_mid_R_62 = new THREE.Mesh(
    mesh_cuff_mid_R_62Geometry,
    materialMap["cloth-teal"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_cuff_mid_R_62.name = "Cuff scrunchie mid R";
  if (endpoint_cuff_mid_R_62) {
    mesh_cuff_mid_R_62.position.copy(endpoint_cuff_mid_R_62.midpoint);
    mesh_cuff_mid_R_62.quaternion.copy(endpoint_cuff_mid_R_62.quaternion);
  }
  mesh_cuff_mid_R_62.castShadow = options.castShadow ?? true;
  mesh_cuff_mid_R_62.receiveShadow = options.receiveShadow ?? true;
  mesh_cuff_mid_R_62.userData.sculptComponent = {"id": "cuff-mid-R", "name": "Cuff scrunchie mid R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-mid-R", "attachment": {"parentSocket": "arm-mid-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0003, 0.3048, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-teal", "materialLayers": ["cloth-teal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_cuff_mid_R_62.add(mesh_cuff_mid_R_62);
  meshes["cuff-mid-R"] = mesh_cuff_mid_R_62;
  colliders["cuff-mid-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_cuff_mid_R_62);

  const attachment_arm_low_R_63 = {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_arm_low_R_63 = makeAttachmentEndpoint(attachment_arm_low_R_63);
  const node_arm_low_R_63 = new THREE.Group();
  node_arm_low_R_63.name = "Arm low R__pivot";
  node_arm_low_R_63.scale.set(1, 1, 1);
  if (endpoint_arm_low_R_63) {
    node_arm_low_R_63.position.copy(endpoint_arm_low_R_63.start);
    node_arm_low_R_63.rotation.set(-0.0, 0.0, 2.048);
  } else {
    node_arm_low_R_63.position.set(-0.89, -0.417, 0.0);
    node_arm_low_R_63.rotation.set(-0.0, 0.0, 2.048);
  }
  node_arm_low_R_63.userData.sculptComponent = {"id": "arm-low-R", "name": "Arm low R", "level": "meso", "role": "arm", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4285714285714286, "height": 0.7769498184620185, "depth": 0.4285714285714286, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.89, -0.417, 0.0], "rotation": [-0.0, 0.0, 2.048]}, "actionProfile": {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_arm_low_R_63.userData.actionProfile = {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_arm_low_R_63);
  nodes["arm-low-R"] = node_arm_low_R_63;
  const mesh_arm_low_R_63Geometry = endpoint_arm_low_R_63
    ? new THREE.CylinderGeometry(endpoint_arm_low_R_63.endRadius, endpoint_arm_low_R_63.baseRadius, endpoint_arm_low_R_63.length, 32, 12)
    : buildWatertightCapsule(0.35, 0.7, 16, 32, 1);
  if (!endpoint_arm_low_R_63) {
    mesh_arm_low_R_63Geometry.scale(0.4285714285714286, 0.7769498184620185, 0.4285714285714286);
  }
  const mesh_arm_low_R_63 = new THREE.Mesh(
    mesh_arm_low_R_63Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_arm_low_R_63.name = "Arm low R";
  if (endpoint_arm_low_R_63) {
    mesh_arm_low_R_63.position.copy(endpoint_arm_low_R_63.midpoint);
    mesh_arm_low_R_63.quaternion.copy(endpoint_arm_low_R_63.quaternion);
  }
  mesh_arm_low_R_63.castShadow = options.castShadow ?? true;
  mesh_arm_low_R_63.receiveShadow = options.receiveShadow ?? true;
  mesh_arm_low_R_63.userData.sculptComponent = {"id": "arm-low-R", "name": "Arm low R", "level": "meso", "role": "arm", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4285714285714286, "height": 0.7769498184620185, "depth": 0.4285714285714286, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.89, -0.417, 0.0], "rotation": [-0.0, 0.0, 2.048]}, "actionProfile": {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_arm_low_R_63.add(mesh_arm_low_R_63);
  meshes["arm-low-R"] = mesh_arm_low_R_63;
  colliders["arm-low-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_arm_low_R_63);

  const endpoint_hand_low_R_64 = makeAttachmentEndpoint(null);
  const node_hand_low_R_64 = new THREE.Group();
  node_hand_low_R_64.name = "Hand low R__pivot";
  node_hand_low_R_64.scale.set(1, 1, 1);
  if (endpoint_hand_low_R_64) {
    node_hand_low_R_64.position.copy(endpoint_hand_low_R_64.start);
    node_hand_low_R_64.rotation.set(-0.0, 0.0, -2.048);
  } else {
    node_hand_low_R_64.position.set(-0.0002, 0.5439, 0.0);
    node_hand_low_R_64.rotation.set(-0.0, 0.0, -2.048);
  }
  node_hand_low_R_64.userData.sculptComponent = {"id": "hand-low-R", "name": "Hand low R", "level": "meso", "role": "hand", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-low-R", "attachment": {"parentSocket": "arm-low-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.38, "height": 0.38, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0002, 0.5439, 0.0], "rotation": [-0.0, 0.0, -2.048]}, "actionProfile": {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hand_low_R_64.userData.actionProfile = {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["arm-low-R"] ?? root).add(node_hand_low_R_64);
  nodes["hand-low-R"] = node_hand_low_R_64;
  const mesh_hand_low_R_64Geometry = endpoint_hand_low_R_64
    ? new THREE.CylinderGeometry(endpoint_hand_low_R_64.endRadius, endpoint_hand_low_R_64.baseRadius, endpoint_hand_low_R_64.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_hand_low_R_64) {
    mesh_hand_low_R_64Geometry.scale(0.38, 0.38, 0.2);
  }
  const mesh_hand_low_R_64 = new THREE.Mesh(
    mesh_hand_low_R_64Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_hand_low_R_64.name = "Hand low R";
  if (endpoint_hand_low_R_64) {
    mesh_hand_low_R_64.position.copy(endpoint_hand_low_R_64.midpoint);
    mesh_hand_low_R_64.quaternion.copy(endpoint_hand_low_R_64.quaternion);
  }
  mesh_hand_low_R_64.castShadow = options.castShadow ?? true;
  mesh_hand_low_R_64.receiveShadow = options.receiveShadow ?? true;
  mesh_hand_low_R_64.userData.sculptComponent = {"id": "hand-low-R", "name": "Hand low R", "level": "meso", "role": "hand", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-low-R", "attachment": {"parentSocket": "arm-low-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.38, "height": 0.38, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0002, 0.5439, 0.0], "rotation": [-0.0, 0.0, -2.048]}, "actionProfile": {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hand_low_R_64.add(mesh_hand_low_R_64);
  meshes["hand-low-R"] = mesh_hand_low_R_64;
  colliders["hand-low-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_hand_low_R_64);

  const endpoint_cuff_low_R_65 = makeAttachmentEndpoint(null);
  const node_cuff_low_R_65 = new THREE.Group();
  node_cuff_low_R_65.name = "Cuff scrunchie low R__pivot";
  node_cuff_low_R_65.scale.set(1, 1, 1);
  if (endpoint_cuff_low_R_65) {
    node_cuff_low_R_65.position.copy(endpoint_cuff_low_R_65.start);
    node_cuff_low_R_65.rotation.set(-0.0, 0.0, 0.0);
  } else {
    node_cuff_low_R_65.position.set(0.0, 0.3636, 0.0);
    node_cuff_low_R_65.rotation.set(-0.0, 0.0, 0.0);
  }
  node_cuff_low_R_65.userData.sculptComponent = {"id": "cuff-low-R", "name": "Cuff scrunchie low R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-low-R", "attachment": {"parentSocket": "arm-low-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.3636, 0.0], "rotation": [-0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_cuff_low_R_65.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["arm-low-R"] ?? root).add(node_cuff_low_R_65);
  nodes["cuff-low-R"] = node_cuff_low_R_65;
  const mesh_cuff_low_R_65Geometry = endpoint_cuff_low_R_65
    ? new THREE.CylinderGeometry(endpoint_cuff_low_R_65.endRadius, endpoint_cuff_low_R_65.baseRadius, endpoint_cuff_low_R_65.length, 32, 12)
    : new THREE.TorusGeometry(0.45, 0.08, 24, 96);
  if (!endpoint_cuff_low_R_65) {
    mesh_cuff_low_R_65Geometry.scale(0.2, 0.2, 0.1);
  }
  const mesh_cuff_low_R_65 = new THREE.Mesh(
    mesh_cuff_low_R_65Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_cuff_low_R_65.name = "Cuff scrunchie low R";
  if (endpoint_cuff_low_R_65) {
    mesh_cuff_low_R_65.position.copy(endpoint_cuff_low_R_65.midpoint);
    mesh_cuff_low_R_65.quaternion.copy(endpoint_cuff_low_R_65.quaternion);
  }
  mesh_cuff_low_R_65.castShadow = options.castShadow ?? true;
  mesh_cuff_low_R_65.receiveShadow = options.receiveShadow ?? true;
  mesh_cuff_low_R_65.userData.sculptComponent = {"id": "cuff-low-R", "name": "Cuff scrunchie low R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-low-R", "attachment": {"parentSocket": "arm-low-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0, 0.3636, 0.0], "rotation": [-0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_cuff_low_R_65.add(mesh_cuff_low_R_65);
  meshes["cuff-low-R"] = mesh_cuff_low_R_65;
  colliders["cuff-low-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_cuff_low_R_65);

  const attachment_arm_up_L_66 = {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_arm_up_L_66 = makeAttachmentEndpoint(attachment_arm_up_L_66);
  const node_arm_up_L_66 = new THREE.Group();
  node_arm_up_L_66.name = "Arm up L__pivot";
  node_arm_up_L_66.scale.set(1, 1, 1);
  if (endpoint_arm_up_L_66) {
    node_arm_up_L_66.position.copy(endpoint_arm_up_L_66.start);
    node_arm_up_L_66.rotation.set(-0.0, 0.0, -1.49);
  } else {
    node_arm_up_L_66.position.set(0.974, 0.133, 0.0);
    node_arm_up_L_66.rotation.set(-0.0, 0.0, -1.49);
  }
  node_arm_up_L_66.userData.sculptComponent = {"id": "arm-up-L", "name": "Arm up L", "level": "meso", "role": "arm", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4285714285714286, "height": 0.8836060739252256, "depth": 0.4285714285714286, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.974, 0.133, 0.0], "rotation": [-0.0, 0.0, -1.49]}, "actionProfile": {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_arm_up_L_66.userData.actionProfile = {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_arm_up_L_66);
  nodes["arm-up-L"] = node_arm_up_L_66;
  const mesh_arm_up_L_66Geometry = endpoint_arm_up_L_66
    ? new THREE.CylinderGeometry(endpoint_arm_up_L_66.endRadius, endpoint_arm_up_L_66.baseRadius, endpoint_arm_up_L_66.length, 32, 12)
    : buildWatertightCapsule(0.35, 0.7, 16, 32, 1);
  if (!endpoint_arm_up_L_66) {
    mesh_arm_up_L_66Geometry.scale(0.4285714285714286, 0.8836060739252256, 0.4285714285714286);
  }
  const mesh_arm_up_L_66 = new THREE.Mesh(
    mesh_arm_up_L_66Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_arm_up_L_66.name = "Arm up L";
  if (endpoint_arm_up_L_66) {
    mesh_arm_up_L_66.position.copy(endpoint_arm_up_L_66.midpoint);
    mesh_arm_up_L_66.quaternion.copy(endpoint_arm_up_L_66.quaternion);
  }
  mesh_arm_up_L_66.castShadow = options.castShadow ?? true;
  mesh_arm_up_L_66.receiveShadow = options.receiveShadow ?? true;
  mesh_arm_up_L_66.userData.sculptComponent = {"id": "arm-up-L", "name": "Arm up L", "level": "meso", "role": "arm", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4285714285714286, "height": 0.8836060739252256, "depth": 0.4285714285714286, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.974, 0.133, 0.0], "rotation": [-0.0, 0.0, -1.49]}, "actionProfile": {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_arm_up_L_66.add(mesh_arm_up_L_66);
  meshes["arm-up-L"] = mesh_arm_up_L_66;
  colliders["arm-up-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_arm_up_L_66);

  const endpoint_hand_up_L_67 = makeAttachmentEndpoint(null);
  const node_hand_up_L_67 = new THREE.Group();
  node_hand_up_L_67.name = "Hand up L__pivot";
  node_hand_up_L_67.scale.set(1, 1, 1);
  if (endpoint_hand_up_L_67) {
    node_hand_up_L_67.position.copy(endpoint_hand_up_L_67.start);
    node_hand_up_L_67.rotation.set(-0.0, 0.0, 1.49);
  } else {
    node_hand_up_L_67.position.set(-0.0001, 0.618, 0.0);
    node_hand_up_L_67.rotation.set(-0.0, 0.0, 1.49);
  }
  node_hand_up_L_67.userData.sculptComponent = {"id": "hand-up-L", "name": "Hand up L", "level": "meso", "role": "hand", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-up-L", "attachment": {"parentSocket": "arm-up-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.38, "height": 0.38, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0001, 0.618, 0.0], "rotation": [-0.0, 0.0, 1.49]}, "actionProfile": {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hand_up_L_67.userData.actionProfile = {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["arm-up-L"] ?? root).add(node_hand_up_L_67);
  nodes["hand-up-L"] = node_hand_up_L_67;
  const mesh_hand_up_L_67Geometry = endpoint_hand_up_L_67
    ? new THREE.CylinderGeometry(endpoint_hand_up_L_67.endRadius, endpoint_hand_up_L_67.baseRadius, endpoint_hand_up_L_67.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_hand_up_L_67) {
    mesh_hand_up_L_67Geometry.scale(0.38, 0.38, 0.2);
  }
  const mesh_hand_up_L_67 = new THREE.Mesh(
    mesh_hand_up_L_67Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_hand_up_L_67.name = "Hand up L";
  if (endpoint_hand_up_L_67) {
    mesh_hand_up_L_67.position.copy(endpoint_hand_up_L_67.midpoint);
    mesh_hand_up_L_67.quaternion.copy(endpoint_hand_up_L_67.quaternion);
  }
  mesh_hand_up_L_67.castShadow = options.castShadow ?? true;
  mesh_hand_up_L_67.receiveShadow = options.receiveShadow ?? true;
  mesh_hand_up_L_67.userData.sculptComponent = {"id": "hand-up-L", "name": "Hand up L", "level": "meso", "role": "hand", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-up-L", "attachment": {"parentSocket": "arm-up-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.38, "height": 0.38, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0001, 0.618, 0.0], "rotation": [-0.0, 0.0, 1.49]}, "actionProfile": {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hand_up_L_67.add(mesh_hand_up_L_67);
  meshes["hand-up-L"] = mesh_hand_up_L_67;
  colliders["hand-up-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_hand_up_L_67);

  const endpoint_cuff_up_L_68 = makeAttachmentEndpoint(null);
  const node_cuff_up_L_68 = new THREE.Group();
  node_cuff_up_L_68.name = "Cuff scrunchie up L__pivot";
  node_cuff_up_L_68.scale.set(1, 1, 1);
  if (endpoint_cuff_up_L_68) {
    node_cuff_up_L_68.position.copy(endpoint_cuff_up_L_68.start);
    node_cuff_up_L_68.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_cuff_up_L_68.position.set(0.0004, 0.4384, 0.0);
    node_cuff_up_L_68.rotation.set(-0.0, 0.0, -0.0);
  }
  node_cuff_up_L_68.userData.sculptComponent = {"id": "cuff-up-L", "name": "Cuff scrunchie up L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-up-L", "attachment": {"parentSocket": "arm-up-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0004, 0.4384, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_cuff_up_L_68.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["arm-up-L"] ?? root).add(node_cuff_up_L_68);
  nodes["cuff-up-L"] = node_cuff_up_L_68;
  const mesh_cuff_up_L_68Geometry = endpoint_cuff_up_L_68
    ? new THREE.CylinderGeometry(endpoint_cuff_up_L_68.endRadius, endpoint_cuff_up_L_68.baseRadius, endpoint_cuff_up_L_68.length, 32, 12)
    : new THREE.TorusGeometry(0.45, 0.08, 24, 96);
  if (!endpoint_cuff_up_L_68) {
    mesh_cuff_up_L_68Geometry.scale(0.2, 0.2, 0.1);
  }
  const mesh_cuff_up_L_68 = new THREE.Mesh(
    mesh_cuff_up_L_68Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_cuff_up_L_68.name = "Cuff scrunchie up L";
  if (endpoint_cuff_up_L_68) {
    mesh_cuff_up_L_68.position.copy(endpoint_cuff_up_L_68.midpoint);
    mesh_cuff_up_L_68.quaternion.copy(endpoint_cuff_up_L_68.quaternion);
  }
  mesh_cuff_up_L_68.castShadow = options.castShadow ?? true;
  mesh_cuff_up_L_68.receiveShadow = options.receiveShadow ?? true;
  mesh_cuff_up_L_68.userData.sculptComponent = {"id": "cuff-up-L", "name": "Cuff scrunchie up L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-up-L", "attachment": {"parentSocket": "arm-up-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0004, 0.4384, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_cuff_up_L_68.add(mesh_cuff_up_L_68);
  meshes["cuff-up-L"] = mesh_cuff_up_L_68;
  colliders["cuff-up-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_cuff_up_L_68);

  const attachment_arm_mid_L_69 = {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_arm_mid_L_69 = makeAttachmentEndpoint(attachment_arm_mid_L_69);
  const node_arm_mid_L_69 = new THREE.Group();
  node_arm_mid_L_69.name = "Arm mid L__pivot";
  node_arm_mid_L_69.scale.set(1, 1, 1);
  if (endpoint_arm_mid_L_69) {
    node_arm_mid_L_69.position.copy(endpoint_arm_mid_L_69.start);
    node_arm_mid_L_69.rotation.set(-0.0, 0.0, -2.073);
  } else {
    node_arm_mid_L_69.position.set(0.815, -0.283, 0.0);
    node_arm_mid_L_69.rotation.set(-0.0, 0.0, -2.073);
  }
  node_arm_mid_L_69.userData.sculptComponent = {"id": "arm-mid-L", "name": "Arm mid L", "level": "meso", "role": "arm", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4285714285714286, "height": 0.6927426267660852, "depth": 0.4285714285714286, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.815, -0.283, 0.0], "rotation": [-0.0, 0.0, -2.073]}, "actionProfile": {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_arm_mid_L_69.userData.actionProfile = {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_arm_mid_L_69);
  nodes["arm-mid-L"] = node_arm_mid_L_69;
  const mesh_arm_mid_L_69Geometry = endpoint_arm_mid_L_69
    ? new THREE.CylinderGeometry(endpoint_arm_mid_L_69.endRadius, endpoint_arm_mid_L_69.baseRadius, endpoint_arm_mid_L_69.length, 32, 12)
    : buildWatertightCapsule(0.35, 0.7, 16, 32, 1);
  if (!endpoint_arm_mid_L_69) {
    mesh_arm_mid_L_69Geometry.scale(0.4285714285714286, 0.6927426267660852, 0.4285714285714286);
  }
  const mesh_arm_mid_L_69 = new THREE.Mesh(
    mesh_arm_mid_L_69Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_arm_mid_L_69.name = "Arm mid L";
  if (endpoint_arm_mid_L_69) {
    mesh_arm_mid_L_69.position.copy(endpoint_arm_mid_L_69.midpoint);
    mesh_arm_mid_L_69.quaternion.copy(endpoint_arm_mid_L_69.quaternion);
  }
  mesh_arm_mid_L_69.castShadow = options.castShadow ?? true;
  mesh_arm_mid_L_69.receiveShadow = options.receiveShadow ?? true;
  mesh_arm_mid_L_69.userData.sculptComponent = {"id": "arm-mid-L", "name": "Arm mid L", "level": "meso", "role": "arm", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4285714285714286, "height": 0.6927426267660852, "depth": 0.4285714285714286, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.815, -0.283, 0.0], "rotation": [-0.0, 0.0, -2.073]}, "actionProfile": {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_arm_mid_L_69.add(mesh_arm_mid_L_69);
  meshes["arm-mid-L"] = mesh_arm_mid_L_69;
  colliders["arm-mid-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_arm_mid_L_69);

  const endpoint_hand_mid_L_70 = makeAttachmentEndpoint(null);
  const node_hand_mid_L_70 = new THREE.Group();
  node_hand_mid_L_70.name = "Hand mid L__pivot";
  node_hand_mid_L_70.scale.set(1, 1, 1);
  if (endpoint_hand_mid_L_70) {
    node_hand_mid_L_70.position.copy(endpoint_hand_mid_L_70.start);
    node_hand_mid_L_70.rotation.set(-0.0, 0.0, 2.073);
  } else {
    node_hand_mid_L_70.position.set(0.0005, 0.4852, 0.0);
    node_hand_mid_L_70.rotation.set(-0.0, 0.0, 2.073);
  }
  node_hand_mid_L_70.userData.sculptComponent = {"id": "hand-mid-L", "name": "Hand mid L", "level": "meso", "role": "hand", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-mid-L", "attachment": {"parentSocket": "arm-mid-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.38, "height": 0.38, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0005, 0.4852, 0.0], "rotation": [-0.0, 0.0, 2.073]}, "actionProfile": {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hand_mid_L_70.userData.actionProfile = {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["arm-mid-L"] ?? root).add(node_hand_mid_L_70);
  nodes["hand-mid-L"] = node_hand_mid_L_70;
  const mesh_hand_mid_L_70Geometry = endpoint_hand_mid_L_70
    ? new THREE.CylinderGeometry(endpoint_hand_mid_L_70.endRadius, endpoint_hand_mid_L_70.baseRadius, endpoint_hand_mid_L_70.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_hand_mid_L_70) {
    mesh_hand_mid_L_70Geometry.scale(0.38, 0.38, 0.2);
  }
  const mesh_hand_mid_L_70 = new THREE.Mesh(
    mesh_hand_mid_L_70Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_hand_mid_L_70.name = "Hand mid L";
  if (endpoint_hand_mid_L_70) {
    mesh_hand_mid_L_70.position.copy(endpoint_hand_mid_L_70.midpoint);
    mesh_hand_mid_L_70.quaternion.copy(endpoint_hand_mid_L_70.quaternion);
  }
  mesh_hand_mid_L_70.castShadow = options.castShadow ?? true;
  mesh_hand_mid_L_70.receiveShadow = options.receiveShadow ?? true;
  mesh_hand_mid_L_70.userData.sculptComponent = {"id": "hand-mid-L", "name": "Hand mid L", "level": "meso", "role": "hand", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-mid-L", "attachment": {"parentSocket": "arm-mid-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.38, "height": 0.38, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0005, 0.4852, 0.0], "rotation": [-0.0, 0.0, 2.073]}, "actionProfile": {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hand_mid_L_70.add(mesh_hand_mid_L_70);
  meshes["hand-mid-L"] = mesh_hand_mid_L_70;
  colliders["hand-mid-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_hand_mid_L_70);

  const endpoint_cuff_mid_L_71 = makeAttachmentEndpoint(null);
  const node_cuff_mid_L_71 = new THREE.Group();
  node_cuff_mid_L_71.name = "Cuff scrunchie mid L__pivot";
  node_cuff_mid_L_71.scale.set(1, 1, 1);
  if (endpoint_cuff_mid_L_71) {
    node_cuff_mid_L_71.position.copy(endpoint_cuff_mid_L_71.start);
    node_cuff_mid_L_71.rotation.set(-0.0, 0.0, 0.0);
  } else {
    node_cuff_mid_L_71.position.set(0.0003, 0.3048, 0.0);
    node_cuff_mid_L_71.rotation.set(-0.0, 0.0, 0.0);
  }
  node_cuff_mid_L_71.userData.sculptComponent = {"id": "cuff-mid-L", "name": "Cuff scrunchie mid L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-mid-L", "attachment": {"parentSocket": "arm-mid-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0003, 0.3048, 0.0], "rotation": [-0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-teal", "materialLayers": ["cloth-teal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_cuff_mid_L_71.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["arm-mid-L"] ?? root).add(node_cuff_mid_L_71);
  nodes["cuff-mid-L"] = node_cuff_mid_L_71;
  const mesh_cuff_mid_L_71Geometry = endpoint_cuff_mid_L_71
    ? new THREE.CylinderGeometry(endpoint_cuff_mid_L_71.endRadius, endpoint_cuff_mid_L_71.baseRadius, endpoint_cuff_mid_L_71.length, 32, 12)
    : new THREE.TorusGeometry(0.45, 0.08, 24, 96);
  if (!endpoint_cuff_mid_L_71) {
    mesh_cuff_mid_L_71Geometry.scale(0.2, 0.2, 0.1);
  }
  const mesh_cuff_mid_L_71 = new THREE.Mesh(
    mesh_cuff_mid_L_71Geometry,
    materialMap["cloth-teal"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_cuff_mid_L_71.name = "Cuff scrunchie mid L";
  if (endpoint_cuff_mid_L_71) {
    mesh_cuff_mid_L_71.position.copy(endpoint_cuff_mid_L_71.midpoint);
    mesh_cuff_mid_L_71.quaternion.copy(endpoint_cuff_mid_L_71.quaternion);
  }
  mesh_cuff_mid_L_71.castShadow = options.castShadow ?? true;
  mesh_cuff_mid_L_71.receiveShadow = options.receiveShadow ?? true;
  mesh_cuff_mid_L_71.userData.sculptComponent = {"id": "cuff-mid-L", "name": "Cuff scrunchie mid L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-mid-L", "attachment": {"parentSocket": "arm-mid-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0003, 0.3048, 0.0], "rotation": [-0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-teal", "materialLayers": ["cloth-teal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(31, 138, 134, 1.0)", "secondaryAlbedo": "rgba(31, 138, 134, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_cuff_mid_L_71.add(mesh_cuff_mid_L_71);
  meshes["cuff-mid-L"] = mesh_cuff_mid_L_71;
  colliders["cuff-mid-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_cuff_mid_L_71);

  const attachment_arm_low_L_72 = {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_arm_low_L_72 = makeAttachmentEndpoint(attachment_arm_low_L_72);
  const node_arm_low_L_72 = new THREE.Group();
  node_arm_low_L_72.name = "Arm low L__pivot";
  node_arm_low_L_72.scale.set(1, 1, 1);
  if (endpoint_arm_low_L_72) {
    node_arm_low_L_72.position.copy(endpoint_arm_low_L_72.start);
    node_arm_low_L_72.rotation.set(-0.0, 0.0, -2.048);
  } else {
    node_arm_low_L_72.position.set(0.89, -0.417, 0.0);
    node_arm_low_L_72.rotation.set(-0.0, 0.0, -2.048);
  }
  node_arm_low_L_72.userData.sculptComponent = {"id": "arm-low-L", "name": "Arm low L", "level": "meso", "role": "arm", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4285714285714286, "height": 0.7769498184620185, "depth": 0.4285714285714286, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.89, -0.417, 0.0], "rotation": [-0.0, 0.0, -2.048]}, "actionProfile": {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_arm_low_L_72.userData.actionProfile = {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["torso"] ?? root).add(node_arm_low_L_72);
  nodes["arm-low-L"] = node_arm_low_L_72;
  const mesh_arm_low_L_72Geometry = endpoint_arm_low_L_72
    ? new THREE.CylinderGeometry(endpoint_arm_low_L_72.endRadius, endpoint_arm_low_L_72.baseRadius, endpoint_arm_low_L_72.length, 32, 12)
    : buildWatertightCapsule(0.35, 0.7, 16, 32, 1);
  if (!endpoint_arm_low_L_72) {
    mesh_arm_low_L_72Geometry.scale(0.4285714285714286, 0.7769498184620185, 0.4285714285714286);
  }
  const mesh_arm_low_L_72 = new THREE.Mesh(
    mesh_arm_low_L_72Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_arm_low_L_72.name = "Arm low L";
  if (endpoint_arm_low_L_72) {
    mesh_arm_low_L_72.position.copy(endpoint_arm_low_L_72.midpoint);
    mesh_arm_low_L_72.quaternion.copy(endpoint_arm_low_L_72.quaternion);
  }
  mesh_arm_low_L_72.castShadow = options.castShadow ?? true;
  mesh_arm_low_L_72.receiveShadow = options.receiveShadow ?? true;
  mesh_arm_low_L_72.userData.sculptComponent = {"id": "arm-low-L", "name": "Arm low L", "level": "meso", "role": "arm", "importance": 1.0, "confidence": 0.7, "primitive": "capsule", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "torso", "attachment": {"parentSocket": "torso-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4285714285714286, "height": 0.7769498184620185, "depth": 0.4285714285714286, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.89, -0.417, 0.0], "rotation": [-0.0, 0.0, -2.048]}, "actionProfile": {"animationRole": "arm", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_arm_low_L_72.add(mesh_arm_low_L_72);
  meshes["arm-low-L"] = mesh_arm_low_L_72;
  colliders["arm-low-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_arm_low_L_72);

  const endpoint_hand_low_L_73 = makeAttachmentEndpoint(null);
  const node_hand_low_L_73 = new THREE.Group();
  node_hand_low_L_73.name = "Hand low L__pivot";
  node_hand_low_L_73.scale.set(1, 1, 1);
  if (endpoint_hand_low_L_73) {
    node_hand_low_L_73.position.copy(endpoint_hand_low_L_73.start);
    node_hand_low_L_73.rotation.set(-0.0, 0.0, 2.048);
  } else {
    node_hand_low_L_73.position.set(0.0002, 0.5439, 0.0);
    node_hand_low_L_73.rotation.set(-0.0, 0.0, 2.048);
  }
  node_hand_low_L_73.userData.sculptComponent = {"id": "hand-low-L", "name": "Hand low L", "level": "meso", "role": "hand", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-low-L", "attachment": {"parentSocket": "arm-low-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.38, "height": 0.38, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0002, 0.5439, 0.0], "rotation": [-0.0, 0.0, 2.048]}, "actionProfile": {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hand_low_L_73.userData.actionProfile = {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["arm-low-L"] ?? root).add(node_hand_low_L_73);
  nodes["hand-low-L"] = node_hand_low_L_73;
  const mesh_hand_low_L_73Geometry = endpoint_hand_low_L_73
    ? new THREE.CylinderGeometry(endpoint_hand_low_L_73.endRadius, endpoint_hand_low_L_73.baseRadius, endpoint_hand_low_L_73.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_hand_low_L_73) {
    mesh_hand_low_L_73Geometry.scale(0.38, 0.38, 0.2);
  }
  const mesh_hand_low_L_73 = new THREE.Mesh(
    mesh_hand_low_L_73Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_hand_low_L_73.name = "Hand low L";
  if (endpoint_hand_low_L_73) {
    mesh_hand_low_L_73.position.copy(endpoint_hand_low_L_73.midpoint);
    mesh_hand_low_L_73.quaternion.copy(endpoint_hand_low_L_73.quaternion);
  }
  mesh_hand_low_L_73.castShadow = options.castShadow ?? true;
  mesh_hand_low_L_73.receiveShadow = options.receiveShadow ?? true;
  mesh_hand_low_L_73.userData.sculptComponent = {"id": "hand-low-L", "name": "Hand low L", "level": "meso", "role": "hand", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-low-L", "attachment": {"parentSocket": "arm-low-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.38, "height": 0.38, "depth": 0.2, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.0002, 0.5439, 0.0], "rotation": [-0.0, 0.0, 2.048]}, "actionProfile": {"animationRole": "hand", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_hand_low_L_73.add(mesh_hand_low_L_73);
  meshes["hand-low-L"] = mesh_hand_low_L_73;
  colliders["hand-low-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_hand_low_L_73);

  const endpoint_cuff_low_L_74 = makeAttachmentEndpoint(null);
  const node_cuff_low_L_74 = new THREE.Group();
  node_cuff_low_L_74.name = "Cuff scrunchie low L__pivot";
  node_cuff_low_L_74.scale.set(1, 1, 1);
  if (endpoint_cuff_low_L_74) {
    node_cuff_low_L_74.position.copy(endpoint_cuff_low_L_74.start);
    node_cuff_low_L_74.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_cuff_low_L_74.position.set(-0.0, 0.3636, 0.0);
    node_cuff_low_L_74.rotation.set(-0.0, 0.0, -0.0);
  }
  node_cuff_low_L_74.userData.sculptComponent = {"id": "cuff-low-L", "name": "Cuff scrunchie low L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-low-L", "attachment": {"parentSocket": "arm-low-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0, 0.3636, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_cuff_low_L_74.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["arm-low-L"] ?? root).add(node_cuff_low_L_74);
  nodes["cuff-low-L"] = node_cuff_low_L_74;
  const mesh_cuff_low_L_74Geometry = endpoint_cuff_low_L_74
    ? new THREE.CylinderGeometry(endpoint_cuff_low_L_74.endRadius, endpoint_cuff_low_L_74.baseRadius, endpoint_cuff_low_L_74.length, 32, 12)
    : new THREE.TorusGeometry(0.45, 0.08, 24, 96);
  if (!endpoint_cuff_low_L_74) {
    mesh_cuff_low_L_74Geometry.scale(0.2, 0.2, 0.1);
  }
  const mesh_cuff_low_L_74 = new THREE.Mesh(
    mesh_cuff_low_L_74Geometry,
    materialMap["cloth-orange"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_cuff_low_L_74.name = "Cuff scrunchie low L";
  if (endpoint_cuff_low_L_74) {
    mesh_cuff_low_L_74.position.copy(endpoint_cuff_low_L_74.midpoint);
    mesh_cuff_low_L_74.quaternion.copy(endpoint_cuff_low_L_74.quaternion);
  }
  mesh_cuff_low_L_74.castShadow = options.castShadow ?? true;
  mesh_cuff_low_L_74.receiveShadow = options.receiveShadow ?? true;
  mesh_cuff_low_L_74.userData.sculptComponent = {"id": "cuff-low-L", "name": "Cuff scrunchie low L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "arm-low-L", "attachment": {"parentSocket": "arm-low-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.2, "height": 0.2, "depth": 0.1, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.0, 0.3636, 0.0], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-orange", "materialLayers": ["cloth-orange"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(240, 138, 28, 1.0)", "secondaryAlbedo": "rgba(240, 138, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_cuff_low_L_74.add(mesh_cuff_low_L_74);
  meshes["cuff-low-L"] = mesh_cuff_low_L_74;
  colliders["cuff-low-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_cuff_low_L_74);

  const endpoint_prop_ball_75 = makeAttachmentEndpoint(null);
  const node_prop_ball_75 = new THREE.Group();
  node_prop_ball_75.name = "Star ball__pivot";
  node_prop_ball_75.scale.set(1, 1, 1);
  if (endpoint_prop_ball_75) {
    node_prop_ball_75.position.copy(endpoint_prop_ball_75.start);
    node_prop_ball_75.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_prop_ball_75.position.set(-0.267, 0.384, 0.05);
    node_prop_ball_75.rotation.set(-0.0, 0.0, -0.0);
  }
  node_prop_ball_75.userData.sculptComponent = {"id": "prop-ball", "name": "Star ball", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "hand-low-R", "attachment": {"parentSocket": "hand-low-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4, "height": 0.4, "depth": 0.4, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.267, 0.384, 0.05], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "ball-stars", "kind": "decal", "note": "orange star print on purple juggling ball"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_prop_ball_75.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["hand-low-R"] ?? root).add(node_prop_ball_75);
  nodes["prop-ball"] = node_prop_ball_75;
  const mesh_prop_ball_75Geometry = endpoint_prop_ball_75
    ? new THREE.CylinderGeometry(endpoint_prop_ball_75.endRadius, endpoint_prop_ball_75.baseRadius, endpoint_prop_ball_75.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_prop_ball_75) {
    mesh_prop_ball_75Geometry.scale(0.4, 0.4, 0.4);
  }
  const mesh_prop_ball_75 = new THREE.Mesh(
    mesh_prop_ball_75Geometry,
    materialMap["cloth-purple"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_prop_ball_75.name = "Star ball";
  if (endpoint_prop_ball_75) {
    mesh_prop_ball_75.position.copy(endpoint_prop_ball_75.midpoint);
    mesh_prop_ball_75.quaternion.copy(endpoint_prop_ball_75.quaternion);
  }
  mesh_prop_ball_75.castShadow = options.castShadow ?? true;
  mesh_prop_ball_75.receiveShadow = options.receiveShadow ?? true;
  mesh_prop_ball_75.userData.sculptComponent = {"id": "prop-ball", "name": "Star ball", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "hand-low-R", "attachment": {"parentSocket": "hand-low-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.4, "height": 0.4, "depth": 0.4, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.267, 0.384, 0.05], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "cloth-purple", "materialLayers": ["cloth-purple"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "ball-stars", "kind": "decal", "note": "orange star print on purple juggling ball"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(106, 63, 160, 1.0)", "secondaryAlbedo": "rgba(106, 63, 160, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_prop_ball_75.add(mesh_prop_ball_75);
  meshes["prop-ball"] = mesh_prop_ball_75;
  colliders["prop-ball"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_prop_ball_75);

  const endpoint_prop_cards_76 = makeAttachmentEndpoint(null);
  const node_prop_cards_76 = new THREE.Group();
  node_prop_cards_76.name = "Playing cards__pivot";
  node_prop_cards_76.scale.set(1, 1, 1);
  if (endpoint_prop_cards_76) {
    node_prop_cards_76.position.copy(endpoint_prop_cards_76.start);
    node_prop_cards_76.rotation.set(-0.0, 0.0, -0.3);
  } else {
    node_prop_cards_76.position.set(0.254, 0.217, 0.05);
    node_prop_cards_76.rotation.set(-0.0, 0.0, -0.3);
  }
  node_prop_cards_76.userData.sculptComponent = {"id": "prop-cards", "name": "Playing cards", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "box", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "hand-low-L", "attachment": {"parentSocket": "hand-low-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.45, "height": 0.5, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.254, 0.217, 0.05], "rotation": [-0.0, 0.0, -0.3]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_prop_cards_76.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["hand-low-L"] ?? root).add(node_prop_cards_76);
  nodes["prop-cards"] = node_prop_cards_76;
  const mesh_prop_cards_76Geometry = endpoint_prop_cards_76
    ? new THREE.CylinderGeometry(endpoint_prop_cards_76.endRadius, endpoint_prop_cards_76.baseRadius, endpoint_prop_cards_76.length, 32, 12)
    : new THREE.BoxGeometry(1, 1, 1, 12, 12, 12);
  if (!endpoint_prop_cards_76) {
    mesh_prop_cards_76Geometry.scale(0.45, 0.5, 0.03);
  }
  const mesh_prop_cards_76 = new THREE.Mesh(
    mesh_prop_cards_76Geometry,
    materialMap["ink-white"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_prop_cards_76.name = "Playing cards";
  if (endpoint_prop_cards_76) {
    mesh_prop_cards_76.position.copy(endpoint_prop_cards_76.midpoint);
    mesh_prop_cards_76.quaternion.copy(endpoint_prop_cards_76.quaternion);
  }
  mesh_prop_cards_76.castShadow = options.castShadow ?? true;
  mesh_prop_cards_76.receiveShadow = options.receiveShadow ?? true;
  mesh_prop_cards_76.userData.sculptComponent = {"id": "prop-cards", "name": "Playing cards", "level": "meso", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "box", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "hand-low-L", "attachment": {"parentSocket": "hand-low-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.45, "height": 0.5, "depth": 0.03, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.254, 0.217, 0.05], "rotation": [-0.0, 0.0, -0.3]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "ink-white", "materialLayers": ["ink-white"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(255, 255, 255, 1.0)", "secondaryAlbedo": "rgba(255, 255, 255, 1.0)", "materialClass": "plastic", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_prop_cards_76.add(mesh_prop_cards_76);
  meshes["prop-cards"] = mesh_prop_cards_76;
  colliders["prop-cards"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_prop_cards_76);

  const attachment_tail_R_77 = {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_tail_R_77 = makeAttachmentEndpoint(attachment_tail_R_77);
  const node_tail_R_77 = new THREE.Group();
  node_tail_R_77.name = "Tail R__pivot";
  node_tail_R_77.scale.set(1, 1, 1);
  if (endpoint_tail_R_77) {
    node_tail_R_77.position.copy(endpoint_tail_R_77.start);
    node_tail_R_77.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_tail_R_77.position.set(-0.25, 1.35, -0.35);
    node_tail_R_77.rotation.set(-0.0, 0.0, -0.0);
  }
  node_tail_R_77.userData.sculptComponent = {"id": "tail-R", "name": "Tail R", "level": "meso", "role": "tail", "importance": 1.0, "confidence": 0.7, "primitive": "tube", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[0.0, 0.0, 0.0], [-0.5, -0.1, -0.1], [-0.95, 0.2, -0.15], [-1.25, 0.5, -0.1]], "radius": 0.04, "closed": false}}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 1, "height": 1, "depth": 1, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.25, 1.35, -0.35], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "tail", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_tail_R_77.userData.actionProfile = {"animationRole": "tail", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["root"] ?? root).add(node_tail_R_77);
  nodes["tail-R"] = node_tail_R_77;
  const mesh_tail_R_77Geometry = endpoint_tail_R_77
    ? new THREE.CylinderGeometry(endpoint_tail_R_77.endRadius, endpoint_tail_R_77.baseRadius, endpoint_tail_R_77.length, 32, 12)
    : buildTubeGeometry({"points": [[0.0, 0.0, 0.0], [-0.5, -0.1, -0.1], [-0.95, 0.2, -0.15], [-1.25, 0.5, -0.1]], "radius": 0.04, "closed": false});
  if (!endpoint_tail_R_77) {
    mesh_tail_R_77Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_tail_R_77 = new THREE.Mesh(
    mesh_tail_R_77Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_tail_R_77.name = "Tail R";
  if (endpoint_tail_R_77) {
    mesh_tail_R_77.position.copy(endpoint_tail_R_77.midpoint);
    mesh_tail_R_77.quaternion.copy(endpoint_tail_R_77.quaternion);
  }
  mesh_tail_R_77.castShadow = options.castShadow ?? true;
  mesh_tail_R_77.receiveShadow = options.receiveShadow ?? true;
  mesh_tail_R_77.userData.sculptComponent = {"id": "tail-R", "name": "Tail R", "level": "meso", "role": "tail", "importance": 1.0, "confidence": 0.7, "primitive": "tube", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[0.0, 0.0, 0.0], [-0.5, -0.1, -0.1], [-0.95, 0.2, -0.15], [-1.25, 0.5, -0.1]], "radius": 0.04, "closed": false}}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 1, "height": 1, "depth": 1, "units": "relative", "confidence": 0.7}, "transform": {"position": [-0.25, 1.35, -0.35], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "tail", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_tail_R_77.add(mesh_tail_R_77);
  meshes["tail-R"] = mesh_tail_R_77;
  colliders["tail-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_tail_R_77);

  const endpoint_charm_R_78 = makeAttachmentEndpoint(null);
  const node_charm_R_78 = new THREE.Group();
  node_charm_R_78.name = "Tail charm R__pivot";
  node_charm_R_78.scale.set(1, 1, 1);
  if (endpoint_charm_R_78) {
    node_charm_R_78.position.copy(endpoint_charm_R_78.start);
    node_charm_R_78.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_charm_R_78.position.set(-1.25, 0.45, 0.05);
    node_charm_R_78.rotation.set(-0.0, 0.0, -0.0);
  }
  node_charm_R_78.userData.sculptComponent = {"id": "charm-R", "name": "Tail charm R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "tail-R", "attachment": {"parentSocket": "tail-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.14, "height": 0.14, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [-1.25, 0.45, 0.05], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "gold-metal", "materialLayers": ["gold-metal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(217, 165, 32, 1.0)", "secondaryAlbedo": "rgba(217, 165, 32, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_charm_R_78.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["tail-R"] ?? root).add(node_charm_R_78);
  nodes["charm-R"] = node_charm_R_78;
  const mesh_charm_R_78Geometry = endpoint_charm_R_78
    ? new THREE.CylinderGeometry(endpoint_charm_R_78.endRadius, endpoint_charm_R_78.baseRadius, endpoint_charm_R_78.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_charm_R_78) {
    mesh_charm_R_78Geometry.scale(0.14, 0.14, 0.05);
  }
  const mesh_charm_R_78 = new THREE.Mesh(
    mesh_charm_R_78Geometry,
    materialMap["gold-metal"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_charm_R_78.name = "Tail charm R";
  if (endpoint_charm_R_78) {
    mesh_charm_R_78.position.copy(endpoint_charm_R_78.midpoint);
    mesh_charm_R_78.quaternion.copy(endpoint_charm_R_78.quaternion);
  }
  mesh_charm_R_78.castShadow = options.castShadow ?? true;
  mesh_charm_R_78.receiveShadow = options.receiveShadow ?? true;
  mesh_charm_R_78.userData.sculptComponent = {"id": "charm-R", "name": "Tail charm R", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "tail-R", "attachment": {"parentSocket": "tail-R-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.14, "height": 0.14, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [-1.25, 0.45, 0.05], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "gold-metal", "materialLayers": ["gold-metal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(217, 165, 32, 1.0)", "secondaryAlbedo": "rgba(217, 165, 32, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_charm_R_78.add(mesh_charm_R_78);
  meshes["charm-R"] = mesh_charm_R_78;
  colliders["charm-R"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_charm_R_78);

  const attachment_tail_L_79 = {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02};
  const endpoint_tail_L_79 = makeAttachmentEndpoint(attachment_tail_L_79);
  const node_tail_L_79 = new THREE.Group();
  node_tail_L_79.name = "Tail L__pivot";
  node_tail_L_79.scale.set(1, 1, 1);
  if (endpoint_tail_L_79) {
    node_tail_L_79.position.copy(endpoint_tail_L_79.start);
    node_tail_L_79.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_tail_L_79.position.set(0.25, 1.35, -0.35);
    node_tail_L_79.rotation.set(-0.0, 0.0, -0.0);
  }
  node_tail_L_79.userData.sculptComponent = {"id": "tail-L", "name": "Tail L", "level": "meso", "role": "tail", "importance": 1.0, "confidence": 0.7, "primitive": "tube", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[0.0, 0.0, 0.0], [0.5, -0.1, -0.1], [0.95, 0.2, -0.15], [1.25, 0.5, -0.1]], "radius": 0.04, "closed": false}}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 1, "height": 1, "depth": 1, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.25, 1.35, -0.35], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "tail", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tail-charm-hook", "kind": "contour", "note": "tail curls outward ending in a hanging crescent/star charm"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_tail_L_79.userData.actionProfile = {"animationRole": "tail", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["root"] ?? root).add(node_tail_L_79);
  nodes["tail-L"] = node_tail_L_79;
  const mesh_tail_L_79Geometry = endpoint_tail_L_79
    ? new THREE.CylinderGeometry(endpoint_tail_L_79.endRadius, endpoint_tail_L_79.baseRadius, endpoint_tail_L_79.length, 32, 12)
    : buildTubeGeometry({"points": [[0.0, 0.0, 0.0], [0.5, -0.1, -0.1], [0.95, 0.2, -0.15], [1.25, 0.5, -0.1]], "radius": 0.04, "closed": false});
  if (!endpoint_tail_L_79) {
    mesh_tail_L_79Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_tail_L_79 = new THREE.Mesh(
    mesh_tail_L_79Geometry,
    materialMap["fur-black"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_tail_L_79.name = "Tail L";
  if (endpoint_tail_L_79) {
    mesh_tail_L_79.position.copy(endpoint_tail_L_79.midpoint);
    mesh_tail_L_79.quaternion.copy(endpoint_tail_L_79.quaternion);
  }
  mesh_tail_L_79.castShadow = options.castShadow ?? true;
  mesh_tail_L_79.receiveShadow = options.receiveShadow ?? true;
  mesh_tail_L_79.userData.sculptComponent = {"id": "tail-L", "name": "Tail L", "level": "meso", "role": "tail", "importance": 1.0, "confidence": 0.7, "primitive": "tube", "topologyClass": "assembled-solid", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[0.0, 0.0, 0.0], [0.5, -0.1, -0.1], [0.95, 0.2, -0.15], [1.25, 0.5, -0.1]], "radius": 0.04, "closed": false}}, "parent": "root", "attachment": {"parentSocket": "root-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 1, "height": 1, "depth": 1, "units": "relative", "confidence": 0.7}, "transform": {"position": [0.25, 1.35, -0.35], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "tail", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "fur-black", "materialLayers": ["fur-black"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "tail-charm-hook", "kind": "contour", "note": "tail curls outward ending in a hanging crescent/star charm"}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(22, 22, 28, 1.0)", "secondaryAlbedo": "rgba(22, 22, 28, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_tail_L_79.add(mesh_tail_L_79);
  meshes["tail-L"] = mesh_tail_L_79;
  colliders["tail-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_tail_L_79);

  const endpoint_charm_L_80 = makeAttachmentEndpoint(null);
  const node_charm_L_80 = new THREE.Group();
  node_charm_L_80.name = "Tail charm L__pivot";
  node_charm_L_80.scale.set(1, 1, 1);
  if (endpoint_charm_L_80) {
    node_charm_L_80.position.copy(endpoint_charm_L_80.start);
    node_charm_L_80.rotation.set(-0.0, 0.0, -0.0);
  } else {
    node_charm_L_80.position.set(1.25, 0.45, 0.05);
    node_charm_L_80.rotation.set(-0.0, 0.0, -0.0);
  }
  node_charm_L_80.userData.sculptComponent = {"id": "charm-L", "name": "Tail charm L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "tail-L", "attachment": {"parentSocket": "tail-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.14, "height": 0.14, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [1.25, 0.45, 0.05], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "gold-metal", "materialLayers": ["gold-metal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(217, 165, 32, 1.0)", "secondaryAlbedo": "rgba(217, 165, 32, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_charm_L_80.userData.actionProfile = {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}};
  (nodes["tail-L"] ?? root).add(node_charm_L_80);
  nodes["charm-L"] = node_charm_L_80;
  const mesh_charm_L_80Geometry = endpoint_charm_L_80
    ? new THREE.CylinderGeometry(endpoint_charm_L_80.endRadius, endpoint_charm_L_80.baseRadius, endpoint_charm_L_80.length, 32, 12)
    : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_charm_L_80) {
    mesh_charm_L_80Geometry.scale(0.14, 0.14, 0.05);
  }
  const mesh_charm_L_80 = new THREE.Mesh(
    mesh_charm_L_80Geometry,
    materialMap["gold-metal"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_charm_L_80.name = "Tail charm L";
  if (endpoint_charm_L_80) {
    mesh_charm_L_80.position.copy(endpoint_charm_L_80.midpoint);
    mesh_charm_L_80.quaternion.copy(endpoint_charm_L_80.quaternion);
  }
  mesh_charm_L_80.castShadow = options.castShadow ?? true;
  mesh_charm_L_80.receiveShadow = options.receiveShadow ?? true;
  mesh_charm_L_80.userData.sculptComponent = {"id": "charm-L", "name": "Tail charm L", "level": "micro", "role": "part", "importance": 1.0, "confidence": 0.7, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "tail-L", "attachment": {"parentSocket": "tail-L-socket", "contactType": "overlap", "localStart": [0, 0, 0], "localEnd": [0, 0, 0], "embed": 0.02, "overlap": 0.04, "gapTolerance": 0.02}, "dimensions": {"width": 0.14, "height": 0.14, "depth": 0.05, "units": "relative", "confidence": 0.7}, "transform": {"position": [1.25, 0.45, 0.05], "rotation": [-0.0, 0.0, -0.0]}, "actionProfile": {"animationRole": "part", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.5}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "root", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "base"}}, "material": "gold-metal", "materialLayers": ["gold-metal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"dominantAlbedo": "rgba(217, 165, 32, 1.0)", "secondaryAlbedo": "rgba(217, 165, 32, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.7, "evidenceRef": "reference.jpg flat toon fills (single front view)"}};
  node_charm_L_80.add(mesh_charm_L_80);
  meshes["charm-L"] = mesh_charm_L_80;
  colliders["charm-L"] = {"type": "box", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "Replace with sphere/capsule/compound proxy when the object shape demands it."};
  destructionGroups["root"] ??= [];
  destructionGroups["root"].push(node_charm_L_80);

  root.userData.sculptRuntime = { nodes, meshes, sockets, colliders, destructionGroups } satisfies ProceduralModelRuntime;
  root.userData.lookDevTargets = {"qualityPriority": "reference-fidelity", "materialPass": {"albedoPaletteRequired": true, "roughnessVariationRequired": true, "normalOrBumpRequired": true, "localOverridesRequired": true, "minimumTextureResolution": 1024, "preferredTextureResolution": 2048, "independentMapChannels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "requiredSurfaceFrequencyBands": ["macro", "meso", "micro"], "geometryReliefRequiredWhenSilhouetteAffected": true, "referencePbrExtraction": {"requiredWhenSourceImagePresent": true, "targetThreshold": 0.7, "stopOnLowConfidence": true, "script": "forge/stage1_intake/extract_pbr_evidence.py", "acceptedLimitation": "single-image extraction is reference-derived inference, not exact photogrammetry"}, "mustAvoid": ["single flat albedo per material", "uniform roughness", "albedo texture reused as roughness/height/normal/AO", "single-frequency random noise", "plastic-looking smooth bark, stone, cloth, foliage, or aged material", "local color/detail described only in prose without material masks", "claiming exact PBR recovery when confidence is below the target threshold"]}, "lightingPass": {"requiredTerms": ["key light", "fill light", "rim or environment light", "exposure", "tone mapping", "background", "contact shadow"], "mustAvoid": ["ambient-only lighting", "flat value range", "missing contact shadow", "reference lighting copied without separating material readability"]}, "screenshotReview": ["Compare albedo palette and local color zones.", "Compare roughness/normal/bump response under light.", "Compare cavity dirt, edge wear, stains, moss, scratches, or other local masks.", "Compare key/fill/rim structure, exposure, tone mapping, background, and contact shadows.", "Capture a neutral-light render to verify material readability without reference lighting.", "Capture a grazing-light close-up to expose flat normals, uniform roughness, tiling, and plastic highlights.", "Capture a reference-matched render from the same camera framing as the source."]};
  root.userData.actionReadiness = {
    note: 'Use root.userData.sculptRuntime.nodes for transforms, sockets for attachments, colliders for physics proxies, and destructionGroups for breakable sets.',
  };
  return root;
}

export function createMouseJesterLookDevLights(
  mode: 'neutral' | 'grazing' | 'reference' = 'neutral',
): THREE.Group {
  const lights = new THREE.Group();
  lights.name = "Mouse Jester look-dev lights";
  const hemi = new THREE.HemisphereLight(
    mode === 'reference' ? 0xfff0d6 : 0xf2f4ff,
    0x363b42,
    mode === 'grazing' ? 0.28 : mode === 'reference' ? 0.72 : 0.85,
  );
  lights.add(hemi);
  const key = new THREE.DirectionalLight(
    mode === 'reference' ? 0xffcf8a : 0xfff4e8,
    mode === 'grazing' ? 4.2 : mode === 'reference' ? 2.6 : 2.15,
  );
  if (mode === 'grazing') key.position.set(7.5, 1.1, 4.0);
  else if (mode === 'reference') key.position.set(-4.5, 7.5, 5.0);
  else key.position.set(-4.0, 6.0, 5.5);
  key.castShadow = true;
  key.shadow.mapSize.set(4096, 4096);
  key.shadow.bias = -0.00025;
  key.shadow.normalBias = 0.018;
  key.shadow.radius = 7;
  key.shadow.blurSamples = 24;
  key.shadow.camera.near = 0.5;
  key.shadow.camera.far = 30;
  key.shadow.camera.left = -2.6;
  key.shadow.camera.right = 2.6;
  key.shadow.camera.top = 2.6;
  key.shadow.camera.bottom = -2.6;
  key.shadow.camera.updateProjectionMatrix();
  lights.add(key);
  const fill = new THREE.DirectionalLight(0xa8c4ff, mode === 'grazing' ? 0.12 : 0.42);
  fill.position.set(4.0, 3.0, 3.5);
  lights.add(fill);
  const rim = new THREE.DirectionalLight(0xfff1c4, mode === 'grazing' ? 0.28 : 0.85);
  rim.position.set(0.5, 4.5, -6.0);
  lights.add(rim);
  lights.userData.reviewMode = mode;
  lights.userData.lightingFromPhoto = ["key: soft frontal-top directional (flat toon, intensity ~1.0), warm white", "fill: hemisphere ambient 0.7 sky/ground cream", "rim: faint back-light to separate black fur from background", "exposure 1.0, ACES filmic tone mapping, sRGB output", "contact shadow: soft ground shadow ellipse under shoes (ambient occlusion on floor)"];
  lights.userData.lookDevTargets = {"qualityPriority": "reference-fidelity", "materialPass": {"albedoPaletteRequired": true, "roughnessVariationRequired": true, "normalOrBumpRequired": true, "localOverridesRequired": true, "minimumTextureResolution": 1024, "preferredTextureResolution": 2048, "independentMapChannels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "requiredSurfaceFrequencyBands": ["macro", "meso", "micro"], "geometryReliefRequiredWhenSilhouetteAffected": true, "referencePbrExtraction": {"requiredWhenSourceImagePresent": true, "targetThreshold": 0.7, "stopOnLowConfidence": true, "script": "forge/stage1_intake/extract_pbr_evidence.py", "acceptedLimitation": "single-image extraction is reference-derived inference, not exact photogrammetry"}, "mustAvoid": ["single flat albedo per material", "uniform roughness", "albedo texture reused as roughness/height/normal/AO", "single-frequency random noise", "plastic-looking smooth bark, stone, cloth, foliage, or aged material", "local color/detail described only in prose without material masks", "claiming exact PBR recovery when confidence is below the target threshold"]}, "lightingPass": {"requiredTerms": ["key light", "fill light", "rim or environment light", "exposure", "tone mapping", "background", "contact shadow"], "mustAvoid": ["ambient-only lighting", "flat value range", "missing contact shadow", "reference lighting copied without separating material readability"]}, "screenshotReview": ["Compare albedo palette and local color zones.", "Compare roughness/normal/bump response under light.", "Compare cavity dirt, edge wear, stains, moss, scratches, or other local masks.", "Compare key/fill/rim structure, exposure, tone mapping, background, and contact shadows.", "Capture a neutral-light render to verify material readability without reference lighting.", "Capture a grazing-light close-up to expose flat normals, uniform roughness, tiling, and plastic highlights.", "Capture a reference-matched render from the same camera framing as the source."]};
  return lights;
}

// PBR materials (clearcoat/iridescence/transmission/anisotropy) need an environment
// map to visually behave as intended — call this once per renderer and assign the
// result to scene.environment before rendering. No external HDR asset required.
export function createMouseJesterEnvironment(renderer: THREE.WebGLRenderer): THREE.Texture {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const texture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();
  return texture;
}

// Plan 1.3 §3.2 — auto-framing by bounding box. The Divine Eye can only compare a
// render to the reference if the object is FRAMED consistently (an object framed
// differently scores as wrong even when its shape is right). This positions the camera
// deterministically from the object's bounding box so it fills the frame at a stable
// margin, and sets near/far to the object scale. Call after adding the model to the
// scene, and again on resize (after updating camera.aspect).
export function frameMouseJesterCamera(
  camera: THREE.PerspectiveCamera,
  object: THREE.Object3D,
  options: { margin?: number; azimuthDeg?: number; elevationDeg?: number } = {},
): void {
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const margin = options.margin ?? 1.15;
  const maxDim = Math.max(size.x, size.y, size.z) * margin;
  const fov = (camera.fov * Math.PI) / 180;
  // distance so the largest object dimension fits vertically in the frame
  const distance = (maxDim / 2) / Math.tan(fov / 2);
  const az = ((options.azimuthDeg ?? 0) * Math.PI) / 180;
  const el = ((options.elevationDeg ?? 0) * Math.PI) / 180;
  const dir = new THREE.Vector3(
    Math.sin(az) * Math.cos(el),
    Math.sin(el),
    Math.cos(az) * Math.cos(el),
  );
  camera.position.copy(center).addScaledVector(dir, distance);
  camera.near = Math.max(0.01, distance - maxDim);
  camera.far = distance + maxDim * 2;
  camera.lookAt(center);
  camera.updateProjectionMatrix();
}

// Plan 1.3 §3.2c — PRESENTATION composer (DOF + bloom). CRITICAL (R-POSTFX): this is
// for the showcase/hero render ONLY. The Divine Eye's EVALUATION render MUST use a
// plain renderer with NO composer — bloom blows highlights and DOF blurs edges, which
// would corrupt the deterministic IoU/DCD/edge/blowout signals. Enable dof/bloom ONLY
// when the reference photo actually exhibits them (detect_reference_effects.py authorizes).
export function createMouseJesterPresentationComposer(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  options: { dof?: boolean; bloom?: boolean; bloomStrength?: number; dofFocus?: number; dofAperture?: number } = {},
): EffectComposer {
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  if (options.dof) {
    composer.addPass(new BokehPass(scene, camera, {
      focus: options.dofFocus ?? 10.0,
      aperture: options.dofAperture ?? 0.0002,
      maxblur: 0.01,
    }));
  }
  if (options.bloom) {
    const size = new THREE.Vector2();
    renderer.getSize(size);
    composer.addPass(new UnrealBloomPass(size, options.bloomStrength ?? 0.4, 0.4, 0.85));
  }
  return composer;
}

export function configureMouseJesterRenderer(renderer: THREE.WebGLRenderer): void {
  // Load-bearing for view-dependent finishes (anodized / Doppler): without ACES + sRGB
  // the environment reflection reads flat/washed instead of a believable metal response.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
}

export function createMouseJesterInspectControls(
  camera: THREE.Camera,
  domElement: HTMLElement,
): OrbitControls {
  // View-dependent finishes only read correctly once the user orbits — their color
  // comes from the environment reflection, not albedo, so free rotation matters here.
  const controls = new OrbitControls(camera, domElement);
  controls.enableDamping = true;
  controls.minDistance = 1.0;
  controls.maxDistance = 8.0;
  controls.autoRotate = false;
  return controls;
}

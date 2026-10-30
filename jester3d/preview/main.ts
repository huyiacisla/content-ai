import * as THREE from 'three';
import { createMouseJesterModel, createMouseJesterLookDevLights, frameMouseJesterCamera, configureMouseJesterRenderer } from '../src/createObjectModel';

const W = 1000, H = 1000;
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(W, H);
configureMouseJesterRenderer(renderer);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xefe6d6);
const model = createMouseJesterModel();
scene.add(model); (window as any).__model = model;
scene.add(createMouseJesterLookDevLights('reference'));
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
(window as any).shot = (az: number, el = 0) => {
  frameMouseJesterCamera(camera, model, { margin: 1.12, azimuthDeg: az, elevationDeg: el });
  renderer.render(scene, camera);
  return renderer.domElement.toDataURL('image/png');
};
(window as any).ready = true;

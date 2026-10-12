/** Drawing helpers shared by entity views and tools: fat lines, point sprites, materials. */
import * as THREE from "three";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import type { BSplineCurve2Def, Frame, Segment2D, Vec2, Vec3 } from "../core/types";
import { frameToWorld } from "../core/vec";

// Fat-line materials need the viewport resolution; keep a registry and update on resize.
const lineMaterials = new Set<LineMaterial>();
const resolution = new THREE.Vector2(1, 1);

export function setLineResolution(w: number, h: number): void {
  resolution.set(w, h);
  for (const m of lineMaterials) m.resolution.copy(resolution);
}

export function lineMaterial(opts: {
  color: THREE.ColorRepresentation;
  width?: number;
  dashed?: boolean;
  dashSize?: number;
  opacity?: number;
  depthTest?: boolean;
}): LineMaterial {
  const m = new LineMaterial({
    color: new THREE.Color(opts.color).getHex(),
    linewidth: opts.width ?? 2,
    dashed: !!opts.dashed,
    dashSize: opts.dashSize ?? 3,
    gapSize: (opts.dashSize ?? 3) * 0.7,
    transparent: (opts.opacity ?? 1) < 1 || opts.depthTest === false,
    opacity: opts.opacity ?? 1,
    depthTest: opts.depthTest ?? true,
    worldUnits: false,
  });
  m.resolution.copy(resolution);
  lineMaterials.add(m);
  const dispose = m.dispose.bind(m);
  m.dispose = () => {
    lineMaterials.delete(m);
    dispose();
  };
  return m;
}

/** Fat line segments from a flat array of segment endpoint pairs (xyz xyz ...). */
export function fatSegments(pairs: ArrayLike<number>, material: LineMaterial): LineSegments2 {
  const g = new LineSegmentsGeometry();
  g.setPositions(pairs instanceof Float32Array ? pairs : Float32Array.from(pairs));
  const l = new LineSegments2(g, material);
  if (material.dashed) l.computeLineDistances();
  return l;
}

/** Convert polylines to segment pairs. */
export function polylinesToPairs(polylines: { points: Vec3[]; closed?: boolean }[]): Float32Array {
  let n = 0;
  for (const pl of polylines) n += Math.max(0, pl.points.length - 1) + (pl.closed && pl.points.length > 2 ? 1 : 0);
  const out = new Float32Array(n * 6);
  let k = 0;
  for (const pl of polylines) {
    const pts = pl.points;
    const cnt = pts.length - 1 + (pl.closed && pts.length > 2 ? 1 : 0);
    for (let i = 0; i < cnt; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      out[k++] = a[0]; out[k++] = a[1]; out[k++] = a[2];
      out[k++] = b[0]; out[k++] = b[1]; out[k++] = b[2];
    }
  }
  return out;
}

let dotTexture: THREE.Texture | null = null;
function getDotTexture(): THREE.Texture {
  if (dotTexture) return dotTexture;
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const ctx = c.getContext("2d")!;
  ctx.beginPath();
  ctx.arc(32, 32, 26, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.lineWidth = 8;
  ctx.strokeStyle = "rgba(0,0,0,0.55)";
  ctx.stroke();
  dotTexture = new THREE.CanvasTexture(c);
  return dotTexture;
}

/** Screen-sized round point sprites. */
export function pointSprites(
  positions: ArrayLike<number>,
  color: THREE.ColorRepresentation,
  sizePx = 8,
  opts: { depthTest?: boolean; colors?: Float32Array } = {},
): THREE.Points {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(positions instanceof Float32Array ? positions : Float32Array.from(positions), 3));
  if (opts.colors) g.setAttribute("color", new THREE.BufferAttribute(opts.colors, 3));
  const m = new THREE.PointsMaterial({
    color: opts.colors ? 0xffffff : color,
    size: sizePx * Math.min(window.devicePixelRatio, 2),
    sizeAttenuation: false,
    map: getDotTexture(),
    alphaTest: 0.4,
    transparent: true,
    depthTest: opts.depthTest ?? true,
    vertexColors: !!opts.colors,
  });
  return new THREE.Points(g, m);
}

/** Plain small dots (no texture) for dense point sets such as section points. */
export function pointCloudObject(positions: Float32Array, color: THREE.ColorRepresentation, sizePx = 2.5, colors?: Uint8Array): THREE.Points {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  if (colors) g.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
  const m = new THREE.PointsMaterial({ color: colors ? 0xffffff : color, size: sizePx, sizeAttenuation: false, vertexColors: !!colors });
  return new THREE.Points(g, m);
}

export function translucentMaterial(color: THREE.ColorRepresentation, opacity = 0.35): THREE.MeshPhongMaterial {
  return new THREE.MeshPhongMaterial({
    color,
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
    depthWrite: false,
    shininess: 30,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
}

// ---------------------------------------------------------------------------
// 2D segment sampling (sketch display)
// ---------------------------------------------------------------------------

function deBoor2(c: BSplineCurve2Def, t: number): Vec2 {
  const p = c.degree;
  const U = c.knots;
  const n = c.controlPoints.length - 1;
  let k = p;
  if (t >= U[n + 1]) k = n;
  else while (k < n && !(t >= U[k] && t < U[k + 1])) k++;
  const w = c.weights;
  const d: [number, number, number][] = [];
  for (let j = 0; j <= p; j++) {
    const cp = c.controlPoints[j + k - p];
    const wt = w ? w[j + k - p] : 1;
    d.push([cp[0] * wt, cp[1] * wt, wt]);
  }
  for (let r = 1; r <= p; r++)
    for (let j = p; j >= r; j--) {
      const i = j + k - p;
      const denom = U[i + p - r + 1] - U[i];
      const a = denom === 0 ? 0 : (t - U[i]) / denom;
      d[j] = [(1 - a) * d[j - 1][0] + a * d[j][0], (1 - a) * d[j - 1][1] + a * d[j][1], (1 - a) * d[j - 1][2] + a * d[j][2]];
    }
  return [d[p][0] / d[p][2], d[p][1] / d[p][2]];
}

/** Sample a 2D segment into a polyline with roughly `tol` chordal accuracy. */
export function sampleSegment2D(s: Segment2D, tol = 0.05): Vec2[] {
  switch (s.type) {
    case "line":
      return [s.a, s.b];
    case "circle": {
      const n = Math.max(24, Math.min(256, Math.ceil(Math.PI / Math.acos(Math.max(-1, 1 - tol / Math.max(s.radius, 1e-9))))));
      const out: Vec2[] = [];
      for (let i = 0; i <= n; i++) {
        const a = (i / n) * Math.PI * 2;
        out.push([s.center[0] + s.radius * Math.cos(a), s.center[1] + s.radius * Math.sin(a)]);
      }
      return out;
    }
    case "arc": {
      const { start, mid, end } = s;
      const d = 2 * (start[0] * (mid[1] - end[1]) + mid[0] * (end[1] - start[1]) + end[0] * (start[1] - mid[1]));
      if (Math.abs(d) < 1e-12) return [start, end];
      const a2 = start[0] ** 2 + start[1] ** 2, b2 = mid[0] ** 2 + mid[1] ** 2, c2 = end[0] ** 2 + end[1] ** 2;
      const cx = (a2 * (mid[1] - end[1]) + b2 * (end[1] - start[1]) + c2 * (start[1] - mid[1])) / d;
      const cy = (a2 * (end[0] - mid[0]) + b2 * (start[0] - end[0]) + c2 * (mid[0] - start[0])) / d;
      const r = Math.hypot(start[0] - cx, start[1] - cy);
      const a0 = Math.atan2(start[1] - cy, start[0] - cx);
      const am = Math.atan2(mid[1] - cy, mid[0] - cx);
      const a1 = Math.atan2(end[1] - cy, end[0] - cx);
      const norm = (x: number) => ((x % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      // CCW sweep from a0 to a1; if mid is not inside, go CW
      let sweep = norm(a1 - a0);
      if (norm(am - a0) > sweep) sweep = sweep - 2 * Math.PI;
      const n = Math.max(8, Math.min(256, Math.ceil((Math.abs(sweep) / (2 * Math.PI)) * 96)));
      const out: Vec2[] = [];
      for (let i = 0; i <= n; i++) {
        const a = a0 + (sweep * i) / n;
        out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
      }
      return out;
    }
    case "bspline": {
      const c = s.curve;
      const t0 = c.knots[c.degree], t1 = c.knots[c.knots.length - c.degree - 1];
      const n = Math.max(16, c.controlPoints.length * 12);
      const out: Vec2[] = [];
      for (let i = 0; i <= n; i++) out.push(deBoor2(c, t0 + ((t1 - t0) * i) / n));
      return out;
    }
  }
}

export function segment2DToWorld(frame: Frame, s: Segment2D, tol?: number): Vec3[] {
  return sampleSegment2D(s, tol).map((p) => frameToWorld(frame, p));
}

export function disposeObject(o: THREE.Object3D): void {
  o.traverse((c) => {
    const anyC = c as THREE.Mesh;
    anyC.geometry?.dispose?.();
    const mat = anyC.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
    else mat?.dispose?.();
  });
}

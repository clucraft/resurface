/**
 * Segmentation of a triangle mesh into regions that each follow one analytic surface (plane,
 * cylinder, cone, sphere, torus). Designed for meshes exported from CAD (vertices lie on the
 * true surfaces, triangles never straddle two faces) and tolerant enough for clean scans when
 * the tolerance is raised.
 *
 *  1. Planes, largest facets first. A small flat patch is rejected when a curved surface explains
 *     it together with its surroundings (the flat strips of a coarsely tessellated cylinder).
 *  2. Curved surfaces from seed patches: every kind is fitted to the patch, grown over edge
 *     neighbours while vertices stay within tolerance and facet normals agree, and the kind that
 *     claims the most area wins (ties go to the simpler kind).
 *  3. Leftovers are absorbed into an adjacent region when they fit it; what remains is faceted.
 *  4. Adjacent regions that fit one common surface are merged.
 */
import type { MeshData, ProgressFn, Vec3 } from "../core/types";
import { buildTriangleNeighbors } from "../mesh/topology";
import { fitSurface, minCurvatureRadius, surfDistance, surfNormal, surfaceScale, SURF_DOF, type ConvSurface, type FitInputEx, type SurfKind } from "./surfaces";

export interface SegmentOptions {
  /** max distance of a vertex from its region's surface (mm) */
  tolerance: number;
  /** base angle allowed between a facet normal and the surface normal (deg) */
  angleTolDeg: number;
  /** model size (bounding-box diagonal), for sanity checks on fitted radii */
  size: number;
  kinds?: SurfKind[];
  neighbors?: Int32Array;
  onProgress?: ProgressFn;
}

export interface Region {
  id: number;
  surface: ConvSurface;
  triangles: number;
  area: number;
  rms: number;
  maxDev: number;
}

export interface Segmentation {
  /** region id per triangle, -1 = faceted (no surface held) */
  labels: Int32Array;
  regions: Region[];
}

const DEBUG = typeof process !== "undefined" && !!process.env?.DEBUG_CONVERT;
const ALL_KINDS: SurfKind[] = ["plane", "cylinder", "cone", "sphere", "torus"];
const DEG = Math.PI / 180;

/** a candidate region during model selection */
interface Cand {
  kind: SurfKind;
  area: number;
  /** RMS chord-midpoint distance */
  sag: number;
  /** total turning of the facet normals (radians) */
  turn: number;
}

interface Grown {
  tris: number[];
  surface: ConvSurface;
  area: number;
}

class Segmenter {
  readonly P: Float32Array;
  readonly I: Uint32Array;
  readonly nt: number;
  readonly nbr: Int32Array;
  readonly fn: Float64Array;
  readonly area: Float64Array;
  readonly maxEdge: Float64Array;
  readonly centre: Float64Array;
  readonly label: Int32Array;
  private markT: Int32Array;
  private markV: Int32Array;
  private stamp = 0;
  readonly regions: Region[] = [];
  readonly tol: number;
  readonly baseAng: number;
  readonly tinyArea: number;
  readonly kinds: SurfKind[];
  readonly curved: SurfKind[];
  private assigned = 0;
  get assignedFraction(): number {
    return this.assigned / Math.max(1, this.nt);
  }

  constructor(mesh: MeshData, readonly opts: SegmentOptions) {
    this.P = mesh.positions;
    this.I = mesh.indices;
    this.nt = this.I.length / 3;
    this.nbr = opts.neighbors ?? buildTriangleNeighbors(mesh);
    this.tol = opts.tolerance;
    this.baseAng = opts.angleTolDeg * DEG;
    this.kinds = opts.kinds ?? ALL_KINDS;
    this.curved = this.kinds.filter((k) => k !== "plane");
    const nt = this.nt, P = this.P, I = this.I;
    this.fn = new Float64Array(3 * nt);
    this.area = new Float64Array(nt);
    this.maxEdge = new Float64Array(nt);
    this.centre = new Float64Array(3 * nt);
    let maxArea = 0;
    for (let t = 0; t < nt; t++) {
      const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
      const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
      const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const l = Math.hypot(nx, ny, nz);
      this.area[t] = l / 2;
      if (l > 0) {
        this.fn[3 * t] = nx / l;
        this.fn[3 * t + 1] = ny / l;
        this.fn[3 * t + 2] = nz / l;
      }
      const wx = P[c] - P[b], wy = P[c + 1] - P[b + 1], wz = P[c + 2] - P[b + 2];
      this.maxEdge[t] = Math.max(Math.hypot(ux, uy, uz), Math.hypot(vx, vy, vz), Math.hypot(wx, wy, wz));
      this.centre[3 * t] = (P[a] + P[b] + P[c]) / 3;
      this.centre[3 * t + 1] = (P[a + 1] + P[b + 1] + P[c + 1]) / 3;
      this.centre[3 * t + 2] = (P[a + 2] + P[b + 2] + P[c + 2]) / 3;
      maxArea = Math.max(maxArea, l / 2);
    }
    // slivers have unreliable normals: skip the normal test for them
    this.tinyArea = maxArea * 1e-8;
    this.label = new Int32Array(nt).fill(-1);
    this.markT = new Int32Array(nt);
    this.markV = new Int32Array(this.P.length / 3);
    this.midTol = Math.max(3 * this.tol, 1.25 * this.estimateDeflection());
  }

  /** allowed distance of an edge midpoint from the surface (chord sag of the STL export + tolerance) */
  readonly midTol: number;

  /**
   * Chord deviation the exporter allowed, estimated from neighbouring facets on curved areas:
   * two facets meeting at angle θ with centres d apart approximate an arc whose chord sag is
   * about d·θ/8.
   */
  private estimateDeflection(): number {
    const est: number[] = [];
    const step = Math.max(1, Math.floor(this.nt / 20000));
    for (let t = 0; t < this.nt; t += step) {
      for (let k = 0; k < 3; k++) {
        const u = this.nbr[3 * t + k];
        if (u <= t) continue;
        const c = this.fn[3 * t] * this.fn[3 * u] + this.fn[3 * t + 1] * this.fn[3 * u + 1] + this.fn[3 * t + 2] * this.fn[3 * u + 2];
        const th = Math.acos(Math.max(-1, Math.min(1, c)));
        if (th < 0.5 * DEG || th > 45 * DEG) continue;
        const d = Math.hypot(this.centre[3 * t] - this.centre[3 * u], this.centre[3 * t + 1] - this.centre[3 * u + 1], this.centre[3 * t + 2] - this.centre[3 * u + 2]);
        est.push((d * th) / 8);
      }
    }
    if (!est.length) return 0;
    est.sort((a, b) => a - b);
    // the coarsest curved faces matter: take a high percentile
    return est[Math.floor(est.length * 0.98)];
  }

  private nextStamp(): number {
    if (++this.stamp > 2_000_000_000) {
      this.markT.fill(0);
      this.markV.fill(0);
      this.stamp = 1;
    }
    return this.stamp;
  }

  private progress(msg: string): void {
    this.opts.onProgress?.(Math.min(0.99, this.assigned / Math.max(1, this.nt)), msg);
  }

  /** Points, facet normals and centres of a set of triangles (sub-sampled for large sets). */
  fitInput(tris: readonly number[], maxTris = 4000): FitInputEx {
    const st = this.nextStamp();
    const verts: number[] = [];
    const normals: Vec3[] = [];
    const weights: number[] = [];
    const centres: Vec3[] = [];
    const stride = Math.max(1, tris.length / maxTris);
    for (let f = 0; f < tris.length; f += stride) {
      const t = tris[Math.floor(f)];
      for (let k = 0; k < 3; k++) {
        const v = this.I[3 * t + k];
        if (this.markV[v] !== st) {
          this.markV[v] = st;
          verts.push(v);
        }
      }
      if (this.area[t] > this.tinyArea) {
        normals.push([this.fn[3 * t], this.fn[3 * t + 1], this.fn[3 * t + 2]]);
        weights.push(this.area[t]);
        centres.push([this.centre[3 * t], this.centre[3 * t + 1], this.centre[3 * t + 2]]);
      }
    }
    return { positions: this.P, verts, normals, weights, centres };
  }

  maxDev(s: ConvSurface, verts: ArrayLike<number>): number {
    const d = surfDistance(s);
    const P = this.P;
    let m = 0;
    for (let k = 0; k < verts.length; k++) {
      const i = verts[k];
      const r = Math.abs(d(P[3 * i], P[3 * i + 1], P[3 * i + 2]));
      if (!(r <= m)) m = r;
      if (!Number.isFinite(r)) return Infinity;
    }
    return m;
  }

  sane(s: ConvSurface): boolean {
    const size = this.opts.size;
    switch (s.kind) {
      case "plane":
        return true;
      case "cylinder":
      case "sphere":
        return s.radius > 2 * this.tol && s.radius < 5 * size;
      case "cone":
        // near-flat or near-cylindrical cones are artefacts of tangent junctions
        return s.halfAngle > 1 * DEG && s.halfAngle < 84 * DEG && surfaceScale(s) < 10 * size;
      case "torus":
        return s.minor > 2 * this.tol && s.major < 1.5 * size && s.minor < 0.75 * size && s.major > s.minor;
    }
  }

  /**
   * Triangles reachable from `start` over unassigned triangles within `rings` edge rings, crossing
   * only smooth edges (facet normals within maxSpan) so patches do not mix faces across sharp edges.
   */
  ringAround(start: readonly number[], rings: number, max: number): number[] {
    const st = this.nextStamp();
    const out = [...start];
    for (const t of start) this.markT[t] = st;
    const cosSpan = Math.cos(this.maxSpan);
    const fn = this.fn;
    let lo = 0;
    for (let r = 0; r < rings && out.length < max; r++) {
      const hi = out.length;
      for (let i = lo; i < hi && out.length < max; i++) {
        const t = out[i];
        for (let k = 0; k < 3; k++) {
          const u = this.nbr[3 * t + k];
          if (u < 0 || this.markT[u] === st || this.label[u] !== -1) continue;
          if (fn[3 * t] * fn[3 * u] + fn[3 * t + 1] * fn[3 * u + 1] + fn[3 * t + 2] * fn[3 * u + 2] < cosSpan) continue;
          this.markT[u] = st;
          out.push(u);
        }
      }
      lo = hi;
    }
    return out;
  }

  /**
   * Seed patch around t: a priority flood that always crosses the smoothest edge next (cosine of
   * the dihedral angle), so the patch stays on one surface as long as possible.
   */
  seedPatch(t: number, max: number): number[] {
    const st = this.nextStamp();
    const fn = this.fn;
    const out: number[] = [t];
    this.markT[t] = st;
    const cand: { u: number; c: number }[] = [];
    // a CAD tessellation is roughly uniform within one curved face: stay among similar triangles
    const lo = this.maxEdge[t] / 3, hi = this.maxEdge[t] * 3;
    const push = (from: number) => {
      for (let k = 0; k < 3; k++) {
        const u = this.nbr[3 * from + k];
        if (u < 0 || this.markT[u] === st || this.label[u] !== -1) continue;
        if (this.maxEdge[u] < lo || this.maxEdge[u] > hi) continue;
        const c = fn[3 * from] * fn[3 * u] + fn[3 * from + 1] * fn[3 * u + 1] + fn[3 * from + 2] * fn[3 * u + 2];
        if (c < Math.cos(this.maxSpan)) continue;
        cand.push({ u, c });
      }
    };
    push(t);
    while (out.length < max && cand.length) {
      let bi = 0;
      for (let i = 1; i < cand.length; i++) if (cand[i].c > cand[bi].c) bi = i;
      const { u } = cand[bi];
      cand[bi] = cand[cand.length - 1];
      cand.pop();
      if (this.markT[u] === st) continue;
      this.markT[u] = st;
      out.push(u);
      push(u);
    }
    return out;
  }

  private nb4 = new Float64Array(12);
  /** a tessellated curved surface never turns its normal by more than this across one triangle */
  readonly maxSpan = 50 * DEG;

  /**
   * Does triangle t lie on the surface? Its vertices must be within tolerance, the surface normal
   * may not turn implausibly much across it (sparse CAD vertices fit many wrong surfaces, but the
   * wrong ones bend wildly between them), and the facet normal must agree with the surface.
   */
  triangleFits(t: number, dist: (x: number, y: number, z: number) => number, nrm: (x: number, y: number, z: number, o: Float64Array) => void, sign: number): boolean {
    const P = this.P, I = this.I, tol = this.tol, nb = this.nb4;
    const o = [nb.subarray(0, 3), nb.subarray(3, 6), nb.subarray(6, 9)];
    for (let k = 0; k < 3; k++) {
      const v = 3 * I[3 * t + k];
      if (!(Math.abs(dist(P[v], P[v + 1], P[v + 2])) <= tol)) return false;
      nrm(P[v], P[v + 1], P[v + 2], o[k]);
    }
    // the mesh between the vertices is part of the shape too: chords must stay near the surface
    for (let k = 0; k < 3; k++) {
      const a = 3 * I[3 * t + k], b = 3 * I[3 * t + ((k + 1) % 3)];
      if (!(Math.abs(dist((P[a] + P[b]) / 2, (P[a + 1] + P[b + 1]) / 2, (P[a + 2] + P[b + 2]) / 2)) <= this.midTol)) return false;
    }
    // slivers whose height is down at coordinate precision have meaningless facet normals:
    // positions decide for them (long thin CAD strips still have exact normals)
    if (!(this.area[t] > this.tinyArea) || (2 * this.area[t]) / this.maxEdge[t] < 1e-5 * this.opts.size) return true;
    const d01 = nb[0] * nb[3] + nb[1] * nb[4] + nb[2] * nb[5];
    const d02 = nb[0] * nb[6] + nb[1] * nb[7] + nb[2] * nb[8];
    const d12 = nb[3] * nb[6] + nb[4] * nb[7] + nb[5] * nb[8];
    const span = Math.acos(Math.max(-1, Math.min(1, Math.min(d01, d02, d12))));
    if (span > this.maxSpan) return false;
    const c = nb.subarray(9, 12);
    nrm(this.centre[3 * t], this.centre[3 * t + 1], this.centre[3 * t + 2], c);
    const d = sign * (this.fn[3 * t] * c[0] + this.fn[3 * t + 1] * c[1] + this.fn[3 * t + 2] * c[2]);
    return d >= Math.cos(this.baseAng + span);
  }

  /** Orientation of triangles relative to the surface normal (area-weighted majority). */
  orientation(tris: readonly number[], nrm: (x: number, y: number, z: number, o: Float64Array) => void): number {
    const nb = new Float64Array(3);
    let s0 = 0;
    for (const t of tris) {
      nrm(this.centre[3 * t], this.centre[3 * t + 1], this.centre[3 * t + 2], nb);
      s0 += this.area[t] * (this.fn[3 * t] * nb[0] + this.fn[3 * t + 1] * nb[1] + this.fn[3 * t + 2] * nb[2]);
    }
    return s0 >= 0 ? 1 : -1;
  }

  /**
   * Flood-fill from `seed` over unassigned edge neighbours that fit `surface`; the surface is
   * refitted as the region doubles and the rejected boundary re-tested after each refit.
   */
  grow(seed: readonly number[], surface: ConvSurface, allowSwitch = false): Grown | null {
    const tol = this.tol;
    let surf = surface;
    let dist = surfDistance(surf);
    let nrm = surfNormal(surf);
    const nb = new Float64Array(3);
    // orientation of the region relative to the surface normal (from the seed)
    let sign = this.orientation(seed, nrm);
    const st = this.nextStamp();
    const rejectStamp = -st;
    const inRegion = (t: number) => this.markT[t] === st;
    const accept = (t: number): boolean => {
      if (this.label[t] !== -1 || this.markT[t] === st) return false;
      return this.triangleFits(t, dist, nrm, sign);
    };
    const tris: number[] = [];
    for (const t of seed) {
      if (!accept(t)) return null;
      this.markT[t] = st;
      tris.push(t);
    }
    let rejected: number[] = [];
    let lastFit = Math.max(tris.length, 6);
    const refit = (): boolean => {
      const inp = this.fitInput(tris);
      const s2 = fitSurface(surf.kind, inp, surf);
      if (!s2 || !this.sane(s2) || this.maxDev(s2, inp.verts) > tol) return false;
      surf = s2;
      dist = surfDistance(surf);
      nrm = surfNormal(surf);
      return true;
    };
    let head = 0;
    for (let round = 0; round < 6; round++) {
      while (head < tris.length) {
        const t = tris[head++];
        for (let k = 0; k < 3; k++) {
          const u = this.nbr[3 * t + k];
          if (u < 0 || inRegion(u) || this.label[u] !== -1) continue;
          if (!accept(u)) {
            if (this.markT[u] !== rejectStamp) {
              this.markT[u] = rejectStamp;
              rejected.push(u);
            }
            continue;
          }
          this.markT[u] = st;
          tris.push(u);
        }
        if (tris.length >= 2 * lastFit) {
          refit();
          lastFit = tris.length;
        }
      }
      if (!rejected.length) break;
      let switched = false;
      if (allowSwitch && surf.kind !== "plane") {
        // a stalled region may be a more general surface (a fillet torus grown as its tube cylinder)
        const inp = this.fitInput(tris);
        const allV = tris.length > 4000 ? this.fitInput(tris, Infinity).verts : inp.verts;
        for (const kind of this.curved) {
          if (SURF_DOF[kind] <= SURF_DOF[surf.kind]) continue;
          const s2 = fitSurface(kind, inp);
          if (!s2 || !this.sane(s2) || this.maxDev(s2, allV) > tol) continue;
          const d2 = surfDistance(s2), n2 = surfNormal(s2);
          const sg = this.orientation(tris, n2);
          if (!rejected.some((u) => this.label[u] === -1 && this.triangleFits(u, d2, n2, sg))) continue;
          surf = s2;
          dist = d2;
          nrm = n2;
          sign = sg;
          switched = true;
          break;
        }
      }
      if (!switched && !refit()) break;
      const retry = rejected;
      rejected = [];
      let added = 0;
      for (const u of retry) {
        if (this.markT[u] === st) continue;
        this.markT[u] = 0;
        if (accept(u)) {
          this.markT[u] = st;
          tris.push(u);
          added++;
        } else {
          this.markT[u] = rejectStamp;
          rejected.push(u);
        }
      }
      if (!added) break;
    }
    let area = 0;
    for (const t of tris) area += this.area[t];
    return { tris, surface: surf, area };
  }

  /** Area-weighted mean angular disagreement (1 - cos) of facet normals with the surface. */
  normalError(g: Grown): number {
    const nrm = surfNormal(g.surface);
    const nb = new Float64Array(3);
    let s = 0, w = 0, sgn = 0;
    for (const t of g.tris) {
      nrm(this.centre[3 * t], this.centre[3 * t + 1], this.centre[3 * t + 2], nb);
      sgn += this.area[t] * (this.fn[3 * t] * nb[0] + this.fn[3 * t + 1] * nb[1] + this.fn[3 * t + 2] * nb[2]);
    }
    const sign = sgn >= 0 ? 1 : -1;
    for (const t of g.tris) {
      if (!(this.area[t] > this.tinyArea)) continue;
      nrm(this.centre[3 * t], this.centre[3 * t + 1], this.centre[3 * t + 2], nb);
      s += this.area[t] * (1 - sign * (this.fn[3 * t] * nb[0] + this.fn[3 * t + 1] * nb[1] + this.fn[3 * t + 2] * nb[2]));
      w += this.area[t];
    }
    return w > 0 ? s / w : 0;
  }

  /**
   * Is a flat patch genuinely flat rather than one facet strip of the curved surface `s`? A strip
   * of a tessellated curved surface meets its neighbours at about the angle the surface turns
   * across it; a flat land between two fillets meets them almost tangentially although `s`
   * would have to turn a lot across it.
   */
  genuineFlat(tris: readonly number[], s: ConvSurface): boolean {
    if (s.kind === "plane" || !tris.length) return false;
    const nrm = surfNormal(s);
    const P = this.P, I = this.I;
    const c = new Float64Array(3), o = new Float64Array(3);
    nrm(this.centre[3 * tris[0]], this.centre[3 * tris[0] + 1], this.centre[3 * tris[0] + 2], c);
    let minDot = 1;
    for (const t of tris)
      for (let k = 0; k < 3; k++) {
        const v = 3 * I[3 * t + k];
        nrm(P[v], P[v + 1], P[v + 2], o);
        minDot = Math.min(minDot, c[0] * o[0] + c[1] * o[1] + c[2] * o[2]);
      }
    const predicted = 2 * Math.acos(Math.max(-1, Math.min(1, minDot)));
    const inside = new Set(tris);
    const cosSpan = Math.cos(this.maxSpan);
    let observed = 0;
    for (const t of tris)
      for (let k = 0; k < 3; k++) {
        const u = this.nbr[3 * t + k];
        if (u < 0 || inside.has(u)) continue;
        const d = this.fn[3 * t] * this.fn[3 * u] + this.fn[3 * t + 1] * this.fn[3 * u + 1] + this.fn[3 * t + 2] * this.fn[3 * u + 2];
        if (d < cosSpan) continue; // a sharp edge says nothing about this surface
        observed = Math.max(observed, Math.acos(Math.min(1, d)));
      }
    return predicted > 3 * observed + 1 * DEG;
  }

  /** Largest angle between a facet normal and the region's mean normal, doubled (total turning). */
  turning(tris: readonly number[]): number {
    const m = [0, 0, 0];
    for (const t of tris) for (let k = 0; k < 3; k++) m[k] += this.area[t] * this.fn[3 * t + k];
    const l = Math.hypot(m[0], m[1], m[2]);
    if (!(l > 0)) return Math.PI;
    let minDot = 1;
    for (const t of tris) {
      if (!(this.area[t] > this.tinyArea)) continue;
      minDot = Math.min(minDot, (this.fn[3 * t] * m[0] + this.fn[3 * t + 1] * m[1] + this.fn[3 * t + 2] * m[2]) / l);
    }
    return 2 * Math.acos(Math.max(-1, Math.min(1, minDot)));
  }

  /**
   * RMS distance of the triangles' edge midpoints from the surface. A CAD tessellator only leaves
   * an edge long where the true surface is (nearly) straight along it, so the right surface keeps
   * the chords close, while a wrong surface that merely passes through the vertices (a sphere or a
   * bent tube through the two vertex rows of a cylinder) bows away between them.
   */
  sagError(tris: readonly number[], s: ConvSurface): number {
    const d = surfDistance(s);
    const P = this.P, I = this.I;
    let ss = 0, n = 0;
    const stride = Math.max(1, tris.length / 3000);
    for (let f = 0; f < tris.length; f += stride) {
      const t = tris[Math.floor(f)];
      for (let k = 0; k < 3; k++) {
        const a = 3 * I[3 * t + k], b = 3 * I[3 * t + ((k + 1) % 3)];
        const e = d((P[a] + P[b]) / 2, (P[a + 1] + P[b + 1]) / 2, (P[a + 2] + P[b + 2]) / 2);
        ss += e * e;
        n++;
      }
    }
    return n ? Math.sqrt(ss / n) : 0;
  }

  /** Is candidate g a better explanation than `best`? (area first, unless much less clean) */
  private better(g: Cand, best: Cand | null): boolean {
    if (!best) return true;
    // a "curved" surface that hardly turns is a plane plus slivers of its tangent neighbours
    if (g.kind === "plane" && best.kind !== "plane" && best.turn < 15 * DEG && g.area >= 0.75 * best.area) return true;
    if (best.kind === "plane" && g.kind !== "plane" && g.turn < 15 * DEG && best.area >= 0.75 * g.area) return false;
    const eps = 0.02 * this.tol;
    const gs = g.sag + eps, bs = best.sag + eps;
    // chords bowing out beyond the tolerance are strong evidence against a surface
    if (g.sag > 2 * this.tol && bs < 0.5 * gs && best.area >= 0.3 * g.area) return false;
    if (best.sag > 2 * this.tol && gs < 0.5 * bs && g.area >= 0.3 * best.area) return true;
    if (g.area >= best.area * 0.85 && gs < 0.5 * bs) return true; // nearly as large and much cleaner
    if (best.area >= g.area * 0.85 && bs < 0.5 * gs) return false;
    if (g.area > best.area * 1.02) return true;
    if (g.area < best.area * 0.98) return false;
    return gs < 0.8 * bs || (gs <= 1.25 * bs && SURF_DOF[g.kind] < SURF_DOF[best.kind]);
  }

  /**
   * Re-test a grown region against the other kinds: the seed patch may have been polluted, and on
   * two rows of vertices a sphere or torus fits a cylinder's points as well as the cylinder —
   * only the facet normals tell them apart.
   */
  reconsider(g: Grown & Cand): Grown & Cand {
    let best = g;
    for (const kind of this.curved) {
      if (kind === g.kind) continue;
      const r = this.fitRobust(kind, g.tris);
      if (!r || r.inliers.length < 0.5 * g.tris.length) continue;
      if (g.kind === "plane" && this.genuineFlat(g.tris, r.surface)) continue;
      const g2 = this.grow(r.inliers, r.surface);
      if (!g2) continue;
      const cand: Grown & Cand = { ...g2, kind: g2.surface.kind, sag: this.sagError(g2.tris, g2.surface), turn: this.turning(g2.tris) };
      if (DEBUG) console.log(`reconsider ${best.kind}(${best.tris.length}, sag ${best.sag.toExponential(2)}) as ${kind}: ${cand.tris.length} tris sag ${cand.sag.toExponential(2)}`);
      if (this.better(cand, best)) best = cand;
    }
    return best;
  }

  commit(g: Grown): Region {
    const id = this.regions.length;
    for (const t of g.tris) this.label[t] = id;
    this.assigned += g.tris.length;
    const r: Region = { id, surface: g.surface, triangles: g.tris.length, area: g.area, rms: 0, maxDev: 0 };
    this.regions.push(r);
    return r;
  }

  vertexCount(tris: readonly number[]): number {
    const st = this.nextStamp();
    let n = 0;
    for (const t of tris)
      for (let k = 0; k < 3; k++) {
        const v = this.I[3 * t + k];
        if (this.markV[v] !== st) {
          this.markV[v] = st;
          n++;
        }
      }
    return n;
  }

  /**
   * Fit `kind` to a patch that may contain a few triangles of neighbouring faces (tangent fillets
   * make clean patches rare): fit, keep the best-fitting triangles, refit — then return the surface
   * with the patch triangles that lie on it.
   */
  fitRobust(kind: SurfKind, tris: readonly number[]): { surface: ConvSurface; inliers: number[] } | null {
    const P = this.P, I = this.I;
    let cur = tris;
    let s = fitSurface(kind, this.fitInput(cur));
    for (let iter = 0; iter < 4 && s; iter++) {
      if (!this.sane(s)) return null;
      const d = surfDistance(s);
      const nrm = surfNormal(s);
      const worst = tris.map((t) => {
        let m = 0;
        for (let k = 0; k < 3; k++) {
          const v = 3 * I[3 * t + k];
          m = Math.max(m, Math.abs(d(P[v], P[v + 1], P[v + 2])));
        }
        return m;
      });
      const sign = this.orientation(cur, nrm);
      const inliers = tris.filter((t, i) => worst[i] <= this.tol && this.triangleFits(t, d, nrm, sign));
      if (inliers.length === tris.length || (iter > 0 && inliers.length === cur.length)) return inliers.length ? { surface: s, inliers } : null;
      // keep the better-fitting part and refit from it
      const sorted = worst.slice().sort((x, y) => x - y);
      const cut = Math.max(this.tol, sorted[Math.max(0, Math.floor(sorted.length * 0.6) - 1)]);
      const next = tris.filter((_, i) => worst[i] <= cut);
      if (next.length < 2 || next.length === cur.length) {
        return inliers.length >= 2 ? { surface: s, inliers } : null;
      }
      cur = next;
      // refit from scratch first: cylinder/cone axes come from the (now clean) facet normals
      const inp = this.fitInput(cur);
      const fresh = fitSurface(kind, inp);
      s = fresh && this.maxDev(fresh, inp.verts) <= this.tol ? fresh : (fitSurface(kind, inp, s) ?? fresh);
    }
    return null;
  }

  /** Does some curved surface explain these flat triangles together with their surroundings? */
  private curvedExplains(tris: readonly number[]): boolean {
    const ring = this.ringAround(tris, 3, Math.max(60, tris.length * 4));
    if (ring.length <= tris.length + 1) return false;
    // a curved explanation must cover clearly more than the flat patch itself (otherwise a large
    // plane plus a sliver of a tangent fillet "fits" a huge-radius cylinder)
    let own = 0;
    for (const t of tris) own += this.area[t];
    const mine = new Set(tris);
    for (const kind of this.curved) {
      const r = this.fitRobust(kind, ring);
      if (!r) continue;
      let covered = 0, mineIn = 0;
      for (const t of r.inliers) {
        covered += this.area[t];
        if (mine.has(t)) mineIn++;
      }
      if (DEBUG) console.log(`explain ${tris.length} ring ${ring.length} ${kind} inliers ${r.inliers.length} mine ${mineIn} cover ${(covered / own).toFixed(2)}`);
      if (mineIn === tris.length && covered >= 1.5 * own && !this.genuineFlat(tris, r.surface)) return true;
    }
    return false;
  }

  /** total surface area */
  get totalArea(): number {
    let a = 0;
    for (let t = 0; t < this.nt; t++) a += this.area[t];
    return a;
  }

  /**
   * Grow planes from the largest facets. Only planes of at least `minArea` are committed here;
   * smaller flat patches are decided later, competing with curved surfaces.
   */
  planes(checkCurved: boolean, minTris: number, minArea = 0): void {
    if (!this.kinds.includes("plane")) return;
    const order = this.byArea();
    const tried = new Uint8Array(this.nt);
    let n = 0;
    for (const t of order) {
      if (this.label[t] !== -1 || tried[t] || !(this.area[t] > this.tinyArea)) continue;
      if ((++n & 255) === 0) this.progress("Finding planes");
      const c: Vec3 = [this.centre[3 * t], this.centre[3 * t + 1], this.centre[3 * t + 2]];
      const g = this.grow([t], { kind: "plane", origin: c, normal: [this.fn[3 * t], this.fn[3 * t + 1], this.fn[3 * t + 2]] });
      if (!g || g.tris.length < minTris || g.area < minArea) {
        if (g) for (const u of g.tris) tried[u] = 1;
        else tried[t] = 1;
        continue;
      }
      if (checkCurved && this.curved.length && g.tris.length <= 256 && this.curvedExplains(g.tris)) {
        for (const u of g.tris) tried[u] = 1;
        continue;
      }
      this.commit(g);
    }
  }

  curvedRegions(budgetMs = 45_000): void {
    if (!this.curved.length) return;
    const order = this.byArea();
    const deadline = performance.now() + budgetMs;
    for (const rings of [2, 1]) {
      for (let pass = 0; pass < 8; pass++) {
        let committed = 0;
        const tried = new Uint8Array(this.nt);
        let n = 0;
        for (const t of order) {
          if (this.label[t] !== -1 || tried[t]) continue;
          if ((++n & 63) === 0) {
            this.progress("Fitting curved surfaces");
            // very noisy or organic meshes: stop searching, the rest stays faceted
            if (performance.now() > deadline) return;
          }
          const patch = rings === 2 ? this.seedPatch(t, 24) : this.ringAround([t], 1, 12);
          if (patch.length < 3) {
            tried[t] = 1;
            continue;
          }
          let best: (Grown & Cand) | null = null;
          let planeCand: (Grown & Cand) | null = null;
          if (this.kinds.includes("plane") && this.area[t] > this.tinyArea) {
            const c: Vec3 = [this.centre[3 * t], this.centre[3 * t + 1], this.centre[3 * t + 2]];
            const g = this.grow([t], { kind: "plane", origin: c, normal: [this.fn[3 * t], this.fn[3 * t + 1], this.fn[3 * t + 2]] });
            if (g) best = planeCand = { ...g, kind: "plane", sag: this.sagError(g.tris, g.surface), turn: 0 };
          }
          for (const kind of this.curved) {
            const r = this.fitRobust(kind, patch);
            if (!r || !r.inliers.includes(t)) continue;
            const g = this.grow(r.inliers, r.surface);
            if (!g) continue;
            if (this.vertexCount(g.tris) < SURF_DOF[kind] + 2) continue;
            const cand: Grown & Cand = { ...g, kind, sag: this.sagError(g.tris, g.surface), turn: this.turning(g.tris) };
            if (DEBUG) console.log(`seed ${t} ${kind} area ${g.area.toFixed(2)} tris ${g.tris.length} sag ${cand.sag.toExponential(2)}`);
            if (this.better(cand, best)) best = cand;
          }
          if (best && planeCand && best !== planeCand && best.kind !== "plane" && planeCand.area >= 0.3 * best.area && this.genuineFlat(planeCand.tris, best.surface)) best = planeCand;
          if (!best) {
            tried[t] = 1;
            continue;
          }
          const chosen = this.reconsider(best);
          if (DEBUG) console.log(`commit ${chosen.kind} ${chosen.tris.length} tris area ${chosen.area.toFixed(1)} sag ${chosen.sag.toExponential(2)}`);
          this.commit(chosen);
          committed++;
        }
        if (!committed) break;
      }
    }
  }

  /** Debug: why does triangle t not fit `s`? */
  explainFit(t: number, s: ConvSurface, sign: number): string {
    const d = surfDistance(s), nrm = surfNormal(s);
    const P = this.P, I = this.I;
    const ds = [0, 1, 2].map((k) => { const v = 3 * I[3 * t + k]; return d(P[v], P[v + 1], P[v + 2]); });
    const n = [0, 1, 2].map((k) => { const v = 3 * I[3 * t + k]; const o = new Float64Array(3); nrm(P[v], P[v + 1], P[v + 2], o); return o; });
    const c = new Float64Array(3);
    nrm(this.centre[3 * t], this.centre[3 * t + 1], this.centre[3 * t + 2], c);
    const dd = (a: Float64Array, b: Float64Array) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const span = Math.acos(Math.min(1, Math.min(dd(n[0], n[1]), dd(n[0], n[2]), dd(n[1], n[2])))) / DEG;
    const fdev = Math.acos(Math.min(1, sign * (this.fn[3 * t] * c[0] + this.fn[3 * t + 1] * c[1] + this.fn[3 * t + 2] * c[2]))) / DEG;
    return `t${t} area ${this.area[t].toExponential(2)} dists ${ds.map((x) => x.toExponential(1))} span ${span.toFixed(2)} fdev ${fdev.toFixed(2)} allowed ${(this.baseAng / DEG + 0.6 * span).toFixed(2)}`;
  }

  /** Assign leftover triangles to an adjacent region whose surface they fit. */
  absorb(): void {
    for (let pass = 0; pass < 4; pass++) {
      let changed = 0;
      for (let t = 0; t < this.nt; t++) {
        if (this.label[t] !== -1) continue;
        for (let k = 0; k < 3; k++) {
          const u = this.nbr[3 * t + k];
          if (u < 0 || this.label[u] < 0) continue;
          const reg = this.regions[this.label[u]];
          const nrm = surfNormal(reg.surface);
          const sign = this.orientation([u], nrm);
          if (!this.triangleFits(t, surfDistance(reg.surface), nrm, sign)) {
            if (DEBUG && reg.surface.kind !== "plane") console.log(`absorb-fail into ${reg.surface.kind}: ${this.explainFit(t, reg.surface, sign)}`);
            continue;
          }
          this.label[t] = reg.id;
          reg.triangles++;
          reg.area += this.area[t];
          this.assigned++;
          changed++;
          break;
        }
      }
      if (!changed) break;
    }
  }

  /**
   * Merge adjacent regions whose union one surface explains (vertices and facet normals): rows of
   * a surface of revolution that were claimed as sphere bands, flat strips of a cylinder, faces
   * split by the growth order. Among fitting kinds the best normal agreement wins.
   */
  mergeAdjacent(budgetMs = 8000): void {
    const nR = this.regions.length;
    const parent = Array.from({ length: nR }, (_, i) => i);
    const find = (x: number): number => {
      while (parent[x] !== x) x = parent[x] = parent[parent[x]];
      return x;
    };
    const tri: number[][] = this.regions.map(() => []);
    for (let t = 0; t < this.nt; t++) if (this.label[t] >= 0) tri[this.label[t]].push(t);
    const area = this.regions.map((r) => r.area);
    const total = this.totalArea;
    const failed = new Set<string>();
    // a region's pairs are re-tested after it changed
    const ver = new Int32Array(nR);
    const keyOf = (a: number, b: number) => `${a}:${ver[a]},${b}:${ver[b]}`;
    const deadline = performance.now() + budgetMs;
    for (let round = 0; round < 64 && performance.now() < deadline; round++) {
      const pairs = new Map<string, [number, number]>();
      for (let t = 0; t < this.nt; t++) {
        if (this.label[t] < 0) continue;
        const a = find(this.label[t]);
        for (let k = 0; k < 3; k++) {
          const u = this.nbr[3 * t + k];
          if (u < 0 || this.label[u] < 0) continue;
          const b = find(this.label[u]);
          if (a === b) continue;
          const [x, y] = a < b ? [a, b] : [b, a];
          const key = keyOf(x, y);
          if (!failed.has(key)) pairs.set(key, [x, y]);
        }
      }
      const order = [...pairs.values()].sort((x, y) => area[y[0]] + area[y[1]] - (area[x[0]] + area[x[1]]));
      let merged = 0;
      const touched = new Set<number>();
      for (const [a, b] of order) {
        if (touched.has(a) || touched.has(b)) continue;
        if (performance.now() > deadline) break;
        const ra = this.regions[a].surface, rb = this.regions[b].surface;
        const bigIsA = area[a] >= area[b];
        const big = bigIsA ? ra : rb, small = bigIsA ? rb : ra;
        const bigArea = Math.max(area[a], area[b]);
        const smallPlane = bigArea < 0.005 * total;
        // an established cylinder/cone/torus keeps its kind (otherwise a giant torus can swallow a
        // cylinder plus a few rows of its tangent fillet); planes and sphere bands are the
        // degenerate explanations of strips and rows, and may become something more general
        const curvedFrom = (minDof: number) => this.curved.filter((k) => SURF_DOF[k] >= minDof);
        let kinds: SurfKind[];
        if (big.kind === "plane") kinds = small.kind === "plane" ? ["plane", ...(smallPlane ? this.curved : [])] : smallPlane ? curvedFrom(SURF_DOF[small.kind]) : [];
        else if (big.kind === "sphere") kinds = curvedFrom(small.kind === "plane" ? 0 : SURF_DOF[small.kind]);
        else kinds = [big.kind];
        // locked kind: the small region's vertices must already lie near the big surface
        const smallTris = bigIsA ? tri[b] : tri[a];
        if (!kinds.length || (kinds.length === 1 && kinds[0] === big.kind && this.maxDev(big, this.fitInput(smallTris, 400).verts) > 3 * this.tol)) {
          failed.add(keyOf(a, b));
          continue;
        }
        const union = tri[a].concat(tri[b]);
        const inp = this.fitInput(union);
        const allV = union.length > 4000 ? this.fitInput(union, Infinity).verts : inp.verts;
        let best: { surface: ConvSurface; sag: number } | null = null;
        for (const kind of kinds) {
          const init = big.kind === kind ? big : ra.kind === kind ? ra : rb.kind === kind ? rb : undefined;
          let sf = fitSurface(kind, inp, init);
          if ((!sf || this.maxDev(sf, allV) > this.tol) && init) sf = fitSurface(kind, inp);
          if (!sf || !this.sane(sf) || this.maxDev(sf, allV) > this.tol) continue;
          const dist = surfDistance(sf), nrm = surfNormal(sf);
          const sign = this.orientation(union, nrm);
          if (!union.every((t) => this.triangleFits(t, dist, nrm, sign))) continue;
          if (kind !== "plane" && ((ra.kind === "plane" && this.genuineFlat(tri[a], sf)) || (rb.kind === "plane" && this.genuineFlat(tri[b], sf)))) continue;
          const sag = this.sagError(union, sf) + 0.02 * this.tol;
          if (!best || sag < best.sag * 0.8 || (sag <= best.sag * 1.25 && SURF_DOF[kind] < SURF_DOF[best.surface.kind])) best = { surface: sf, sag };
        }
        if (!best) {
          failed.add(keyOf(a, b));
          continue;
        }
        parent[b] = a;
        tri[a] = union;
        tri[b] = [];
        area[a] += area[b];
        this.regions[a] = { ...this.regions[a], surface: best.surface };
        for (const t of union) this.label[t] = a;
        ver[a]++;
        touched.add(a);
        touched.add(b);
        merged++;
      }
      if (!merged) break;
    }
    // compact
    const remap = new Int32Array(nR).fill(-1);
    const out: Region[] = [];
    this.regions.forEach((_, i) => {
      if (find(i) !== i) return;
      remap[i] = out.length;
      out.push({ ...this.regions[i], id: out.length });
    });
    for (let t = 0; t < this.nt; t++) if (this.label[t] >= 0) this.label[t] = remap[find(this.label[t])];
    this.regions.length = 0;
    this.regions.push(...out);
  }

  /**
   * Sharpen boundaries between tangent regions: near a tangent junction the first rows of a
   * fillet also lie within tolerance of its neighbour, so growth order decides where they end up.
   * Move each boundary triangle to the adjacent surface its facet normal agrees with best.
   */
  relaxBoundaries(): void {
    const nR = this.regions.length;
    if (!nR) return;
    const dists = this.regions.map((r) => surfDistance(r.surface));
    const nrms = this.regions.map((r) => surfNormal(r.surface));
    const tri: number[][] = this.regions.map(() => []);
    for (let t = 0; t < this.nt; t++) if (this.label[t] >= 0) tri[this.label[t]].push(t);
    const signs = this.regions.map((_, i) => (tri[i].length ? this.orientation(tri[i], nrms[i]) : 1));
    const nb = new Float64Array(3);
    const dev = (t: number, r: number): number => {
      nrms[r](this.centre[3 * t], this.centre[3 * t + 1], this.centre[3 * t + 2], nb);
      const d = signs[r] * (this.fn[3 * t] * nb[0] + this.fn[3 * t + 1] * nb[1] + this.fn[3 * t + 2] * nb[2]);
      return Math.acos(Math.max(-1, Math.min(1, d)));
    };
    const margin = 0.5 * DEG;
    for (let pass = 0; pass < 8; pass++) {
      let moved = 0;
      for (let t = 0; t < this.nt; t++) {
        const cur = this.label[t];
        if (cur < 0 || !(this.area[t] > this.tinyArea)) continue;
        let best = cur, bestDev = dev(t, cur);
        for (let k = 0; k < 3; k++) {
          const u = this.nbr[3 * t + k];
          if (u < 0) continue;
          const r = this.label[u];
          if (r < 0 || r === best || r === cur) continue;
          const d = dev(t, r);
          if (d + margin < bestDev && this.triangleFits(t, dists[r], nrms[r], signs[r])) {
            best = r;
            bestDev = d;
          }
        }
        if (best !== cur) {
          this.label[t] = best;
          moved++;
        }
      }
      if (!moved) break;
    }
  }

  /**
   * Every region must be one connected patch (a B-rep face has one outer boundary). Islands are
   * given to the surrounding region when they fit it, kept as their own region when sizeable,
   * otherwise left faceted.
   */
  splitComponents(): void {
    const comp = new Int32Array(this.nt).fill(-1);
    const comps: { region: number; tris: number[] }[] = [];
    for (let t0 = 0; t0 < this.nt; t0++) {
      if (this.label[t0] < 0 || comp[t0] >= 0) continue;
      const r = this.label[t0];
      const id = comps.length;
      const tris = [t0];
      comp[t0] = id;
      for (let i = 0; i < tris.length; i++) {
        const t = tris[i];
        for (let k = 0; k < 3; k++) {
          const u = this.nbr[3 * t + k];
          if (u < 0 || comp[u] >= 0 || this.label[u] !== r) continue;
          comp[u] = id;
          tris.push(u);
        }
      }
      comps.push({ region: r, tris });
    }
    // the largest component of each region keeps it
    const main = new Int32Array(this.regions.length).fill(-1);
    comps.forEach((c, i) => {
      if (main[c.region] < 0 || comps[main[c.region]].tris.length < c.tris.length) main[c.region] = i;
    });
    comps.forEach((c, i) => {
      if (main[c.region] === i) return;
      // neighbouring regions around the island
      const around = new Map<number, number>();
      for (const t of c.tris)
        for (let k = 0; k < 3; k++) {
          const u = this.nbr[3 * t + k];
          if (u >= 0 && this.label[u] >= 0 && this.label[u] !== c.region) around.set(this.label[u], (around.get(this.label[u]) ?? 0) + 1);
        }
      const order = [...around.entries()].sort((a, b) => b[1] - a[1]).map((e) => e[0]);
      for (const r of order) {
        const s2 = this.regions[r].surface;
        const d = surfDistance(s2), n = surfNormal(s2);
        const sign = this.orientation(c.tris, n);
        if (c.tris.every((t) => this.triangleFits(t, d, n, sign))) {
          for (const t of c.tris) this.label[t] = r;
          return;
        }
      }
      if (c.tris.length >= 20) {
        const id = this.regions.length;
        this.regions.push({ ...this.regions[c.region], id, triangles: c.tris.length });
        for (const t of c.tris) this.label[t] = id;
      } else for (const t of c.tris) this.label[t] = -1;
    });
  }

  /** Final refit of every region on all its vertices, and residual statistics. */
  finish(): void {
    const tri: number[][] = this.regions.map(() => []);
    for (let t = 0; t < this.nt; t++) if (this.label[t] >= 0) tri[this.label[t]].push(t);
    const P = this.P;
    this.regions.forEach((r, i) => {
      const all = this.fitInput(tri[i], Infinity);
      const s = fitSurface(r.surface.kind, this.fitInput(tri[i]), r.surface);
      if (s && this.sane(s) && this.maxDev(s, all.verts) <= Math.max(this.tol, this.maxDev(r.surface, all.verts))) r.surface = s;
      const d = surfDistance(r.surface);
      let ss = 0, mx = 0;
      for (let k = 0; k < all.verts.length; k++) {
        const v = all.verts[k];
        const e = d(P[3 * v], P[3 * v + 1], P[3 * v + 2]);
        ss += e * e;
        mx = Math.max(mx, Math.abs(e));
      }
      r.rms = Math.sqrt(ss / Math.max(1, all.verts.length));
      r.maxDev = mx;
      r.triangles = tri[i].length;
      let area = 0;
      for (const t of tri[i]) area += this.area[t];
      r.area = area;
    });
  }

  private orderCache: Uint32Array | null = null;
  private byArea(): Uint32Array {
    if (!this.orderCache) {
      const o = new Uint32Array(this.nt);
      for (let i = 0; i < this.nt; i++) o[i] = i;
      const a = this.area;
      this.orderCache = o.sort((x, y) => a[y] - a[x]);
    }
    return this.orderCache;
  }
}

export function segmentMesh(mesh: MeshData, opts: SegmentOptions): Segmentation {
  const s = new Segmenter(mesh, opts);
  const steps: [string, () => void][] = [
    ["Finding planes", () => s.planes(true, 1, s.totalArea * 0.01)],
    ["Fitting curved surfaces", () => s.curvedRegions()],
    ["Absorbing leftovers", () => s.absorb()],
    ["Finding small planes", () => s.planes(false, 2)],
    ["Absorbing leftovers", () => s.absorb()],
    ["Merging regions", () => s.mergeAdjacent()],
    // merging replaces small wrong regions by the right surface: leftovers may fit it now
    ["Absorbing leftovers", () => s.absorb()],
    ["Merging regions", () => s.mergeAdjacent()],
    ["Sharpening boundaries", () => s.relaxBoundaries()],
    ["Sharpening boundaries", () => s.splitComponents()],
    ["Absorbing leftovers", () => s.absorb()],
    ["Refitting surfaces", () => s.finish()],
  ];
  for (const [msg, fn] of steps) {
    const t0 = DEBUG ? performance.now() : 0;
    opts.onProgress?.(Math.min(0.99, s.assignedFraction), msg);
    fn();
    if (DEBUG) console.log(`segment step ${msg}: ${(performance.now() - t0).toFixed(0)} ms, regions ${s.regions.length}`);
  }
  opts.onProgress?.(1, "Segmented");
  return { labels: s.label, regions: s.regions };
}

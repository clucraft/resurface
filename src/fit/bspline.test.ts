import { describe, expect, it } from "vitest";
import type { BSplineCurveDef, BSplineSurfaceDef, Frame, Vec3 } from "../core/types";
import { cross, dot, normalize } from "../core/vec";
import {
  averagedKnots,
  basisFuns,
  clampedUniformKnots,
  closestPointOnCurve,
  closestPointOnSurface,
  curveDerivs,
  dersBasisFuns,
  evalCurve,
  evalSurface,
  findSpan,
  fitBSplineCurve,
  fitBSplineSurface,
  interpolateBSplineCurve,
  surfaceDerivs,
  surfaceDerivs2,
  tessellateSurface,
} from "./bspline";
import { heightField, makeRand, randomRigid, transformCloud } from "./testing/synthetic";

const U = [0, 0, 0, 0, 0.2, 0.5, 0.5, 0.8, 1, 1, 1, 1]; // degree 3, 8 control points, a double knot

describe("basis functions", () => {
  it("findSpan / partition of unity / non-negativity", () => {
    const p = 3, n = 7;
    expect(findSpan(n, p, 0, U)).toBe(3);
    expect(findSpan(n, p, 0.2, U)).toBe(4);
    expect(findSpan(n, p, 0.5, U)).toBe(6);
    expect(findSpan(n, p, 1, U)).toBe(7);
    for (let k = 0; k <= 100; k++) {
      const u = k / 100;
      const span = findSpan(n, p, u, U);
      expect(U[span] <= u && (u < U[span + 1] || u === 1)).toBe(true);
      const N = basisFuns(span, u, p, U);
      expect(N.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
      for (const v of N) expect(v).toBeGreaterThanOrEqual(-1e-15);
    }
  });

  it("dersBasisFuns matches finite differences", () => {
    const p = 3, n = 7;
    for (const u of [0.05, 0.3, 0.55, 0.77, 0.95]) {
      const span = findSpan(n, p, u, U);
      const d = dersBasisFuns(span, u, p, 3, U);
      const h = 1e-6;
      const Np = basisFuns(span, u + h, p, U), Nm = basisFuns(span, u - h, p, U);
      for (let j = 0; j <= p; j++) {
        expect(d[0][j]).toBeCloseTo(basisFuns(span, u, p, U)[j], 12);
        expect(d[1][j]).toBeCloseTo((Np[j] - Nm[j]) / (2 * h), 5);
      }
      const dp = dersBasisFuns(span, u + h, p, 1, U), dm = dersBasisFuns(span, u - h, p, 1, U);
      for (let j = 0; j <= p; j++) expect(d[2][j]).toBeCloseTo((dp[1][j] - dm[1][j]) / (2 * h), 3);
      // order above the degree is zero
      expect(dersBasisFuns(span, u, p, 4, U)[4].every((v) => v === 0)).toBe(true);
    }
  });

  it("knot vectors", () => {
    expect(clampedUniformKnots(5, 3)).toEqual([0, 0, 0, 0, 0.5, 1, 1, 1, 1]);
    expect(() => clampedUniformKnots(3, 3)).toThrow();
    const t = [0, 0.1, 0.3, 0.6, 1];
    const Ui = averagedKnots(t, 5, 3);
    expect(Ui.length).toBe(9);
    expect(Ui[4]).toBeCloseTo((0.1 + 0.3 + 0.6) / 3, 12);
    const params = Array.from({ length: 50 }, (_, i) => (i / 49) ** 2);
    const Ua = averagedKnots(params, 10, 3);
    for (let i = 1; i < Ua.length; i++) expect(Ua[i]).toBeGreaterThanOrEqual(Ua[i - 1]);
  });
});

describe("curves", () => {
  it("rational quarter circle evaluates exactly and derivatives match finite differences", () => {
    const w = Math.SQRT1_2;
    const c: BSplineCurveDef = { degree: 2, knots: [0, 0, 0, 1, 1, 1], controlPoints: [[1, 0, 0], [1, 1, 0], [0, 1, 0]], weights: [1, w, 1] };
    for (let k = 0; k <= 10; k++) {
      const P = evalCurve(c, k / 10);
      expect(Math.hypot(P[0], P[1])).toBeCloseTo(1, 12);
    }
    const t = 0.37, h = 1e-6;
    const [, d1, d2] = curveDerivs(c, t, 2);
    const a = evalCurve(c, t + h), b = evalCurve(c, t - h), m = evalCurve(c, t);
    for (let i = 0; i < 3; i++) {
      expect(d1[i]).toBeCloseTo((a[i] - b[i]) / (2 * h), 6);
      expect(d2[i]).toBeCloseTo((a[i] - 2 * m[i] + b[i]) / (h * h), 2);
    }
  });

  it("interpolation passes through the points", () => {
    const pts: Vec3[] = Array.from({ length: 25 }, (_, i) => [Math.cos(i * 0.3) * 5, Math.sin(i * 0.3) * 5, i * 0.2] as Vec3);
    const c = interpolateBSplineCurve(pts, 3);
    expect(c.controlPoints.length).toBe(25);
    // check via closest point (parameters are internal)
    for (const p of pts) expect(closestPointOnCurve(c, p).distance).toBeLessThan(1e-9);
    expect(evalCurve(c, 0)).toEqual(pts[0]);
    const e = evalCurve(c, 1);
    for (let i = 0; i < 3; i++) expect(e[i]).toBeCloseTo(pts[24][i], 12);
  });

  it("least-squares fit of a noisy helix (end points interpolated)", () => {
    const r = makeRand(3);
    const N = 600, noise = 0.005;
    const pts: Vec3[] = Array.from({ length: N }, (_, i) => {
      const t = (i / (N - 1)) * 4 * Math.PI;
      return [3 * Math.cos(t) + noise * r.gauss(), 3 * Math.sin(t) + noise * r.gauss(), 0.5 * t + noise * r.gauss()] as Vec3;
    });
    const fit = fitBSplineCurve(pts, { nCtrl: 24, degree: 3, iterations: 2 });
    expect(fit.curve.controlPoints[0]).toEqual(pts[0]);
    expect(fit.curve.controlPoints[23]).toEqual(pts[N - 1]);
    expect(fit.stats.rms).toBeLessThan(noise * 1.8);
    // deviation from the true helix
    let maxDev = 0;
    for (let k = 0; k <= 200; k++) {
      const P = evalCurve(fit.curve, k / 200);
      const rr = Math.hypot(P[0], P[1]);
      maxDev = Math.max(maxDev, Math.abs(rr - 3));
    }
    expect(maxDev).toBeLessThan(0.01);
    // smoothing trades accuracy for fairness
    const smooth = fitBSplineCurve(pts, { nCtrl: 24, smoothing: 1e-2 });
    expect(smooth.stats.rms).toBeGreaterThan(fit.stats.rms);
    expect(() => fitBSplineCurve(pts.slice(0, 5), { nCtrl: 10 })).toThrow(/exceeds/);
    expect(fitBSplineCurve(pts.slice(0, 5), { nCtrl: 10, smoothing: 1e-3 }).curve.controlPoints.length).toBe(10);
  });
});

// ---------------------------------------------------------------------------

function bumpSurface(): BSplineSurfaceDef {
  const nu = 5, nv = 4;
  const controlPoints: Vec3[] = [];
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) controlPoints.push([i, j, Math.sin(i) * Math.cos(j)]);
  return { degreeU: 3, degreeV: 2, knotsU: clampedUniformKnots(nu, 3), knotsV: clampedUniformKnots(nv, 2), nu, nv, controlPoints };
}

describe("surfaces", () => {
  it("bilinear patch and rational patch evaluation", () => {
    const s: BSplineSurfaceDef = {
      degreeU: 1,
      degreeV: 1,
      knotsU: [0, 0, 1, 1],
      knotsV: [0, 0, 1, 1],
      nu: 2,
      nv: 2,
      controlPoints: [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]],
    };
    expect(evalSurface(s, 0.5, 0.5)).toEqual([0.5, 0.5, 0.25]);
    const rs = { ...bumpSurface(), weights: Array.from({ length: 20 }, (_, i) => 1 + 0.1 * (i % 3)) };
    // rational evaluation = Σ w N P / Σ w N
    const u = 0.3, v = 0.6;
    const su = findSpan(4, 3, u, rs.knotsU), sv = findSpan(3, 2, v, rs.knotsV);
    const Nu = basisFuns(su, u, 3, rs.knotsU), Nv = basisFuns(sv, v, 2, rs.knotsV);
    const acc = [0, 0, 0, 0];
    for (let a = 0; a <= 3; a++)
      for (let b = 0; b <= 2; b++) {
        const idx = (su - 3 + a) * 4 + sv - 2 + b;
        const f = Nu[a] * Nv[b] * rs.weights[idx];
        acc[0] += f * rs.controlPoints[idx][0];
        acc[1] += f * rs.controlPoints[idx][1];
        acc[2] += f * rs.controlPoints[idx][2];
        acc[3] += f;
      }
    const P = evalSurface(rs, u, v);
    for (let c = 0; c < 3; c++) expect(P[c]).toBeCloseTo(acc[c] / acc[3], 12);
  });

  it("derivatives (incl. rational) match finite differences", () => {
    for (const s of [bumpSurface(), { ...bumpSurface(), weights: Array.from({ length: 20 }, (_, i) => 1 + 0.2 * Math.sin(i)) }]) {
      const u = 0.41, v = 0.63, h = 1e-6;
      const d = surfaceDerivs2(s, u, v);
      const pu = evalSurface(s, u + h, v), mu = evalSurface(s, u - h, v);
      const pv = evalSurface(s, u, v + h), mv = evalSurface(s, u, v - h);
      const p0 = evalSurface(s, u, v);
      for (let c = 0; c < 3; c++) {
        expect(d.du[c]).toBeCloseTo((pu[c] - mu[c]) / (2 * h), 5);
        expect(d.dv[c]).toBeCloseTo((pv[c] - mv[c]) / (2 * h), 5);
        expect(d.duu[c]).toBeCloseTo((pu[c] - 2 * p0[c] + mu[c]) / (h * h), 1);
        expect(d.dvv[c]).toBeCloseTo((pv[c] - 2 * p0[c] + mv[c]) / (h * h), 1);
      }
      const d1u = surfaceDerivs(s, u, v + h).du, d1m = surfaceDerivs(s, u, v - h).du;
      for (let c = 0; c < 3; c++) expect(d.duv[c]).toBeCloseTo((d1u[c] - d1m[c]) / (2 * h), 4);
      expect(dot(d.normal, d.du)).toBeCloseTo(0, 10);
    }
  });

  it("tessellation", () => {
    const s = bumpSurface();
    const m = tessellateSurface(s, 10, 6);
    expect(m.positions.length).toBe(3 * 11 * 7);
    expect(m.indices.length).toBe(6 * 10 * 6);
    // triangle orientation agrees with the analytic normals
    for (let t = 0; t < m.indices.length; t += 3) {
      const [a, b, c] = [m.indices[t], m.indices[t + 1], m.indices[t + 2]];
      const P = (i: number): Vec3 => [m.positions[3 * i], m.positions[3 * i + 1], m.positions[3 * i + 2]];
      const fn = cross([P(b)[0] - P(a)[0], P(b)[1] - P(a)[1], P(b)[2] - P(a)[2]], [P(c)[0] - P(a)[0], P(c)[1] - P(a)[1], P(c)[2] - P(a)[2]]);
      const vn: Vec3 = [m.normals![3 * a], m.normals![3 * a + 1], m.normals![3 * a + 2]];
      expect(dot(fn, vn)).toBeGreaterThan(0);
    }
  });

  it("closest point on surface", () => {
    const s = bumpSurface();
    for (const [u, v, off] of [
      [0.3, 0.7, 0.2],
      [0.9, 0.1, -0.15],
      [0.5, 0.5, 0.05],
    ]) {
      const d = surfaceDerivs(s, u, v);
      const q: Vec3 = [d.point[0] + off * d.normal[0], d.point[1] + off * d.normal[1], d.point[2] + off * d.normal[2]];
      const c = closestPointOnSurface(s, q);
      expect(c.u).toBeCloseTo(u, 6);
      expect(c.v).toBeCloseTo(v, 6);
      expect(c.signedDistance).toBeCloseTo(off, 8);
    }
  });
});

describe("fitBSplineSurface", () => {
  const f = (x: number, y: number) => 0.4 * Math.sin(1.5 * x) * Math.cos(y) + 0.1 * x * y;
  const M = (() => {
    const m = randomRigid(makeRand(4), 0.6, [10, -5, 3]);
    return m;
  })();

  it("fits a rotated noisy free-form patch to the noise level", () => {
    const cloud = transformCloud(M, heightField(f, { x0: -2, x1: 2, y0: -1.5, y1: 1.5, nx: 120, ny: 90, jitter: 0.9, noise: 0.002, seed: 7 }));
    const r = fitBSplineSurface(cloud.positions, { nu: 20, nv: 16, normals: cloud.normals });
    expect(r.stats.count).toBe(120 * 90);
    expect(r.stats.rms).toBeLessThan(0.0023);
    expect(r.residuals.length).toBe(120 * 90);
    // deviation from the exact (noise-free) surface
    const exact = transformCloud(M, heightField(f, { x0: -1.9, x1: 1.9, y0: -1.4, y1: 1.4, nx: 25, ny: 25, seed: 8 }));
    let maxDev = 0;
    for (let i = 0; i < exact.positions.length / 3; i++) {
      const c = closestPointOnSurface(r.surface, [exact.positions[3 * i], exact.positions[3 * i + 1], exact.positions[3 * i + 2]]);
      maxDev = Math.max(maxDev, c.distance);
    }
    expect(maxDev).toBeLessThan(0.002);
    // surface normal agrees with the point normals (frame oriented by normals)
    const sd = surfaceDerivs(r.surface, 0.5, 0.5);
    const up = normalize([M[8], M[9], M[10]]);
    expect(dot(sd.normal, up)).toBeGreaterThan(0.5);
  });

  it("bridges a hole and the extension with the thin-plate term", () => {
    const hole = (x: number, y: number) => Math.hypot(x - 0.3, y + 0.2) < 0.6;
    const cloud = heightField(f, { x0: -2, x1: 2, y0: -1.5, y1: 1.5, nx: 120, ny: 90, jitter: 0.9, noise: 0.001, seed: 9, skip: hole });
    const frame: Frame = { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 1, 0], zAxis: [0, 0, 1] };
    const r = fitBSplineSurface(cloud.positions, { nu: 24, nv: 18, frame, extend: 0.1 });
    expect(r.stats.rms).toBeLessThan(0.0015);
    // inside the hole: close to the true function
    let maxDev = 0;
    for (let k = 0; k < 30; k++) {
      const a = (k / 30) * 2 * Math.PI;
      const x = 0.3 + 0.4 * Math.cos(a), y = -0.2 + 0.4 * Math.sin(a);
      const c = closestPointOnSurface(r.surface, [x, y, f(x, y)]);
      maxDev = Math.max(maxDev, c.distance);
    }
    expect(maxDev).toBeLessThan(0.02);
    // the surface extends beyond the data by ~10 %
    const corner = evalSurface(r.surface, 0, 0);
    expect(corner[0]).toBeLessThan(-2.3);
    expect(corner[1]).toBeLessThan(-1.75);
  });

  it("handles a 60×60 control net on 100k points", () => {
    const cloud = heightField((x, y) => 0.2 * Math.sin(3 * x) * Math.sin(2 * y), { x0: -3, x1: 3, y0: -3, y1: 3, nx: 320, ny: 320, jitter: 0.9, noise: 0.001, seed: 10 });
    const t0 = performance.now();
    const r = fitBSplineSurface(cloud.positions, { nu: 60, nv: 60, iterations: 1 });
    const ms = performance.now() - t0;
    expect(r.surface.controlPoints.length).toBe(3600);
    expect(r.stats.rms).toBeLessThan(0.0013);
    expect(ms).toBeLessThan(20000);
  });

  it("rejects degenerate input", () => {
    const line = new Float32Array(Array.from({ length: 30 }, (_, i) => [i, 2 * i, 3 * i]).flat());
    expect(() => fitBSplineSurface(line, { nu: 4, nv: 4 })).toThrow(/collinear/);
    expect(() => fitBSplineSurface(line, { nu: 3, nv: 4 })).toThrow(/degree/);
    const flat = heightField(() => 0, { x0: 0, x1: 1, y0: 0, y1: 1, nx: 10, ny: 10 });
    const edgeOn: Frame = { origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 0, 1], zAxis: [0, -1, 0] };
    expect(() => fitBSplineSurface(flat.positions, { nu: 4, nv: 4, frame: edgeOn })).toThrow(/degenerate/);
  });
});

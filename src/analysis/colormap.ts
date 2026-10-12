/**
 * Colour maps for deviation and curvature displays. The deviation map is the classic
 * metrology scale: blue (below) → cyan → green (within tolerance) → yellow → red (above).
 */

const STOPS: [number, [number, number, number]][] = [
  [0.0, [32, 64, 230]],
  [0.25, [40, 200, 235]],
  [0.5, [60, 205, 100]],
  [0.75, [245, 215, 50]],
  [1.0, [235, 50, 45]],
];

function ramp(t: number): [number, number, number] {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1];
      const [t1, c1] = STOPS[i];
      const f = (t - t0) / (t1 - t0);
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return STOPS[STOPS.length - 1][1];
}

const GREEN: [number, number, number] = [60, 205, 100];

/**
 * Map signed values to RGB bytes. `range` is the symmetric max |value| shown at the ends of
 * the scale; values within ±tolerance are drawn solid green. NaN → grey (no data).
 */
export function deviationColors(values: ArrayLike<number>, range: number, tolerance = 0, out?: Uint8Array): Uint8Array {
  const n = values.length;
  const colors = out ?? new Uint8Array(n * 3);
  const r = Math.max(range, 1e-12);
  for (let i = 0; i < n; i++) {
    const v = values[i];
    let c: [number, number, number];
    if (!Number.isFinite(v)) c = [120, 120, 128];
    else if (Math.abs(v) <= tolerance) c = GREEN;
    else c = ramp(0.5 + 0.5 * (v / r));
    colors[3 * i] = c[0];
    colors[3 * i + 1] = c[1];
    colors[3 * i + 2] = c[2];
  }
  return colors;
}

/** Sequential map for unsigned quantities (curvature): uses the upper half of the ramp. */
export function sequentialColors(values: ArrayLike<number>, max: number, out?: Uint8Array): Uint8Array {
  const n = values.length;
  const colors = out ?? new Uint8Array(n * 3);
  const m = Math.max(max, 1e-12);
  for (let i = 0; i < n; i++) {
    const v = values[i];
    const c = Number.isFinite(v) ? ramp(0.25 + 0.75 * Math.min(1, Math.abs(v) / m)) : ([120, 120, 128] as [number, number, number]);
    colors[3 * i] = c[0];
    colors[3 * i + 1] = c[1];
    colors[3 * i + 2] = c[2];
  }
  return colors;
}

/** CSS gradient (top = max) for the legend bar; optional tolerance band as a fraction of range. */
export function colormapCss(tolFraction?: number): string {
  const stops = STOPS.map(([t, c]) => `rgb(${c.join(",")}) ${((1 - t) * 100).toFixed(1)}%`).reverse();
  if (tolFraction && tolFraction > 0) {
    const a = (50 - tolFraction * 50).toFixed(1), b = (50 + tolFraction * 50).toFixed(1);
    const g = `rgb(${GREEN.join(",")})`;
    return `linear-gradient(to top, rgb(${STOPS[0][1].join(",")}) 0%, rgb(${STOPS[1][1].join(",")}) 25%, ${g} ${a}%, ${g} ${b}%, rgb(${STOPS[3][1].join(",")}) 75%, rgb(${STOPS[4][1].join(",")}) 100%)`;
  }
  return `linear-gradient(to top, ${stops.join(", ")})`;
}

/** Robust symmetric range for a set of signed deviations (98th percentile of |v|). */
export function robustRange(values: ArrayLike<number>, percentile = 0.98): number {
  const n = values.length;
  if (!n) return 1;
  const step = Math.max(1, Math.floor(n / 200000));
  const sample: number[] = [];
  for (let i = 0; i < n; i += step) {
    const v = values[i];
    if (Number.isFinite(v)) sample.push(Math.abs(v));
  }
  if (!sample.length) return 1;
  sample.sort((a, b) => a - b);
  return sample[Math.min(sample.length - 1, Math.floor(sample.length * percentile))] || 1e-3;
}

/** Binary STL writer for indexed triangle meshes (little-endian, 50 bytes per facet). */

export interface StlPart {
  positions: Float32Array;
  indices: Uint32Array;
}

export function writeBinarySTL(parts: readonly StlPart[], header = "Resurface binary STL"): Uint8Array<ArrayBuffer> {
  let triCount = 0;
  for (const p of parts) triCount += p.indices.length / 3;
  const buf = new ArrayBuffer(84 + 50 * triCount);
  const bytes = new Uint8Array(buf);
  const dv = new DataView(buf);
  for (let i = 0; i < Math.min(80, header.length); i++) bytes[i] = header.charCodeAt(i) & 0x7f;
  dv.setUint32(80, triCount, true);
  let o = 84;
  for (const { positions: v, indices: t } of parts) {
    for (let i = 0; i < t.length; i += 3) {
      const a = 3 * t[i], b = 3 * t[i + 1], c = 3 * t[i + 2];
      const ux = v[b] - v[a], uy = v[b + 1] - v[a + 1], uz = v[b + 2] - v[a + 2];
      const wx = v[c] - v[a], wy = v[c + 1] - v[a + 1], wz = v[c + 2] - v[a + 2];
      let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
      const l = Math.hypot(nx, ny, nz) || 1;
      nx /= l; ny /= l; nz /= l;
      dv.setFloat32(o, nx, true); dv.setFloat32(o + 4, ny, true); dv.setFloat32(o + 8, nz, true);
      o += 12;
      for (const k of [a, b, c]) {
        dv.setFloat32(o, v[k], true);
        dv.setFloat32(o + 4, v[k + 1], true);
        dv.setFloat32(o + 8, v[k + 2], true);
        o += 12;
      }
      dv.setUint16(o, 0, true);
      o += 2;
    }
  }
  return bytes;
}

/**
 * Renders a (potentially very large) triangle mesh as spatially coherent chunks that share
 * one position/normal/colour buffer. Each chunk has its own index range and bounding sphere,
 * so off-screen chunks are frustum-culled. While the camera moves, an optional decimated
 * proxy is drawn instead of the full-resolution chunks.
 *
 * Triangle selection is drawn as per-chunk overlay index buffers (only dirty chunks rebuild).
 */
import * as THREE from "three";
import type { MeshData } from "../core/types";
import type { PickHit, PickTarget, Viewer } from "./Viewer";
import { uniqueTriangleEdges } from "./meshEdges";

export interface MeshChunk {
  triStart: number;
  triCount: number;
  bounds: { min: [number, number, number]; max: [number, number, number] };
}

export type MeshColorMode = "solid" | "scan" | "map";

export interface MeshViewOptions {
  color?: string;
  /** used for picking; returns world hit for a ray */
  raycast?: (origin: [number, number, number], dir: [number, number, number]) => {
    point: [number, number, number];
    normal: [number, number, number];
    distance: number;
    faceIndex: number;
  } | null;
}

const SELECT_COLOR = new THREE.Color("#ff7a1a");
/** above this many triangles the LOD proxy is used during navigation */
export const LOD_THRESHOLD = 3_000_000;
/** the triangle-edge overlay is only built for meshes up to this many triangles */
export const TRIANGLE_EDGE_LIMIT = 1_500_000;
/** above this many triangles the edge overlay is hidden while the camera moves */
const EDGES_WHILE_MOVING_LIMIT = 600_000;

function patchBackfaceTint(mat: THREE.Material, tint = "vec3(0.42, 0.36, 0.30)") {
  mat.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <dithering_fragment>",
      `#include <dithering_fragment>
       if (!gl_FrontFacing) gl_FragColor.rgb = mix(gl_FragColor.rgb, ${tint} * (0.35 + 0.65 * gl_FragColor.rgb), 0.75);`,
    );
  };
  mat.customProgramCacheKey = () => "backface-tint";
}

export class MeshView {
  readonly group = new THREE.Group();
  private fullGroup = new THREE.Group();
  private selGroup = new THREE.Group();
  private proxyObj: THREE.Mesh | null = null;
  private geometryBase: THREE.BufferGeometry;
  private chunkMeshes: THREE.Mesh[] = [];
  private selMeshes: (THREE.Mesh | null)[] = [];
  readonly material: THREE.MeshPhongMaterial;
  private selMaterial: THREE.MeshPhongMaterial;
  private wireMaterial: THREE.MeshBasicMaterial | null = null;
  private removePick: () => void;
  private unsubInteraction: () => void;
  colorMode: MeshColorMode = "solid";
  private mapColors: Uint8Array | null = null;
  private visible = true;
  private wireframe = false;
  /** triangle-edge overlay ("faceted STL" look), built lazily on first use */
  private edgeObj: THREE.LineSegments | null = null;
  private edgeMaterial: THREE.LineBasicMaterial | null = null;
  private triangleEdges = false;

  constructor(
    private viewer: Viewer,
    readonly entityId: string,
    readonly mesh: MeshData,
    readonly chunks: MeshChunk[],
    private opts: MeshViewOptions = {},
  ) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(mesh.positions, 3));
    if (mesh.normals) g.setAttribute("normal", new THREE.BufferAttribute(mesh.normals, 3));
    this.geometryBase = g;

    this.material = new THREE.MeshPhongMaterial({
      color: opts.color ?? "#b9c2cc",
      specular: 0x222222,
      shininess: 18,
      side: THREE.DoubleSide,
      flatShading: !mesh.normals,
    });
    patchBackfaceTint(this.material);
    this.selMaterial = new THREE.MeshPhongMaterial({
      color: SELECT_COLOR,
      emissive: new THREE.Color("#5a2200"),
      specular: 0x111111,
      shininess: 10,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -2,
      flatShading: !mesh.normals,
    });

    for (const c of chunks) {
      const cg = new THREE.BufferGeometry();
      for (const [name, attr] of Object.entries(g.attributes)) cg.setAttribute(name, attr);
      cg.setIndex(new THREE.BufferAttribute(mesh.indices.subarray(c.triStart * 3, (c.triStart + c.triCount) * 3), 1));
      const box = new THREE.Box3(new THREE.Vector3(...c.bounds.min), new THREE.Vector3(...c.bounds.max));
      cg.boundingBox = box;
      cg.boundingSphere = box.getBoundingSphere(new THREE.Sphere());
      const m = new THREE.Mesh(cg, this.material);
      m.matrixAutoUpdate = false;
      this.chunkMeshes.push(m);
      this.fullGroup.add(m);
    }
    this.selMeshes = chunks.map(() => null);
    this.group.add(this.fullGroup, this.selGroup);
    this.group.userData.entityId = entityId;
    viewer.layers.meshes.add(this.group);

    this.removePick = viewer.addPickTarget(this.pickTarget());
    this.unsubInteraction = viewer.on("interaction", (on) => this.updateLod(on));
    viewer.markSceneChanged();
  }

  get triangleCount(): number {
    return this.mesh.indices.length / 3;
  }

  private pickTarget(): PickTarget {
    return {
      entityId: this.entityId,
      priority: 0,
      layer: "mesh",
      enabled: () => this.visible && !!this.opts.raycast,
      raycast: (ray): PickHit | null => {
        const hit = this.opts.raycast!(
          [ray.origin.x, ray.origin.y, ray.origin.z],
          [ray.direction.x, ray.direction.y, ray.direction.z],
        );
        if (!hit) return null;
        return { entityId: this.entityId, point: hit.point, normal: hit.normal, distance: hit.distance, faceIndex: hit.faceIndex };
      },
    };
  }

  setRaycast(fn: MeshViewOptions["raycast"]): void {
    this.opts.raycast = fn;
  }

  setVisible(v: boolean): void {
    this.visible = v;
    this.group.visible = v;
    this.viewer.markSceneChanged();
  }

  setColor(css: string): void {
    this.material.color.set(css);
    this.viewer.requestRender();
  }

  setOpacity(alpha: number): void {
    this.material.transparent = alpha < 1;
    this.material.opacity = alpha;
    this.material.depthWrite = alpha >= 1;
    this.material.needsUpdate = true;
    this.viewer.requestRender();
  }

  setWireframe(on: boolean): void {
    this.wireframe = on;
    if (on && !this.wireMaterial) {
      this.wireMaterial = new THREE.MeshBasicMaterial({ color: 0x1b2028, wireframe: true, transparent: true, opacity: 0.35 });
    }
    for (const m of this.chunkMeshes) {
      m.children.forEach((c) => m.remove(c));
      if (on) m.add(new THREE.Mesh(m.geometry, this.wireMaterial!));
    }
    this.viewer.requestRender();
  }

  /**
   * Draw every triangle edge over the shaded mesh (the faceted STL look of the compare view).
   * Built lazily the first time it is needed; skipped above TRIANGLE_EDGE_LIMIT triangles.
   * Returns whether the edges are shown.
   */
  setTriangleEdges(on: boolean): boolean {
    const show = on && this.triangleCount > 0 && this.triangleCount <= TRIANGLE_EDGE_LIMIT;
    if (show === this.triangleEdges) return show;
    this.triangleEdges = show;
    if (show && !this.edgeObj) this.buildTriangleEdges();
    this.updateEdgeVisibility(this.viewer.interacting);
    // push the faces back a little so edges lying exactly on them pass the depth test
    this.material.polygonOffset = show;
    this.material.polygonOffsetFactor = show ? 1 : 0;
    this.material.polygonOffsetUnits = show ? 1 : 0;
    this.viewer.requestRender();
    return show;
  }

  private buildTriangleEdges(): void {
    const pos = this.mesh.positions;
    const pairs = uniqueTriangleEdges(this.mesh.indices, pos.length / 3);
    // typical edge length, used to fade edges that are only a few pixels long on screen
    let sum = 0;
    for (let i = 0; i < pairs.length; i += 2) {
      const a = 3 * pairs[i], b = 3 * pairs[i + 1];
      sum += Math.hypot(pos[a] - pos[b], pos[a + 1] - pos[b + 1], pos[a + 2] - pos[b + 2]);
    }
    const edgeLength = pairs.length ? sum / (pairs.length / 2) : 1;

    const g = new THREE.BufferGeometry();
    // share the position buffer with the chunks; only the index is new
    g.setAttribute("position", this.geometryBase.getAttribute("position"));
    g.setIndex(new THREE.BufferAttribute(pairs, 1));
    const box = new THREE.Box3();
    for (const m of this.chunkMeshes) if (m.geometry.boundingBox) box.union(m.geometry.boundingBox);
    g.boundingBox = box;
    g.boundingSphere = box.getBoundingSphere(new THREE.Sphere());

    const mat = new THREE.LineBasicMaterial({ color: 0x10151c, transparent: true, opacity: 0.5, depthWrite: false });
    const uniforms = { uEdgeLen: { value: edgeLength }, uPxScale: { value: 1 } };
    // Fade by on-screen edge length: where triangles are only a pixel or two wide (a dense scan
    // seen from afar) full-strength edges would turn the surface dark, so they become a faint
    // texture there and reach full strength once the facets are clearly visible.
    mat.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nuniform float uEdgeLen;\nuniform float uPxScale;\nvarying float vEdgePx;")
        .replace("#include <project_vertex>", "#include <project_vertex>\nvEdgePx = uEdgeLen / max(-mvPosition.z * uPxScale, 1e-9);");
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", "#include <common>\nvarying float vEdgePx;")
        .replace("#include <alphatest_fragment>", "diffuseColor.a *= mix(0.16, 1.0, smoothstep(2.5, 9.0, vEdgePx));\n#include <alphatest_fragment>");
    };
    mat.customProgramCacheKey = () => "mesh-triangle-edges";
    this.edgeMaterial = mat;

    const lines = new THREE.LineSegments(g, mat);
    lines.matrixAutoUpdate = false;
    lines.renderOrder = 1;
    lines.userData.noBounds = true;
    lines.userData.noPick = true;
    lines.onBeforeRender = (_r, _s, camera) => {
      // world size of one CSS pixel per unit of view depth
      const cam = camera as THREE.PerspectiveCamera;
      if (cam.isPerspectiveCamera) uniforms.uPxScale.value = (2 * Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2)) / Math.max(1, this.viewer.size.height);
      // follow section clipping applied to the mesh material by tools
      if (mat.clippingPlanes !== this.material.clippingPlanes) mat.clippingPlanes = this.material.clippingPlanes;
    };
    this.edgeObj = lines;
    this.fullGroup.add(lines);
  }

  private updateEdgeVisibility(interacting: boolean): void {
    if (!this.edgeObj) return;
    this.edgeObj.visible = this.triangleEdges && !(interacting && this.triangleCount > EDGES_WHILE_MOVING_LIMIT);
  }

  /** Switch between plain colour, scan (per-vertex RGB from the file) and a computed colour map. */
  setColorMode(mode: MeshColorMode, mapColors?: Uint8Array | null): void {
    if (mapColors !== undefined) this.mapColors = mapColors;
    this.colorMode = mode;
    let colors: Uint8Array | null = null;
    if (mode === "scan") colors = this.mesh.colors ?? null;
    if (mode === "map") colors = this.mapColors;
    const attrs = [this.geometryBase, ...this.chunkMeshes.map((m) => m.geometry), ...this.selMeshes.filter(Boolean).map((m) => m!.geometry)];
    if (colors) {
      const attr = new THREE.BufferAttribute(colors, 3, true);
      for (const g of attrs) g.setAttribute("color", attr);
      this.material.vertexColors = true;
      this.material.color.set("#ffffff");
    } else {
      for (const g of attrs) g.deleteAttribute("color");
      this.material.vertexColors = false;
      this.material.color.set(this.opts.color ?? "#b9c2cc");
    }
    this.material.needsUpdate = true;
    this.viewer.requestRender();
  }

  /** Update the colour-map buffer in place (e.g. live deviation updates). */
  updateMapColors(colors: Uint8Array): void {
    if (this.mapColors && this.mapColors.length === colors.length && this.colorMode === "map") {
      this.mapColors.set(colors);
      const attr = this.chunkMeshes[0]?.geometry.getAttribute("color") as THREE.BufferAttribute | undefined;
      if (attr) attr.needsUpdate = true;
      this.viewer.requestRender();
    } else {
      this.setColorMode("map", colors);
    }
  }

  /**
   * Rebuild selection overlays. `mask` is per triangle (1 = selected). When `dirtyChunks` is
   * given only those chunks are rebuilt.
   */
  setSelection(mask: Uint8Array | null, dirtyChunks?: Iterable<number>): void {
    const idx = this.mesh.indices;
    const which = dirtyChunks ? [...dirtyChunks] : this.chunks.map((_, i) => i);
    for (const ci of which) {
      const c = this.chunks[ci];
      if (!c) continue;
      let count = 0;
      if (mask) for (let t = c.triStart, e = c.triStart + c.triCount; t < e; t++) count += mask[t] ? 1 : 0;
      const old = this.selMeshes[ci];
      if (count === 0) {
        if (old) {
          this.selGroup.remove(old);
          old.geometry.dispose();
          this.selMeshes[ci] = null;
        }
        continue;
      }
      const sel = new Uint32Array(count * 3);
      let k = 0;
      for (let t = c.triStart, e = c.triStart + c.triCount; t < e; t++) {
        if (!mask![t]) continue;
        sel[k++] = idx[3 * t];
        sel[k++] = idx[3 * t + 1];
        sel[k++] = idx[3 * t + 2];
      }
      let m = old;
      if (!m) {
        const g = new THREE.BufferGeometry();
        for (const [name, attr] of Object.entries(this.geometryBase.attributes)) if (name !== "color") g.setAttribute(name, attr);
        g.boundingSphere = this.chunkMeshes[ci].geometry.boundingSphere;
        g.boundingBox = this.chunkMeshes[ci].geometry.boundingBox;
        m = new THREE.Mesh(g, this.selMaterial);
        m.matrixAutoUpdate = false;
        m.renderOrder = 1;
        this.selMeshes[ci] = m;
        this.selGroup.add(m);
      } else {
        m.geometry.index = null;
      }
      m.geometry.setIndex(new THREE.BufferAttribute(sel, 1));
    }
    this.viewer.requestRender();
  }

  /** Hide the orange selection overlay (e.g. while a colour map shows the selection instead). */
  setSelectionVisible(on: boolean): void {
    this.selGroup.visible = on;
    this.viewer.requestRender();
  }

  /** Chunk index containing a triangle (chunks are contiguous and sorted). */
  chunkOfTriangle(tri: number): number {
    let lo = 0, hi = this.chunks.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.chunks[mid].triStart <= tri) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  /** Provide a decimated proxy shown while navigating (only used for very large meshes). */
  setProxy(proxy: MeshData | null): void {
    if (this.proxyObj) {
      this.group.remove(this.proxyObj);
      this.proxyObj.geometry.dispose();
      this.proxyObj = null;
    }
    if (!proxy) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(proxy.positions, 3));
    if (proxy.normals) g.setAttribute("normal", new THREE.BufferAttribute(proxy.normals, 3));
    g.setIndex(new THREE.BufferAttribute(proxy.indices, 1));
    this.proxyObj = new THREE.Mesh(g, this.material);
    this.proxyObj.visible = false;
    this.group.add(this.proxyObj);
  }

  private updateLod(interacting: boolean): void {
    this.updateEdgeVisibility(interacting);
    if (!this.proxyObj || this.triangleCount < LOD_THRESHOLD) return;
    this.proxyObj.visible = interacting;
    this.fullGroup.visible = !interacting;
    this.selGroup.visible = !interacting;
  }

  dispose(): void {
    this.removePick();
    this.unsubInteraction();
    this.viewer.layers.meshes.remove(this.group);
    for (const m of this.chunkMeshes) m.geometry.dispose();
    for (const m of this.selMeshes) m?.geometry.dispose();
    this.proxyObj?.geometry.dispose();
    this.edgeObj?.geometry.dispose();
    this.geometryBase.dispose();
    this.material.dispose();
    this.selMaterial.dispose();
    this.wireMaterial?.dispose();
    this.edgeMaterial?.dispose();
    this.viewer.markSceneChanged();
  }
}

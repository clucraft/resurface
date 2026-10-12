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
    this.geometryBase.dispose();
    this.material.dispose();
    this.selMaterial.dispose();
    this.wireMaterial?.dispose();
    this.viewer.markSceneChanged();
  }
}

import { afterAll, describe, expect, it } from "vitest";
import type { Loop2D, Vec2 } from "../core/types";
import { WORLD_FRAME } from "../core/vec";
import { KernelClient, type KernelTransport } from "./KernelClient";
import { collectTransferables, KERNEL_METHODS, serveKernel, type MessagePortLike } from "./kernelRpc";
import type { KernelApi } from "./kernelTypes";
import { OcctKernel } from "./OcctKernel";

const rect = (x0: number, y0: number, x1: number, y1: number): Loop2D => {
  const p: Vec2[] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  return p.map((a, i) => ({ type: "line" as const, a, b: p[(i + 1) % 4] }));
};

/** Client <-> "worker" over a real MessageChannel (structured clone + transfer semantics). */
function connect(kernel: Promise<KernelApi>): { client: KernelClient; close: () => void } {
  const { port1, port2 } = new MessageChannel();
  serveKernel(port2 as unknown as MessagePortLike, kernel);
  const client = new KernelClient(port1 as unknown as KernelTransport);
  return { client, close: () => { port1.close(); port2.close(); } };
}

const closers: (() => void)[] = [];
afterAll(() => closers.forEach((c) => c()));

describe("KernelClient RPC", () => {
  it("exposes one async method per kernel method", () => {
    for (const m of KERNEL_METHODS) expect(typeof (KernelClient.prototype as unknown as Record<string, unknown>)[m]).toBe("function");
    for (const m of KERNEL_METHODS) expect(typeof (OcctKernel.prototype as unknown as Record<string, unknown>)[m]).toBe("function");
  });

  it("runs kernel calls through the worker protocol and transfers typed arrays", async () => {
    const { client, close } = connect(OcctKernel.create());
    closers.push(close);
    await client.ready();
    const id = await client.extrude({ frame: WORLD_FRAME, profile: { loops: [rect(0, 0, 10, 10)] }, distance: 5 });
    expect(typeof id).toBe("number");
    const info = await client.info(id);
    expect(info.volume).toBeCloseTo(500, 6);
    const viaCall = await client.call("info", id);
    expect(viaCall.faces).toBe(6);
    const mesh = await client.tessellate(id, { linearDeflection: 0.1, angularDeflectionDeg: 15 });
    expect(mesh.positions).toBeInstanceOf(Float32Array);
    expect(mesh.indices).toBeInstanceOf(Uint32Array);
    expect(mesh.faceTriangleRanges.length).toBe(12);
    expect(mesh.edges.length).toBe(12);
    const step = await client.exportSTEP([{ id, name: "Block" }]);
    expect(step).toContain("Block");
    const stl = await client.exportSTL([{ id }], { linearDeflection: 0.1 });
    expect(stl.byteLength).toBe(84 + 50 * 12);
    // concurrent calls resolve independently and in order
    const [a, b] = await Promise.all([client.count(), client.info(id)]);
    expect(a).toBeGreaterThan(0);
    expect(b.type).toBe("solid");
    await client.release(id);
    expect(client.pendingCount).toBe(0);
  });

  it("propagates kernel errors as rejections with the worker message", async () => {
    const { client, close } = connect(OcctKernel.create());
    closers.push(close);
    await expect(client.extrude({ frame: WORLD_FRAME, profile: { loops: [] }, distance: 5 })).rejects.toThrow(
      "Extrude failed: profile has no closed loop",
    );
    await expect(client.info(987654)).rejects.toThrow(/unknown shape id 987654/);
    await expect((client.call as (m: string, ...a: unknown[]) => Promise<unknown>)("nope")).rejects.toThrow(/unknown kernel method 'nope'/);
  });

  it("rejects ready() and calls when the kernel fails to initialise", async () => {
    const { client, close } = connect(Promise.reject(new Error("wasm missing")));
    closers.push(close);
    await expect(client.ready()).rejects.toThrow(/wasm missing/);
    await expect(client.count()).rejects.toThrow(/wasm missing/);
  });

  it("collects only buffers owned by typed arrays", () => {
    const owned = new Float32Array(4);
    const view = new Uint8Array(new ArrayBuffer(16), 4, 4);
    const t = collectTransferables({ a: owned, b: [view, { c: owned }], d: "x" });
    expect(t).toEqual([owned.buffer]);
  });
});

import { Trash2, Pencil, Eye, EyeOff } from "lucide-react";
import { app } from "../app/App";
import type { BodyEntity, Entity, MeshEntity, PrimitiveEntity } from "../core/entities";
import { KIND_LABEL } from "../core/entities";
import { Section, Stats, TextInput, Field, NumberInput, Check, mm, deg, int } from "./widgets";
import { useStore, useUi } from "./hooks";
import { editEntity } from "../tools/registry";
import type { Vec3 } from "../core/types";

const v3 = (v: Vec3, d = 3) => `${v[0].toFixed(d)}, ${v[1].toFixed(d)}, ${v[2].toFixed(d)}`;

export function EntityInspector({ entity: e }: { entity: Entity }) {
  return (
    <>
      <Section title={KIND_LABEL[e.kind]}>
        <Field label="Name">
          <TextInput value={e.name} onChange={(v) => app.rename(e.id, v)} />
        </Field>
        <div className="btn-row" style={{ marginTop: 4 }}>
          <button className="btn small" onClick={() => app.setVisible(e.id, !e.visible)}>
            {e.visible ? <EyeOff size={13} /> : <Eye size={13} />} {e.visible ? "Hide" : "Show"}
          </button>
          {(e.kind === "sketch" || e.kind === "body" || e.kind === "sketch3d") && (
            <button className="btn small" onClick={() => void editEntity(app, e.id)}>
              <Pencil size={13} /> Edit
            </button>
          )}
          <button className="btn small danger" onClick={() => app.deleteEntities([e.id])}>
            <Trash2 size={13} /> Delete
          </button>
        </div>
      </Section>
      {e.kind === "mesh" && <MeshInfo e={e} />}
      {e.kind === "primitive" && <PrimitiveInfo e={e} />}
      {e.kind === "refline" && (
        <Section title="Line">
          <Stats rows={[["Origin", v3(e.def.origin)], ["Direction", v3(e.def.direction, 4)]]} />
        </Section>
      )}
      {e.kind === "refpoint" && (
        <Section title="Point">
          <Stats rows={[["Position", v3(e.def.position)]]} />
        </Section>
      )}
      {e.kind === "sketch" && (
        <Section title="Sketch">
          <Stats
            rows={[
              ["Entities", int(e.sketch?.entities?.length ?? 0)],
              ["Constraints", int(e.sketch?.constraints?.length ?? 0)],
              ["Section curves", int(e.sections.length)],
              ["Plane origin", v3(e.frame.origin, 2)],
              ["Plane normal", v3(e.frame.zAxis, 3)],
            ]}
          />
        </Section>
      )}
      {e.kind === "body" && <BodyInfo e={e} />}
    </>
  );
}

function MeshInfo({ e }: { e: MeshEntity }) {
  const sel = useUi((s) => (s.activeMeshId === e.id ? s.selectedTriangles : 0));
  const p = e.mesh.positions;
  let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  const n = p.length / 3;
  const step = Math.max(1, Math.floor(n / 500000));
  for (let i = 0; i < n; i += step) {
    for (let k = 0; k < 3; k++) {
      const v = p[3 * i + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  return (
    <Section title="Mesh">
      <Stats
        rows={[
          ["Triangles", int(e.mesh.indices.length / 3)],
          ["Vertices", int(n)],
          ["Selected", int(sel)],
          ["Size X", mm(max[0] - min[0], 2)],
          ["Size Y", mm(max[1] - min[1], 2)],
          ["Size Z", mm(max[2] - min[2], 2)],
          ["Colours", e.mesh.colors ? "yes" : "no"],
          ...(e.source ? ([["Source", e.source]] as [string, string][]) : []),
        ]}
      />
    </Section>
  );
}

function PrimitiveInfo({ e }: { e: PrimitiveEntity }) {
  const d = e.def;
  const rows: [string, string][] = [];
  if (d.kind === "plane") rows.push(["Origin", v3(d.origin)], ["Normal", v3(d.normal, 4)]);
  if (d.kind === "sphere") rows.push(["Centre", v3(d.center)], ["Radius", mm(d.radius)], ["Diameter", mm(2 * d.radius)]);
  if (d.kind === "cylinder") rows.push(["Axis point", v3(d.origin)], ["Axis", v3(d.axis, 4)], ["Radius", mm(d.radius)], ["Diameter", mm(2 * d.radius)], ["Length", mm(d.hMax - d.hMin)]);
  if (d.kind === "cone") rows.push(["Apex", v3(d.apex)], ["Axis", v3(d.axis, 4)], ["Half angle", deg(d.halfAngle)], ["Length", mm(d.hMax - d.hMin)]);
  return (
    <>
      <Section title="Geometry">
        <Stats rows={rows} />
      </Section>
      {e.stats && (
        <Section title="Fit quality">
          <Stats
            rows={[
              ["RMS deviation", mm(e.stats.rms), e.stats.rms < 0.05 ? "good" : e.stats.rms < 0.2 ? "warn" : "bad"],
              ["Max deviation", mm(e.stats.maxAbs)],
              ["Points used", int(e.stats.count)],
            ]}
          />
        </Section>
      )}
      {d.kind === "plane" && (
        <Section title="Display">
          <Field label="Size">
            <NumberInput value={e.displaySize ?? 50} unit="mm" min={1} onChange={(v) => app.doc.transact("Plane size", () => app.doc.update<PrimitiveEntity>(e.id, { displaySize: v }))} />
          </Field>
        </Section>
      )}
    </>
  );
}

function BodyInfo({ e }: { e: BodyEntity }) {
  const errors = useStore(app.bodyErrors);
  const f = e.feature;
  const set = (patch: Partial<typeof f>) =>
    app.doc.transact("Edit feature", () => app.doc.update<BodyEntity>(e.id, { feature: { ...f, ...patch } as BodyEntity["feature"] }));
  return (
    <Section title={`Feature: ${f.type}`}>
      {errors[e.id] && <div className="note err" style={{ marginBottom: 8 }}>{errors[e.id]}</div>}
      {f.type === "extrude" && (
        <>
          <Field label="Distance">
            <NumberInput value={f.distance} unit="mm" onChange={(v) => set({ distance: v })} />
          </Field>
          <Field label="Opposite">
            <NumberInput value={f.distance2} unit="mm" onChange={(v) => set({ distance2: v })} />
          </Field>
          <Check checked={f.solid} onChange={(v) => set({ solid: v })}>
            Solid (unchecked = surface)
          </Check>
        </>
      )}
      {f.type === "revolve" && (
        <>
          <Field label="Angle">
            <NumberInput value={f.angleDeg} unit="°" min={-360} max={360} onChange={(v) => set({ angleDeg: v })} />
          </Field>
          <Check checked={f.solid} onChange={(v) => set({ solid: v })}>
            Solid
          </Check>
        </>
      )}
      {(f.type === "fillet" || f.type === "chamfer") && (
        <Field label={f.type === "fillet" ? "Radius" : "Distance"}>
          <NumberInput
            value={f.type === "fillet" ? f.radius : f.distance}
            unit="mm"
            min={0.001}
            onChange={(v) => set(f.type === "fillet" ? ({ radius: v } as never) : ({ distance: v } as never))}
          />
        </Field>
      )}
      {(f.type === "fitSurface" || f.type === "freeform") && f.stats && (
        <Stats rows={[["RMS deviation", mm(f.stats.rms)], ["Max deviation", mm(f.stats.maxAbs)]]} />
      )}
    </Section>
  );
}

/** Small form controls used by tool panels and the property inspector. */
import { useEffect, useState, type ReactNode } from "react";

export function Section({ title, children, right }: { title?: ReactNode; children: ReactNode; right?: ReactNode }) {
  return (
    <div className="props-section">
      {title && (
        <div className="props-title">
          {title}
          {right && <span style={{ marginLeft: "auto" }}>{right}</span>}
        </div>
      )}
      {children}
    </div>
  );
}

export function Field({ label, children, wide }: { label?: ReactNode; children: ReactNode; wide?: boolean }) {
  return (
    <div className={`field${wide ? " wide" : ""}`}>
      {label !== undefined && <label>{label}</label>}
      <div>{children}</div>
    </div>
  );
}

const fmt = (v: number, digits: number) => {
  if (!Number.isFinite(v)) return "";
  const s = v.toFixed(digits);
  return digits > 0 ? s.replace(/\.?0+$/, "") || "0" : s;
};

/** Numeric input that commits on Enter/blur and supports arrow-key nudging and simple math. */
export function NumberInput({
  value,
  onChange,
  unit,
  step = 1,
  min,
  max,
  digits = 3,
  disabled,
}: {
  value: number;
  onChange: (v: number) => void;
  unit?: string;
  step?: number;
  min?: number;
  max?: number;
  digits?: number;
  disabled?: boolean;
}) {
  const [text, setText] = useState(fmt(value, digits));
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (!focused) setText(fmt(value, digits));
  }, [value, digits, focused]);
  const clamp = (v: number) => Math.min(max ?? Infinity, Math.max(min ?? -Infinity, v));
  const commit = (t: string) => {
    let v = Number(t);
    if (!Number.isFinite(v)) {
      // allow simple expressions like "10/2" or "3*4+1"
      if (/^[\d\s+\-*/().]+$/.test(t)) {
        try {
          v = Function(`"use strict";return (${t})`)() as number;
        } catch {
          v = NaN;
        }
      }
    }
    if (Number.isFinite(v)) {
      const c = clamp(v);
      onChange(c);
      setText(fmt(c, digits));
    } else setText(fmt(value, digits));
  };
  return (
    <div className="num-wrap">
      <input
        className="num-input"
        value={text}
        disabled={disabled}
        onFocus={(e) => {
          setFocused(true);
          e.target.select();
        }}
        onBlur={(e) => {
          setFocused(false);
          commit(e.target.value);
        }}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          e.stopPropagation();
          if (e.key === "Enter") commit((e.target as HTMLInputElement).value);
          if (e.key === "ArrowUp" || e.key === "ArrowDown") {
            e.preventDefault();
            const mult = e.shiftKey ? 10 : e.altKey ? 0.1 : 1;
            const v = clamp(value + (e.key === "ArrowUp" ? 1 : -1) * step * mult);
            onChange(v);
            setText(fmt(v, digits));
          }
        }}
      />
      {unit && <span className="unit">{unit}</span>}
    </div>
  );
}

export function Slider({
  value,
  onChange,
  min,
  max,
  step = 1,
  format,
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  format?: (v: number) => string;
}) {
  return (
    <div className="slider-row">
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="val">{format ? format(value) : value}</span>
    </div>
  );
}

export function Check({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="check">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{children}</span>
    </label>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: ReactNode; title?: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="seg">
      {options.map((o) => (
        <button key={o.value} className={o.value === value ? "on" : ""} title={o.title} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Select<T extends string>({
  value,
  options,
  onChange,
  placeholder,
}: {
  value: T | "";
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  placeholder?: string;
}) {
  return (
    <select className="select-input" value={value} onChange={(e) => onChange(e.target.value as T)}>
      {placeholder !== undefined && (
        <option value="" disabled>
          {placeholder}
        </option>
      )}
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function TextInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const [t, setT] = useState(value);
  useEffect(() => setT(value), [value]);
  return (
    <input
      className="text-input"
      value={t}
      onChange={(e) => setT(e.target.value)}
      onBlur={() => t !== value && onChange(t)}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
      }}
    />
  );
}

export function Stats({ rows }: { rows: [ReactNode, ReactNode, ("good" | "warn" | "bad")?][] }) {
  return (
    <div className="stats">
      {rows.map(([k, v, cls], i) => (
        <div key={i} style={{ display: "contents" }}>
          <span className="k">{k}</span>
          <span className={`v ${cls ?? ""}`}>{v}</span>
        </div>
      ))}
    </div>
  );
}

export const mm = (v: number | undefined, digits = 3) => (v === undefined || !Number.isFinite(v) ? "—" : `${v.toFixed(digits)} mm`);
export const deg = (rad: number, digits = 2) => `${((rad * 180) / Math.PI).toFixed(digits)}°`;
export const int = (n: number) => n.toLocaleString("en-US");

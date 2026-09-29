// Camera preview frames ride the browser's NBF1 binary frames (browser/frames.ts) with their own
// kind byte, so one socket carries both and neither card ever paints the other's pixels.
import { HEADER_LEN } from "../browser/frames";
import type { CamBox } from "./types";

/** NBF1 `kind` for camera preview frames (browser: 1 = live page, 2 = still). */
export const KIND_CAMERA = 3;
const MAGIC = 0x4e424631; // "NBF1", big-endian

/** A binary message that is a v1 NBF1 camera frame. */
export function isCameraFrame(buf: ArrayBuffer): boolean {
  if (buf.byteLength < HEADER_LEN) return false;
  const dv = new DataView(buf);
  return dv.getUint32(0, false) === MAGIC && dv.getUint8(4) === 1 && dv.getUint8(5) === KIND_CAMERA;
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** Labels as the core may send them — "cup", ["cup", 0.9], {label, score|conf} — as short strings. */
export function normLabels(raw: unknown[] | undefined, max = 6): string[] {
  const out: string[] = [];
  for (const r of raw ?? []) {
    let s = "";
    if (typeof r === "string") s = r;
    else if (Array.isArray(r) && typeof r[0] === "string") s = r[0];
    else if (r && typeof r === "object" && typeof (r as { label?: unknown }).label === "string") s = (r as { label: string }).label;
    s = s.trim().slice(0, 40);
    if (s && !out.some((o) => o.toLowerCase() === s.toLowerCase())) out.push(s);
    if (out.length >= max) break;
  }
  return out;
}

/** Boxes in 0..1 frame coordinates; anything malformed or empty is dropped. Also accepts `box: [x, y, w, h]`. */
export function normBoxes(raw: unknown[] | undefined, max = 12): CamBox[] {
  const out: CamBox[] = [];
  for (const r of raw ?? []) {
    if (!r || typeof r !== "object") continue;
    const o = r as Record<string, unknown>;
    const arr = Array.isArray(o.box) ? (o.box as unknown[]) : null;
    const x = num(o.x) ?? num(arr?.[0]), y = num(o.y) ?? num(arr?.[1]);
    const w = num(o.w) ?? num(arr?.[2]), h = num(o.h) ?? num(arr?.[3]);
    if (x === null || y === null || w === null || h === null) continue;
    const bx = clamp01(x), by = clamp01(y);
    const bw = clamp01(Math.min(w, 1 - bx)), bh = clamp01(Math.min(h, 1 - by));
    if (bw <= 0.005 || bh <= 0.005) continue;
    const label = typeof o.label === "string" ? o.label.trim().slice(0, 32) : "";
    const id = num(o.id);
    out.push({ x: bx, y: by, w: bw, h: bh, label, score: num(o.score) ?? num(o.conf) ?? 0, ...(id !== null ? { id } : {}) });
    if (out.length >= max) break;
  }
  return out;
}

/** Where a box sits on a preview that may be mirrored (selfie view): CSS percentages. */
export function boxStyle(b: CamBox, mirrored: boolean): { left: string; top: string; width: string; height: string } {
  const x = mirrored ? 1 - b.x - b.w : b.x;
  const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
  return { left: pct(x), top: pct(b.y), width: pct(b.w), height: pct(b.h) };
}

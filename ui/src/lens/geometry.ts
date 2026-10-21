// Search Screen's selection, as geometry: one gesture for everything. The selection is always the
// box around the pointer's path — a straight drag is a rectangle, a circle or a scribble gives its
// box (padded), and a click (or a still press) is smart pick. Then its corners, edges and inside
// adjust it. All in display-local points; nothing here touches the DOM.

import type { Rect } from "./wire.ts";

export interface Box { x: number; y: number; w: number; h: number }
export interface Pt { x: number; y: number }

/** Moved less than this (pt): a click. */
export const CLICK_MOVE = 6;
/** A box smaller than this on either side is taken for a click at its middle. */
export const MIN_SIDE = 16;
/** A path this many times longer than its box's diagonal curved back on itself (a circle, a scribble). */
export const LASSO_RATIO = 2.2;
export const LASSO_PAD = 6;
/** Corner hit squares and edge bands, in points. */
export const CORNER_HIT = 28;
export const EDGE_HIT = 20;

export const box = (x: number, y: number, w: number, h: number): Box => ({ x, y, w, h });
export const toRect = (b: Box): Rect => [b.x, b.y, b.w, b.h];
export const fromRect = (r: Rect): Box => ({ x: r[0], y: r[1], w: r[2], h: r[3] });
export const centre = (b: Box): Pt => ({ x: b.x + b.w / 2, y: b.y + b.h / 2 });
export const area = (b: Box) => Math.max(0, b.w) * Math.max(0, b.h);
export const contains = (b: Box, p: Pt) => p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h;

export function bbox(pts: readonly Pt[]): Box {
  if (!pts.length) return box(0, 0, 0, 0);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of pts) {
    x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y);
    x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y);
  }
  return box(x0, y0, x1 - x0, y1 - y0);
}

export function pathLength(pts: readonly Pt[]): number {
  let n = 0;
  for (let i = 1; i < pts.length; i++) n += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  return n;
}

/** The farthest any point got from the first. */
export function travel(pts: readonly Pt[]): number {
  if (!pts.length) return 0;
  let m = 0;
  for (const p of pts) m = Math.max(m, Math.hypot(p.x - pts[0].x, p.y - pts[0].y));
  return m;
}

/** Inside the display (W × H), whatever it was. */
export function clampBox(b: Box, W: number, H: number): Box {
  const x0 = Math.min(Math.max(b.x, 0), W), y0 = Math.min(Math.max(b.y, 0), H);
  const x1 = Math.min(Math.max(b.x + b.w, 0), W), y1 = Math.min(Math.max(b.y + b.h, 0), H);
  return box(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
}

export function padBox(b: Box, pad: number, W: number, H: number): Box {
  return clampBox(box(b.x - pad, b.y - pad, b.w + 2 * pad, b.h + 2 * pad), W, H);
}

export type Gesture = { kind: "click"; at: Pt } | { kind: "box"; box: Box; how: "drag" | "lasso" };

/** What a press-move-release was: a click (smart pick) or a box, and which kind of box. */
export function classify(pts: readonly Pt[], W: number, H: number): Gesture {
  if (!pts.length) return { kind: "click", at: { x: 0, y: 0 } };
  if (travel(pts) < CLICK_MOVE) return { kind: "click", at: pts[0] };
  const raw = bbox(pts);
  const diag = Math.hypot(raw.w, raw.h);
  const lasso = diag > 0 && pathLength(pts) > LASSO_RATIO * diag;
  const b = lasso ? padBox(raw, LASSO_PAD, W, H) : clampBox(raw, W, H);
  if (b.w < MIN_SIDE || b.h < MIN_SIDE) return { kind: "click", at: centre(clampBox(raw, W, H)) };
  return { kind: "box", box: b, how: lasso ? "lasso" : "drag" };
}

export type Hit = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w" | "inside" | "outside";

/** Which part of the selection a point is on: a corner, an edge, the inside, or outside it. */
export function hitTest(b: Box, p: Pt): Hit {
  const c = CORNER_HIT / 2, e = EDGE_HIT / 2;
  const near = (a: number, v: number, r: number) => Math.abs(a - v) <= r;
  const [l, r, t, bt] = [b.x, b.x + b.w, b.y, b.y + b.h];
  if (near(p.x, l, c) && near(p.y, t, c)) return "nw";
  if (near(p.x, r, c) && near(p.y, t, c)) return "ne";
  if (near(p.x, r, c) && near(p.y, bt, c)) return "se";
  if (near(p.x, l, c) && near(p.y, bt, c)) return "sw";
  const inX = p.x >= l - e && p.x <= r + e, inY = p.y >= t - e && p.y <= bt + e;
  if (inX && near(p.y, t, e)) return "n";
  if (inX && near(p.y, bt, e)) return "s";
  if (inY && near(p.x, l, e)) return "w";
  if (inY && near(p.x, r, e)) return "e";
  return contains(b, p) ? "inside" : "outside";
}

export const CURSOR: Record<Hit, string> = {
  nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize",
  n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", inside: "grab", outside: "crosshair",
};

/** The selection dragged by one of its parts (`start` is how it was at the press). Never smaller
 *  than MIN_SIDE, never off the display; dragging an edge past the other one flips it. */
export function dragBox(start: Box, hit: Hit, dx: number, dy: number, W: number, H: number): Box {
  if (hit === "outside") return start;
  if (hit === "inside") {
    const x = Math.min(Math.max(start.x + dx, 0), Math.max(0, W - start.w));
    const y = Math.min(Math.max(start.y + dy, 0), Math.max(0, H - start.h));
    return box(x, y, start.w, start.h);
  }
  let [l, t, r, b] = [start.x, start.y, start.x + start.w, start.y + start.h];
  if (hit.includes("w")) l += dx;
  if (hit.includes("e")) r += dx;
  if (hit === "n" || hit === "nw" || hit === "ne") t += dy;
  if (hit === "s" || hit === "sw" || hit === "se") b += dy;
  const out = clampBox(box(Math.min(l, r), Math.min(t, b), Math.abs(r - l), Math.abs(b - t)), W, H);
  return atLeast(out, W, H);
}

/** Grown to MIN_SIDE on a side that is shorter, inside the display. */
export function atLeast(b: Box, W: number, H: number): Box {
  const w = Math.max(b.w, Math.min(MIN_SIDE, W)), h = Math.max(b.h, Math.min(MIN_SIDE, H));
  return box(Math.min(Math.max(b.x, 0), W - w), Math.min(Math.max(b.y, 0), H - h), w, h);
}

/** The arrow keys: moved by (dx, dy), or (`resize`) grown from the bottom-right corner. */
export function nudge(b: Box, dx: number, dy: number, resize: boolean, W: number, H: number): Box {
  if (resize) return atLeast(clampBox(box(b.x, b.y, Math.max(MIN_SIDE, b.w + dx), Math.max(MIN_SIDE, b.h + dy)), W, H), W, H);
  return dragBox(b, "inside", dx, dy, W, H);
}

export function iou(a: Box, b: Box): number {
  const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const union = area(a) + area(b) - inter;
  return union > 0 ? inter / union : 0;
}

/** Points no closer than `min` to the previous one (the last is always kept). */
export function simplify(pts: readonly Pt[], min = 2): Pt[] {
  if (pts.length < 3) return [...pts];
  const out: Pt[] = [pts[0]];
  for (let i = 1; i < pts.length - 1; i++) {
    const p = out[out.length - 1];
    if (Math.hypot(pts[i].x - p.x, pts[i].y - p.y) >= min) out.push(pts[i]);
  }
  out.push(pts[pts.length - 1]);
  return out;
}

const r1 = (v: number) => Math.round(v * 10) / 10;

/** A smooth SVG path through the points (Catmull-Rom as cubic Béziers). */
export function smoothPath(pts: readonly Pt[]): string {
  const p = simplify(pts);
  if (!p.length) return "";
  if (p.length === 1) return `M${r1(p[0].x)} ${r1(p[0].y)}`;
  let d = `M${r1(p[0].x)} ${r1(p[0].y)}`;
  for (let i = 0; i < p.length - 1; i++) {
    const p0 = p[i - 1] ?? p[i], p1 = p[i], p2 = p[i + 1], p3 = p[i + 2] ?? p2;
    const c1 = { x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6 };
    const c2 = { x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6 };
    d += ` C${r1(c1.x)} ${r1(c1.y)} ${r1(c2.x)} ${r1(c2.y)} ${r1(p2.x)} ${r1(p2.y)}`;
  }
  return d;
}

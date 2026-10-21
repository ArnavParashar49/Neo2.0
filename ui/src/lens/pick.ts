// Smart pick: what a click selects, from what the core found in the still (lens_regions) — the
// line of text under the pointer, the thing (a foreground instance, else a salient object), the
// window, else a box around the point. A double-click on a line takes its paragraph. Tab walks
// the same regions in reading order.

import { area, box, clampBox, contains, fromRect, padBox, type Box, type Pt } from "./geometry.ts";
import type { Regions } from "./wire.ts";

export type Source = "line" | "block" | "instance" | "object" | "window" | "default";
export interface Picked { box: Box; source: Source }

/** A line's box grows this much (its letters' tops and tails). */
export const LINE_PAD = 4;
/** An instance covering more of the display than this is the background, not a thing. */
export const MAX_INSTANCE = 0.6;
export const MIN_WINDOW = { w: 120, h: 80 };
export const DEFAULT_BOX = { w: 280, h: 200 };

function smallest(boxes: Box[], p: Pt): Box | null {
  let best: Box | null = null;
  for (const b of boxes) if (contains(b, p) && (!best || area(b) < area(best))) best = b;
  return best;
}

/** The region a click at `p` means, without the last resort (null: none under it). */
export function regionAt(r: Regions | null, p: Pt, W: number, H: number, double = false): Picked | null {
  if (!r) return null;
  const line = r.lines.find((l) => contains(fromRect(l.box), p));
  if (line) {
    const blk = double && line.block >= 0 ? r.blocks[line.block] : undefined;
    if (blk) return { box: padBox(fromRect(blk), LINE_PAD, W, H), source: "block" };
    return { box: padBox(fromRect(line.box), LINE_PAD, W, H), source: "line" };
  }
  const limit = MAX_INSTANCE * W * H;
  const inst = smallest(r.instances.map(fromRect).filter((b) => area(b) <= limit), p);
  if (inst) return { box: clampBox(inst, W, H), source: "instance" };
  const objc = smallest(r.objects.map(fromRect), p);
  if (objc) return { box: clampBox(objc, W, H), source: "object" };
  const win = windowAt(r, p);
  if (win) return { box: clampBox(win, W, H), source: "window" };
  return null;
}

/** The front-most window under the point that is big enough to mean something. */
export function windowAt(r: Regions | null, p: Pt): Box | null {
  if (!r) return null;
  for (const w of r.windows) {
    const b = fromRect(w);
    if (b.w >= MIN_WINDOW.w && b.h >= MIN_WINDOW.h && contains(b, p)) return b; // listed front first
  }
  return null;
}

/** A box of the default size around the point, inside the display. */
export function around(p: Pt, W: number, H: number): Box {
  const w = Math.min(DEFAULT_BOX.w, W), h = Math.min(DEFAULT_BOX.h, H);
  const x = Math.min(Math.max(p.x - w / 2, 0), W - w), y = Math.min(Math.max(p.y - h / 2, 0), H - h);
  return box(x, y, w, h);
}

/** What a click at `p` selects. */
export function pick(r: Regions | null, p: Pt, W: number, H: number, double = false): Picked {
  return regionAt(r, p, W, H, double) ?? { box: around(p, W, H), source: "default" };
}

/** Top to bottom, then left to right (boxes whose tops are within `row` points share a row). */
export function readingOrder(boxes: Box[], row = 12): Box[] {
  return [...boxes].sort((a, b) => (Math.abs(a.y - b.y) <= row ? a.x - b.x : a.y - b.y));
}

/** Tab's stops: the things (instances and objects), then the paragraphs, then the windows. */
export function tabStops(r: Regions | null, W: number, H: number): Box[] {
  if (!r) return [];
  const limit = MAX_INSTANCE * W * H;
  const things = [...r.instances.map(fromRect).filter((b) => area(b) <= limit), ...r.objects.map(fromRect)];
  const blocks = r.blocks.map(fromRect);
  const windows = r.windows.map(fromRect).filter((b) => b.w >= MIN_WINDOW.w && b.h >= MIN_WINDOW.h);
  const out: Box[] = [];
  for (const b of [...readingOrder(things), ...readingOrder(blocks), ...readingOrder(windows)]) {
    const c = clampBox(b, W, H);
    if (c.w > 0 && c.h > 0 && !out.some((o) => o.x === c.x && o.y === c.y && o.w === c.w && o.h === c.h)) out.push(c);
  }
  return out;
}

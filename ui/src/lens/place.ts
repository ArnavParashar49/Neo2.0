// Where the results sheet goes: beside the selection (right, else left), else under it, else above
// it, else — a selection that fills the screen — floating inside it, bottom-right. Anchored at its
// top: as it grows it moves up rather than jumping to another side.

import type { Box } from "./geometry.ts";

export const MARGIN = 16;
export const GAP = 14;
export type Side = "right" | "left" | "below" | "above" | "inside";
export interface Placed { x: number; y: number; w: number; maxH: number; side: Side }

/** The sheet's width on a display this wide. */
export const sheetWidth = (W: number) => (W < 1000 ? 320 : 372);

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), Math.max(lo, hi));

/**
 * `h`: the sheet's height now (it scrolls past 70% of the display). `prev`: the side it is on, kept
 * while it still fits there (the sheet never jumps sides because its answer grew).
 */
export function placeSheet(sel: Box, h: number, W: number, H: number, safeTop = 0, prev?: Side): Placed {
  const w = Math.min(sheetWidth(W), W - 2 * MARGIN);
  const maxH = Math.round(H * 0.7);
  const sh = Math.min(h, maxH);
  const top = safeTop + MARGIN, bottom = H - MARGIN;
  const fits: Record<Side, () => Placed | null> = {
    right: () => {
      const x = sel.x + sel.w + GAP;
      return x + w <= W - MARGIN ? { x, y: clamp(sel.y, top, bottom - sh), w, maxH, side: "right" } : null;
    },
    left: () => {
      const x = sel.x - GAP - w;
      return x >= MARGIN ? { x, y: clamp(sel.y, top, bottom - sh), w, maxH, side: "left" } : null;
    },
    below: () => {
      const y = sel.y + sel.h + GAP;
      return y + sh <= bottom ? { x: clamp(sel.x + sel.w / 2 - w / 2, MARGIN, W - MARGIN - w), y, w, maxH, side: "below" } : null;
    },
    above: () => {
      const y = sel.y - GAP - sh;
      return y >= top ? { x: clamp(sel.x + sel.w / 2 - w / 2, MARGIN, W - MARGIN - w), y, w, maxH, side: "above" } : null;
    },
    inside: () => {
      const x = clamp(Math.min(sel.x + sel.w, W) - MARGIN - w, MARGIN, W - MARGIN - w);
      const y = clamp(Math.min(sel.y + sel.h, H) - MARGIN - sh, top, bottom - sh);
      return { x, y, w, maxH, side: "inside" };
    },
  };
  // A side beside the selection stays while the sheet fits beside it at any height (it slides up);
  // below and above stay while the sheet fits there at its height now.
  if (prev && prev !== "inside") {
    const kept = fits[prev]();
    if (kept) return kept;
  }
  for (const side of ["right", "left", "below", "above"] as const) {
    const p = fits[side]();
    if (p) return p;
  }
  return fits.inside() as Placed;
}

/** Whether two boxes overlap (the tests check the sheet never covers the selection when there's room). */
export function overlaps(a: Box, b: Box): boolean {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

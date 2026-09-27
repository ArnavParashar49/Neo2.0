// Window sizes while browsing (§1.2) and the page viewport policy (§1.4).
import type { Area, Size } from "../win";

export const CARD: Size = { w: 400, h: 600 };
export const PILL: Size = { w: 84, h: 84 };
/** Header 52 + card chrome 90 + strip 96 + input 57 + gaps 21. */
export const DOCKED_CHROME = 316;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export interface BrowserSizes { browse: Size; focus: Size; compact: boolean }

export function browserSizes(a: Area | null): BrowserSizes {
  if (!a) return { browse: { w: 847, h: 788 }, focus: { w: 1119, h: 861 }, compact: false };
  let browse = { w: Math.min(a.w - 16, clamp(Math.round(a.w * 0.56), 760, 1000)), h: Math.min(a.h - 16, clamp(Math.round(a.h * 0.86), 580, 860)) };
  let focus = { w: Math.min(a.w - 16, clamp(Math.round(a.w * 0.74), 900, 1400)), h: Math.min(a.h - 16, clamp(Math.round(a.h * 0.94), 620, 1040)) };
  const compact = a.w < 776 || browse.h - DOCKED_CHROME < 400;
  if (compact) {
    browse = { w: Math.min(a.w - 16, focus.w), h: a.h - 16 };
    focus = { w: a.w - 16, h: a.h - 16 };
  }
  return { browse, focus, compact };
}

/** The page's CSS viewport for a view of this size: desktop layouts at ≈0.8 scale, same aspect. */
export function pageViewport(viewW: number, viewH: number): { w: number; h: number } {
  const w = clamp(Math.round(viewW / 0.8), 1024, Math.max(1440, viewW));
  const h = clamp(Math.round((w * viewH) / Math.max(1, viewW)), 560, 1600);
  return { w, h };
}

/** Pop-out window default frame. */
export function popoutSize(a: Area | null): Size {
  if (!a) return { w: 1100, h: 760 };
  return { w: Math.max(720, clamp(Math.round(a.w * 0.7), 900, 1440)), h: Math.max(520, clamp(Math.round(a.h * 0.82), 620, 1000)) };
}

/** Contain-fit a w×h page into a view: the frame rect everything on the page maps into. */
export function fitRect(viewW: number, viewH: number, pw: number, ph: number) {
  if (pw <= 0 || ph <= 0 || viewW <= 0 || viewH <= 0) return { x: 0, y: 0, w: Math.max(0, viewW), h: Math.max(0, viewH), s: 1 };
  const s = Math.min(viewW / pw, viewH / ph);
  const w = pw * s, h = ph * s;
  return { x: (viewW - w) / 2, y: (viewH - h) / 2, w, h, s };
}

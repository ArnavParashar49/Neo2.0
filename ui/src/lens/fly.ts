// The area flying into the notch, as numbers (LensApp's `Fly` plays them). When NotchHub shows the
// results, the overlay hands the area over and closes: the selected crop lifts off the still and
// arcs up into the notch — it moves sideways early (x eases out) and rises late and fast (y eases
// in), shrinking to a 36 pt speck that vanishes as it arrives — while the still, the veil and the
// glow fade, so the live screen is back at once and NotchHub's panel springs open as the crop gets
// there. On another display than NotchHub's island it only lifts and fades where it was; with
// Reduce Motion it fades in place. Only transforms and opacity animate, on layers the size of the
// area or smaller.

import type { Box } from "./geometry.ts";

export const FLY = { ms: 380, stillMs: 200, bracketsMs: 120, targetPt: 36, reducedMs: 120, liftMs: 240, liftPt: 48 } as const;
/** The page resets this long after the overlay starts closing into the notch (the shell orders the
 *  window out at 440 ms). */
export const FLY_RESET_MS = FLY.ms + 80;
export const FLY_RESET_REDUCED_MS = 200;

export type FlyMode = "notch" | "lift" | "fade";

export interface FlyPlan {
  mode: FlyMode;
  /** Where it starts: exactly over the area in the still. */
  box: Box;
  /** To the target's centre. */
  dx: number;
  dy: number;
  /** targetPt / the area's longer side (at least 0.02). */
  scale: number;
  ms: number;
}

/** The three curves (cubic-bezier control points): sideways early, up late, shrinking in between. */
export const EASE = {
  x: [0.2, 0, 0, 1] as const,
  y: [0.55, 0, 0.8, 0.2] as const,
  s: [0.4, 0, 0.2, 1] as const,
};
const css = (c: readonly number[]) => `cubic-bezier(${c.join(", ")})`;

/** Where the notch is on this display: its centre (`safeTop` is the notch's height; none: 12 pt down). */
export function notchCentre(W: number, safeTop: number): { x: number; y: number } {
  return { x: W / 2, y: safeTop > 0 ? safeTop / 2 : 12 };
}

/**
 * The flight for an area. `sameDisplay`: NotchHub's island is on this overlay's display (or either
 * display is unknown — then the notch is assumed here).
 */
export function flyPlan(box: Box, W: number, H: number, safeTop: number, sameDisplay: boolean, reduced: boolean): FlyPlan {
  const b = { ...box };
  const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
  if (reduced) return { mode: "fade", box: b, dx: 0, dy: 0, scale: 1, ms: FLY.reducedMs };
  if (!sameDisplay || !(W > 0) || !(H > 0)) {
    return { mode: "lift", box: b, dx: 0, dy: -Math.min(FLY.liftPt, Math.max(0, cy)), scale: 0.9, ms: FLY.liftMs };
  }
  const t = notchCentre(W, safeTop);
  const long = Math.max(b.w, b.h, 1);
  return { mode: "notch", box: b, dx: t.x - cx, dy: t.y - cy, scale: Math.max(0.02, Math.min(1, FLY.targetPt / long)), ms: FLY.ms };
}

const r = (v: number) => Math.round(v * 100) / 100;

/** The plan as keyframes: the outer layer moves sideways (x), the middle one up (y), the inner one
 *  shrinks (s) and fades (o). The fade keeps the rise's timing, so the crop stays whole until it is
 *  nearly at the notch and vanishes into it. */
export function flyFrames(p: FlyPlan): {
  x: Keyframe[]; y: Keyframe[]; s: Keyframe[]; o: Keyframe[];
  easing: { x: string; y: string; s: string; o: string };
} {
  if (p.mode === "fade") {
    return { x: [], y: [], s: [], o: [{ opacity: 1 }, { opacity: 0 }], easing: { x: "linear", y: "linear", s: "linear", o: "linear" } };
  }
  const x: Keyframe[] = p.dx ? [{ transform: "translate3d(0px, 0px, 0px)" }, { transform: `translate3d(${r(p.dx)}px, 0px, 0px)` }] : [];
  const y: Keyframe[] = [{ transform: "translate3d(0px, 0px, 0px)" }, { transform: `translate3d(0px, ${r(p.dy)}px, 0px)` }];
  const s: Keyframe[] = [{ transform: "scale(1)" }, { transform: `scale(${Math.round(p.scale * 1000) / 1000})` }];
  if (p.mode === "lift") {
    return { x, y, s, o: [{ opacity: 1 }, { opacity: 0 }], easing: { x: css(EASE.x), y: css(EASE.x), s: css(EASE.s), o: css(EASE.s) } };
  }
  return {
    x, y, s,
    // Whole until 72% of the rise, gone as it reaches the notch.
    o: [{ opacity: 1, offset: 0 }, { opacity: 1, offset: 0.72 }, { opacity: 0, offset: 1 }],
    easing: { x: css(EASE.x), y: css(EASE.y), s: css(EASE.s), o: css(EASE.y) },
  };
}

/** A CSS cubic-bezier timing function: time fraction → progress (for the tests, and for anyone who
 *  needs to know where the crop is). */
export function bezier(x1: number, y1: number, x2: number, y2: number): (t: number) => number {
  const at = (a: number, b: number, u: number) => 3 * a * u * (1 - u) ** 2 + 3 * b * u ** 2 * (1 - u) + u ** 3;
  return (t: number) => {
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    let lo = 0, hi = 1, u = t;
    for (let i = 0; i < 60; i++) {
      u = (lo + hi) / 2;
      if (at(x1, x2, u) < t) lo = u;
      else hi = u;
    }
    return at(y1, y2, u);
  };
}

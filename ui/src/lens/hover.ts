// Smart pick's hover: a thin outline round the line, paragraph or thing under a resting pointer.
//
// v1 drew it as a 6% white wash with a bright border, took it away on every move and brought it
// back 150 ms after each rest — over a maximized app that is the whole window, so moving the mouse
// blinked the screen white. Now it is an outline with no fill, never for a window, the default box
// or anything over HOVER.maxArea of the display; it moves from one region to the next at once
// (no fade between them, no blink), and it never goes through the overlay's reducer: one element,
// written by `write`, and all the work of any number of pointer moves done once per animation frame.

import { area, type Box, type Pt } from "./geometry.ts";
import type { Picked } from "./pick.ts";

export const HOVER = { restMs: 150, maxArea: 0.35, fadeInMs: 120, fadeOutMs: 90 } as const;

/** The box a hover may show for a pick, or null: lines, paragraphs, things — never a window, never
 *  the default box, never anything over HOVER.maxArea of the display. */
export function hoverable(p: Picked | null, W: number, H: number): Box | null {
  if (!p || p.source === "window" || p.source === "default") return null;
  const b = p.box;
  if (!(W > 0) || !(H > 0) || !(b.w > 0) || !(b.h > 0)) return null;
  return area(b) <= HOVER.maxArea * W * H ? b : null;
}

export interface HoverDeps {
  raf(cb: () => void): number;
  cancelRaf(id: number): void;
  setTimeout(cb: () => void, ms: number): number;
  clearTimeout(id: number): void;
  /** pick.regionAt bound to the current regions (null: nothing under the point, or no regions yet). */
  regionAt(p: Pt): Picked | null;
  /** The display's size now, in points. */
  size(): { W: number; H: number };
  /** The one DOM write (LensApp: the hover element's transform, size and opacity). */
  write(box: Box | null): void;
}

export interface Hover {
  /** The pointer moved (any number of times a frame: the work runs once, in the next frame). */
  move(p: Pt): void;
  /** Gone at once: pointer down, a selection, a Tab candidate, a card, closing. */
  hide(): void;
  dispose(): void;
}

const same = (a: Box, b: Box) => a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;

export function makeHover(d: HoverDeps): Hover {
  let at: Pt | null = null;
  let frame: number | null = null;
  let rest: number | null = null;
  let shown: Box | null = null;
  let dead = false;

  const look = (p: Pt): Box | null => {
    const { W, H } = d.size();
    return hoverable(d.regionAt(p), W, H);
  };

  const rested = () => {
    rest = null;
    if (dead || !at) return;
    const b = look(at);
    if (b) {
      shown = b;
      d.write(b);
    }
  };

  const tick = () => {
    frame = null;
    if (dead || !at) return;
    if (!shown) {
      // Nothing shown yet: no hit test while the pointer moves, one once it has rested.
      if (rest !== null) d.clearTimeout(rest);
      rest = d.setTimeout(rested, HOVER.restMs);
      return;
    }
    const b = look(at);
    if (b && same(b, shown)) return; // still over the same region
    shown = b;
    d.write(b); // another region: the outline moves there at once; none: it fades out
  };

  const cancel = () => {
    if (frame !== null) d.cancelRaf(frame);
    if (rest !== null) d.clearTimeout(rest);
    frame = rest = null;
  };

  return {
    move(p) {
      if (dead) return;
      at = p;
      if (frame === null) frame = d.raf(tick);
    },
    hide() {
      cancel();
      if (shown) {
        shown = null;
        d.write(null);
      }
    },
    dispose() {
      dead = true;
      cancel();
    },
  };
}

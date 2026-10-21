// The selected area: clear inside with rounded corners, the rest of the screen dimmed, a hairline
// along its edges and four white corner brackets that snap into place when it is picked. Tab's
// candidate uses the same pieces, lighter.
//
// Every piece is small and composited, and follows the box by `transform` (plus a length for the
// veil and the hairlines): four solid veil rectangles round the box (a layer of one colour has no
// backing store to repaint), four 14 pt caps that round the hole's corners, four hairlines and four
// bracket corners (a dark under-stroke gives them contrast; no filter). Nothing is display-sized and
// nothing paints while a corner is dragged — `place(box)` writes the transforms directly, and React
// calls the same `place` when it renders, so the two never disagree.

import { forwardRef, useImperativeHandle, useLayoutEffect, useRef } from "react";
import type { Box } from "./geometry";

export interface SelectionHandle {
  place(box: Box): void;
}

interface Props {
  box: Box;
  /** The display's size (the veil reaches its edges). */
  W: number;
  H: number;
  candidate?: boolean;
  label?: string;
}

/** The hole's corner radius, in points. */
export const RADIUS = 14;
/** A bracket corner's SVG: 30 × 30, the corner itself PAD in from its outer edges. */
const SIZE = 30;
const PAD = 4;

/** A bracket's arm for a box this size (as v1's bracketPath). */
export const armFor = (b: Box) => Math.max(4, Math.min(22, b.w / 2.5, b.h / 2.5));

type Corner = "nw" | "ne" | "se" | "sw";
const CORNERS: Corner[] = ["nw", "ne", "se", "sw"];

/** One corner of the brackets, in its own 30 × 30 box (the corner point at PAD, PAD from its outside). */
export function cornerPath(c: Corner, arm: number): string {
  const a = arm, r = Math.max(0, Math.min(12, a - 1));
  const f = (v: number) => Math.round(v * 10) / 10;
  const lo = PAD, hi = SIZE - PAD;
  switch (c) {
    case "nw": return `M${lo} ${f(lo + a)}V${f(lo + r)}A${r} ${r} 0 0 1 ${f(lo + r)} ${lo}H${f(lo + a)}`;
    case "ne": return `M${f(hi - a)} ${lo}H${f(hi - r)}A${r} ${r} 0 0 1 ${hi} ${f(lo + r)}V${f(lo + a)}`;
    case "se": return `M${hi} ${f(hi - a)}V${f(hi - r)}A${r} ${r} 0 0 1 ${f(hi - r)} ${hi}H${f(hi - a)}`;
    case "sw": return `M${f(lo + a)} ${hi}H${f(lo + r)}A${r} ${r} 0 0 1 ${lo} ${f(hi - r)}V${f(hi - a)}`;
  }
}

const px = (v: number) => `${Math.round(v * 100) / 100}px`;
const at = (el: HTMLElement | SVGElement | null | undefined, x: number, y: number, extra = "") => {
  if (el) el.style.transform = `translate3d(${px(x)}, ${px(y)}, 0)${extra}`;
};
const size = (el: HTMLElement | null | undefined, w: number, h: number) => {
  if (!el) return;
  el.style.width = px(Math.max(0, w));
  el.style.height = px(Math.max(0, h));
};

export const SelectionView = forwardRef<SelectionHandle, Props>(function SelectionView({ box, W, H, candidate = false, label }, ref) {
  const veil = useRef<(HTMLDivElement | null)[]>([]);
  const caps = useRef<(HTMLDivElement | null)[]>([]);
  const hair = useRef<(HTMLDivElement | null)[]>([]);
  const brackets = useRef<(SVGSVGElement | null)[]>([]);
  const snap = useRef<HTMLDivElement>(null);
  const dims = useRef({ W, H });
  dims.current = { W, H };
  /** What was last written (a call with the same box on the same display writes nothing). */
  const last = useRef<{ key: string; arm: number }>({ key: "", arm: -1 });

  const place = (b: Box) => {
    const { W: dw, H: dh } = dims.current;
    const key = `${b.x},${b.y},${b.w},${b.h},${dw},${dh}`;
    if (key === last.current.key) return;
    // Whole points: pieces that meet on a fractional edge would leave a lighter seam between them.
    const x0 = Math.round(b.x), y0 = Math.round(b.y), x1 = Math.round(b.x + b.w), y1 = Math.round(b.y + b.h);
    const bw = x1 - x0, bh = y1 - y0;
    const v = veil.current;
    // Above, below (full width), then left and right of the box (its height only).
    at(v[0], 0, 0); size(v[0], dw, y0);
    at(v[1], 0, y1); size(v[1], dw, dh - y1);
    at(v[2], 0, y0); size(v[2], x0, bh);
    at(v[3], x1, y0); size(v[3], dw - x1, bh);
    // The hole's rounded corners (smaller on a box too small for 14 pt ones).
    const k = Math.min(1, bw / (2 * RADIUS), bh / (2 * RADIUS));
    const sc = k < 1 ? ` scale(${Math.round(k * 1000) / 1000})` : "";
    const c = caps.current;
    at(c[0], x0, y0, sc);
    at(c[1], x1 - RADIUS, y0, sc);
    at(c[2], x1 - RADIUS, y1 - RADIUS, sc);
    at(c[3], x0, y1 - RADIUS, sc);
    // The hairlines between the brackets' arms, just inside the box.
    const arm = armFor(b);
    const hl = hair.current;
    at(hl[0], x0 + arm, y0); size(hl[0], bw - 2 * arm, 1);
    at(hl[1], x1 - 1, y0 + arm); size(hl[1], 1, bh - 2 * arm);
    at(hl[2], x0 + arm, y1 - 1); size(hl[2], bw - 2 * arm, 1);
    at(hl[3], x0, y0 + arm); size(hl[3], 1, bh - 2 * arm);
    // The brackets: four corners, moved; their paths change only with the arm's length.
    const br = brackets.current;
    at(br[0], x0 - PAD, y0 - PAD);
    at(br[1], x1 - SIZE + PAD, y0 - PAD);
    at(br[2], x1 - SIZE + PAD, y1 - SIZE + PAD);
    at(br[3], x0 - PAD, y1 - SIZE + PAD);
    if (arm !== last.current.arm) {
      CORNERS.forEach((cn, i) => {
        const d = cornerPath(cn, arm);
        br[i]?.querySelectorAll("path").forEach((p) => p.setAttribute("d", d));
      });
    }
    // The snap scales about the box's centre.
    if (snap.current) snap.current.style.transformOrigin = `${px(x0 + bw / 2)} ${px(y0 + bh / 2)}`;
    last.current = { key, arm };
  };

  useImperativeHandle(ref, () => ({ place }), []);
  // The declarative position is the same call (a render after a drag ends shows the same box).
  useLayoutEffect(() => { place(box); }, [box.x, box.y, box.w, box.h, W, H]); // eslint-disable-line react-hooks/exhaustive-deps

  const kind = candidate ? " candidate" : "";
  return (
    <div className={"lx-selv" + kind} role="img" aria-label={label ?? `Selected area, ${Math.round(box.w)} by ${Math.round(box.h)} points`}>
      {[0, 1, 2, 3].map((i) => <div key={`v${i}`} className="lx-veil" ref={(n) => { veil.current[i] = n; }} />)}
      {CORNERS.map((c, i) => <div key={`c${c}`} className={`lx-cap ${c}`} ref={(n) => { caps.current[i] = n; }} />)}
      <div className={"lx-snap" + (candidate ? "" : " enter")} ref={snap}>
        {[0, 1, 2, 3].map((i) => <div key={`h${i}`} className="lx-hair" ref={(n) => { hair.current[i] = n; }} />)}
        {CORNERS.map((c, i) => (
          <svg key={`b${c}`} className="lx-bracket" width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} aria-hidden="true" ref={(n) => { brackets.current[i] = n; }}>
            <path className="under" />
            <path className="over" />
          </svg>
        ))}
      </div>
    </div>
  );
});

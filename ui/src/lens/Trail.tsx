// The glowing trail while the user circles or drags, and faint brackets round what it encloses so
// far (a rectangle-dragger sees a rectangle forming). Drawn straight into the SVG on each pointer
// move — no React render per point — and faded out on release.

import { forwardRef, useImperativeHandle, useRef } from "react";
import { bbox, smoothPath, type Box, type Pt } from "./geometry";

export interface TrailHandle {
  draw(pts: readonly Pt[]): void;
  fade(): void;
}

/** Four corner brackets round a box (arms `arm` long, rounded corners), as one SVG path. */
export function bracketPath(b: Box, arm = 22, r = 12): string {
  const a = Math.max(4, Math.min(arm, b.w / 2.5, b.h / 2.5));
  const rr = Math.max(0, Math.min(r, a - 1));
  const [x0, y0, x1, y1] = [b.x, b.y, b.x + b.w, b.y + b.h];
  const f = (v: number) => Math.round(v * 10) / 10;
  return [
    `M${f(x0)} ${f(y0 + a)}V${f(y0 + rr)}A${rr} ${rr} 0 0 1 ${f(x0 + rr)} ${f(y0)}H${f(x0 + a)}`,
    `M${f(x1 - a)} ${f(y0)}H${f(x1 - rr)}A${rr} ${rr} 0 0 1 ${f(x1)} ${f(y0 + rr)}V${f(y0 + a)}`,
    `M${f(x1)} ${f(y1 - a)}V${f(y1 - rr)}A${rr} ${rr} 0 0 1 ${f(x1 - rr)} ${f(y1)}H${f(x1 - a)}`,
    `M${f(x0 + a)} ${f(y1)}H${f(x0 + rr)}A${rr} ${rr} 0 0 1 ${f(x0)} ${f(y1 - rr)}V${f(y1 - a)}`,
  ].join("");
}

export const Trail = forwardRef<TrailHandle, { reduced: boolean }>(function Trail({ reduced }, ref) {
  const svg = useRef<SVGSVGElement>(null);
  const grad = useRef<SVGLinearGradientElement>(null);
  const paths = useRef<(SVGPathElement | null)[]>([]);
  const box = useRef<SVGPathElement>(null);
  const head = useRef<SVGCircleElement>(null);

  useImperativeHandle(ref, () => ({
    draw(pts) {
      const s = svg.current;
      if (!s || !pts.length) return;
      s.classList.remove("fading");
      const d = smoothPath(pts);
      for (const p of paths.current) p?.setAttribute("d", d);
      const a = pts[0], z = pts[pts.length - 1];
      const g = grad.current;
      if (g) {
        g.setAttribute("x1", String(a.x)); g.setAttribute("y1", String(a.y));
        g.setAttribute("x2", String(z.x === a.x && z.y === a.y ? z.x + 1 : z.x)); g.setAttribute("y2", String(z.y));
      }
      const b = bbox(pts);
      box.current?.setAttribute("d", b.w > 8 && b.h > 8 ? bracketPath(b) : "");
      if (head.current) { head.current.setAttribute("cx", String(z.x)); head.current.setAttribute("cy", String(z.y)); }
    },
    fade() {
      svg.current?.classList.add("fading");
      window.setTimeout(() => {
        if (!svg.current?.classList.contains("fading")) return;
        for (const p of paths.current) p?.setAttribute("d", "");
        box.current?.setAttribute("d", "");
        head.current?.setAttribute("cx", "-100");
      }, 220);
    },
  }), []);

  const stops = ["--lens-1", "--lens-2", "--lens-3", "--lens-4", "--lens-5"];
  return (
    <svg className="lx-trail" ref={svg} aria-hidden="true">
      <defs>
        <linearGradient id="lx-trail-g" ref={grad} gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="1" y2="0">
          {stops.map((v, i) => <stop key={v} offset={`${(i / (stops.length - 1)) * 100}%`} style={{ stopColor: `var(${v})` }} />)}
        </linearGradient>
        <radialGradient id="lx-head-g">
          <stop offset="0%" style={{ stopColor: "#fff", stopOpacity: 1 }} />
          <stop offset="100%" style={{ stopColor: "#fff", stopOpacity: 0 }} />
        </radialGradient>
      </defs>
      <path ref={box} className="live-box" style={{ fill: "none", stroke: "#fff", strokeWidth: 3, strokeLinecap: "round" }} />
      <path ref={(n) => { paths.current[0] = n; }} className="halo" stroke="url(#lx-trail-g)" />
      <path ref={(n) => { paths.current[1] = n; }} className="body" stroke="url(#lx-trail-g)" />
      <path ref={(n) => { paths.current[2] = n; }} className="core" />
      {!reduced && <circle ref={head} cx="-100" cy="-100" r="5" fill="url(#lx-head-g)" />}
    </svg>
  );
});

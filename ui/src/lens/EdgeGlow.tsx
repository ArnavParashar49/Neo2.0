// The glow along the screen's edges. Four soft bands and four crisp rims, each holding a long strip
// of NEO's five colours that slides along its edge; with the strips' offsets set from the screen's
// perimeter, the colours travel round the screen as one ring, clockwise. Only transforms and
// opacity animate (the compositor's work: nothing repaints per frame), and each phase's curve
// (./glow.ts) is sampled into keyframes from wherever the phase already is.

import { useLayoutEffect, useMemo, useRef } from "react";
import { BAND, OPACITY, PERIOD, breathes, glowAt, glowFrames, type GlowCtx, type GlowPhase, type Look } from "./glow.ts";

type Edge = "top" | "right" | "bottom" | "left";
const EDGES: Edge[] = ["top", "right", "bottom", "left"];
const COLORS = "var(--lens-1), var(--lens-2), var(--lens-3), var(--lens-4), var(--lens-5), var(--lens-1)";

interface Props {
  phase: GlowPhase;
  /** When the phase began (Date.now() clock). */
  phaseAt: number;
  toArmMs: number;
  via: string;
  reduced: boolean;
  selected: boolean;
  W: number;
  H: number;
}

/** Where each edge's strip sits and which way its colours run, so the four join up at the corners. */
function strips(W: number, H: number) {
  const P = 2 * (W + H);
  const L = Math.max(200, P / 2); // two turns of the colours round the screen
  const mod = (v: number) => ((v % L) + L) % L;
  return {
    L,
    top: { style: { left: -L, width: W + L, backgroundImage: `linear-gradient(to right, ${COLORS})`, backgroundSize: `${L}px 100%`, backgroundPosition: "0 0" }, axis: "X", dir: 1 },
    right: { style: { top: -L, height: H + L, backgroundImage: `linear-gradient(to bottom, ${COLORS})`, backgroundSize: `100% ${L}px`, backgroundPosition: `0 ${-mod(W)}px` }, axis: "Y", dir: 1 },
    bottom: { style: { left: 0, width: W + L, backgroundImage: `linear-gradient(to left, ${COLORS})`, backgroundSize: `${L}px 100%`, backgroundPosition: `${mod(2 * W + H)}px 0` }, axis: "X", dir: -1 },
    left: { style: { top: 0, height: H + L, backgroundImage: `linear-gradient(to top, ${COLORS})`, backgroundSize: `100% ${L}px`, backgroundPosition: `0 ${mod(P)}px` }, axis: "Y", dir: -1 },
  } as const;
}

const scaleFor = (edge: Edge, band: number) => {
  const k = Math.max(0.001, band / BAND.max);
  return edge === "top" || edge === "bottom" ? `scaleY(${k})` : `scaleX(${k})`;
};

export function EdgeGlow({ phase, phaseAt, toArmMs, via, reduced, selected, W, H }: Props) {
  const root = useRef<HTMLDivElement>(null);
  const bands = useRef<Record<Edge, HTMLDivElement | null>>({ top: null, right: null, bottom: null, left: null });
  const flowAnims = useRef<Animation[]>([]);
  const phaseAnims = useRef<Animation[]>([]);
  /** The colours' speed the phase last set (0: still — Reduce Motion, or no glow). */
  const rate = useRef(1);
  const last = useRef<{ phase: GlowPhase; at: number; ctx: GlowCtx } | null>(null);
  const geo = useMemo(() => strips(W || 1, H || 1), [W, H]);

  // The colours' travel: one infinite slide per strip; its speed follows the phase.
  useLayoutEffect(() => {
    flowAnims.current.forEach((a) => a.cancel());
    flowAnims.current = [];
    const els = [...(root.current?.querySelectorAll<HTMLDivElement>(".lx-flow") ?? [])];
    if (!W || !H || typeof els[0]?.animate !== "function") return;
    for (const el of els) {
      const edge = el.dataset.edge as Edge;
      const g = geo[edge];
      const to = `translate${g.axis}(${g.dir * geo.L}px)`;
      const a = el.animate([{ transform: `translate${g.axis}(0px)` }, { transform: to }], { duration: (PERIOD.open * 1000) / 2, iterations: Infinity, easing: "linear" });
      // Made again (the window moved to another display): at the phase's speed, still under Reduce Motion.
      if (rate.current === 0) a.pause();
      else a.playbackRate = rate.current;
      flowAnims.current.push(a);
    }
    return () => { flowAnims.current.forEach((a) => a.cancel()); flowAnims.current = []; };
  }, [geo, W, H]);

  // Each phase: the bands' width and the glow's brightness follow the curve from where it is now.
  useLayoutEffect(() => {
    const now = Date.now();
    const prev = last.current;
    const was = prev ? glowAt(prev.phase, now - prev.at, prev.ctx) : undefined;
    const carry = phase === "cancel" || phase === "closing" || (phase === "open" && prev?.phase === "armed");
    const ctx: GlowCtx = { toArmMs, reduced, via, ...(carry && was ? { from: was } : {}) };
    last.current = { phase, at: phaseAt, ctx };
    const { frames, ms } = glowFrames(phase, Math.max(0, now - phaseAt), ctx);
    phaseAnims.current.forEach((a) => a.cancel());
    phaseAnims.current = [];
    const end: Look = frames[frames.length - 1];
    const el = root.current;
    if (!el) return;
    const opts: KeyframeAnimationOptions = { duration: Math.max(1, ms), fill: "forwards", easing: "linear" };
    if (typeof el.animate === "function") {
      phaseAnims.current.push(el.animate(frames.map((f) => ({ opacity: f.opacity, offset: f.offset })), opts));
      for (const e of EDGES) {
        const b = bands.current[e];
        if (b) phaseAnims.current.push(b.animate(frames.map((f) => ({ transform: scaleFor(e, f.band), offset: f.offset })), opts));
      }
    } else {
      el.style.opacity = String(end.opacity);
    }
    // Still colours under Reduce Motion; otherwise as fast as the phase says.
    rate.current = end.period > 0 ? (PERIOD.open / end.period) : 0;
    for (const a of flowAnims.current) {
      if (rate.current === 0) a.pause();
      else {
        a.playbackRate = rate.current; // (keeps its place: the colours speed up, they don't jump)
        if (a.playState === "paused") a.play();
      }
    }
  }, [phase, phaseAt, toArmMs, reduced, via]);

  const strip = (edge: Edge) => <div className="lx-flow" data-edge={edge} style={geo[edge].style} />;
  return (
    <div className="lx-glow" ref={root} aria-hidden="true">
      <div className="lx-glow-dim" style={{ opacity: selected ? OPACITY.selected : 1 }}>
        <div className={"lx-glow-breathe" + (breathes(phase, reduced, selected) ? " on" : "")}>
          {EDGES.map((e) => (
            <div key={`b-${e}`} className={`lx-band ${e}`} ref={(n) => { bands.current[e] = n; }} style={{ transform: scaleFor(e, 0) }}>
              {strip(e)}
            </div>
          ))}
          {EDGES.map((e) => (
            <div key={`r-${e}`} className={`lx-rim ${e}`}>{strip(e)}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

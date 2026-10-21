// The glow along the screen's edges, as numbers: how wide its soft band is, how bright, and how
// fast its colours travel round the screen, at each moment of each phase. Glow.tsx samples these
// curves into the compositor's animations (nothing repaints per frame), so a phase that arrives
// late starts where the curve already is instead of lagging behind it.

export type GlowPhase = "idle" | "warm" | "preview" | "armed" | "open" | "cancel" | "closing";

export interface Look {
  /** The soft band's width, in points (the crisp 3 pt edge is drawn whenever it shows). */
  band: number;
  opacity: number;
  /** Seconds for the colours to travel once round the screen (0: still). */
  period: number;
}

export interface GlowCtx {
  /** Preview: how long until it arms, from the preview's start. */
  toArmMs: number;
  /** Reduce Motion: no travel, no breathing, no bloom, no spring; ≤ 120 ms fades. */
  reduced: boolean;
  /** What opened it: a hold has already built its glow; the menu or the voice blooms it. */
  via: string;
  /** Where the glow was when this phase began (cancel, closing and open carry on from it). */
  from?: Look;
}

export const BAND = { max: 56, preview: [8, 28] as const, armed: [28, 56] as const, open: 36 };
export const OPACITY = { preview: 0.65, armed: 1, selected: 0.55, breatheLow: 0.85 };
export const PERIOD = { building: 2.4, open: 8 };
/** Durations, in ms (design §4.3). */
export const DUR = { armUp: 140, armSettle: 420, bloom: 260, settle: 200, cancel: 140, closing: 160, reduced: 120, breathe: 3200 };

const NONE: Look = { band: 0, opacity: 0, period: 0 };
const clamp01 = (v: number) => Math.min(1, Math.max(0, Number.isFinite(v) ? v : 1));
export const easeIn = (t: number) => clamp01(t) ** 3;
export const easeOut = (t: number) => 1 - (1 - clamp01(t)) ** 3;
/** A spring settling on 1 with one small overshoot (0 → 1, passing ~1.08). */
export const spring = (t: number) => {
  const x = clamp01(t);
  return 1 - Math.exp(-6 * x) * Math.cos(9 * x) * (1 - x);
};
const mix = (a: number, b: number, t: number) => a + (b - a) * t;

/** How long a phase's curve runs before it holds still (ms). */
export function glowSpan(phase: GlowPhase, ctx: GlowCtx): number {
  if (ctx.reduced) return phase === "preview" || phase === "armed" || phase === "open" || phase === "cancel" || phase === "closing" ? DUR.reduced : 0;
  switch (phase) {
    case "preview": return Math.max(0, ctx.toArmMs);
    case "armed": return DUR.armUp + DUR.armSettle;
    case "open": return ctx.from ? DUR.settle : ctx.via === "hold" ? 0 : DUR.bloom;
    case "cancel": return DUR.cancel;
    case "closing": return DUR.closing;
    default: return 0;
  }
}

/** The glow `ms` into a phase. */
export function glowAt(phase: GlowPhase, ms: number, ctx: GlowCtx): Look {
  const t = Math.max(0, ms);
  const from = ctx.from;
  if (ctx.reduced) {
    const k = clamp01(t / DUR.reduced);
    const start = from?.opacity ?? 0;
    switch (phase) {
      case "preview": return { band: 28, opacity: mix(start, 0.4, k), period: 0 };
      case "armed":
      case "open": return { band: BAND.open, opacity: mix(start, 1, k), period: 0 };
      case "cancel":
      case "closing": return { band: from?.band ?? BAND.open, opacity: mix(start, 0, k), period: 0 };
      default: return NONE;
    }
  }
  switch (phase) {
    case "preview": {
      const p = ctx.toArmMs > 0 ? clamp01(t / ctx.toArmMs) : 1;
      // Faint for the first half: a slow tap that lets go there sees only a glimmer.
      return { band: mix(BAND.preview[0], BAND.preview[1], p), opacity: OPACITY.preview * easeIn(p), period: PERIOD.building };
    }
    case "armed": {
      if (t < DUR.armUp) return { band: mix(BAND.armed[0], BAND.armed[1], easeOut(t / DUR.armUp)), opacity: 1, period: PERIOD.building };
      const s = spring((t - DUR.armUp) / DUR.armSettle);
      return { band: mix(BAND.armed[1], BAND.open, s), opacity: 1, period: PERIOD.building };
    }
    case "open": {
      if (from) {
        const k = easeOut(t / DUR.settle);
        return { band: mix(from.band, BAND.open, k), opacity: mix(from.opacity, 1, k), period: PERIOD.open };
      }
      if (ctx.via === "hold") return { band: BAND.open, opacity: 1, period: PERIOD.open };
      const k = easeOut(t / DUR.bloom); // opened from the menu or the voice: it blooms
      return { band: BAND.open * k, opacity: k, period: PERIOD.open };
    }
    case "cancel": {
      const f = from ?? { band: BAND.preview[1], opacity: 0.3, period: PERIOD.building };
      return { band: f.band, opacity: f.opacity * (1 - easeIn(t / DUR.cancel)), period: f.period }; // retracts into the edges
    }
    case "closing": {
      const f = from ?? { band: BAND.open, opacity: 1, period: PERIOD.open };
      return { band: f.band, opacity: f.opacity * (1 - clamp01(t / DUR.closing)), period: f.period };
    }
    default:
      return NONE;
  }
}

/** The curve as keyframes (offsets 0..1), for the compositor: `n` steps over the rest of the span. */
export function glowFrames(phase: GlowPhase, startMs: number, ctx: GlowCtx, n = 12): { frames: (Look & { offset: number })[]; ms: number } {
  const span = glowSpan(phase, ctx);
  const left = Math.max(0, span - startMs);
  if (left <= 0) return { frames: [{ ...glowAt(phase, span, ctx), offset: 0 }, { ...glowAt(phase, span, ctx), offset: 1 }], ms: 0 };
  const frames = Array.from({ length: n + 1 }, (_, i) => ({ ...glowAt(phase, startMs + (left * i) / n, ctx), offset: i / n }));
  return { frames, ms: left };
}

/** Whether the glow breathes (open, nothing selected, motion allowed). */
export const breathes = (phase: GlowPhase, reduced: boolean, selected: boolean) => phase === "open" && !reduced && !selected;

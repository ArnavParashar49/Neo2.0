// The edge glow's curves (src/lens/glow.ts): nothing while warm, a faint-then-building glow while
// held (0 → 0.65), full and springy once armed, a calm breathing glow while open, retracting on a
// cancel; with Reduce Motion, still colours and short fades only.
import { test } from "node:test";
import assert from "node:assert/strict";
import { BAND, DUR, OPACITY, PERIOD, breathes, glowAt, glowFrames, glowSpan, spring, type GlowCtx } from "../src/lens/glow.ts";

const ctx: GlowCtx = { toArmMs: 550, reduced: false, via: "hold" };

test("warm shows nothing", () => {
  assert.deepEqual(glowAt("warm", 100, ctx), { band: 0, opacity: 0, period: 0 });
  assert.deepEqual(glowAt("idle", 100, ctx), { band: 0, opacity: 0, period: 0 });
});

test("the preview builds from nothing to 0.65 at the arm, monotonic, faint for its first half", () => {
  assert.equal(glowAt("preview", 0, ctx).opacity, 0);
  assert.equal(glowAt("preview", 550, ctx).opacity, OPACITY.preview);
  assert.equal(glowAt("preview", 0, ctx).band, BAND.preview[0]);
  assert.equal(glowAt("preview", 550, ctx).band, BAND.preview[1]);
  let last = -1;
  for (let ms = 0; ms <= 550; ms += 10) {
    const o = glowAt("preview", ms, ctx).opacity;
    assert.ok(o >= last, `${ms}`);
    last = o;
  }
  assert.ok(glowAt("preview", 275, ctx).opacity < 0.1, "a slow tap sees only a glimmer");
  assert.equal(glowAt("preview", 9999, ctx).opacity, OPACITY.preview);
  assert.equal(glowAt("preview", 100, ctx).period, PERIOD.building);
});

test("armed swells to 56 in 140 ms and springs back to 36", () => {
  assert.equal(glowAt("armed", 0, ctx).band, BAND.armed[0]);
  assert.ok(Math.abs(glowAt("armed", DUR.armUp - 0.001, ctx).band - BAND.armed[1]) < 0.01);
  assert.ok(Math.abs(glowAt("armed", DUR.armUp + DUR.armSettle, ctx).band - BAND.open) < 1e-9);
  const lows = [...Array(42).keys()].map((i) => glowAt("armed", DUR.armUp + i * 10, ctx).band);
  assert.ok(Math.min(...lows) < BAND.open, "it overshoots a little, like a spring");
  assert.equal(glowAt("armed", 50, ctx).opacity, 1);
  assert.equal(spring(0), 0);
  assert.equal(spring(1), 1);
});

test("open: steady after a hold, a bloom from the menu or the voice, slower colours", () => {
  assert.deepEqual(glowAt("open", 0, ctx), { band: BAND.open, opacity: 1, period: PERIOD.open });
  const menu = { ...ctx, via: "menu" };
  assert.deepEqual(glowAt("open", 0, menu), { band: 0, opacity: 0, period: PERIOD.open });
  assert.equal(glowAt("open", DUR.bloom, menu).band, BAND.open);
  assert.equal(glowSpan("open", menu), DUR.bloom);
  assert.equal(glowSpan("open", ctx), 0);
  assert.equal(glowSpan("open", { ...ctx, from: { band: 40, opacity: 1, period: 2.4 } }), DUR.settle);
  assert.ok(breathes("open", false, false));
  assert.ok(!breathes("open", false, true) && !breathes("open", true, false) && !breathes("armed", false, false));
});

test("a cancel retracts from wherever it was in 140 ms; closing fades in 160", () => {
  const from = { band: 20, opacity: 0.4, period: PERIOD.building };
  assert.equal(glowAt("cancel", 0, { ...ctx, from }).opacity, 0.4);
  assert.equal(glowAt("cancel", DUR.cancel, { ...ctx, from }).opacity, 0);
  assert.equal(glowSpan("cancel", ctx), 140);
  assert.equal(glowSpan("closing", ctx), 160);
  assert.equal(glowAt("closing", 160, { ...ctx, from: { band: 36, opacity: 1, period: 8 } }).opacity, 0);
});

test("Reduce Motion: still colours, no swell or bloom, fades of at most 120 ms", () => {
  const r = { ...ctx, reduced: true };
  for (const phase of ["preview", "armed", "open"] as const) {
    assert.equal(glowAt(phase, 1000, r).period, 0, phase);
    assert.ok(glowSpan(phase, r) <= 120, phase);
  }
  assert.equal(glowAt("armed", 30, r).band, BAND.open);
  assert.equal(glowAt("armed", 120, r).opacity, 1);
  assert.equal(glowAt("open", 0, { ...r, via: "menu" }).band, BAND.open); // no bloom
  assert.equal(glowAt("cancel", 120, { ...r, from: { band: 28, opacity: 0.4, period: 0 } }).opacity, 0);
});

test("the curve becomes keyframes from where a late phase already is", () => {
  const { frames, ms } = glowFrames("preview", 200, ctx, 10);
  assert.equal(ms, 350);
  assert.equal(frames.length, 11);
  assert.equal(frames[0].offset, 0);
  assert.equal(frames[10].offset, 1);
  assert.deepEqual({ ...frames[0], offset: undefined }, { ...glowAt("preview", 200, ctx), offset: undefined });
  assert.equal(frames[10].opacity, OPACITY.preview);
  const done = glowFrames("preview", 999, ctx);
  assert.equal(done.ms, 0);
  assert.equal(done.frames[1].opacity, OPACITY.preview);
});

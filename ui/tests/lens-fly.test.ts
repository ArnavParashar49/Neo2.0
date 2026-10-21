// The area's flight into the notch (src/lens/fly.ts): it heads for the notch's centre, shrinks to
// 36 pt, curves (sideways early, up late), lifts and fades where it is on a display without the
// island, fades in place under Reduce Motion, and only ever animates transform and opacity.
import { test } from "node:test";
import assert from "node:assert/strict";
import { EASE, FLY, FLY_RESET_MS, bezier, flyFrames, flyPlan, notchCentre } from "../src/lens/fly.ts";
import { box } from "../src/lens/geometry.ts";

const W = 1512, H = 982, NOTCH = 32;

test("the plan heads for the notch's centre", () => {
  const b = box(200, 500, 400, 300);
  const p = flyPlan(b, W, H, NOTCH, true, false);
  assert.equal(p.mode, "notch");
  assert.equal(p.ms, FLY.ms);
  assert.deepEqual(p.box, b);
  assert.equal(b.x + b.w / 2 + p.dx, W / 2);
  assert.equal(b.y + b.h / 2 + p.dy, NOTCH / 2);
  // No notch on this display: 12 pt down from the top, centred.
  assert.deepEqual(notchCentre(W, 0), { x: W / 2, y: 12 });
  const q = flyPlan(b, W, H, 0, true, false);
  assert.equal(b.y + b.h / 2 + q.dy, 12);
  assert.equal(FLY_RESET_MS, 460);
});

test("the scale fits 36 pt", () => {
  assert.equal(flyPlan(box(0, 0, 720, 300), W, H, NOTCH, true, false).scale, 36 / 720);
  assert.equal(flyPlan(box(0, 0, 200, 900), W, H, NOTCH, true, false).scale, 36 / 900);
  assert.equal(flyPlan(box(0, 0, W * 40, H), W * 40, H, NOTCH, true, false).scale, 0.02, "never below 0.02");
  assert.equal(flyPlan(box(10, 10, 20, 16), W, H, NOTCH, true, false).scale, 1, "never grows");
});

test("the arc: x eases out and y eases in, both monotonic", () => {
  const x = bezier(...EASE.x), y = bezier(...EASE.y), s = bezier(...EASE.s);
  let px = 0, py = 0, ps = 0;
  for (let i = 1; i < 100; i++) {
    const t = i / 100;
    const [vx, vy, vs] = [x(t), y(t), s(t)];
    assert.ok(vx >= px && vy >= py && vs >= ps, `monotonic at ${t}`);
    assert.ok(vx > vy, `sideways ahead of up at ${t}: ${vx} ${vy}`);
    [px, py, ps] = [vx, vy, vs];
  }
  assert.ok(x(0.25) > 0.6, "most of the sideways travel early");
  assert.ok(y(0.5) < 0.25, "little of the rise by half time");
  assert.equal(x(0), 0);
  assert.equal(y(1), 1);
  const f = flyFrames(flyPlan(box(200, 500, 400, 300), W, H, NOTCH, true, false));
  assert.equal(f.easing.x, "cubic-bezier(0.2, 0, 0, 1)");
  assert.equal(f.easing.y, "cubic-bezier(0.55, 0, 0.8, 0.2)");
  assert.equal(f.easing.s, "cubic-bezier(0.4, 0, 0.2, 1)");
  // Whole until 72% of the rise, gone as it reaches the notch (the fade keeps the rise's timing).
  assert.deepEqual(f.o.map((k) => [k.offset, k.opacity]), [[0, 1], [0.72, 1], [1, 0]]);
  assert.equal(f.easing.o, f.easing.y);
  assert.deepEqual(f.s.map((k) => k.transform), ["scale(1)", "scale(0.09)"]);
});

test("another display lifts and fades where it is; Reduce Motion fades in place", () => {
  const b = box(300, 30, 200, 100);
  const lift = flyPlan(b, W, H, NOTCH, false, false);
  assert.equal(lift.mode, "lift");
  assert.equal(lift.dx, 0);
  assert.equal(lift.dy, -48);
  assert.equal(flyPlan(box(300, 0, 200, 40), W, H, NOTCH, false, false).dy, -20, "never above the screen's top");
  assert.equal(lift.scale, 0.9);
  assert.equal(lift.ms, FLY.liftMs);
  const fade = flyPlan(b, W, H, NOTCH, true, true);
  assert.deepEqual({ mode: fade.mode, dx: fade.dx, dy: fade.dy, scale: fade.scale, ms: fade.ms }, { mode: "fade", dx: 0, dy: 0, scale: 1, ms: FLY.reducedMs });
  const ff = flyFrames(fade);
  assert.deepEqual(ff.x, []);
  assert.deepEqual(ff.y, []);
  assert.deepEqual(ff.s, []);
  assert.deepEqual(ff.o, [{ opacity: 1 }, { opacity: 0 }]);
  assert.ok(FLY.reducedMs <= 120);
});

test("only transform and opacity in the keyframes", () => {
  const plans = [
    flyPlan(box(200, 500, 400, 300), W, H, NOTCH, true, false),
    flyPlan(box(200, 500, 400, 300), W, H, NOTCH, false, false),
    flyPlan(box(200, 500, 400, 300), W, H, NOTCH, true, true),
  ];
  for (const p of plans) {
    const f = flyFrames(p);
    for (const k of [...f.x, ...f.y, ...f.s, ...f.o]) {
      for (const key of Object.keys(k)) assert.ok(["transform", "opacity", "offset"].includes(key), `${p.mode}: ${key}`);
    }
    for (const k of f.x) assert.match(String(k.transform), /^translate3d\(-?[\d.]+px, 0px, 0px\)$/, "x moves sideways only");
    for (const k of f.y) assert.match(String(k.transform), /^translate3d\(0px, -?[\d.]+px, 0px\)$/, "y moves up only");
  }
});

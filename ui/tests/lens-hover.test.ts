// Smart pick's hover (src/lens/hover.ts), with fake frames, timers and regions: any number of
// pointer moves in a frame are one frame's work; nothing shows until the pointer rests; the outline
// moves from region to region at once and never shows for a window, the default box or anything
// big. Nothing here can reach the overlay's reducer: the hover has no way to.
import { test } from "node:test";
import assert from "node:assert/strict";
import { HOVER, hoverable, makeHover, type HoverDeps } from "../src/lens/hover.ts";
import { box, type Box, type Pt } from "../src/lens/geometry.ts";
import type { Picked } from "../src/lens/pick.ts";

const W = 1512, H = 982;

function fakes(regionAt: (p: Pt) => Picked | null) {
  let id = 0;
  const frames = new Map<number, () => void>();
  const timers = new Map<number, { cb: () => void; at: number }>();
  let clock = 0;
  const writes: (Box | null)[] = [];
  let hits = 0;
  const deps: HoverDeps = {
    raf: (cb) => { frames.set(++id, cb); return id; },
    cancelRaf: (i) => { frames.delete(i); },
    setTimeout: (cb, ms) => { timers.set(++id, { cb, at: clock + ms }); return id; },
    clearTimeout: (i) => { timers.delete(i); },
    regionAt: (p) => { hits++; return regionAt(p); },
    size: () => ({ W, H }),
    write: (b) => { writes.push(b); },
  };
  const frame = () => { const fs = [...frames.values()]; frames.clear(); fs.forEach((f) => f()); };
  const advance = (ms: number) => {
    clock += ms;
    for (const [k, t] of [...timers]) if (t.at <= clock) { timers.delete(k); t.cb(); }
  };
  return { deps, frame, advance, writes, hits: () => hits, frames: () => frames.size, timers: () => timers.size };
}

const line: Picked = { box: box(100, 100, 300, 20), source: "line" };
const thing: Picked = { box: box(600, 300, 200, 160), source: "object" };
/** A line left of x = 500, a thing right of it. */
const two = (p: Pt) => (p.x < 500 ? line : thing);

test("a hundred moves in one frame make one hit test and at most one write, and never reach the reducer", () => {
  const f = fakes(two);
  const h = makeHover(f.deps);
  for (let i = 0; i < 100; i++) h.move({ x: 110 + i, y: 105 });
  assert.equal(f.frames(), 1, "one frame asked for");
  f.frame();
  assert.equal(f.hits(), 0, "moving, nothing shown yet: no hit test at all");
  f.advance(HOVER.restMs);
  assert.equal(f.hits(), 1);
  assert.deepEqual(f.writes, [line.box]);
  // Shown: a hundred more moves in a frame, still one hit test, and no write (the same line).
  for (let i = 0; i < 100; i++) h.move({ x: 120 + i, y: 106 });
  f.frame();
  assert.equal(f.hits(), 2);
  assert.equal(f.writes.length, 1);
  // The hover's whole world is HoverDeps: frames, timers, regions, a size and one write.
  assert.deepEqual(Object.keys(f.deps).sort(), ["cancelRaf", "clearTimeout", "raf", "regionAt", "setTimeout", "size", "write"]);
});

test("nothing shows until the pointer rests 150 ms", () => {
  const f = fakes(two);
  const h = makeHover(f.deps);
  for (let t = 0; t < 10; t++) {
    h.move({ x: 110 + t * 3, y: 105 });
    f.frame();
    f.advance(100); // moving again before it rested
  }
  assert.deepEqual(f.writes, []);
  assert.equal(f.hits(), 0);
  f.advance(HOVER.restMs - 100 - 1);
  assert.deepEqual(f.writes, []);
  f.advance(1);
  assert.deepEqual(f.writes, [line.box]);
});

test("moving inside the same region writes nothing", () => {
  const f = fakes(two);
  const h = makeHover(f.deps);
  h.move({ x: 110, y: 105 }); f.frame(); f.advance(HOVER.restMs);
  assert.equal(f.writes.length, 1);
  for (let x = 120; x < 400; x += 7) { h.move({ x, y: 108 }); f.frame(); f.advance(16); }
  assert.equal(f.writes.length, 1);
});

test("a new region while shown moves the outline at once (one write, no hide between)", () => {
  const f = fakes(two);
  const h = makeHover(f.deps);
  h.move({ x: 110, y: 105 }); f.frame(); f.advance(HOVER.restMs);
  h.move({ x: 650, y: 350 }); f.frame();
  assert.deepEqual(f.writes, [line.box, thing.box], "no null between the two");
  // Off every region: it goes (and the next one needs a rest again).
  const g = fakes((p) => (p.x < 500 ? line : null));
  const k = makeHover(g.deps);
  k.move({ x: 110, y: 105 }); g.frame(); g.advance(HOVER.restMs);
  k.move({ x: 900, y: 900 }); g.frame();
  assert.deepEqual(g.writes, [line.box, null]);
  k.move({ x: 120, y: 105 }); g.frame();
  assert.equal(g.writes.length, 2, "not straight back: a rest first");
  g.advance(HOVER.restMs);
  assert.deepEqual(g.writes, [line.box, null, line.box]);
});

test("windows, the default box and regions over 35% of the display never show", () => {
  const big = box(0, 0, W * 0.7, H * 0.6); // 42% of the display
  assert.equal(hoverable({ box: box(0, 0, W, H - 40), source: "window" }, W, H), null);
  assert.equal(hoverable({ box: box(10, 10, 280, 200), source: "window" }, W, H), null, "a small window too");
  assert.equal(hoverable({ box: box(10, 10, 280, 200), source: "default" }, W, H), null);
  assert.equal(hoverable({ box: big, source: "object" }, W, H), null);
  assert.equal(hoverable({ box: big, source: "instance" }, W, H), null);
  assert.equal(hoverable({ box: box(0, 0, 900, 30), source: "block" }, W, H)?.w, 900);
  assert.equal(hoverable(null, W, H), null);
  assert.equal(hoverable(line, 0, 0), null);
  const ok = box(0, 0, Math.sqrt(HOVER.maxArea * W * H), Math.sqrt(HOVER.maxArea * W * H));
  assert.deepEqual(hoverable({ box: ok, source: "object" }, W, H), ok, "35% exactly still shows");
  // Through the hover: a resting pointer over a window shows nothing, ever.
  const f = fakes(() => ({ box: box(0, 25, W, H - 25), source: "window" }));
  const h = makeHover(f.deps);
  h.move({ x: 700, y: 400 }); f.frame(); f.advance(1000);
  h.move({ x: 710, y: 400 }); f.frame(); f.advance(1000);
  assert.deepEqual(f.writes, []);
});

test("hide() cancels a pending rest and writes null once", () => {
  const f = fakes(two);
  const h = makeHover(f.deps);
  h.move({ x: 110, y: 105 }); f.frame(); f.advance(HOVER.restMs);
  h.move({ x: 650, y: 350 }); // a frame is pending
  h.hide();
  assert.deepEqual(f.writes, [line.box, null]);
  f.frame(); f.advance(1000);
  h.hide();
  assert.deepEqual(f.writes, [line.box, null], "nothing more: no rest fired, a second hide wrote nothing");
  // A rest pending, nothing shown: it never fires.
  const g = fakes(two);
  const k = makeHover(g.deps);
  k.move({ x: 110, y: 105 }); g.frame();
  assert.equal(g.timers(), 1);
  k.hide();
  g.advance(1000);
  assert.deepEqual(g.writes, []);
  assert.equal(g.hits(), 0);
  // Disposed: nothing at all.
  k.move({ x: 110, y: 105 }); g.frame(); g.advance(1000);
  assert.equal(g.writes.length, 1, "(moving again after hide works)");
  k.dispose();
  k.move({ x: 650, y: 350 }); g.frame(); g.advance(1000);
  assert.equal(g.writes.length, 1);
});

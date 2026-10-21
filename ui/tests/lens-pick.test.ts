// Smart pick (src/lens/pick.ts): a click takes the line under it, else the thing, else the window,
// else a box around the point; a double-click on a line takes its paragraph; Tab walks things,
// then paragraphs, then windows, in reading order.
import { test } from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_BOX, LINE_PAD, around, pick, readingOrder, regionAt, tabStops, windowAt } from "../src/lens/pick.ts";
import { box } from "../src/lens/geometry.ts";
import type { Regions } from "../src/lens/wire.ts";

const W = 1512, H = 982;
const R: Regions = {
  lines: [{ box: [100, 100, 300, 20], block: 0 }, { box: [100, 124, 280, 20], block: 0 }, { box: [700, 600, 200, 18], block: 1 }],
  blocks: [[100, 100, 300, 44], [700, 600, 200, 18]],
  instances: [[500, 300, 200, 200], [0, 0, 1500, 980]], // a shoe; the whole background (too big to be a thing)
  objects: [[480, 280, 260, 260], [1000, 100, 200, 150]],
  windows: [[50, 50, 900, 700], [0, 0, W, H], [1100, 800, 60, 40]],
  texts: [],
};

test("a line under the pointer wins, padded", () => {
  const p = pick(R, { x: 150, y: 110 }, W, H);
  assert.equal(p.source, "line");
  assert.deepEqual(p.box, box(100 - LINE_PAD, 100 - LINE_PAD, 300 + 2 * LINE_PAD, 20 + 2 * LINE_PAD));
});

test("a double-click on a line takes its paragraph", () => {
  const p = pick(R, { x: 150, y: 130 }, W, H, true);
  assert.equal(p.source, "block");
  assert.deepEqual(p.box, box(96, 96, 308, 52));
});

test("then the smallest thing, an instance before a salient object", () => {
  assert.equal(pick(R, { x: 600, y: 400 }, W, H).source, "instance");
  assert.deepEqual(pick(R, { x: 600, y: 400 }, W, H).box, box(500, 300, 200, 200));
  assert.equal(pick(R, { x: 490, y: 290 }, W, H).source, "object"); // the object's edge, outside the shoe
  assert.equal(pick(R, { x: 1100, y: 150 }, W, H).source, "object");
});

test("then the front window big enough to mean something, then a box around the point", () => {
  const p = pick(R, { x: 200, y: 600 }, W, H);
  assert.equal(p.source, "window");
  assert.deepEqual(p.box, box(50, 50, 900, 700));
  // Only the whole-screen window and a tiny one are left there: the whole screen (front first).
  assert.equal(windowAt(R, { x: 1120, y: 810 })?.w, W);
  const none: Regions = { lines: [], blocks: [], instances: [], objects: [], windows: [], texts: [] };
  const d = pick(none, { x: 5, y: 5 }, W, H);
  assert.equal(d.source, "default");
  assert.deepEqual(d.box, box(0, 0, DEFAULT_BOX.w, DEFAULT_BOX.h));
  assert.deepEqual(around({ x: W, y: H }, W, H), box(W - 280, H - 200, 280, 200));
});

test("without regions yet, the click still picks something", () => {
  assert.equal(regionAt(null, { x: 1, y: 1 }, W, H), null);
  assert.equal(pick(null, { x: 400, y: 400 }, W, H).source, "default");
});

test("Tab's stops are things, then paragraphs, then windows, each in reading order", () => {
  const stops = tabStops(R, W, H);
  assert.deepEqual(stops.slice(0, 3), [box(1000, 100, 200, 150), box(480, 280, 260, 260), box(500, 300, 200, 200)]);
  assert.deepEqual(stops[3], box(100, 100, 300, 44));
  assert.deepEqual(stops[4], box(700, 600, 200, 18));
  assert.deepEqual(stops.slice(5), [box(0, 0, W, H), box(50, 50, 900, 700)]);
  assert.deepEqual(tabStops(null, W, H), []);
  assert.deepEqual(readingOrder([box(300, 10, 5, 5), box(10, 15, 5, 5), box(0, 100, 5, 5)]), [box(10, 15, 5, 5), box(300, 10, 5, 5), box(0, 100, 5, 5)]);
});

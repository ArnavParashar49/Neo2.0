// Search Screen's one gesture (src/lens/geometry.ts): a straight drag is the marquee, a circle or a
// scribble gives its box padded, a click (or a tiny box) is smart pick, and the box's corners,
// edges and inside adjust it — never off the display, never smaller than 16 pt.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  CLICK_MOVE, LASSO_PAD, MIN_SIDE, atLeast, bbox, box, clampBox, classify, dragBox, hitTest, iou, nudge, pathLength,
  simplify, smoothPath, type Pt,
} from "../src/lens/geometry.ts";

const W = 1512, H = 982;
const line = (a: Pt, b: Pt, n = 20): Pt[] => Array.from({ length: n + 1 }, (_, i) => ({ x: a.x + ((b.x - a.x) * i) / n, y: a.y + ((b.y - a.y) * i) / n }));
const circle = (cx: number, cy: number, r: number, n = 64): Pt[] =>
  Array.from({ length: n + 1 }, (_, i) => ({ x: cx + r * Math.cos((2 * Math.PI * i) / n), y: cy + r * Math.sin((2 * Math.PI * i) / n) }));

test("a straight drag is exactly the marquee", () => {
  const g = classify(line({ x: 100, y: 120 }, { x: 400, y: 300 }), W, H);
  assert.deepEqual(g, { kind: "box", box: box(100, 120, 300, 180), how: "drag" });
  // Dragged up and to the left: the same box.
  const back = classify(line({ x: 400, y: 300 }, { x: 100, y: 120 }), W, H);
  assert.deepEqual(back, { kind: "box", box: box(100, 120, 300, 180), how: "drag" });
});

test("a circle or a scribble gives its box, padded", () => {
  const g = classify(circle(500, 400, 80), W, H);
  assert.equal(g.kind, "box");
  if (g.kind !== "box") return;
  assert.equal(g.how, "lasso");
  assert.ok(Math.abs(g.box.x - (420 - LASSO_PAD)) < 0.5 && Math.abs(g.box.w - (160 + 2 * LASSO_PAD)) < 0.5, JSON.stringify(g.box));
  // A zigzag over an area is a lasso too.
  const zig: Pt[] = [];
  for (let i = 0; i < 12; i++) zig.push({ x: 200 + (i % 2) * 150, y: 200 + i * 10 });
  assert.equal((classify(zig, W, H) as { how: string }).how, "lasso");
});

test("a click, a still press and a tiny box are smart pick", () => {
  assert.deepEqual(classify([{ x: 10, y: 10 }], W, H), { kind: "click", at: { x: 10, y: 10 } });
  assert.deepEqual(classify(line({ x: 50, y: 50 }, { x: 50 + CLICK_MOVE - 1, y: 50 }), W, H), { kind: "click", at: { x: 50, y: 50 } });
  const tiny = classify(line({ x: 100, y: 100 }, { x: 130, y: 110 }), W, H); // 30 × 10: too thin
  assert.deepEqual(tiny, { kind: "click", at: { x: 115, y: 105 } });
});

test("a box is clamped to the display", () => {
  assert.deepEqual(clampBox(box(-50, -20, 200, 100), W, H), box(0, 0, 150, 80));
  assert.deepEqual(clampBox(box(1400, 900, 300, 300), W, H), box(1400, 900, 112, 82));
  const g = classify(line({ x: -100, y: 500 }, { x: 300, y: 1200 }), W, H);
  assert.deepEqual(g, { kind: "box", box: box(0, 500, 300, 482), how: "drag" });
  assert.deepEqual(atLeast(box(10, 10, 4, 4), W, H), box(10, 10, MIN_SIDE, MIN_SIDE));
  assert.deepEqual(atLeast(box(W - 2, H - 2, 2, 2), W, H), box(W - MIN_SIDE, H - MIN_SIDE, MIN_SIDE, MIN_SIDE));
});

test("corners, edges and the inside are where they should be", () => {
  const b = box(100, 100, 300, 200);
  assert.equal(hitTest(b, { x: 100, y: 100 }), "nw");
  assert.equal(hitTest(b, { x: 112, y: 88 }), "nw");
  assert.equal(hitTest(b, { x: 400, y: 300 }), "se");
  assert.equal(hitTest(b, { x: 395, y: 105 }), "ne");
  assert.equal(hitTest(b, { x: 105, y: 297 }), "sw");
  assert.equal(hitTest(b, { x: 250, y: 92 }), "n");
  assert.equal(hitTest(b, { x: 250, y: 308 }), "s");
  assert.equal(hitTest(b, { x: 92, y: 200 }), "w");
  assert.equal(hitTest(b, { x: 408, y: 200 }), "e");
  assert.equal(hitTest(b, { x: 250, y: 200 }), "inside");
  assert.equal(hitTest(b, { x: 600, y: 600 }), "outside");
});

test("adjusting moves, resizes, flips and stays on the display", () => {
  const b = box(100, 100, 300, 200);
  assert.deepEqual(dragBox(b, "inside", 50, -20, W, H), box(150, 80, 300, 200));
  assert.deepEqual(dragBox(b, "inside", -500, -500, W, H), box(0, 0, 300, 200));
  assert.deepEqual(dragBox(b, "se", 20, 30, W, H), box(100, 100, 320, 230));
  assert.deepEqual(dragBox(b, "n", 0, -40, W, H), box(100, 60, 300, 240));
  assert.deepEqual(dragBox(b, "w", 400, 0, W, H), box(400, 100, 100, 200)); // past the right edge: flipped
  assert.deepEqual(dragBox(b, "e", -299, 0, W, H).w, MIN_SIDE);
  assert.deepEqual(dragBox(b, "outside", 10, 10, W, H), b);
});

test("arrow keys nudge by a point, shift by ten, option resizes", () => {
  const b = box(100, 100, 300, 200);
  assert.deepEqual(nudge(b, 1, 0, false, W, H), box(101, 100, 300, 200));
  assert.deepEqual(nudge(b, 0, -10, false, W, H), box(100, 90, 300, 200));
  assert.deepEqual(nudge(b, 10, 1, true, W, H), box(100, 100, 310, 201));
  assert.deepEqual(nudge(b, -400, 0, true, W, H).w, MIN_SIDE);
});

test("IoU says how much an adjusted box still is the old one", () => {
  const a = box(0, 0, 100, 100);
  assert.equal(iou(a, a), 1);
  assert.equal(iou(a, box(200, 200, 10, 10)), 0);
  assert.ok(Math.abs(iou(a, box(50, 0, 100, 100)) - 1 / 3) < 1e-9);
  assert.ok(iou(a, box(1, 1, 100, 100)) > 0.92);
  assert.ok(iou(a, box(10, 0, 100, 100)) < 0.92);
});

test("the trail is smoothed and thinned", () => {
  const pts = line({ x: 0, y: 0 }, { x: 10, y: 0 }, 100);
  const s = simplify(pts);
  assert.ok(s.length <= 7 && s.length >= 5, String(s.length));
  assert.deepEqual(s[s.length - 1], { x: 10, y: 0 });
  assert.equal(smoothPath([]), "");
  assert.equal(smoothPath([{ x: 1, y: 2 }]), "M1 2");
  const d = smoothPath(circle(100, 100, 50, 16));
  assert.ok(d.startsWith("M150 100") && (d.match(/C/g) ?? []).length === 16, d);
  assert.equal(pathLength(line({ x: 0, y: 0 }, { x: 3, y: 4 })), 5);
  assert.deepEqual(bbox([]), box(0, 0, 0, 0));
});

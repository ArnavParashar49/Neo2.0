// Where the results sheet goes (src/lens/place.ts): right of the area, else left, below, above,
// else floating inside a near full-screen one; below the notch; never over the area when there is
// room; growing upward at the bottom instead of jumping sides.
import { test } from "node:test";
import assert from "node:assert/strict";
import { GAP, MARGIN, overlaps, placeSheet, sheetWidth } from "../src/lens/place.ts";
import { box } from "../src/lens/geometry.ts";

const W = 1512, H = 982;
const sheetBox = (p: { x: number; y: number; w: number }, h: number) => box(p.x, p.y, p.w, h);

test("right of the area first, top aligned", () => {
  const sel = box(200, 300, 300, 200);
  const p = placeSheet(sel, 400, W, H);
  assert.equal(p.side, "right");
  assert.deepEqual([p.x, p.y, p.w], [200 + 300 + GAP, 300, 372]);
  assert.equal(p.maxH, Math.round(H * 0.7));
});

test("left when the right has no room, then below, then above", () => {
  assert.equal(placeSheet(box(1100, 300, 300, 200), 300, W, H).side, "left");
  const wide = box(100, 100, 1300, 300);
  const below = placeSheet(wide, 300, W, H);
  assert.equal(below.side, "below");
  assert.equal(below.y, 100 + 300 + GAP);
  assert.equal(below.x, 100 + 650 - 186);
  assert.equal(placeSheet(box(100, 600, 1300, 300), 300, W, H).side, "above");
});

test("a near full-screen area gets the sheet floating inside it", () => {
  const p = placeSheet(box(10, 10, W - 20, H - 20), 300, W, H);
  assert.equal(p.side, "inside");
  assert.ok(p.x + p.w <= W - MARGIN && p.y + 300 <= H - MARGIN);
});

test("it never covers the area when there is room, and keeps below the notch", () => {
  for (const sel of [box(200, 300, 300, 200), box(1100, 300, 300, 200), box(100, 100, 1300, 300), box(100, 600, 1300, 300), box(40, 0, 1400, 60)]) {
    for (const h of [120, 300, 600]) {
      const p = placeSheet(sel, h, W, H, 38);
      if (p.side !== "inside") assert.ok(!overlaps(sheetBox(p, Math.min(h, p.maxH)), sel), `${JSON.stringify(sel)} h=${h} ${p.side}`);
      assert.ok(p.y >= 38 + MARGIN - 0.001 || p.side === "inside", `${p.y}`);
    }
  }
  assert.equal(placeSheet(box(200, 0, 200, 40), 200, W, H, 38).y, 38 + MARGIN);
});

test("as it grows at the bottom it moves up, on the same side", () => {
  const sel = box(200, 800, 300, 150);
  const small = placeSheet(sel, 100, W, H);
  assert.equal(small.side, "right");
  assert.equal(small.y, 800);
  const tall = placeSheet(sel, 500, W, H, 0, small.side);
  assert.equal(tall.side, "right");
  assert.equal(tall.y, H - MARGIN - 500);
});

test("a narrow display gets the narrow sheet", () => {
  assert.equal(sheetWidth(1512), 372);
  assert.equal(sheetWidth(900), 320);
  assert.equal(placeSheet(box(10, 10, 100, 100), 100, 900, 700).w, 320);
});

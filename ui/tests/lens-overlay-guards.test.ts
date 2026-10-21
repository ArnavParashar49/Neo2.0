// Source guards for Search Screen's overlay (there is no DOM here to watch it render): the pointer
// moving never reaches React state, the still is a memo of three props on its own layer, hover and
// the dragging of a corner are no longer state, and lens.css paints nothing display-sized when
// something moves — no 200vmax shadow, no filter, no filled hover, no animated background, no
// sheet that moves by left/top.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const read = (rel: string) => readFileSync(new URL(`../src/lens/${rel}`, import.meta.url), "utf8");
const app = read("LensApp.tsx");
const state = read("state.ts");
const css = read("lens.css");
/** lens.css without its comments, as rules. */
const rules = css.replace(/\/\*[\s\S]*?\*\//g, "");
/** The declarations of every rule whose selector list names `sel` (a class, e.g. ".lx-still"). */
const blocks = (sel: string) => {
  const out: string[] = [];
  const re = /([^{}]+)\{([^{}]*)\}/g;
  for (let m = re.exec(rules); m; m = re.exec(rules)) {
    const sels = m[1].split(",").map((x) => x.trim());
    if (sels.some((x) => new RegExp(`${sel.replace(".", "\\.")}(?![\\w-])`).test(x))) out.push(m[2]);
  }
  return out;
};

test("the pointer moving never touches React state", () => {
  const body = app.split("const onMove")[1]?.split("const onUp")[0] ?? "";
  assert.ok(body.length > 200, "onMove found");
  for (const call of ["dispatch(", "setS(", "setQ(", "setState("]) assert.ok(!body.includes(call), call);
  // Its work runs in an animation frame (the cursor, the trail, a dragged corner) and in the hover.
  assert.ok(body.includes("nextFrame()") && body.includes("hover.current?.move(p)"));
  const frame = app.split("const pointerFrame")[1]?.split("const onDown")[0] ?? "";
  assert.ok(frame.includes("trail.current?.draw(") && frame.includes("selView.current?.place(") && frame.includes("setCursor("));
  for (const call of ["dispatch(", "setS(", "setQ("]) assert.ok(!frame.includes(call), call);
});

test("the still is a memo of exactly src, shown and leaving", () => {
  const m = /const Still = memo\(function Still\(\{([^}]*)\}: \{([^}]*)\}\)/.exec(app);
  assert.ok(m, "declared with memo(");
  const props = m![2].split(";").map((x) => x.trim().split(":")[0]?.trim()).filter(Boolean).sort();
  assert.deepEqual(props, ["leaving", "shown", "src"]);
  assert.ok(app.includes("<Still src={s.still}"), "and rendered as one");
  assert.ok(!/<img[^>]*lx-still/.test(app.split("const Still")[1]?.split("});")[1] ?? ""), "no other still");
});

test("hover and a corner being dragged are not state", () => {
  assert.ok(!/["']hover["']/.test(state), "no hover action");
  assert.ok(!/\bhover\s*:/.test(state), "no hover field");
  assert.ok(!/["']moving["']/.test(state), "no moving action");
  assert.ok(!/["'](hover|moving)["']/.test(app), "LensApp dispatches neither");
});

test("lens.css: nothing display-sized repaints when something moves", () => {
  assert.ok(!rules.includes("200vmax"));
  // No filter anywhere (a backdrop filter is the glass of the sheet, the pill and the cards).
  assert.ok(!/(^|[^-\w])filter\s*:/.test(rules), "no filter:");
  assert.ok(!/drop-shadow/.test(rules));
  const hover = blocks(".lx-hover");
  assert.ok(hover.length > 0);
  for (const b of hover) assert.ok(!/(^|[;\s])background(-color|-image)?\s*:/.test(b), "the hover has no background");
  assert.ok(blocks(".lx-still").some((b) => b.includes("will-change")), "the still has its own layer");
  assert.ok(blocks(".lx-scrim").some((b) => b.includes("will-change")));
  // No keyframes move a background.
  const frames = rules.match(/@keyframes[^{]+\{([^{}]*\{[^{}]*\})*[^{}]*\}/g) ?? [];
  assert.ok(frames.length >= 4);
  for (const k of frames) assert.ok(!k.includes("background-position"), k.slice(0, 40));
  // The sheet moves by transform, never by left/top.
  for (const b of blocks(".lx-sheet")) assert.ok(!/transition[^;]*\b(left|top)\b/.test(b), b);
  assert.ok(blocks(".lx-sheet-at").some((b) => /transition:\s*transform/.test(b)));
  // The trail is not displayed between drags.
  assert.ok(blocks(".lx-trail").some((b) => /display:\s*none/.test(b)));
});

test("an area on its way to the notch shows no sheet, takes no adjusting, and flies on transforms alone", () => {
  const sheet = app.split("<Sheet ")[0].split("\n").slice(-2).join("\n");
  assert.ok(sheet.includes("!handingOff(s.sel)") && sheet.includes('s.sel.handoff !== "ok"'), sheet);
  const down = app.split("const onDown")[1]?.split("const onMove")[0] ?? "";
  assert.ok(down.indexOf("isDouble(") < down.indexOf("handingOff(sel)"), "a double-click still makes a picked line its paragraph");
  assert.ok(app.includes("<Still src={s.still} shown={showStill} leaving={!!s.fly} />"), "the still fades as the area flies");
  for (const b of blocks(".lx-fly").concat(blocks(".lx-fly-y"), blocks(".lx-fly-s"))) assert.ok(!/transition|animation|filter/.test(b), b);
  assert.ok(blocks(".lx-still").some((b) => /transition:\s*opacity 200ms/.test(b)), "the still's 200 ms fade");
});

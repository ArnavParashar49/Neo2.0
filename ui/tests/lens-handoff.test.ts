// The overlay handing its area to the notch (src/lens/state.ts), with fake shell phases and core
// messages: only when the core's hello says NotchHub could show it; a drag at once, a pick once a
// double-click can't make it a paragraph, live mode once captured; no sheet and no adjusting while
// it waits; yes → it closes with "notch" and flies; no (or no answer) → the sheet, as before.
import { test } from "node:test";
import assert from "node:assert/strict";
import { HANDOFF_GIVE_UP_MS, HANDOFF_PICK_MS, FADE_MS, NOTCH_CHECK_MS, handingOff, initial, reduce, type Action, type Effect, type LensState } from "../src/lens/state.ts";
import { FLY_RESET_MS, FLY_RESET_REDUCED_MS } from "../src/lens/fly.ts";
import { DOUBLE_MS } from "../src/lens/pick.ts";
import { box } from "../src/lens/geometry.ts";
import { parseMsg, send, type CoreMsg, type LensHello } from "../src/lens/wire.ts";

const T = "ab".repeat(32);
const W = 1512, H = 982;

function run(s: LensState, ...actions: Action[]): [LensState, Effect[]] {
  const all: Effect[] = [];
  for (const a of actions) {
    const [n, fx] = reduce(s, a);
    s = n;
    all.push(...fx);
  }
  return [s, all];
}
const phase = (g: number, p: string, extra: Record<string, unknown> = {}): Action => ({ t: "phase", ev: { g, phase: p, ...extra }, now: 1000 });
const shell = (g: number, p: string): Action => ({ t: "shell", hello: { g, phase: p, ticket: T, core: true, sc: true, since_ms: 0 }, now: 1000 });
const msg = (m: CoreMsg): Action => ({ t: "msg", msg: m });
const hello = (over: Partial<LensHello> = {}): LensHello => ({
  v: 1, open: true, frozen: "pending", display: { w: W, h: H, scale: 2 }, via: "hold", perms: { screen: true },
  eyes: { label: "Gemini", local: false }, eyesWhy: "", ask: true, search: null, langs: [], prefill: null, notch: true, ...over,
});
const sends = (fx: Effect[]) => fx.filter((e): e is Extract<Effect, { t: "send" | "send_live" }> => e.t === "send" || e.t === "send_live").map((e) => e.cmd);
const handoffs = (fx: Effect[]) => sends(fx).filter((c) => c.cmd === "lens_handoff");
const later = (fx: Effect[], t: string) => fx.filter((e): e is Extract<Effect, { t: "later" }> => e.t === "later" && e.action.t === t);
const reply = (sel: number, ok: boolean, extra: Record<string, unknown> = {}): Action => {
  const m = parseMsg(JSON.stringify({ type: "lens_handoff", data: { sel, ok, ...extra } }));
  assert.ok(m);
  return msg(m!);
};

/** Open, connected, the still in (or not: live), on display 7 with a 32 pt notch. */
function opened(over: Partial<LensHello> = {}, still = true): LensState {
  const frozen: CoreMsg = still ? { t: "lens_frozen", data: { ok: true, ms: 100 } } : { t: "lens_frozen", data: { ok: false, why: "no_sck" } };
  const [s] = run(initial(W, H, "en"), phase(1, "armed", { display: { id: 7, x: 0, y: 0, w: W, h: H, scale: 2 }, safe_top: 32 }), shell(1, "armed"),
    { t: "socket", status: "open" }, msg({ t: "lens_hello", data: hello(over) }), msg(frozen), phase(1, "open"));
  return s;
}

test("the wire: the hello's notch, the reply's display and why, the command's language", () => {
  const h = parseMsg(JSON.stringify({ type: "lens_hello", data: { v: 1, notch: true } }));
  assert.equal(h?.t === "lens_hello" && h.data.notch, true);
  const old = parseMsg(JSON.stringify({ type: "lens_hello", data: { v: 1 } }));
  assert.equal(old?.t === "lens_hello" && old.data.notch, false, "an older core: no hand-off");
  assert.deepEqual(parseMsg(JSON.stringify({ type: "lens_handoff", data: { sel: 3, ok: true, display: 69734208 } }))?.data, { sel: 3, ok: true, display: 69734208, why: "" });
  assert.deepEqual(parseMsg(JSON.stringify({ type: "lens_handoff", data: { sel: 3, ok: false, why: "unshown", display: -1 } }))?.data, { sel: 3, ok: false, display: null, why: "unshown" });
  assert.equal((parseMsg(JSON.stringify({ type: "lens_handoff", data: { sel: 3, ok: false, why: "<b>", display: 2 ** 40 } }))?.data as { why: string }).why, "");
  assert.equal(parseMsg(JSON.stringify({ type: "lens_handoff", data: { ok: true } })), null, "no area, no reply");
  assert.deepEqual(send.handoff(4, "en-GB"), { cmd: "lens_handoff", sel: 4, to: "en-GB" });
  assert.deepEqual(send.handoff(4, null), { cmd: "lens_handoff", sel: 4, to: "" });
  assert.deepEqual(send.handoff(4, "not a language!"), { cmd: "lens_handoff", sel: 4, to: "" });
});

test("notch: false never sends lens_handoff — the sheet, exactly as before", () => {
  let [s, fx] = run(opened({ notch: false }), { t: "select", box: box(100, 100, 300, 200), how: "drag" });
  assert.deepEqual(handoffs(fx), []);
  assert.equal(s.sel?.handoff, "none");
  assert.equal(handingOff(s.sel), false);
  [s, fx] = run(s, { t: "click", at: { x: 700, y: 500 }, double: false }, { t: "pick_now", g: 1 }, { t: "handoff_due", n: 2 });
  assert.deepEqual(handoffs(fx), []);
  assert.deepEqual(later(fx, "handoff_due"), []);
});

test("a drag hands off right after lens_select, and its crop is cut then", () => {
  const [s, fx] = run(opened(), { t: "select", box: box(100, 100, 300, 200), how: "drag" });
  const cmds = sends(fx).map((c) => c.cmd);
  assert.deepEqual(cmds, ["lens_select", "lens_handoff"]);
  assert.deepEqual(handoffs(fx), [{ cmd: "lens_handoff", sel: 1, to: "en" }]);
  assert.deepEqual(fx.filter((e) => e.t === "fly_prep"), [{ t: "fly_prep", box: box(100, 100, 300, 200) }]);
  assert.equal(later(fx, "handoff_late")[0]?.ms, HANDOFF_GIVE_UP_MS);
  assert.equal(s.sel?.handoff, "waiting");
  // ⌘A and a key's window hand off at once too.
  const [, afx] = run(opened(), { t: "key", key: { key: "a", meta: true } });
  assert.equal(handoffs(afx).length, 1);
});

test("a pick waits 370 ms; a double-click hands off the paragraph's area instead", () => {
  const regions = { lines: [{ box: [100, 100, 300, 20] as [number, number, number, number], block: 0 }], blocks: [[100, 100, 300, 60] as [number, number, number, number]], objects: [], instances: [], windows: [], texts: [] };
  let [s, fx] = run(opened(), msg({ t: "lens_regions", data: regions }), { t: "click", at: { x: 150, y: 110 }, double: false });
  assert.equal(HANDOFF_PICK_MS, DOUBLE_MS + 20);
  assert.deepEqual(handoffs(fx), []);
  assert.deepEqual(later(fx, "handoff_due").map((e) => [e.ms, e.action]), [[370, { t: "handoff_due", n: 1 }]]);
  assert.equal(s.sel?.handoff, "due");
  assert.ok(handingOff(s.sel), "no sheet while it waits for a double-click");
  // The second click: the paragraph, handed off at once; the line's timer then does nothing.
  [s, fx] = run(s, { t: "click", at: { x: 150, y: 110 }, double: true });
  assert.equal(s.sel?.n, 2);
  assert.deepEqual(handoffs(fx), [{ cmd: "lens_handoff", sel: 2, to: "en" }]);
  [s, fx] = run(s, { t: "handoff_due", n: 1 });
  assert.deepEqual(fx, []);
  // A single click: once the timer is due.
  [s, fx] = run(opened(), msg({ t: "lens_regions", data: regions }), { t: "click", at: { x: 150, y: 110 }, double: false }, { t: "handoff_due", n: 1 });
  assert.deepEqual(handoffs(fx), [{ cmd: "lens_handoff", sel: 1, to: "en" }]);
  assert.equal(s.sel?.handoff, "waiting");
});

test("waiting shows no sheet and takes no adjustment, no nudge and no typing", () => {
  let [s] = run(opened(), { t: "select", box: box(100, 100, 300, 200), how: "drag" });
  assert.ok(handingOff(s.sel));
  let fx: Effect[];
  [s, fx] = run(s, { t: "adjusted", box: box(160, 100, 300, 200) }, { t: "move_due", v: s.moveV + 1 });
  assert.deepEqual(s.sel?.box, box(100, 100, 300, 200));
  assert.deepEqual(sends(fx), []);
  [s, fx] = run(s, { t: "key", key: { key: "ArrowLeft" } }, { t: "key", key: { key: "x" } });
  assert.deepEqual(s.sel?.box, box(100, 100, 300, 200));
  assert.deepEqual(fx, []);
  // Results that arrive meanwhile are kept (they are the sheet's if the notch says no).
  [s] = run(s, msg({ t: "lens_answer", data: { sel: 1, state: "done", title: "Croissant", actions: [] } }));
  assert.equal(s.sel?.answer?.title, "Croissant");
  // Esc still closes.
  assert.deepEqual(run(s, { t: "key", key: { key: "Escape" } })[1], [{ t: "close", why: "esc" }]);
});

test("yes: it closes with \"notch\" and flies — to the notch on this display, lifting on another", () => {
  let [s, fx] = run(opened(), { t: "select", box: box(100, 300, 400, 200), how: "drag" }, reply(1, true, { display: 7 }));
  assert.deepEqual(fx.filter((e) => e.t === "close"), [{ t: "close", why: "notch" }]);
  assert.equal(s.sel?.handoff, "ok");
  assert.equal(s.fly?.mode, "notch");
  assert.equal(300 + s.fly!.dx, W / 2);
  assert.equal(400 + s.fly!.dy, 16, "the notch's centre (32 pt notch)");
  // Unknown display either side: the notch is assumed here.
  [s] = run(opened(), { t: "select", box: box(100, 300, 400, 200), how: "drag" }, reply(1, true));
  assert.equal(s.fly?.mode, "notch");
  // NotchHub's island on another display: it lifts and fades where it is.
  [s] = run(opened(), { t: "select", box: box(100, 300, 400, 200), how: "drag" }, reply(1, true, { display: 9 }));
  assert.equal(s.fly?.mode, "lift");
  // Reduce Motion: a fade in place.
  const [rm] = run(opened(), phase(1, "open", { reduce_motion: true }), { t: "select", box: box(100, 300, 400, 200), how: "drag" }, reply(1, true));
  assert.equal(rm.fly?.mode, "fade");
  // Flying: no more adjusting, and a second "yes" does nothing.
  [s, fx] = run(s, { t: "adjusted", box: box(0, 0, 50, 50) }, reply(1, true));
  assert.deepEqual(fx, []);
});

test("no — or no answer — and the sheet shows here, complete", () => {
  for (const why of ["no_notch", "unshown", "timeout", "stale"]) {
    const [s, fx] = run(opened(), { t: "select", box: box(100, 100, 300, 200), how: "drag" }, reply(1, false, { why }));
    assert.equal(s.sel?.handoff, "no", why);
    assert.equal(handingOff(s.sel), false);
    assert.equal(s.fly, null);
    assert.deepEqual(fx.filter((e) => e.t === "close"), []);
  }
  // Refused once: the rest of this session stays here (an adjustment, a new drag) — no flicker.
  const [no] = run(opened(), { t: "select", box: box(100, 100, 300, 200), how: "drag" }, reply(1, false, { why: "unshown" }));
  assert.equal(no.notchRefused, true);
  let [again, afx] = run(no, { t: "adjusted", box: box(200, 100, 300, 200) }, { t: "move_due", v: no.moveV + 1 });
  assert.equal(again.sel?.n, 2);
  assert.equal(again.sel?.handoff, "none");
  assert.deepEqual(handoffs(afx), []);
  [again, afx] = run(again, { t: "select", box: box(600, 100, 300, 200), how: "drag" });
  assert.deepEqual(handoffs(afx), []);
  assert.equal(run(again, phase(2, "warm"))[0].notchRefused, false, "a new session asks again");
  const [late] = run(opened(), { t: "select", box: box(100, 100, 300, 200), how: "drag" }, { t: "handoff_late", n: 1 });
  assert.equal(late.sel?.handoff, "no");
  // Given up while asked: the core hears it (the card it made closes, the results come here again).
  const [, lfx] = run(opened(), { t: "select", box: box(100, 100, 300, 200), how: "drag" }, { t: "handoff_late", n: 1 });
  assert.deepEqual(sends(lfx).slice(-1), [{ cmd: "lens_handoff", sel: 1, cancel: true }]);
  assert.deepEqual(send.handoffCancel(3), { cmd: "lens_handoff", sel: 3, cancel: true });
  // …but a refusal needs no word back (the core said it).
  const [, rfx] = run(opened(), { t: "select", box: box(100, 100, 300, 200), how: "drag" }, reply(1, false, { why: "unshown" }));
  assert.deepEqual(sends(rfx).filter((c) => c.cancel), []);
  // A late "yes" after giving up changes nothing.
  const [after, fx] = run(late, reply(1, true));
  assert.equal(after.sel?.handoff, "no");
  assert.deepEqual(fx, []);
});

test("live mode (no still): handed off once captured, without coming back; back with the sheet on no", () => {
  let [s, fx] = run(opened({}, false), { t: "select", box: box(10, 10, 200, 100), how: "drag" });
  assert.deepEqual(fx.map((e) => e.t), ["send_live", "later"]);
  assert.equal(s.sel?.handoff, "due");
  [s, fx] = run(s, msg({ t: "lens_captured", data: { sel: 1, ok: true } }));
  assert.deepEqual(fx.filter((e) => e.t === "peek"), [], "the overlay stays out of the way");
  assert.deepEqual(handoffs(fx), [{ cmd: "lens_handoff", sel: 1, to: "en" }]);
  assert.deepEqual(fx.filter((e) => e.t === "fly_prep"), [], "nothing to fly: the overlay is already out of the way");
  let yes: LensState;
  [yes, fx] = run(s, reply(1, true));
  assert.deepEqual(fx, [{ t: "close", why: "notch" }, { t: "later", ms: NOTCH_CHECK_MS, action: { t: "notch_check", n: 1, tries: 1 } }]);
  assert.equal(yes.fly, null);
  [s, fx] = run(s, reply(1, false, { why: "unshown" }));
  assert.deepEqual(fx, [{ t: "peek", hidden: false }]);
  assert.equal(s.sel?.handoff, "no");
  // A capture that failed: back, with the sheet.
  const [f, ffx] = run(opened({}, false), { t: "select", box: box(10, 10, 200, 100), how: "drag" }, msg({ t: "lens_captured", data: { sel: 1, ok: false } }));
  assert.deepEqual(ffx.filter((e) => e.t === "peek"), [{ t: "peek", hidden: false }]);
  assert.equal(f.sel?.handoff, "no");
  assert.deepEqual(handoffs(ffx), []);
});

test("an older area's reply is ignored", () => {
  let [s, fx] = run(opened(), { t: "select", box: box(100, 100, 300, 200), how: "drag" }, { t: "select", box: box(500, 100, 300, 200), how: "drag" });
  assert.equal(s.sel?.n, 2);
  [s, fx] = run(s, reply(1, true));
  assert.deepEqual(fx, []);
  assert.equal(s.fly, null);
  assert.equal(s.sel?.handoff, "waiting");
  [s, fx] = run(s, reply(1, false));
  assert.equal(s.sel?.handoff, "waiting");
  // Closing anyway (Esc just before the answer): no flight.
  const [esc] = run(opened(), { t: "select", box: box(100, 100, 300, 200), how: "drag" }, phase(1, "closing", { why: "esc" }));
  const [after, efx] = run(esc, reply(1, true));
  assert.equal(after.fly, null);
  assert.deepEqual(efx, []);
});

test("the reset after closing into the notch waits for the flight", () => {
  const [s] = run(opened(), { t: "select", box: box(100, 100, 300, 200), how: "drag" }, reply(1, true));
  const [, fx] = run(s, phase(1, "closing", { why: "notch" }));
  assert.deepEqual(later(fx, "reset").map((e) => e.ms), [FLY_RESET_MS]);
  assert.equal(FLY_RESET_MS, 460);
  const [, rfx] = run(s, phase(1, "closing", { why: "notch", reduce_motion: true }));
  assert.deepEqual(later(rfx, "reset").map((e) => e.ms), [FLY_RESET_REDUCED_MS]);
  const [, efx] = run(s, phase(1, "closing", { why: "esc" }));
  assert.deepEqual(later(efx, "reset").map((e) => e.ms), [FADE_MS]);
  // And then everything goes: the flight with the session.
  const [done] = run(run(s, phase(1, "closing", { why: "notch" }))[0], { t: "reset", g: 1 });
  assert.equal(done.fly, null);
  assert.equal(done.sel, null);
});

test("live mode: the capture's timer never gives up on an area already asked about", () => {
  let [s, fx] = run(opened({}, false), { t: "select", box: box(10, 10, 200, 100), how: "drag" });
  const capture = later(fx, "handoff_late");
  assert.deepEqual(capture.map((e) => e.action), [{ t: "handoff_late", n: 1, whileDue: true }]);
  [s, fx] = run(s, msg({ t: "lens_captured", data: { sel: 1, ok: true } }));
  assert.equal(s.sel?.handoff, "waiting");
  const asked = later(fx, "handoff_late");
  assert.deepEqual(asked.map((e) => e.action), [{ t: "handoff_late", n: 1 }]);
  // The capture's 5 s run out first (the capture was slow): the ask keeps its own full wait.
  [s, fx] = run(s, capture[0].action);
  assert.equal(s.sel?.handoff, "waiting");
  assert.deepEqual(fx, []);
  [s, fx] = run(s, asked[0].action);
  assert.equal(s.sel?.handoff, "no");
  assert.deepEqual(sends(fx), [{ cmd: "lens_handoff", sel: 1, cancel: true }]);
  // Never captured at all: the capture's timer gives up.
  const [never] = run(opened({}, false), { t: "select", box: box(10, 10, 200, 100), how: "drag" }, capture[0].action);
  assert.equal(never.sel?.handoff, "no");
});

test("after a yes the overlay must be closing: asked again, then closed plainly", () => {
  const [s, fx] = run(opened(), { t: "select", box: box(100, 100, 300, 200), how: "drag" }, reply(1, true));
  const check = later(fx, "notch_check");
  assert.deepEqual(check.map((e) => [e.ms, e.action]), [[NOTCH_CHECK_MS, { t: "notch_check", n: 1, tries: 1 }]]);
  assert.ok(NOTCH_CHECK_MS > 440, "after the shell's own 440 ms");
  // Closing as it should: the check does nothing.
  const [closing] = run(s, phase(1, "closing", { why: "notch" }));
  assert.deepEqual(run(closing, check[0].action)[1], []);
  // Still open (the shell's close was lost): "notch" again, then a plain close.
  const [, again] = run(s, check[0].action);
  assert.deepEqual(again.filter((e) => e.t === "close"), [{ t: "close", why: "notch" }]);
  const next = later(again, "notch_check");
  assert.deepEqual(next.map((e) => e.action), [{ t: "notch_check", n: 1, tries: 2 }]);
  assert.deepEqual(run(s, next[0].action)[1], [{ t: "close", why: "cancel" }]);
  // An older area's check, or one after the session ended: nothing.
  assert.deepEqual(run(s, { t: "notch_check", n: 9, tries: 1 })[1], []);
  assert.deepEqual(run(run(s, phase(1, "closing", { why: "notch" }), { t: "reset", g: 1 })[0], check[0].action)[1], []);
});

test("the shell's fallback window (no panel) never hands off: the sheet, as in v1", () => {
  const frozen: CoreMsg = { t: "lens_frozen", data: { ok: true, ms: 100 } };
  const [s] = run(initial(W, H, "en"), phase(1, "armed", { panel: false }), shell(1, "armed"), { t: "socket", status: "open" },
    msg({ t: "lens_hello", data: hello() }), msg(frozen), phase(1, "open"));
  assert.equal(s.panel, false);
  const [d, fx] = run(s, { t: "select", box: box(100, 100, 300, 200), how: "drag" });
  assert.deepEqual(handoffs(fx), []);
  assert.equal(d.sel?.handoff, "none");
  // The panel (or an older shell that doesn't say): it hands off.
  assert.equal(opened().panel, true);
  const [p] = run(initial(W, H, "en"), phase(1, "armed", { panel: true }));
  assert.equal(p.panel, true);
});

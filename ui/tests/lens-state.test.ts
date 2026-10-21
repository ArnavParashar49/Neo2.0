// The overlay "with mocks" (src/lens/state.ts): the reducer driven by fake phase events from the
// shell, fake core messages and fake pointer and keys — warm → preview → armed → open, a cancel,
// Esc, the IME's keys, the permission and "starting" cards, the still pending then ready, live
// mode, stale answers dropped, follow-ups, adjustments that re-ask only when they moved enough.
import { test } from "node:test";
import assert from "node:assert/strict";
import { PICK_WAIT_MS, answerProblem, hint, initial, privacyLine, reduce, replyProblem, showsWebGlow, type Action, type Effect, type LensState } from "../src/lens/state.ts";
import { box } from "../src/lens/geometry.ts";
import type { CoreMsg, LensHello, Regions, Sent } from "../src/lens/wire.ts";

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
const shell = (g: number, p: string, extra: Record<string, unknown> = {}): Action => ({ t: "shell", hello: { g, phase: p, ticket: T, core: true, sc: true, since_ms: 0, ...extra }, now: 1000 });
const msg = (m: CoreMsg): Action => ({ t: "msg", msg: m });
const hello = (over: Partial<LensHello> = {}): LensHello => ({
  v: 1, open: true, frozen: "pending", display: { w: W, h: H, scale: 2 }, via: "hold", perms: { screen: true },
  eyes: { label: "Gemini", local: false }, eyesWhy: "", ask: true, search: { word: "google", label: "Google", image: true, text: true },
  langs: [], prefill: null, ...over,
});
const sends = (fx: Effect[]) => fx.filter((e): e is Extract<Effect, { t: "send" | "send_live" }> => e.t === "send" || e.t === "send_live").map((e) => e.cmd);

/** Open, connected, the still in: ready for an area. */
function opened(over: Partial<LensHello> = {}, frozen: CoreMsg = { t: "lens_frozen", data: { ok: true, ms: 100 } }): LensState {
  const [s] = run(initial(W, H), phase(1, "warm"), phase(1, "preview", { to_arm_ms: 550 }), phase(1, "armed"), shell(1, "armed"),
    { t: "socket", status: "open" }, msg({ t: "lens_hello", data: hello(over) }), msg(frozen), phase(1, "open"));
  return s;
}

test("warm → preview → armed → open: the ticket comes from the shell's command, then the line opens", () => {
  let [s, fx] = run(initial(W, H), phase(1, "warm"));
  assert.equal(s.phase, "warm");
  assert.deepEqual(fx, []);
  [s, fx] = run(s, phase(1, "preview", { to_arm_ms: 550 }));
  assert.equal(s.toArmMs, 550);
  [s, fx] = run(s, phase(1, "armed"));
  assert.deepEqual(fx, [{ t: "shell_hello" }]); // the event never carries the ticket
  [s, fx] = run(s, shell(1, "armed"));
  assert.equal(s.ticket, T);
  assert.deepEqual(fx, [{ t: "connect", ticket: T }]);
  [s, fx] = run(s, { t: "socket", status: "open" });
  assert.deepEqual(sends(fx), [{ cmd: "lens_hello", v: 1 }]);
  [s] = run(s, phase(1, "open"));
  assert.equal(s.phase, "open");
  assert.equal(hint(s), "Circle or drag over anything · Click to pick · Esc to close");
});

test("a cancel retracts, lets go of the line and forgets the session", () => {
  let [s, fx] = run(initial(W, H), phase(1, "warm"), phase(1, "preview", { to_arm_ms: 550 }), phase(1, "armed"), shell(1, "armed"), phase(1, "cancel", { why: "key" }));
  assert.equal(s.phase, "cancel");
  assert.ok(fx.some((e) => e.t === "disconnect"));
  const later = fx.find((e) => e.t === "later");
  assert.ok(later && later.t === "later" && later.action.t === "reset");
  [s, fx] = run(s, (later as Extract<Effect, { t: "later" }>).action);
  assert.equal(s.phase, "idle");
  assert.equal(s.ticket, null);
  // A newer session replaces an older one whatever it had; an older one's events count for nothing.
  [s] = run(s, phase(2, "warm"));
  assert.equal(s.g, 2);
  [s, fx] = run(s, phase(1, "open"));
  assert.equal(s.phase, "warm");
  assert.deepEqual(fx, []);
});

test("Esc closes — unless an input method is composing", () => {
  const s = opened();
  assert.deepEqual(reduce(s, { t: "key", key: { key: "Escape" } })[1], [{ t: "close", why: "esc" }]);
  assert.deepEqual(reduce(s, { t: "key", key: { key: "Escape", composing: true } })[1], []);
  assert.deepEqual(reduce(s, { t: "key", key: { key: "Escape", inComposer: true } })[1], [{ t: "close", why: "esc" }]);
  assert.deepEqual(reduce(s, { t: "close", why: "cancel" })[1], [{ t: "close", why: "cancel" }]); // right-click, ×
});

test("no Screen Recording: the permission card, and its button opens the settings", () => {
  let [s] = run(initial(W, H), phase(1, "open", { via: "menu" }), shell(1, "open", { sc: false }));
  assert.equal(s.card, "permission");
  assert.equal(hint(s), null);
  assert.deepEqual(reduce(s, { t: "allow_sc" })[1], [{ t: "allow_sc" }]);
  // The core says so too (it never asks macOS itself).
  [s] = run(initial(W, H), phase(1, "open"), shell(1, "open"), { t: "socket", status: "open" }, msg({ t: "lens_frozen", data: { ok: false, why: "no_permission" } }));
  assert.equal(s.card, "permission");
  // No area is taken while the card shows.
  assert.equal(run(s, { t: "select", box: box(10, 10, 100, 100), how: "drag" })[0].sel, null);
});

test("no core yet: \"NEO is starting\", closing itself after a moment", () => {
  const [s, fx] = run(initial(W, H), phase(1, "open", { via: "menu" }), shell(1, "open", { core: false }));
  assert.equal(s.card, "starting");
  assert.ok(fx.some((e) => e.t === "later" && e.action.t === "close"));
  // The core went while it was open: the shell says so.
  assert.equal(run(opened(), phase(1, "gone"))[0].card, "restarted");
});

test("an area waits for the still, then goes; without a still it goes live, the overlay out of the way", () => {
  let [s, fx] = run(initial(W, H), phase(1, "armed"), shell(1, "armed"), { t: "socket", status: "open" }, msg({ t: "lens_hello", data: hello() }), phase(1, "open"));
  assert.equal(s.frozen, "pending");
  [s, fx] = run(s, { t: "select", box: box(100, 100, 300, 200), how: "drag" });
  assert.deepEqual(sends(fx), []); // the still is still coming
  [s, fx] = run(s, msg({ t: "lens_frozen", data: { ok: true, ms: 120 } }));
  assert.deepEqual(sends(fx), [{ cmd: "lens_select", sel: 1, rect: [100, 100, 300, 200], how: "drag" }]);
  // Live mode.
  let live = opened({}, { t: "lens_frozen", data: { ok: false, why: "no_sck" } });
  [live, fx] = run(live, { t: "select", box: box(10, 10, 200, 100), how: "lasso" });
  assert.deepEqual(fx, [{ t: "send_live", cmd: { cmd: "lens_select", sel: 1, rect: [10, 10, 200, 100], how: "lasso", live: true } }]);
  [live, fx] = run(live, msg({ t: "lens_captured", data: { sel: 1, ok: true } }));
  assert.deepEqual(fx, [{ t: "peek", hidden: false }]);
  assert.equal(live.sel?.captured, "ok");
});

test("an older area's answers are dropped; the newest's are kept", () => {
  let [s] = run(opened(), { t: "select", box: box(10, 10, 200, 100), how: "drag" }, { t: "select", box: box(300, 300, 200, 100), how: "drag" });
  assert.equal(s.sel?.n, 2);
  [s] = run(s, msg({ t: "lens_answer", data: { sel: 1, state: "done", title: "Old", actions: [] } }));
  assert.equal(s.sel?.answer, null);
  [s] = run(s, msg({ t: "lens_answer", data: { sel: 2, state: "done", title: "Croissant", answer: "A pastry.", brain: { label: "Gemini", local: false }, actions: [] } }));
  assert.equal(s.sel?.answer?.title, "Croissant");
  assert.equal(privacyLine(s), "Sent to Gemini: only this area");
  [s] = run(s, msg({ t: "lens_local", data: { sel: 2, text: "Croissant", lines: 1, lang: null, translateTo: "", dark: false, actions: [{ id: "a1", kind: "copy_text", label: "Copy text" }] } }));
  assert.equal(s.sel?.local?.text, "Croissant");
});

test("follow-ups keep the area: asked, thinking, answered; voice answers join the thread", () => {
  let [s, fx] = run(opened(), { t: "select", box: box(10, 10, 200, 100), how: "drag" }, { t: "ask", q: "  how many  calories? " });
  assert.deepEqual(sends(fx).at(-1), { cmd: "lens_ask", sel: 1, qid: 1, q: "how many calories?" });
  assert.deepEqual(s.sel?.thread, [{ qid: 1, q: "how many calories?", state: "thinking" }]);
  [s] = run(s, msg({ t: "lens_reply", data: { sel: 1, qid: 1, state: "done", text: "About 230.", brain: { label: "Gemini", local: false } } }));
  assert.equal(s.sel?.thread[0].text, "About 230.");
  assert.equal(s.sel?.thread[0].q, "how many calories?");
  [s] = run(s, msg({ t: "lens_reply", data: { sel: 1, qid: 100001, state: "done", text: "Yes." } }));
  assert.equal(s.sel?.thread.length, 2);
  assert.deepEqual(reduce(s, { t: "ask", q: "   " })[1], []);
  // Try again (or Ask, with NEO_LENS_ASK off): an empty question re-asks the first answer.
  [s, fx] = run(s, { t: "ask_again" });
  assert.deepEqual(sends(fx), [{ cmd: "lens_ask", sel: 1, qid: 0, q: "" }]);
  assert.equal(s.sel?.answer?.state, "thinking");
});

test("a click waits a moment for smart pick's regions, then picks with what there is", () => {
  let [s, fx] = run(opened(), { t: "click", at: { x: 150, y: 110 }, double: false });
  assert.ok(s.pendingPick);
  assert.ok(fx.some((e) => e.t === "later" && e.action.t === "pick_now"));
  [s, fx] = run(s, msg({ t: "lens_regions", data: { lines: [{ box: [100, 100, 300, 20], block: 0 }], blocks: [[100, 100, 300, 20]], objects: [], instances: [], windows: [], texts: [] } }));
  assert.deepEqual(sends(fx), [{ cmd: "lens_select", sel: 1, rect: [96, 96, 308, 28], how: "pick" }]);
  // Without regions at all (the timer fires): a box around the point.
  [s, fx] = run(opened(), { t: "click", at: { x: 700, y: 500 }, double: false }, { t: "pick_now", g: 1 });
  assert.deepEqual(sends(fx), [{ cmd: "lens_select", sel: 1, rect: [560, 400, 280, 200], how: "pick" }]);
});

test("adjusting re-asks only when the area really changed", () => {
  let [s, fx] = run(opened(), { t: "select", box: box(100, 100, 300, 200), how: "drag" });
  [s, fx] = run(s, { t: "adjusted", box: box(101, 100, 300, 200) });
  [s, fx] = run(s, { t: "move_due", v: s.moveV });
  assert.deepEqual(sends(fx), []);
  assert.equal(s.sel?.n, 1);
  [s, fx] = run(s, { t: "adjusted", box: box(160, 100, 300, 200) });
  const v = s.moveV;
  [s, fx] = run(s, { t: "adjusted", box: box(180, 100, 300, 200) }, { t: "move_due", v }); // superseded
  assert.deepEqual(sends(fx), []);
  [s, fx] = run(s, { t: "move_due", v: s.moveV });
  assert.deepEqual(sends(fx), [{ cmd: "lens_select", sel: 2, rect: [180, 100, 300, 200], how: "adjust" }]);
});

test("the keyboard: Return takes the window under the pointer, Tab walks regions, ⌘A the screen, ⌘C the text, typing asks", () => {
  const regions = { lines: [], blocks: [[600, 100, 300, 200]] as [number, number, number, number][], objects: [], instances: [], windows: [[50, 50, 900, 700]] as [number, number, number, number][], texts: [] };
  let [s, fx] = run(opened(), msg({ t: "lens_regions", data: regions }), { t: "key", key: { key: "Enter", pointer: { x: 100, y: 100 } } });
  assert.deepEqual(sends(fx).at(-1), { cmd: "lens_select", sel: 1, rect: [50, 50, 900, 700], how: "key" });
  [s, fx] = run(opened(), msg({ t: "lens_regions", data: regions }), { t: "key", key: { key: "Tab" } }, { t: "key", key: { key: "Enter" } });
  assert.deepEqual(sends(fx).at(-1), { cmd: "lens_select", sel: 1, rect: [600, 100, 300, 200], how: "key" });
  [s, fx] = run(opened(), { t: "key", key: { key: "a", meta: true } });
  assert.deepEqual(sends(fx).at(-1), { cmd: "lens_select", sel: 1, rect: [0, 0, W, H], how: "all" });
  [s] = run(s, msg({ t: "lens_local", data: { sel: 1, text: "hi", lines: 1, lang: null, translateTo: "", dark: false, actions: [{ id: "a1", kind: "copy_text", label: "Copy text" }] } }));
  assert.deepEqual(sends(reduce(s, { t: "key", key: { key: "c", meta: true } })[1]), [{ cmd: "lens_act", sel: 1, id: "a1" }]);
  assert.deepEqual(reduce(s, { t: "key", key: { key: "c", meta: true, domSelection: true } })[1], []); // selected text copies as usual
  assert.deepEqual(reduce(s, { t: "key", key: { key: "w" } })[1], [{ t: "focus_composer", text: "w" }]);
  assert.equal(reduce(s, { t: "key", key: { key: "Tab" } })[0].candidate, -1); // with an area, Tab is the sheet's
  assert.deepEqual(reduce(s, { t: "key", key: { key: "w", inComposer: true } })[1], []);
  assert.equal(reduce(s, { t: "key", key: { key: "ArrowRight", shift: true } })[0].sel?.box.x, 0); // the whole screen: nowhere to go
  const [small] = run(opened(), { t: "select", box: box(0, 0, 100, 100), how: "drag" });
  const [moved, mfx] = reduce(small, { t: "key", key: { key: "ArrowRight", shift: true } });
  assert.equal(moved.sel?.box.x, 10);
  assert.ok(mfx.some((e) => e.t === "later" && e.action.t === "move_due"));
  assert.equal(reduce(small, { t: "key", key: { key: "ArrowDown", alt: true } })[0].sel?.box.h, 101);
  assert.deepEqual(sends(reduce(s, { t: "key", key: { key: "Enter", meta: true } })[1]), [{ cmd: "lens_continue", sel: 1, text: "" }]);
});

test("chips: an action goes by id, Translate by language, the picture search shows in the privacy line", () => {
  let [s, fx] = run(opened(), { t: "select", box: box(10, 10, 200, 100), how: "drag" }, { t: "chip", kind: "image_search", id: "a3" });
  assert.deepEqual(sends(fx).at(-1), { cmd: "lens_act", sel: 1, id: "a3" });
  assert.equal(s.sel?.notes.a3, "…");
  assert.equal(privacyLine(s), "Sent to Google: only this area");
  [s, fx] = run(s, msg({ t: "lens_done", data: { sel: 1, id: "a3", ok: true, note: "opened", close: true } }));
  assert.equal(s.sel?.notes.a3, "Opened");
  [s, fx] = run(s, { t: "chip", kind: "translate", to: "en" });
  assert.deepEqual(sends(fx), [{ cmd: "lens_translate", sel: 1, to: "en" }]);
  assert.equal(s.sel?.translation?.pending, true);
  [s] = run(s, msg({ t: "lens_translation", data: { sel: 1, to: "en", text: "Hello" } }));
  assert.deepEqual(s.sel?.translation, { to: "en", text: "Hello", error: undefined, pending: false });
  [s] = run(s, { t: "translate_to", to: "hi" });
  assert.equal(s.translateTo, "hi");
});

test("\"search my screen for X\": the pill says it, the first area carries it", () => {
  let [s] = run(opened({ prefill: "price of this" }));
  assert.equal(hint(s), "Circle what you mean — “price of this”");
  [s] = run(s, { t: "select", box: box(10, 10, 200, 100), how: "drag" });
  assert.equal(s.sel?.question, "price of this");
  assert.equal(hint(s), null);
});

test("the answer slot's words for each failure", () => {
  assert.match(answerProblem({ sel: 1, state: "failed", error: "no_brain", actions: [] }) ?? "", /brain that can see/);
  assert.match(answerProblem({ sel: 1, state: "failed", error: "refused", brain: { label: "Claude", local: false }, actions: [] }) ?? "", /^Claude wouldn't/);
  assert.equal(answerProblem({ sel: 1, state: "failed", error: "timeout", actions: [] }), "No answer this time.");
  assert.equal(answerProblem({ sel: 1, state: "done", actions: [] }), null);
  assert.equal(privacyLine(opened()), "Stayed on this Mac");
});

test("a line that can't be had: says so, then closes", () => {
  const [s, fx] = run(opened(), { t: "socket", status: "failed" });
  assert.equal(s.card, "unreachable");
  assert.ok(fx.some((e) => e.t === "later" && e.action.t === "close"));
});

test("the privacy line names every place the area went — the brains that lost the race or failed too", () => {
  const sel: Action = { t: "select", box: box(10, 10, 200, 100), how: "drag" };
  const pic = (label: string, local = false): Sent => ({ label, local, what: "picture" });
  let [s] = run(opened(), sel, msg({ t: "lens_answer", data: { sel: 1, state: "failed", error: "timeout", actions: [], sentTo: [pic("Gemini"), pic("Claude")] } }));
  assert.equal(privacyLine(s), "Sent to Gemini and Claude: only this area"); // (no answer, but it left)
  [s] = run(s, msg({ t: "lens_translation", data: { sel: 1, to: "en", text: "Hi", sentTo: [{ label: "Groq", local: false, what: "text" }] } }));
  assert.equal(privacyLine(s), "Sent to Gemini and Claude: only this area · its text to Groq");
  [s] = run(s, msg({ t: "lens_links", data: { sel: 1, rows: [], sentTo: [{ label: "web search", local: false, what: "query" }] } }));
  assert.equal(privacyLine(s), "Sent to Gemini and Claude: only this area · its text to Groq · a search phrase to the web");
  [s] = run(s, msg({ t: "lens_reply", data: { sel: 1, qid: 1, state: "done", text: "x", sentTo: [pic("Gemini"), pic("OpenAI")] } }));
  assert.equal(privacyLine(s).split(" · ")[0], "Sent to Gemini, Claude and OpenAI: only this area"); // (each once)
  // NEO_LENS_ASK off: nothing left until a follow-up asked a brain.
  let [t] = run(opened({ ask: false }), sel, msg({ t: "lens_answer", data: { sel: 1, state: "failed", error: "off", actions: [], sentTo: [] } }));
  assert.equal(privacyLine(t), "Stayed on this Mac");
  [t] = run(t, msg({ t: "lens_reply", data: { sel: 1, qid: 1, state: "failed", error: "timeout", sentTo: [pic("Claude")] } }));
  assert.equal(privacyLine(t), "Sent to Claude: only this area");
  // A brain on this Mac never counts; the text alone says so.
  let [u] = run(opened(), sel, msg({ t: "lens_answer", data: { sel: 1, state: "done", title: "x", brain: { label: "Ollama", local: true }, actions: [], sentTo: [pic("Ollama", true)] } }));
  assert.equal(privacyLine(u), "Stayed on this Mac");
  [u] = run(u, msg({ t: "lens_translation", data: { sel: 1, to: "en", text: "Hi", sentTo: [{ label: "Groq", local: false, what: "text" }] } }));
  assert.equal(privacyLine(u), "Its text went to Groq");
  // A newer area starts afresh.
  [u] = run(u, { t: "select", box: box(300, 300, 100, 100), how: "drag" });
  assert.equal(privacyLine(u), "Stayed on this Mac");
});

test("a refused key, busy brains, a black area: the words, and what can be pressed", () => {
  assert.match(answerProblem({ sel: 1, state: "failed", error: "key", brain: { label: "Claude", local: false }, actions: [] }) ?? "", /^Claude's key was refused — Brains & Keys/);
  assert.equal(answerProblem({ sel: 1, state: "failed", error: "busy", brain: { label: "Gemini", local: false }, actions: [] }), "Gemini is busy — try again in a minute.");
  assert.equal(answerProblem({ sel: 1, state: "failed", error: "busy", actions: [] }), "The brains are busy — try again in a minute.");
  assert.match(answerProblem({ sel: 1, state: "failed", error: "dark", actions: [] }) ?? "", /came out black/);
  assert.match(replyProblem("key"), /key was refused/);
  assert.match(replyProblem("busy"), /^Busy/);
  assert.equal(replyProblem("whatever"), "No answer this time.");
  // "Ask anyway" on a black area asks the brain after all (the core's empty question).
  const [s] = run(opened(), { t: "select", box: box(10, 10, 200, 100), how: "drag" }, msg({ t: "lens_answer", data: { sel: 1, state: "failed", error: "dark", actions: [] } }));
  assert.deepEqual(sends(reduce(s, { t: "ask_again" })[1]), [{ cmd: "lens_ask", sel: 1, qid: 0, q: "" }]);
});

test("a click waits for smart pick: a line from its first part at once, a thing until the rest is in", () => {
  assert.equal(PICK_WAIT_MS, 3000);
  const first: Regions = { lines: [{ box: [100, 100, 300, 20], block: 0 }], blocks: [[100, 100, 300, 20]], objects: [], instances: [], windows: [[0, 0, W, H]], texts: [], done: false };
  const all: Regions = { ...first, objects: [[600, 400, 200, 200]], done: true };
  // On a line: picked from the first part.
  let [s, fx] = run(opened(), { t: "click", at: { x: 150, y: 110 }, double: false }, msg({ t: "lens_regions", data: first }));
  assert.deepEqual(sends(fx), [{ cmd: "lens_select", sel: 1, rect: [96, 96, 308, 28], how: "pick" }]);
  // On a thing: it waits for the things (never the whole window it would get from the first part)…
  [s, fx] = run(opened(), { t: "click", at: { x: 700, y: 500 }, double: false }, msg({ t: "lens_regions", data: first }));
  assert.ok(s.pendingPick);
  assert.deepEqual(sends(fx), []);
  [s, fx] = run(s, msg({ t: "lens_regions", data: all }));
  assert.deepEqual(sends(fx), [{ cmd: "lens_select", sel: 1, rect: [600, 400, 200, 200], how: "pick" }]);
  // …and a click made after the first part came waits too.
  [s] = run(opened(), msg({ t: "lens_regions", data: first }), { t: "click", at: { x: 700, y: 500 }, double: false });
  assert.ok(s.pendingPick);
  [s] = run(opened(), msg({ t: "lens_regions", data: first }), { t: "click", at: { x: 150, y: 110 }, double: false });
  assert.equal(s.sel?.how, "pick");
});

test("a double-click on a picked line takes its paragraph", () => {
  const r: Regions = { lines: [{ box: [100, 100, 300, 20], block: 0 }, { box: [100, 124, 280, 20], block: 0 }], blocks: [[100, 100, 300, 44]], objects: [], instances: [], windows: [], texts: [] };
  let [s, fx] = run(opened(), msg({ t: "lens_regions", data: r }), { t: "click", at: { x: 150, y: 110 }, double: false });
  assert.deepEqual(sends(fx).at(-1), { cmd: "lens_select", sel: 1, rect: [96, 96, 308, 28], how: "pick" });
  [s, fx] = run(s, { t: "click", at: { x: 151, y: 111 }, double: true });
  assert.deepEqual(sends(fx), [{ cmd: "lens_select", sel: 2, rect: [96, 96, 308, 52], how: "pick" }]);
  assert.equal(s.sel?.n, 2);
});

test("the mic says Listening while the core listens for a question about the area", () => {
  let [s] = run(opened(), { t: "select", box: box(10, 10, 200, 100), how: "drag" }, msg({ t: "lens_listening", data: { on: true } }));
  assert.equal(s.coreState, "listening");
  [s] = run(s, msg({ t: "lens_listening", data: { on: false } }));
  assert.equal(s.coreState, "");
});

test("the shell's glow hears once when an area appears, and only from a page whose glow is native", () => {
  const native = run(opened(), phase(1, "open", { glow: "native", display: { id: 7, x: 0, y: 0, w: W, h: H, scale: 2 } }))[0];
  assert.equal(native.glow, "native");
  assert.equal(native.displayId, 7);
  let [s, fx] = run(native, { t: "select", box: box(100, 100, 300, 200), how: "drag" });
  assert.deepEqual(fx.filter((e) => e.t === "glow"), [{ t: "glow", selected: true }]);
  // A new area replacing it: still selected, nothing more to say.
  [s, fx] = run(s, { t: "select", box: box(500, 100, 300, 200), how: "drag" });
  assert.deepEqual(fx.filter((e) => e.t === "glow"), []);
  // An adjusted area, answers, keys: nothing.
  [s, fx] = run(s, { t: "adjusted", box: box(520, 100, 300, 200) }, msg({ t: "lens_answer", data: { sel: 2, state: "thinking", actions: [] } }));
  assert.deepEqual(fx.filter((e) => e.t === "glow"), []);
  // The page's own glow (the shell couldn't make one): the shell is never asked.
  const [, wfx] = run(run(opened(), phase(1, "open", { glow: "web" }))[0], { t: "select", box: box(100, 100, 300, 200), how: "drag" });
  assert.deepEqual(wfx.filter((e) => e.t === "glow"), []);
  // The glow mode outlives a session (it is the shell's); a phase without the word keeps it.
  assert.equal(run(native, phase(2, "warm"))[0].glow, "native");
});

test("a page whose glow is native never draws one itself", () => {
  assert.equal(showsWebGlow({ glow: "native" }), false);
  assert.equal(showsWebGlow({ glow: "web" }), true);
  assert.equal(showsWebGlow({}), true, "an older shell says nothing: the page draws it");
  assert.equal(showsWebGlow(null), true);
  assert.equal(showsWebGlow(initial(W, H)), true);
  assert.equal(showsWebGlow(run(initial(W, H), phase(1, "warm", { glow: "native" }))[0]), false);
});

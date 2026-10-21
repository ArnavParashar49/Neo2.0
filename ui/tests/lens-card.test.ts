// The card in the notch on NEO's page (src/lens/card.ts), with a fake sender and a fake clock: the
// snapshot becomes the card, another card's messages are dropped, every click is the card's own
// command naming its cid, close sends once and clears, a chat turn takes the panel back (but never
// the one sentence the card's mic asked for), the page says lens_card_hello only in the notch — the
// sheet and the card ask the core for the same things — and the card's size never bounces.
import { test } from "node:test";
import assert from "node:assert/strict";
import { CARD_SIZE, LISTEN_GRACE_MS, LensCardHub, cardHeight, cardWidth, settleHeight, type Clock } from "../src/lens/card.ts";
import { initial, reduce, type Action, type Effect, type LensState } from "../src/lens/state.ts";
import type { CoreMsg, LensHello } from "../src/lens/wire.ts";

const CID = "ab12".repeat(8);
const OTHER = "cd34".repeat(8);

function fakeClock() {
  let t = 1_000_000;
  let n = 0;
  const due = new Map<number, { at: number; cb: () => void }>();
  const clock: Clock = {
    now: () => t,
    set: (cb, ms) => { due.set(++n, { at: t + ms, cb }); return n; },
    clear: (id) => { due.delete(id); },
  };
  const advance = (ms: number) => {
    t += ms;
    for (const [id, d] of [...due]) if (d.at <= t) { due.delete(id); d.cb(); }
  };
  return { clock, advance, pending: () => due.size };
}

function hub(inNotch = true) {
  const c = fakeClock();
  const h = new LensCardHub(inNotch, c.clock);
  const sent: Record<string, unknown>[] = [];
  h.setSender((cmd) => { sent.push(cmd); return true; });
  return { h, sent, ...c };
}

const snapshot = (over: Record<string, unknown> = {}) => ({
  cid: CID, open: true,
  card: {
    sel: 7, thumb: "data:image/jpeg;base64,AAAA", question: null, translate_to: "en",
    hello: { eyes: { label: "Gemini", local: false }, eyes_why: "", ask: true, search: { word: "google", label: "Google", image: true, text: true }, langs: [{ code: "en", name: "English" }, { code: "de", name: "German" }] },
    local: { text: "Bonjour", lines: 1, lang: { code: "fr", name: "French", conf: 0.9 }, translate_to: "en", actions: [{ id: "a1", kind: "copy_text", label: "Copy text" }, { id: "a2", kind: "image_search", label: "Search with Google" }] },
    answer: null, links: null, thread: [], translation: null, sent_to: [],
    ...over,
  },
});

test("the snapshot becomes the card; its results come tagged with its cid; another card's are dropped", () => {
  const { h } = hub();
  assert.equal(h.snap(), null);
  assert.equal(h.handle("lens_card", snapshot()), true);
  const s = h.snap()!;
  assert.equal(s.cid, CID);
  assert.equal(s.r.n, 7);
  assert.equal(s.r.local?.text, "Bonjour");
  assert.equal(s.thumb, "data:image/jpeg;base64,AAAA");
  assert.equal(s.translateTo, "en");
  assert.equal(s.hello.eyes?.label, "Gemini");
  assert.equal(h.handle("lens_answer", { cid: CID, sel: 7, state: "done", title: "A greeting", actions: [] }), true);
  assert.equal(h.snap()!.r.answer?.title, "A greeting");
  assert.equal(h.handle("lens_answer", { cid: OTHER, sel: 7, state: "done", title: "Not ours", actions: [] }), true);
  assert.equal(h.snap()!.r.answer?.title, "A greeting");
  assert.equal(h.handle("lens_answer", { sel: 7, state: "done", title: "The overlay's", actions: [] }), false, "no cid: not the card's");
  assert.equal(h.handle("lens_continue", {}), false);
  const v = h.snap()!.v;
  h.handle("lens_answer", { cid: CID, sel: 8, state: "done", title: "Another area", actions: [] });
  assert.equal(h.snap()!.v, v, "another area's message changes nothing");
  // It closes (the core says so), for any reason: only that card.
  h.handle("lens_card", { cid: OTHER, open: false, why: "fold" });
  assert.ok(h.snap());
  h.handle("lens_card", { cid: CID, open: false, why: "fold" });
  assert.equal(h.snap(), null);
});

test("each click is the card's own command, naming its cid", () => {
  const { h, sent } = hub();
  h.handle("lens_card", snapshot());
  h.op({ t: "chip", kind: "copy_text", id: "a1" });
  h.op({ t: "chip", kind: "translate", to: "en" });
  h.op({ t: "translate_to", to: "de" });
  h.op({ t: "ask", q: "what does it say?" });
  h.op({ t: "ask_again" });
  h.op({ t: "continue", text: "more" });
  h.op({ t: "listen" });
  assert.deepEqual(sent, [
    { cmd: "lens_card_act", cid: CID, id: "a1" },
    { cmd: "lens_card_translate", cid: CID, to: "en" },
    { cmd: "lens_card_translate", cid: CID, to: "de" },
    { cmd: "lens_card_ask", cid: CID, qid: 1, q: "what does it say?" },
    { cmd: "lens_card_ask", cid: CID, qid: 0, q: "" },
    { cmd: "lens_card_continue", cid: CID, text: "more" },
    { cmd: "lens_card_listen", cid: CID },
  ]);
  const s = h.snap()!;
  assert.equal(s.r.notes.a1, "…");
  assert.equal(s.translateTo, "de");
  assert.equal(s.r.thread[0].q, "what does it say?");
});

test("a finished action's note clears after a moment", () => {
  const { h, advance } = hub();
  h.handle("lens_card", snapshot());
  h.op({ t: "chip", kind: "copy_text", id: "a1" });
  h.handle("lens_done", { cid: CID, sel: 7, id: "a1", ok: true, note: "copied", close: false });
  assert.equal(h.snap()!.r.notes.a1, "Copied");
  advance(1199);
  assert.equal(h.snap()!.r.notes.a1, "Copied");
  advance(2);
  assert.equal(h.snap()!.r.notes.a1, undefined);
});

test("close sends once and clears", () => {
  const { h, sent } = hub();
  h.handle("lens_card", snapshot());
  h.close("esc");
  h.close("x");
  assert.deepEqual(sent, [{ cmd: "lens_card_close", cid: CID, why: "esc" }]);
  assert.equal(h.snap(), null);
});

test("a chat turn takes the panel back: the user's own words, a browser or a song card", () => {
  for (const [type, data] of [
    ["transcript", { role: "user", text: "Neo, what's the weather", final: true }],
    ["browser_open", { sid: "s" }],
    ["song_open", { sid: "s" }],
  ] as const) {
    const { h, sent } = hub();
    h.handle("lens_card", snapshot());
    h.chat(type, data as Record<string, unknown>);
    assert.deepEqual(sent, [{ cmd: "lens_card_close", cid: CID, why: "chat" }], type);
    assert.equal(h.snap(), null);
  }
  const { h, sent } = hub();
  h.handle("lens_card", snapshot());
  h.chat("transcript", { role: "user", text: "Neo, what's", final: false }); // (still being said)
  h.chat("transcript", { role: "assistant", text: "It's sunny.", final: true });
  h.chat("transcript", { role: "user", text: "about this", final: true, attach: "screen" }); // (Continue in NEO's own line)
  assert.deepEqual(sent, []);
  // The hello's replay of the transcript is history, not a turn now.
  h.beginHello();
  h.chat("transcript", { role: "user", text: "old", final: true });
  assert.deepEqual(sent, []);
});

test("the sentence the card's mic asked for never closes it (while listening, or just after)", () => {
  const { h, sent, advance } = hub();
  h.handle("lens_card", snapshot());
  h.handle("lens_listening", { cid: CID, on: true });
  assert.equal(h.snap()!.listening, true);
  h.chat("transcript", { role: "user", text: "what colour is it", final: true });
  h.handle("lens_listening", { cid: CID, on: false });
  advance(LISTEN_GRACE_MS - 10);
  h.chat("transcript", { role: "user", text: "what colour is it", final: true }); // (a voice that says it after)
  assert.deepEqual(sent, []);
  advance(20);
  h.chat("transcript", { role: "user", text: "Neo, set a timer", final: true });
  assert.deepEqual(sent, [{ cmd: "lens_card_close", cid: CID, why: "chat" }]);
});

test("lens_card_hello goes after each hello, in the notch only; a dropped line drops the card", () => {
  const notch = hub(true);
  notch.h.beginHello();
  notch.h.endHello();
  assert.deepEqual(notch.sent, [{ cmd: "lens_card_hello" }]);
  notch.h.handle("lens_card", snapshot());
  notch.h.setConnected(false);
  assert.equal(notch.h.snap(), null);
  assert.equal(notch.sent.length, 1, "nothing sent for it: the core dropped it with the line");
  const win = hub(false); // NEO.app's own window never shows a card
  win.h.beginHello();
  win.h.endHello();
  assert.deepEqual(win.sent, []);
  assert.equal(win.h.handle("lens_answer", { cid: CID, sel: 7, state: "done", actions: [] }), false);
});

test("asked again (the page reloaded), the card keeps its questions' numbers", () => {
  const { h, sent } = hub();
  h.handle("lens_card", snapshot({ thread: [{ qid: 3, q: "how old?", state: "done", text: "Ten years." }, { qid: 100002, q: "", state: "done", text: "Navy." }] }));
  h.op({ t: "ask", q: "and now?" });
  assert.deepEqual(sent.at(-1), { cmd: "lens_card_ask", cid: CID, qid: 4, q: "and now?" });
  assert.deepEqual(h.snap()!.r.thread.map((t) => t.qid), [3, 100002, 4]);
});

// ---- the sheet and the card ask for the same things ----------------------------------------------------

const T = "ab".repeat(32);
const W = 1512, H = 982;
const helloFor = (): LensHello => ({
  v: 1, open: true, frozen: "pending", display: { w: W, h: H, scale: 2 }, via: "hold", perms: { screen: true },
  eyes: { label: "Gemini", local: false }, eyesWhy: "", ask: true, search: { word: "google", label: "Google", image: true, text: true },
  langs: [], prefill: null, notch: false,
});

test("the sheet and the card build the same ops for the same clicks", () => {
  let s: LensState = initial(W, H, "en");
  const fx: Effect[] = [];
  const go = (a: Action) => { const [n, e] = reduce(s, a); s = n; fx.push(...e); };
  go({ t: "phase", ev: { g: 1, phase: "armed" }, now: 1 });
  go({ t: "shell", hello: { g: 1, phase: "armed", ticket: T, core: true, sc: true, since_ms: 0 }, now: 1 });
  go({ t: "socket", status: "open" });
  go({ t: "msg", msg: { t: "lens_hello", data: helloFor() } as CoreMsg });
  go({ t: "msg", msg: { t: "lens_frozen", data: { ok: true, ms: 1 } } });
  go({ t: "phase", ev: { g: 1, phase: "open" }, now: 2 });
  go({ t: "select", box: { x: 10, y: 10, w: 200, h: 100 }, how: "drag" });
  const n = s.sel!.n;
  fx.length = 0;
  const { h, sent } = hub();
  h.handle("lens_card", snapshot({ sel: n }));
  const clicks = [
    { t: "chip", kind: "copy_text", id: "a1" }, { t: "chip", kind: "translate", to: "en" }, { t: "translate_to", to: "de" },
    { t: "ask", q: "what is it?" }, { t: "ask_again" }, { t: "continue", text: "x" }, { t: "listen" },
  ] as const;
  for (const c of clicks) { go(c as Action); h.op(c); }
  const overlay = fx.filter((e): e is Extract<Effect, { t: "send" }> => e.t === "send").map((e) => e.cmd);
  // The same requests, each in its own lane: the overlay's with the area's number, the card's with its id.
  const strip = (c: Record<string, unknown>) => {
    const { cmd, sel: _sel, cid: _cid, ...rest } = c;
    return { cmd: String(cmd).replace("lens_card_", "lens_"), ...rest };
  };
  assert.deepEqual(overlay.map(strip), sent.map(strip));
  assert.ok(overlay.every((c) => c.sel === n) && sent.every((c) => c.cid === CID));
  assert.deepEqual(overlay.map((c) => c.cmd), ["lens_act", "lens_translate", "lens_translate", "lens_ask", "lens_ask", "lens_continue", "lens_listen"]);
});

// ---- its size in the notch ----------------------------------------------------------------------------

test("the size: 372 wide (320 on a narrow screen), 160–560 tall in steps of 8, within the host's room", () => {
  assert.equal(cardWidth(760), 372);
  assert.equal(cardWidth(undefined), 372);
  assert.equal(cardWidth(360), 320);
  assert.equal(cardWidth(300), 300);
  assert.equal(cardHeight(0), CARD_SIZE.minH);
  assert.equal(cardHeight(161), 168);
  assert.equal(cardHeight(296), 296);
  assert.equal(cardHeight(900), 560);
  assert.equal(cardHeight(900, 400), 400);
  assert.equal(cardHeight(100, 120), 120); // (a host with less room than the minimum: all it has)
});

test("it grows at once, and shrinks only by more than 32 pt held for 300 ms", () => {
  assert.deepEqual(settleHeight(null, 240, null, 0), { h: 240, since: null });
  assert.deepEqual(settleHeight(240, 320, null, 0), { h: 320, since: null });
  assert.deepEqual(settleHeight(320, 296, null, 0), { h: 320, since: null }, "24 pt less: it stays");
  const a = settleHeight(320, 240, null, 1000);
  assert.deepEqual(a, { h: 320, since: 1000, recheck: 300 });
  assert.deepEqual(settleHeight(320, 248, a.since, 1100), { h: 320, since: 1000, recheck: 200 });
  assert.deepEqual(settleHeight(320, 240, a.since, 1300), { h: 240, since: null });
  assert.deepEqual(settleHeight(320, 400, a.since, 1100), { h: 400, since: null }, "a rise in between: at once, and the wait is over");
});

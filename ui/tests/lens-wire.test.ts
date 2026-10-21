// Search Screen's wire (src/lens/wire.ts): every message the core sends is read and cut to size,
// anything else is ignored; the still is NBF1 kind 4 and nothing else (the browser's kinds 1–2 and
// the camera's 3 are refused); what the page sends keeps the protocol's limits.
import { test } from "node:test";
import assert from "node:assert/strict";
import { KIND_LENS, MAX, parseMsg, parseStill, send } from "../src/lens/wire.ts";

/** An NBF1 frame as the core packs it (neo2/core/tools/browser_live.py: pack_frame). */
function packFrame(meta: { kind: number; vw: number; vh: number; iw: number; ih: number; dpr: number }, sid: Uint8Array, image: Uint8Array): ArrayBuffer {
  const buf = new ArrayBuffer(64 + image.byteLength);
  const dv = new DataView(buf);
  dv.setUint32(0, 0x4e424631, false);
  dv.setUint8(4, 1);
  dv.setUint8(5, meta.kind);
  dv.setUint16(6, 64, true);
  new Uint8Array(buf, 8, 16).set(sid.subarray(0, 16));
  dv.setUint32(24, 1, true);
  dv.setFloat64(28, 0, true);
  dv.setUint16(36, meta.vw, true);
  dv.setUint16(38, meta.vh, true);
  dv.setUint16(40, meta.iw, true);
  dv.setUint16(42, meta.ih, true);
  dv.setUint16(44, Math.round(meta.dpr * 100), true);
  dv.setUint8(56, 1);
  new Uint8Array(buf, 64).set(image);
  return buf;
}

const msg = (type: string, data: unknown) => parseMsg(JSON.stringify({ type, data, ts: 1 }));

test("the hello, from either door, says what the overlay can do", () => {
  const data = {
    v: 1, open: true, frozen: "pending", display: { w: 1512, h: 982, scale: 2 }, via: "hold",
    perms: { screen: true }, eyes: { label: "Gemini 3 Flash", local: false }, eyes_why: "", ask: true,
    search: { word: "google", label: "Google", image: true, text: true }, langs: [{ code: "en-GB", name: "English" }], prefill: null,
  };
  const a = msg("lens_hello", data);
  const b = msg("hello", { state: "idle", lens: data });
  assert.deepEqual(a, b);
  assert.equal(a?.t, "lens_hello");
  if (a?.t !== "lens_hello") return;
  assert.equal(a.data.frozen, "pending");
  assert.deepEqual(a.data.eyes, { label: "Gemini 3 Flash", local: false });
  assert.deepEqual(a.data.search, { word: "google", label: "Google", image: true, text: true });
  assert.equal(a.data.prefill, null);
  // An off provider, no brain that sees, no permission, a prefilled question.
  const c = msg("lens_hello", { ...data, search: null, eyes: null, eyes_why: "no_brain", perms: { screen: false }, prefill: "price of this" });
  if (c?.t !== "lens_hello") return assert.fail();
  assert.equal(c.data.search, null);
  assert.equal(c.data.eyes, null);
  assert.equal(c.data.eyesWhy, "no_brain");
  assert.equal(c.data.perms.screen, false);
  assert.equal(c.data.prefill, "price of this");
});

test("every Search Screen message parses, cut to its limits", () => {
  assert.deepEqual(msg("lens_frozen", { ok: true, ms: 120 }), { t: "lens_frozen", data: { ok: true, ms: 120, why: undefined } });
  assert.deepEqual(msg("lens_frozen", { ok: false, why: "no_sck" }), { t: "lens_frozen", data: { ok: false, ms: undefined, why: "no_sck" } });
  const r = msg("lens_regions", { lines: [[1, 2, 3, 4, 0], [1, 2, -3, 4, 0], "x"], blocks: [[1, 2, 3, 4]], objects: [], instances: [[0, 0, 5, 5]], windows: [[0, 0, 10, 10]] });
  if (r?.t !== "lens_regions") return assert.fail();
  assert.deepEqual(r.data.lines, [{ box: [1, 2, 3, 4], block: 0 }]);
  assert.deepEqual(r.data.windows, [[0, 0, 10, 10]]);
  const local = msg("lens_local", {
    sel: 3, text: "x".repeat(MAX.text + 50), lines: 2, lang: { code: "fr", name: "French", conf: 0.9 }, translate_to: "en", dark: false,
    actions: [{ id: "a1", kind: "copy_text", label: "Copy text" }, { id: "a2", kind: "open", label: "Open example.com", detail: "example.com", warn: "punycode" },
      { id: "zz", kind: "open", label: "bad id" }, { id: "a3", kind: "run_shell", label: "nope" }],
  });
  if (local?.t !== "lens_local") return assert.fail();
  assert.equal(local.data.text.length, MAX.text);
  assert.deepEqual(local.data.actions.map((a) => a.id), ["a1", "a2"]);
  assert.equal(local.data.actions[1].warn, "punycode");
  assert.equal(local.data.translateTo, "en");
  const ans = msg("lens_answer", { sel: 3, state: "done", kind: "product", title: "Nike Air Max 90", answer: "A running shoe.", query: "nike air max 90", brain: { label: "Gemini", local: false }, actions: [{ id: "a4", kind: "text_search", label: "Search “nike air max 90”" }] });
  if (ans?.t !== "lens_answer") return assert.fail();
  assert.equal(ans.data.title, "Nike Air Max 90");
  assert.equal(ans.data.actions[0].kind, "text_search");
  const failed = msg("lens_answer", { sel: 3, state: "failed", error: "weird" });
  if (failed?.t !== "lens_answer") return assert.fail();
  assert.equal(failed.data.error, "failed"); // an unknown word is the plain failure
  assert.deepEqual(msg("lens_links", { sel: 3, rows: [{ id: "a5", title: "Nike", host: "nike.com" }, { id: "b", title: "x", host: "y" }] }), { t: "lens_links", data: { sel: 3, rows: [{ id: "a5", title: "Nike", host: "nike.com" }] } });
  assert.equal((msg("lens_reply", { sel: 3, qid: 2, state: "done", text: "Yes." }) as { data: { text: string } }).data.text, "Yes.");
  assert.equal(msg("lens_reply", { sel: 3, state: "done" }), null); // no qid: not one
  assert.deepEqual(msg("lens_done", { sel: 3, id: "a1", ok: true, note: "copied", close: false }), { t: "lens_done", data: { sel: 3, id: "a1", ok: true, note: "copied", close: false } });
  assert.deepEqual(msg("lens_captured", { sel: 3, ok: false, error: "failed" }), { t: "lens_captured", data: { sel: 3, ok: false, error: "failed" } });
  assert.deepEqual(msg("lens_translation", { sel: 3, to: "en", text: "Hello" }), { t: "lens_translation", data: { sel: 3, to: "en", text: "Hello", error: undefined } });
  assert.deepEqual(msg("lens_refused", { cmd: "lens_act", error: "stale" }), { t: "lens_refused", data: { cmd: "lens_act", error: "stale" } });
});

test("anything else is ignored", () => {
  for (const bad of ["", "nope", "[]", JSON.stringify({ type: "transcript", data: { text: "hi" } }), JSON.stringify({ type: "browser_frame", data: {} }),
    JSON.stringify({ type: "lens_local", data: { text: "no sel" } }), JSON.stringify({ type: "lens_answer", data: { sel: "3" } })]) {
    assert.equal(parseMsg(bad), null, bad);
  }
  // A thumbnail is a JPEG data URL or nothing.
  const t = msg("lens_local", { sel: 1, actions: [], thumb: "javascript:alert(1)" });
  assert.equal((t as { data: { thumb?: string } }).data.thumb, undefined);
});

test("the still is NBF1 kind 4, and only that", () => {
  const img = new Uint8Array([0xff, 0xd8, 0xff, 0xd9]);
  const meta = { vw: 1512, vh: 982, iw: 3024, ih: 1964, dpr: 2 };
  const still = parseStill(packFrame({ ...meta, kind: KIND_LENS }, new Uint8Array(16), img));
  assert.ok(still);
  assert.deepEqual([still!.vw, still!.vh, still!.iw, still!.ih, still!.dpr, still!.mime], [1512, 982, 3024, 1964, 2, "image/jpeg"]);
  assert.deepEqual([...still!.image], [...img]);
  for (const kind of [1, 2, 3]) assert.equal(parseStill(packFrame({ ...meta, kind }, new Uint8Array(16), img)), null, String(kind));
  assert.equal(parseStill(new ArrayBuffer(10)), null);
  assert.equal(parseStill(packFrame({ ...meta, kind: KIND_LENS }, new Uint8Array(16), new Uint8Array(0))), null);
});

test("what the page sends keeps the protocol's limits", () => {
  assert.deepEqual(send.hello(false), { cmd: "lens_hello", v: 1 });
  assert.deepEqual(send.hello(true), { cmd: "lens_hello", v: 1, texts: true });
  assert.deepEqual(send.select(2, [10.04, 20, 300.06, Number.NaN], "lasso"), { cmd: "lens_select", sel: 2, rect: [10, 20, 300.1, 0], how: "lasso" });
  assert.deepEqual(send.select(2, [0, 0, 5, 5], "drag", true), { cmd: "lens_select", sel: 2, rect: [0, 0, 5, 5], how: "drag", live: true });
  assert.equal(send.ask(1, 1, "  what   is\nthis ".padEnd(900, "x")).q.length, MAX.q);
  assert.equal(send.ask(1, 1, "  what   is\nthis ").q, "what is this");
  assert.equal(send.cont(1, "y".repeat(5000)).text.length, MAX.cont);
  assert.deepEqual(send.act(4, "a3"), { cmd: "lens_act", sel: 4, id: "a3" });
  assert.deepEqual(send.translate(4, "en-GB"), { cmd: "lens_translate", sel: 4, to: "en-GB" });
  assert.deepEqual(send.listen(4), { cmd: "lens_listen", sel: 4 });
});

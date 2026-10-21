// One area's results (src/lens/result.ts), as the sheet and the card in the notch both keep them:
// the core's messages merged in (another area's left alone), the clicks as ops, a note that clears,
// the privacy line's words — and the wire's checks the card reads them through (parseLens, a card's
// snapshot, the card's own commands).
import { test } from "node:test";
import assert from "node:assert/strict";
import { emptyResult, MAX_THREAD, NOTE_MS, privacyLine, resultAction, resultMsg, type Result } from "../src/lens/result.ts";
import { cardSend, parseCard, parseLens, parseMsg, type CoreMsg } from "../src/lens/wire.ts";

const m = (type: string, data: Record<string, unknown>): CoreMsg => {
  const x = parseLens(type, data);
  assert.ok(x, type);
  return x!;
};
const fold = (r: Result, ...msgs: CoreMsg[]) => msgs.reduce((acc, x) => resultMsg(acc, x).r, r);
const gemini = { label: "Gemini", local: false };

test("the core's messages about the area are merged in; another area's are left alone", () => {
  let r = emptyResult(3);
  r = fold(r,
    m("lens_local", { sel: 3, text: "Merci", lines: 1, lang: { code: "fr", name: "French", conf: 0.9 }, translate_to: "en", actions: [{ id: "a1", kind: "copy_text", label: "Copy text" }] }),
    m("lens_answer", { sel: 2, state: "done", title: "Older", actions: [] }),
    m("lens_answer", { sel: 3, state: "done", kind: "text", title: "Thanks", answer: "It means thank you.", brain: gemini, actions: [], sent_to: [{ ...gemini, what: "picture" }] }),
    m("lens_links", { sel: 3, rows: [{ id: "a2", title: "A shop", host: "shop.example" }], sent_to: [{ label: "web search", local: false, what: "query" }] }),
  );
  assert.equal(r.local?.text, "Merci");
  assert.equal(r.answer?.title, "Thanks"); // (area 2's answer dropped)
  assert.deepEqual(r.links, [{ id: "a2", title: "A shop", host: "shop.example" }]);
  assert.deepEqual(r.sentTo.map((w) => w.what), ["picture", "query"]);
  assert.equal(privacyLine(r, null), "Sent to Gemini: only this area · a search phrase to the web");
});

test("a question goes in thinking, its answer replaces it, a voice question arrives with its answer only", () => {
  let out = resultAction(emptyResult(1), { t: "ask", q: "  what   is it? " }, 4);
  assert.deepEqual(out.ops, [{ op: "ask", qid: 4, q: "what is it?" }]);
  assert.equal(out.nextQid, 5);
  let r = out.r;
  assert.deepEqual(r.thread, [{ qid: 4, q: "what is it?", state: "thinking" }]);
  r = fold(r, m("lens_reply", { sel: 1, qid: 4, state: "done", text: "A bag.", brain: gemini }), m("lens_reply", { sel: 1, qid: 100001, state: "done", text: "Navy." }));
  assert.deepEqual(r.thread.map((t) => [t.qid, t.q, t.state, t.text]), [[4, "what is it?", "done", "A bag."], [100001, "", "done", "Navy."]]);
  out = resultAction(r, { t: "ask", q: "   " }, 5);
  assert.deepEqual(out.ops, []);
  for (let i = 0; i < 20; i++) r = resultAction(r, { t: "ask", q: `q${i}` }, 10 + i).r;
  assert.equal(r.thread.length, MAX_THREAD);
});

test("translate goes pending, then done; the menu's language is the user's from then on", () => {
  let out = resultAction(emptyResult(1), { t: "chip", kind: "translate", to: "en" }, 1);
  assert.deepEqual(out.ops, [{ op: "translate", to: "en" }]);
  assert.equal(out.translateTo, undefined);
  assert.deepEqual(out.r.translation, { to: "en", pending: true });
  const r = fold(out.r, m("lens_translation", { sel: 1, to: "en", text: "Thanks", sent_to: [{ label: "Groq", local: false, what: "text" }] }));
  assert.deepEqual(r.translation, { to: "en", text: "Thanks", error: undefined, pending: false });
  assert.equal(privacyLine(r, null), "Its text went to Groq");
  out = resultAction(r, { t: "translate_to", to: "de" }, 1);
  assert.equal(out.translateTo, "de");
  assert.deepEqual(out.ops, [{ op: "translate", to: "de" }]);
});

test("an action shows … then its note, which clears after a moment (not after one that closed)", () => {
  let out = resultAction(emptyResult(2), { t: "chip", kind: "image_search", id: "a4" }, 1);
  assert.deepEqual(out.ops, [{ op: "act", id: "a4" }]);
  assert.equal(out.r.notes.a4, "…");
  assert.equal(out.r.searchedWeb, true);
  const done = resultMsg(out.r, m("lens_done", { sel: 2, id: "a4", ok: true, note: "opened", close: false }));
  assert.equal(done.r.notes.a4, "Opened");
  assert.deepEqual(done.noteClear, { id: "a4", ms: NOTE_MS });
  assert.equal(resultAction(done.r, { t: "note_clear", id: "a4" }, 1).r.notes.a4, undefined);
  const closing = resultMsg(out.r, m("lens_done", { sel: 2, id: "a4", ok: true, note: "opened", close: true }));
  assert.equal(closing.noteClear, undefined);
  assert.deepEqual(resultAction(emptyResult(2), { t: "chip", kind: "open" }, 1).ops, []); // (no id: nothing)
  assert.equal(privacyLine(out.r, { eyes: null, search: { word: "google", label: "Google", image: true, text: true } }), "Sent to Google: only this area");
});

test("ask again, continue and the mic are ops of their own", () => {
  const r = emptyResult(5);
  const again = resultAction({ ...r, answer: { sel: 5, state: "failed", error: "timeout", actions: [] } }, { t: "ask_again" }, 1);
  assert.deepEqual(again.ops, [{ op: "ask_again" }]);
  assert.equal(again.r.answer?.state, "thinking");
  assert.deepEqual(resultAction(r, { t: "continue", text: "and this?" }, 1).ops, [{ op: "continue", text: "and this?" }]);
  assert.deepEqual(resultAction(r, { t: "listen" }, 1).ops, [{ op: "listen" }]);
  assert.equal(privacyLine({ ...r, answer: { sel: 5, state: "thinking", actions: [] } }, { eyes: gemini, search: null }), "Sent to Gemini: only this area");
  assert.equal(privacyLine(null, null), "Stayed on this Mac");
});

test("parseLens reads exactly what parseMsg does, from data already parsed", () => {
  const data = { sel: 1, state: "done", title: "<b>Bag</b>", answer: "x".repeat(5000), actions: [{ id: "a1", kind: "evil", label: "No" }, { id: "a2", kind: "open", label: "Open" }] };
  assert.deepEqual(parseLens("lens_answer", data), parseMsg(JSON.stringify({ type: "lens_answer", data })));
  const a = parseLens("lens_answer", data);
  assert.equal(a?.t === "lens_answer" && a.data.answer?.length, 1200);
  assert.equal(parseLens("lens_answer", { state: "done" }), null); // (no area)
  assert.equal(parseLens("browser_open", { sid: "x" }), null);
});

test("a card's snapshot is checked like every message: its id, its results, its hello", () => {
  const cid = "0123456789abcdef".repeat(2);
  const card = parseCard({
    cid, open: true,
    card: {
      sel: 4, thumb: "data:image/jpeg;base64,AAAA", question: "price of this", translate_to: "de",
      hello: { eyes: gemini, eyes_why: "", ask: true, search: { word: "google", label: "Google", image: true, text: true }, langs: [{ code: "de", name: "German" }] },
      local: { text: "Preis", lines: 1, actions: [{ id: "a1", kind: "copy_text", label: "Copy text" }], sel: 99 },
      answer: { state: "done", title: "A price tag", actions: [] },
      links: { rows: [{ id: "a3", title: "Shop", host: "shop.example" }, { id: "x", title: "bad", host: "h" }] },
      thread: [{ qid: 2, q: "how much?", state: "done", text: "Ten euros." }, { qid: "x" }],
      translation: { to: "de", text: "Preis" },
      sent_to: [{ label: "Gemini", local: false, what: "picture" }, { label: "?", what: "evil" }],
    },
  });
  assert.ok(card && card.open && card.card);
  const c = card.card!;
  assert.equal(c.sel, 4);
  assert.equal(c.local?.sel, 4, "every part is the card's own area");
  assert.equal(c.answer?.title, "A price tag");
  assert.deepEqual(c.links.map((l) => l.id), ["a3"]);
  assert.deepEqual(c.thread.map((t) => [t.qid, t.q, t.text]), [[2, "how much?", "Ten euros."]]);
  assert.equal(c.translation?.text, "Preis");
  assert.deepEqual(c.sentTo, [{ label: "Gemini", local: false, what: "picture" }]);
  assert.equal(c.hello.search?.label, "Google");
  assert.equal(c.translateTo, "de");
  assert.equal(parseCard({ cid: cid.toUpperCase(), open: true, card: {} }), null);
  assert.deepEqual(parseCard({ cid, open: false, why: "<script>" }), { cid, open: false, why: "done" });
  assert.deepEqual(parseCard({ cid, open: false, why: "away" }), { cid, open: false, why: "away" });
  assert.equal(parseCard({ cid, open: true, card: { sel: 4, thumb: "https://evil.example/x.png" } })?.open && (parseCard({ cid, open: true, card: { sel: 4, thumb: "https://evil.example/x.png" } }) as { card: { thumb: string | null } }).card.thumb, null);
});

test("the card's commands each name its cid, within the protocol's limits", () => {
  const cid = "f".repeat(32);
  assert.deepEqual(cardSend.hello(), { cmd: "lens_card_hello" });
  assert.deepEqual(cardSend.ask(cid, 3, "  a   b "), { cmd: "lens_card_ask", cid, qid: 3, q: "a b" });
  assert.equal((cardSend.ask(cid, 3, "q".repeat(900)) as { q: string }).q.length, 500);
  assert.deepEqual(cardSend.translate(cid, "en"), { cmd: "lens_card_translate", cid, to: "en" });
  assert.deepEqual(cardSend.act(cid, "a4"), { cmd: "lens_card_act", cid, id: "a4" });
  assert.equal((cardSend.cont(cid, "t".repeat(3000)) as { text: string }).text.length, 2000);
  assert.deepEqual(cardSend.listen(cid), { cmd: "lens_card_listen", cid });
  assert.deepEqual(cardSend.close(cid, "esc"), { cmd: "lens_card_close", cid, why: "esc" });
});

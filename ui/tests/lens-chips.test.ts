// The sheet's chips (src/lens/chips.ts): on-device first in one order, then Search with Google,
// then the answer's search, then Brains & Keys; at most six, then More; a warning keeps its mark;
// a provider that takes no pictures has no picture search; Translate never into the text's own
// language.
import { test } from "node:test";
import assert from "node:assert/strict";
import { MAX_CHIPS, chipsFor, doneText, langName, translateTarget } from "../src/lens/chips.ts";
import type { Action, AnswerData, LensHello, LocalData } from "../src/lens/wire.ts";

const A = (id: number, kind: Action["kind"], label: string, extra: Partial<Action> = {}): Action => ({ id: `a${id}`, kind, label, ...extra });
const local = (actions: Action[], over: Partial<LocalData> = {}): LocalData => ({
  sel: 1, text: "Bonjour tout le monde", lines: 1, lang: { code: "fr", name: "French", conf: 0.95 }, translateTo: "en", dark: false, actions, ...over,
});
const hello = (image = true): LensHello => ({
  v: 1, open: true, frozen: "ok", display: null, via: "hold", perms: { screen: true }, eyes: null, eyesWhy: "", ask: true,
  search: { word: "google", label: "Google", image, text: true }, langs: [{ code: "en", name: "English" }, { code: "hi", name: "Hindi" }], prefill: null,
});
const answer = (actions: Action[]): AnswerData => ({ sel: 1, state: "done", actions });

test("on-device chips first, in one order, then Search with Google and the answer's search", () => {
  const l = local([A(1, "copy_image", "Copy image"), A(2, "image_search", "Search with Google"), A(3, "open", "Open example.com", { detail: "example.com" }), A(4, "copy_text", "Copy text")]);
  const { shown, more } = chipsFor(l, answer([A(5, "text_search", "Search “croissant”")]), hello());
  assert.deepEqual(shown.map((c) => c.kind), ["copy_text", "translate", "open", "copy_image", "image_search", "text_search"]);
  assert.deepEqual(more, []);
  assert.equal(shown[1].label, "Translate to English");
  assert.equal(shown[2].detail, "example.com");
  assert.ok(shown.find((c) => c.kind === "image_search")?.closes);
  assert.ok(!shown.find((c) => c.kind === "copy_text")?.closes);
});

test("at most six, then More", () => {
  const acts = [A(1, "copy_text", "Copy text"), A(2, "open", "Open a.com"), A(3, "open", "Open b.com"), A(4, "mail", "Email x@y.z"), A(5, "call", "Call 1"), A(6, "wifi", "Copy Wi-Fi password"), A(7, "code", "Copy code"), A(8, "copy_image", "Copy image"), A(9, "image_search", "Search with Google")];
  const { shown, more } = chipsFor(local(acts), null, hello());
  assert.equal(shown.length, MAX_CHIPS);
  assert.deepEqual(more.map((c) => c.kind), ["wifi", "code", "copy_image", "image_search"]);
});

test("a warning stays with its chip, and a provider without pictures gets no picture search", () => {
  const { shown } = chipsFor(local([A(1, "open", "Open xn--pple-43d.com", { warn: "punycode", detail: "xn--pple-43d.com" }), A(2, "image_search", "Search with Bing")]), null, hello(false));
  assert.equal(shown.find((c) => c.kind === "open")?.warn, "punycode");
  assert.ok(!shown.some((c) => c.kind === "image_search"));
});

test("Translate goes into the user's choice, else the core's, never the text's own language", () => {
  assert.equal(translateTarget(local([]), null), "en");
  assert.equal(translateTarget(local([]), "hi"), "hi");
  assert.equal(translateTarget(local([]), "fr-CA"), "en"); // that's the text's own language
  assert.equal(translateTarget(local([], { translateTo: "" }), null), null);
  assert.equal(translateTarget(local([], { text: "  " }), "en"), null);
  assert.equal(translateTarget(null, "en"), null);
  assert.ok(!chipsFor(local([], { lang: { code: "en", name: "English", conf: 1 }, translateTo: "en" }), null, hello()).shown.some((c) => c.kind === "translate"));
  assert.equal(langName("hi", hello().langs), "Hindi");
  assert.equal(langName("en-GB", hello().langs), "English");
});

test("what a finished action says for a moment", () => {
  assert.equal(doneText("copied", true), "Copied");
  assert.equal(doneText("no_browser", false), "No browser to open it in");
  assert.equal(doneText("", false), "Couldn't do that");
});

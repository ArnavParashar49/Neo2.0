// Esc folds NEO's chat as "−" does (src/esc.ts): wherever the page's focus is, after everything on
// the page with an Esc of its own; and — more carefully — when the user's app had the key and
// NEO.app's shell passed it on. The wiring in App.tsx and ws.ts is read from the source.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ESC_SETTLE_MS, awayEscFolds, escFolds, plainEsc, wantsAwayEsc, type AwayEsc, type ChatEsc, type EscKey } from "../src/esc.ts";

const key = (over: Partial<EscKey> = {}): EscKey => ({
  key: "Escape", isComposing: false, keyCode: 27, repeat: false, metaKey: false, ctrlKey: false, altKey: false, defaultPrevented: false, ...over,
});
const chat = (over: Partial<ChatEsc> = {}): ChatEsc => ({ open: true, confirming: false, erasing: false, lensCard: false, ...over });
const away = (over: Partial<AwayEsc> = {}): AwayEsc => ({
  ...chat(), focused: false, pinned: false, holds: false, openForMs: 5_000, ...over,
});
const src = (rel: string) => readFileSync(new URL(`../src/${rel}`, import.meta.url), "utf8");

test("a plain Esc: not a shortcut, not the input method's, not held down (⇧ changes nothing)", () => {
  assert.equal(plainEsc(key()), true);
  assert.equal(plainEsc(key({ key: "Escape", altKey: false, metaKey: false, ctrlKey: false })), true);
  assert.equal(plainEsc({ ...key(), shiftKey: true } as EscKey), true);
  for (const [why, k] of [
    ["another key", key({ key: "Enter" })],
    ["a key named like it", key({ key: "Esc" })],
    ["⌘Esc", key({ metaKey: true })],
    ["⌥Esc", key({ altKey: true })],
    ["⌃Esc", key({ ctrlKey: true })],
    ["an input method composing", key({ isComposing: true })],
    ["the input method's key (229)", key({ keyCode: 229 })],
    ["held down", key({ repeat: true })],
  ] as const) {
    assert.equal(plainEsc(k), false, why);
  }
});

test("on the page, Esc folds the open chat wherever the focus is", () => {
  assert.equal(escFolds(key(), chat()), true);
  // Nothing to fold: the orb (NEO's window, collapsed).
  assert.equal(escFolds(key(), chat({ open: false })), false);
});

test("…after the page's own Esc: a question says No, the erase dialog cancels, Search Screen's card closes", () => {
  assert.equal(escFolds(key(), chat({ confirming: true })), false);
  assert.equal(escFolds(key(), chat({ erasing: true })), false);
  assert.equal(escFolds(key(), chat({ lensCard: true })), false);
  // What prevented the key's default kept it: the browser card handing it to the page the user is
  // driving, a dialog or list that took it.
  assert.equal(escFolds(key({ defaultPrevented: true }), chat()), false);
  // An input method composing keeps its Esc (it cancels the composition).
  assert.equal(escFolds(key({ isComposing: true }), chat()), false);
  assert.equal(escFolds(key({ keyCode: 229 }), chat()), false);
});

test("an Esc the user's app got folds a plain chat on screen", () => {
  assert.equal(awayEscFolds(away()), true);
  assert.equal(awayEscFolds(away({ openForMs: ESC_SETTLE_MS })), true);
});

test("…never one that isn't up, or a key the page had itself", () => {
  assert.equal(awayEscFolds(away({ open: false })), false);
  assert.equal(awayEscFolds(away({ focused: true })), false); // (escFolds had it already)
});

test("…never over a question, the erase dialog, Search Screen's card, a pinned chat or one that must stay in sight", () => {
  for (const [why, s] of [
    ["a question waiting for an OK", away({ confirming: true })],
    ["the erase dialog", away({ erasing: true })],
    ["Search Screen's card", away({ lensCard: true })],
    ["pinned open", away({ pinned: true })],
    ["a login page, the camera, a song, the browser on stage", away({ holds: true })],
  ] as const) {
    assert.equal(awayEscFolds(s), false, why);
  }
});

test("…and never one that came up a moment ago: that Esc was meant for what was there before", () => {
  assert.equal(awayEscFolds(away({ openForMs: 0 })), false);
  assert.equal(awayEscFolds(away({ openForMs: ESC_SETTLE_MS - 1 })), false);
  assert.ok(ESC_SETTLE_MS >= 200 && ESC_SETTLE_MS <= 600);
});

test("the notch's page asks for Esc pressed elsewhere only while it is on screen without the keys", () => {
  assert.equal(wantsAwayEsc({ visible: true, focused: false }), true);
  assert.equal(wantsAwayEsc({ visible: true, focused: true }), false); // the key reaches it (or NotchHub)
  assert.equal(wantsAwayEsc({ visible: false, focused: false }), false); // the panel is closed
  assert.equal(wantsAwayEsc({ visible: false, focused: true }), false);
});

test("App.tsx: one window-level Esc for the page, the input's own gone, the shell's Esc judged by awayEscFolds", () => {
  const app = src("App.tsx");
  const onKey = app.split("const onKey = (e: React.KeyboardEvent) => {")[1].split("\n  };")[0];
  assert.ok(!onKey.includes("Escape"), "the input folds through the window's listener, IME-safe");
  const esc = app.split("// ---- Esc (src/esc.ts)")[1];
  assert.ok(esc, "the Esc section");
  // The page's key: on the window, bubbling (no capture flag), named guards, then the fold.
  assert.match(esc, /escFolds\(e, \{ open: expanded, confirming, erasing: erasing !== false, lensCard: lensLive \}\)\) return;\s+e\.preventDefault\(\);\s+collapse\(\);/);
  assert.match(esc, /window\.addEventListener\("keydown", h\);/);
  // The shell's: NEO's window hears `neo://esc`, the notch the core's `esc` (ws.ts' escAt).
  assert.match(esc, /listen\("neo:\/\/esc", \(\) => awayEsc\.current\(\)\)/);
  assert.match(esc, /if \(neo\.escAt\) awayEsc\.current\(\);/);
  assert.match(esc, /if \(folds\) collapse\(\);/);
  // What holds a chat against an Esc from elsewhere.
  for (const hold of ["pinned", "chatMustShow", "camLive && !csnap.popped", "songHearing", "browsingHolds"]) {
    assert.ok(esc.split("awayEscFolds({")[1].split("});")[0].includes(hold), hold);
  }
  // The notch's page says when it wants it, and again after each (re)connect.
  assert.match(esc, /neo\.send\(\{ cmd: "esc_watch", on \}\)/);
  assert.match(esc, /\}, \[neo\.helloAt, neo\.send\]\);/);
  // The confirm card's own Esc is untouched: still a No.
  assert.match(app, /if \(e\.key === "Escape"\) \{ e\.preventDefault\(\); answer\(false\); \}/);
  // "−" still collapses, its title still says Esc.
  assert.equal(app.split('title="Collapse (Esc)" aria-label="Collapse" onClick={collapse}').length - 1, 3);
});

test("ws.ts: the core's `esc` reaches the page as escAt", () => {
  const ws = src("ws.ts");
  assert.match(ws, /case "esc":\s+setEscAt\(Date\.now\(\)\);\s+break;/);
  assert.match(ws, /lensOpeningAt, escAt, uiHost/);
});

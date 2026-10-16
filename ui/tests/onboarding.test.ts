// "Give NEO a brain" (src/onboard.ts, src/Onboarding.tsx) in NEO's window and the notch: it shows
// while the core says NEO has no model, its button asks the core (never the shell) to open Brains &
// Keys, a core with no NEO.app gets the terminal command, and its words are the core's. The chat
// names a brain by the core's label (no more guessing from model ids), and the composer keeps a
// pasted key from every model.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ALSO, BODY, COMMAND, GEMINI_KEY_URL, GET_KEY, SET_UP, TITLE, afterOpenKeys, afterSetupNeeded, cardNote, isNoModelProblem, needsBrain } from "../src/onboard.ts";
import { show } from "../src/keys/catalog.ts";
import type { BrainsSummary } from "../src/keys/types.ts";

const ROOT = new URL("../../", import.meta.url);
const read = (rel: string) => readFileSync(new URL(rel, ROOT), "utf8");
const role = (label: string, ok: boolean) => ({ ref: "", provider: "", label, ok });
const summary = (ready: boolean): BrainsSummary => ({
  ready, brain: role("Not set up", false), quick: role("Not set up", false), eyes: null, voice: "local", providers: {}, restart_needed: false,
});

test("the card shows while connected and the core says NEO has no model — never for an older core", () => {
  assert.equal(needsBrain(true, summary(false)), true);
  assert.equal(needsBrain(true, summary(true)), false);
  assert.equal(needsBrain(false, summary(false)), false, "offline: the reconnect line says so");
  assert.equal(needsBrain(true, null), false, "an older core sends no brains: no card");
  assert.equal(needsBrain(true, undefined), false);
});

test("its words are the design's, and its command and link are the core's", () => {
  assert.equal(TITLE, "Give NEO a brain");
  assert.equal(BODY, "NEO already does simple things on this Mac (volume, apps, timers). To chat and do tasks, connect an AI model. Gemini has a free tier, no card needed.");
  assert.equal(ALSO, "Also works with Claude, OpenAI, Groq (free), OpenRouter, or a model on this Mac.");
  assert.equal(SET_UP, "Set up a brain");
  assert.equal(GET_KEY, "Get a free Gemini key");
  assert.ok(read("neo2/core/onboarding.py").includes(`COMMAND = "${COMMAND}"`));
  assert.equal(GEMINI_KEY_URL, show("gemini")?.keyUrl);
  // The health line the card says better is hidden while the card is up.
  const health = read("neo2/core/health.py");
  const line = /NO_MODEL = "([^"]+)"/.exec(health)?.[1] ?? "";
  assert.ok(isNoModelProblem(line), line);
  assert.equal(isNoModelProblem("Microphone access is off for NEO."), false);
});

test("after open_keys: opened, already opening, or the terminal command when there's no NEO.app", () => {
  assert.equal(afterOpenKeys(null), "idle");
  assert.equal(afterOpenKeys({ ok: true }), "opened");
  assert.equal(afterOpenKeys({ ok: false, error: "busy" }), "busy");
  assert.equal(afterOpenKeys({ ok: false, error: "no_app" }), "terminal");
  assert.equal(afterOpenKeys({ ok: false, error: "forbidden" }), "terminal");
  assert.equal(afterSetupNeeded({ shell: true, opened: true, command: COMMAND }), "opened");
  assert.equal(afterSetupNeeded({ shell: true, opened: false, command: COMMAND }), "idle", "opened in the last 10 minutes: the button is there");
  assert.equal(afterSetupNeeded({ shell: false, opened: false, command: COMMAND }), "terminal");
  assert.equal(cardNote("terminal"), `In a terminal, run: ${COMMAND}`);
  assert.equal(cardNote("idle"), null);
  assert.match(cardNote("opened") ?? "", /Brains & Keys is open/);
});

test("the card asks the core, never the shell: one command, open_keys, and the page's own link opener", () => {
  const card = read("ui/src/Onboarding.tsx");
  assert.ok(card.includes('send({ cmd: "open_keys" })'));
  assert.ok(card.includes("openExternal(GEMINI_KEY_URL)"));
  assert.ok(!/invoke\(|@tauri-apps|shellAction|keys_ticket/.test(card), "no door to NEO.app's shell from the chat page");
  const shell = read("ui/src/shell.ts");
  assert.ok(!/open_keys|keys_ticket|edit_env/.test(shell), "the page's shell shim gains nothing");
});

test("the chat shows the card in place of the empty state, and hides the line it replaces", () => {
  const app = read("ui/src/App.tsx");
  assert.ok(app.includes("const noBrain = needsBrain(neo.connected, neo.brains);"));
  assert.ok(app.includes("{!noBrain && visibleItems.length === 0"), "the suggestions give way to the card");
  assert.ok(app.includes("noBrain && !showStrip && !live && visibleItems.length === 0 && <Onboarding"));
  assert.ok(app.includes("!(noBrain && isNoModelProblem(p))"));
  const ws = read("ui/src/ws.ts");
  for (const t of ['case "brains":', 'case "setup_needed":', 'case "open_keys_result":', "setBrains(ev.data.brains"]) assert.ok(ws.includes(t), t);
});

test("a turn's brain is the core's label: the model-id guessing is gone", () => {
  const app = read("ui/src/App.tsx");
  assert.ok(!app.includes("function brainName("));
  assert.ok(!/replace\(\/\^claude-\/|Groq gpt-oss-120b|"Cerebras " \+/.test(app));
  assert.ok(app.includes("return m.label || m.model || m.brain"));
  assert.ok(read("ui/src/ws.ts").includes("label?: string"));
  assert.ok(read("neo2/core/agent/session.py").includes("label=brain_label("), "the core sends it");
});

test("the composer keeps a pasted key from every model, and from ↑", () => {
  const app = read("ui/src/App.tsx");
  const submit = app.split("const submit = (t = text) => {")[1].split("\n  };\n")[0];
  const guard = submit.indexOf("if (looksLikeKey(v))");
  assert.ok(guard > 0);
  assert.ok(guard < submit.indexOf("lastInput.current = v"), "checked before ↑ remembers it");
  assert.ok(guard < submit.indexOf('neo.send({ cmd: "text"'), "checked before anything is sent");
  assert.ok(/if \(looksLikeKey\(v\)\) \{ setText\(""\); setKeyGuard\(true\); return; \}/.test(submit), "the input is cleared, nothing is sent");
});

test("index.html#/keys is Brains & Keys, and only it", () => {
  const main = read("ui/src/main.tsx");
  assert.ok(main.includes('const keys = location.hash === "#/keys";'));
  assert.ok(main.includes("{keys ? <KeysApp />"));
  const keysApp = read("ui/src/keys/KeysApp.tsx");
  assert.ok(!/localStorage|sessionStorage|indexedDB/.test(keysApp + read("ui/src/keys/wire.ts")), "nothing kept in the browser's storage");
  assert.ok(!/console\.(log|info|debug)/.test(keysApp + read("ui/src/keys/wire.ts")), "nothing printed");
  // Every key field is a password field the browser won't remember, cleared as it is sent.
  const fields = keysApp.split('<input type="password"').slice(1).map((f) => f.split("/>")[0]);
  assert.ok(fields.length >= 4, String(fields.length));
  for (const f of fields) assert.ok(f.includes("{...SECRET_INPUT}"), f);
  assert.ok(keysApp.includes('const SECRET_INPUT = { autoComplete: "off", spellCheck: false, autoCapitalize: "off"'));
  for (const clear of ['setValue(""); // dropped from the page as it goes', 'setValue(""); setExtra(""); // dropped from the page as they go', 'setKey(""); // dropped from the page as it goes']) {
    assert.ok(keysApp.includes(clear), clear);
  }
});

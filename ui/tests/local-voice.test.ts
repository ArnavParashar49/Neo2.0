// The on-device voice (src/keys/localvoice.ts): NEO downloads it only when a button asks, the chat's
// "Give NEO a brain" card offers it only while NEO has no voice it could use, Brains & Keys shows it
// always, and both say how far the download is from what the core said — never guessing.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  DOWNLOAD_CMD, downloadErrorText, downloadLabel, etaText, localVoiceOf, offerOnCard, percent, sizeText, statusLine, withLocalVoice,
} from "../src/keys/localvoice.ts";
import { RESULT_OF } from "../src/keys/wire.ts";
import { VOICE_LINE, isVoiceSetupProblem } from "../src/onboard.ts";
import type { BrainsSummary, LocalVoice } from "../src/keys/types.ts";

const ROOT = new URL("../../", import.meta.url);
const read = (rel: string) => readFileSync(new URL(rel, ROOT), "utf8");
const role = (label: string, ok: boolean) => ({ ref: "", provider: "", label, ok });
const SIZE = 2_898_977_287;
const summary = (lv: LocalVoice | null | undefined, gemini = false): BrainsSummary => ({
  ready: false, brain: role("Not set up", false), quick: role("Not set up", false), eyes: null, voice: "local",
  providers: { gemini }, restart_needed: false, ...(lv === undefined ? {} : { local_voice: lv }),
});

test("sizes as Finder counts them, and the button names the download", () => {
  assert.equal(sizeText(SIZE), "2.9 GB");
  assert.equal(sizeText(160_000_000), "160 MB");
  assert.equal(sizeText(0), "");
  assert.equal(downloadLabel({ state: "missing", size: SIZE }), "Download on-device voice (2.9 GB)");
  assert.equal(downloadLabel(null), "Download on-device voice");
});

test("the card offers it only while there's no voice NEO could use — never for an older core", () => {
  assert.equal(offerOnCard(summary({ state: "missing", size: SIZE })), true);
  assert.equal(offerOnCard(summary({ state: "downloading", size: SIZE, done: 1, total: SIZE })), true, "it shows how far");
  assert.equal(offerOnCard(summary({ state: "failed", size: SIZE, code: "offline" })), true, "and why it stopped");
  assert.equal(offerOnCard(summary({ state: "ready", size: SIZE })), false);
  assert.equal(offerOnCard(summary({ state: "missing", size: SIZE }, true)), false, "a Gemini key: Gemini Live is the voice");
  assert.equal(offerOnCard(summary(undefined)), false, "an older core doesn't say: nothing offered");
  assert.equal(offerOnCard(null), false);
  assert.equal(localVoiceOf(summary(null)), null);
});

test("how far it is, in words, from the core's own numbers", () => {
  const lv: LocalVoice = { state: "downloading", size: SIZE, done: SIZE / 4, total: SIZE, eta_s: 150 };
  assert.equal(percent(lv), 25);
  assert.equal(statusLine(lv), "Downloading the on-device voice… 25% of 2.9 GB · about 3 minutes left");
  assert.equal(percent({ state: "missing", size: SIZE }), null);
  assert.equal(etaText(30), "less than a minute left");
  assert.equal(etaText(70), "about a minute left");
  assert.equal(etaText(null), "");
  assert.equal(statusLine({ state: "downloading", size: SIZE, done: 0, total: 0 }), "Downloading the on-device voice… 0% of 2.9 GB");
  assert.equal(statusLine({ state: "missing", size: SIZE }), null);
  assert.match(statusLine({ state: "ready", size: SIZE }) ?? "", /^Downloaded\./);
  assert.match(statusLine({ state: "failed", size: SIZE, code: "offline" }) ?? "", /Hugging Face/);
  assert.match(statusLine({ state: "failed", size: SIZE, code: "disk_full" }) ?? "", /free space/);
  assert.equal(statusLine({ state: "failed", size: SIZE, code: "weird" }), "The download didn't finish. Try again.");
});

test("a local_voice event or a voice_download answer updates the summary, and nothing else does", () => {
  const b = summary({ state: "missing", size: SIZE });
  const after = withLocalVoice(b, { state: "downloading", size: SIZE, done: 5, total: SIZE });
  assert.equal(after?.local_voice?.state, "downloading");
  assert.equal(after?.ready, false);
  assert.equal(withLocalVoice(b, null), b);
  assert.equal(withLocalVoice(b, { nope: 1 }), b);
  assert.equal(withLocalVoice(null, { state: "ready", size: 1 }), null, "no summary yet: nothing to fold it into");
});

test("both windows send voice_download, the core's command for it, and say its refusals", () => {
  assert.equal(DOWNLOAD_CMD, "voice_download");
  assert.equal(RESULT_OF.voice_download, "voice_download_result");
  const ws = read("neo2/core/server/ws.py");
  assert.ok(ws.includes('kind == "voice_download"') && ws.includes('"voice_download_result"'));
  assert.ok(read("ui/src/Onboarding.tsx").includes("cmd: DOWNLOAD_CMD"));
  assert.ok(read("ui/src/keys/KeysApp.tsx").includes("ask(DOWNLOAD_CMD"));
  assert.equal(downloadErrorText("busy"), "It's downloading already.");
  assert.equal(downloadErrorText("forbidden"), "This window can't start downloads.");
  assert.match(VOICE_LINE, /^No Gemini key\?/);
  // The size the windows show is the core's (fetch.LOCAL_VOICE_BYTES), not one written here.
  assert.ok(!/2[.,]?[69] ?GB/.test(read("ui/src/Onboarding.tsx") + read("ui/src/keys/KeysApp.tsx")));
});

test("the voice's own health line is the one the card says better", () => {
  const pipeline = read("neo2/core/voice/pipeline.py");
  const line = /problem = \(f"(NEO's voice needs[^"]+)"/.exec(pipeline)?.[1] ?? "";
  assert.ok(line && isVoiceSetupProblem(line), line);
  assert.equal(isVoiceSetupProblem("The on-device voice isn't downloaded yet (2.9 GB)."), false, "chosen on purpose: kept");
  assert.ok(read("ui/src/App.tsx").includes("offerOnCard(neo.brains) && isVoiceSetupProblem(p)"));
});

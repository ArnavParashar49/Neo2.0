// The window's key shapes (src/keys/keyshape.ts) are the core's (neo2/core/keyshape.py): the same
// prefixes, the same soft hints, the same narrow composer guard — the cases below are the core's
// own (tests/test_keycheck.py). And the only pages the window may open are the catalogue's key
// pages, exactly what its capability allows. Fake keys only.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { EXPECTED, KEY_TYPED, PREFIXES, cleanKey, guess, hint, keyish, looksLikeKey } from "../src/keys/keyshape.ts";
import { KEYED, PROVIDERS, QUICK, TEMPLATE_KEY_URLS, brainName, costBadge, keyPage, keyPages, modelOf, providerOf, refFor } from "../src/keys/catalog.ts";

const ROOT = new URL("../../", import.meta.url);
const read = (rel: string) => readFileSync(new URL(rel, ROOT), "utf8");

const KEYS = {
  gemini: "AIzaTEST-gemini-0000-abcdefgh",
  anthropic: "sk-ant-TEST-0000-abcdefghijkl",
  openai: "sk-proj-TEST-0000-abcdefghijkl",
  groq: "gsk_TEST-groq-0000-abcdefghij",
  cerebras: "csk-TEST-cerebras-0000-abcdef",
  openrouter: "sk-or-TEST-0000-abcdefghijklm",
};
const DS_KEY = "sk-TEST-deepseek-0000-abcdefgh";

test("the prefixes are the core's, in the core's order", () => {
  const py = read("neo2/core/keyshape.py");
  const block = py.split("PREFIXES: tuple[tuple[str, str], ...] = (")[1].split("\n)")[0];
  const theirs = [...block.matchAll(/\("([^"]+)", "([^"]+)"\)/g)].map((m) => [m[1], m[2]]);
  assert.deepEqual(PREFIXES.map(([p, id]) => [p, id]), theirs);
  const expected = py.split("EXPECTED: dict[str, tuple[str, ...]] = {")[1].split("\n}")[0];
  for (const [id, prefixes] of Object.entries(EXPECTED)) {
    assert.ok(expected.includes(`"${id}": (${prefixes.map((p) => `"${p}"`).join(", ")}${prefixes.length === 1 ? "," : ""})`), id);
  }
  assert.ok(read("neo2/core/onboarding.py").includes(`KEY_TYPED = "${KEY_TYPED}"`), "the composer says what the core says");
});

test("shape hints never block and name the likely provider", () => {
  assert.equal(hint("anthropic", KEYS.openai), "This looks like an OpenAI key — did you mean OpenAI?");
  assert.equal(hint("openai", KEYS.anthropic), "This looks like an Anthropic key — did you mean Anthropic?");
  assert.equal(hint("groq", KEYS.gemini), "This looks like a Gemini key — did you mean Gemini?");
  assert.equal(hint("gemini", "nope-0000"), "Gemini keys usually start with AIza.");
  assert.equal(hint("openai", "sk-legacy-0000"), null); // OpenAI's older shape
  assert.equal(hint("custom:deepseek", DS_KEY), null); // "sk-" alone names nobody
  assert.equal(hint("custom:xai", "xai-TEST-0000"), null);
  assert.equal(hint("anthropic", `  "${KEYS.anthropic}" `), null);
  assert.equal(hint("anthropic", ""), null);
  assert.equal(guess(KEYS.openrouter), "openrouter");
  assert.equal(guess("hello"), null);
});

test("a key is kept as the core keeps it: quotes and spaces around it go; breaks, spaces in it and length don't", () => {
  assert.deepEqual(cleanKey(KEYS.openai), { ok: true, key: KEYS.openai });
  assert.deepEqual(cleanKey(`  '${KEYS.openai}'\n`), { ok: true, key: KEYS.openai });
  for (const bad of ["sk-proj-a\nb", "sk-proj a", "x".repeat(513), "   ", "sk-TEST\u0007", "sk-TEST x", "sk x"]) {
    const r = cleanKey(bad);
    assert.equal(r.ok, false, JSON.stringify(bad));
    if (!r.ok) assert.ok(!r.problem.includes("TEST") && !r.problem.includes("sk-"), "the words never say the key");
  }
  assert.equal(cleanKey("x".repeat(512)).ok, true);
});

test("the composer guard is narrow: known prefixes with a key's body, never a hash or a word", () => {
  for (const text of [KEYS.anthropic, `${KEYS.openai} please`, KEYS.gemini, KEYS.groq, "xai-" + "a".repeat(40), "sk-" + "A1b2".repeat(12), `"${KEYS.openrouter}"`]) {
    assert.ok(looksLikeKey(text), text);
  }
  for (const text of ["sk-learn is a library", "AIza is a prefix", "my key is gsk_short", "deadbeef".repeat(8), "", "skip it", "sk-" + "a".repeat(20), `my key is ${KEYS.anthropic}`]) {
    assert.ok(!looksLikeKey(text), text);
  }
});

test("the window opens only the catalogue's key pages, and its capability allows exactly those", () => {
  const cap = JSON.parse(read("ui/src-tauri/capabilities/keys.json"));
  assert.deepEqual(cap.windows, ["keys"]);
  const opener = cap.permissions.find((p: unknown) => typeof p === "object" && (p as { identifier?: string }).identifier === "opener:allow-open-url");
  const allowed = (opener.allow as { url: string }[]).map((a) => a.url).sort();
  assert.deepEqual(allowed, [...keyPages()].sort());
  for (const u of allowed) assert.match(u, /^https:\/\/[a-z0-9.-]+(\/[A-Za-z0-9/_-]*)?$/, "an exact https page, no wildcard");
  // Nothing else of the overlay's powers: no webviews of its own, no shell door, no hiding NEO.
  const names = cap.permissions.filter((p: unknown) => typeof p === "string") as string[];
  for (const n of names) assert.ok(!/create-webview|global-shortcut|opener:default|set-position|set-size|hide/.test(n), n);
  assert.equal(keyPage("https://console.anthropic.com/settings/keys"), "https://console.anthropic.com/settings/keys");
  assert.equal(keyPage("https://example.com/"), null);
  assert.equal(keyPage(undefined), null);
});

test("the catalogue's key pages and hints are the core's own", () => {
  const py = read("neo2/core/providers/catalog.py");
  for (const p of PROVIDERS) {
    if (!p.keyUrl) continue;
    assert.ok(py.includes(`key_url="${p.keyUrl}"`), p.id);
    assert.ok(py.includes(`key_hint="${p.keyHint}"`), p.id);
    assert.ok(py.includes(`id="${p.id}"`), p.id);
  }
  for (const [t, url] of Object.entries(TEMPLATE_KEY_URLS)) assert.ok(py.includes(`"${t}"`) && py.includes(`key_url="${url}"`), t);
  assert.deepEqual(KEYED, ["gemini", "anthropic", "openai", "groq", "cerebras", "openrouter"]);
  assert.equal(QUICK[0].id, "gemini", "Gemini first (free, no card)");
  assert.equal(costBadge("free_tier").text, "Free tier");
  assert.equal(costBadge("paid").text, "Paid");
});

test("brain references: the provider, the model after @, and back", () => {
  assert.equal(providerOf("claude@claude-opus-5-5"), "anthropic");
  assert.equal(providerOf("anthropic"), "anthropic");
  assert.equal(providerOf("gemini_lite"), "gemini");
  assert.equal(providerOf("groq_vision"), "groq");
  assert.equal(providerOf("cerebras_vision"), "cerebras");
  assert.equal(providerOf("openrouter@deepseek/deepseek-v4-pro"), "openrouter");
  assert.equal(providerOf("custom:ollama@qwen3:14b"), "custom:ollama");
  assert.equal(providerOf("claud"), "");
  assert.equal(providerOf(""), "");
  assert.equal(modelOf("custom:ollama@qwen3:14b"), "qwen3:14b");
  assert.equal(modelOf("gemini"), null);
  assert.equal(brainName("anthropic"), "claude");
  assert.equal(refFor("anthropic", "claude-haiku-5-5"), "claude@claude-haiku-5-5");
  assert.equal(refFor("openrouter", "x-ai/grok-4.7"), "openrouter@x-ai/grok-4.7");
  assert.equal(refFor("custom:deepseek", null), "custom:deepseek");
});

test("a model id or a name shaped like a key is caught as the core catches it (keyshape.keyish)", () => {
  for (const text of ["sk-or-v1-" + "ab12".repeat(16), `custom:x@${KEYS.anthropic}`, "sk-" + "A1b2".repeat(10), "a1B2".repeat(9), `my ${KEYS.openrouter}`, `x/${KEYS.gemini}`]) {
    assert.ok(keyish(text), text);
  }
  for (const text of ["meta-llama/llama-4-maverick-17b-128e-instruct", "qwen3:14b", "deepseek/deepseek-v4-pro", "x-ai/grok-4",
    "mlx-community/gemma-4-12B-it-4bit", "ft:gpt-6-luna:my-org:custom:A1b2C3d4", "xai-org/grok-1", "Qwen/Qwen2.5-Coder-32B-Instruct",
    "openai/gpt-oss-120b:free", "deadbeef".repeat(8), "Claude Opus 5.5", "DeepSeek", ""]) {
    assert.ok(!keyish(text), text);
  }
});

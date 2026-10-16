// The Brains & Keys window's line to the core (src/keys/wire.ts) and what it sends
// (src/keys/model.ts), with a fake socket and fake timers: it connects only with the shell's
// ticket, each command carries its own id and its answer reaches whoever asked, a refusal is said
// as one, nothing is queued while the core is away, and no key is kept once it has been sent.
// Fake keys only; nothing here opens a socket.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { KeysWire, RESULT_OF, type SocketLike, type WireDeps } from "../src/keys/wire.ts";
import {
  budgetRows, budgetsAfter, endpointBody, endpointProblem, errorText, formFrom, formFromTemplate, lightFor, modelOptions,
  providerChoices, quickSetup, quickSetupWith, rolesChanges, shadowNote, sourceText, testText,
} from "../src/keys/model.ts";
import type { BrainsDetail } from "../src/keys/types.ts";

const TICKET = "ab".repeat(32);
const KEY = "sk-ant-TEST-0000-abcdefghijkl";

class FakeSocket implements SocketLike {
  readyState = 0;
  url: string;
  sent: string[] = [];
  onopen: SocketLike["onopen"] = null;
  onmessage: SocketLike["onmessage"] = null;
  onclose: SocketLike["onclose"] = null;
  onerror: SocketLike["onerror"] = null;
  constructor(url: string) { this.url = url; }
  open() { this.readyState = 1; this.onopen?.(); }
  recv(type: string, data: Record<string, unknown>) { this.onmessage?.({ data: JSON.stringify({ type, data, ts: 1 }) }); }
  send(d: string) { this.sent.push(d); }
  close() { if (this.readyState === 3) return; this.readyState = 3; this.onclose?.(); }
}

function harness(ticket: () => Promise<string> = async () => TICKET) {
  const sockets: FakeSocket[] = [];
  const timers: { fn: () => void; ms: number; live: boolean }[] = [];
  let tickets = 0;
  const deps: WireDeps = {
    ticket: async () => { tickets++; return ticket(); },
    token: async () => "tok-TEST",
    socket: (url) => { const s = new FakeSocket(url); sockets.push(s); return s; },
    later: (fn, ms) => { const t = { fn, ms, live: true }; timers.push(t); return t; },
    cancel: (h) => { (h as { live: boolean }).live = false; },
  };
  const wire = new KeysWire(deps);
  const tick = async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); };
  const run = async (ms?: number) => {
    for (const t of timers.filter((t) => t.live && (ms === undefined || t.ms === ms))) { t.live = false; t.fn(); }
    await tick();
  };
  return { wire, sockets, timers, run, tick, tickets: () => tickets };
}

async function opened() {
  const h = harness();
  h.wire.start();
  await h.tick();
  h.sockets[0].open();
  return h;
}

/** Everything reachable from `root` (objects, arrays, Maps, Sets) that is a string containing `needle`
 *  — past the socket itself (the test's record of what went out over it). */
function holds(root: unknown, needle: string, seen = new Set<unknown>()): boolean {
  if (typeof root === "string") return root.includes(needle);
  if (!root || typeof root !== "object" || seen.has(root) || root instanceof FakeSocket) return false;
  seen.add(root);
  if (root instanceof Map) return [...root.entries()].some(([k, v]) => holds(k, needle, seen) || holds(v, needle, seen));
  if (root instanceof Set) return [...root].some((v) => holds(v, needle, seen));
  return Object.values(root as Record<string, unknown>).some((v) => holds(v, needle, seen));
}

test("it connects with the token and the shell's ticket — and only with a ticket that is one", async () => {
  const h = await opened();
  assert.equal(h.sockets[0].url, `ws://127.0.0.1:8765/?token=tok-TEST&ticket=${TICKET}`);
  assert.equal(h.wire.status, "open");
  for (const bad of ["", "AB".repeat(32), "ab".repeat(31), `${"ab".repeat(32)}&role=host`]) {
    const x = harness(async () => bad);
    x.wire.start();
    await x.tick();
    assert.equal(x.sockets.length, 0, bad);
    assert.equal(x.wire.status, "forbidden");
  }
});

test("why the shell has no ticket decides what the window says, and whether it asks again", async () => {
  for (const [why, status, retries] of [["no_link", "no_link", true], ["no_core", "no_core", true], ["not_ready", "connecting", true], ["forbidden", "forbidden", false], ["command keys_ticket not found", "forbidden", false]] as const) {
    const h = harness(async () => { throw why; });
    h.wire.start();
    await h.tick();
    assert.equal(h.wire.status, status, why);
    assert.equal(h.sockets.length, 0);
    assert.equal(h.timers.some((t) => t.live), retries, why);
  }
  // A core started by hand, then NEO.app's own: the window gets in by itself.
  let first = true;
  const h = harness(async () => { if (first) { first = false; throw "no_link"; } return TICKET; });
  h.wire.start();
  await h.tick();
  await h.run();
  assert.equal(h.sockets.length, 1);
  assert.equal(h.tickets(), 2);
});

test("each command carries its own id, and each answer reaches whoever asked", async () => {
  const h = await opened();
  const a = h.wire.request("models_list", { provider: "openai" });
  const b = h.wire.request("brains_get");
  const sent = h.sockets[0].sent.map((s) => JSON.parse(s));
  assert.deepEqual(sent.map((m) => m.cmd), ["models_list", "brains_get"]);
  assert.equal(sent[0].provider, "openai");
  assert.notEqual(sent[0].id, sent[1].id);
  for (const m of sent) assert.match(m.id, /^[A-Za-z0-9_.:-]{1,32}$/);
  // Out of order, with noise: someone else's id, the wrong type, an event.
  h.sockets[0].recv("brains_detail", { id: "r999", catalog: [] });
  h.sockets[0].recv("key_test_result", { id: sent[1].id, ok: true });
  h.sockets[0].recv("brains_detail", { id: sent[1].id, catalog: [], keys: {} });
  h.sockets[0].recv("models_result", { id: sent[0].id, ok: true, models: [], defaults: {} });
  const [ra, rb] = await Promise.all([a, b]);
  assert.equal((ra as Record<string, unknown>).provider === undefined && ra.ok, true);
  assert.ok(Array.isArray((rb as Record<string, unknown>).catalog));
  assert.equal(h.wire.waiting, 0);
  // The body can't name another command or another id.
  void h.wire.request("key_remove", { cmd: "key_save", id: "mine", provider: "anthropic" });
  const last = JSON.parse(h.sockets[0].sent.at(-1)!);
  assert.equal(last.cmd, "key_remove");
  assert.notEqual(last.id, "mine");
  assert.equal(RESULT_OF.key_save, "key_save_result");
});

test("a refusal is said as one, to the command it refused", async () => {
  const h = await opened();
  const p = h.wire.request("key_save", { provider: "anthropic", key: KEY });
  const id = JSON.parse(h.sockets[0].sent[0]).id;
  h.sockets[0].recv("keys_refused", { id, cmd: "key_save", error: "forbidden" });
  const r = await p;
  assert.equal(r.ok, false);
  assert.equal(r.error, "forbidden");
  assert.equal(errorText(String(r.error)), "This window isn't allowed to change NEO's keys.");
  assert.deepEqual(await h.wire.request("not_a_command"), { ok: false, error: "unknown" }, "only the window's own commands go out");
  assert.equal(h.sockets[0].sent.length, 1);
});

test("no key is kept: not while its answer is awaited, not after", async () => {
  const h = await opened();
  const p = h.wire.request("key_save", { provider: "anthropic", key: KEY, test: true });
  assert.equal(h.wire.waiting, 1);
  assert.ok(!holds(h.wire, KEY), "the waiting command holds only its id");
  const sent = JSON.parse(h.sockets[0].sent[0]);
  assert.equal(sent.key, KEY, "it went out once, in the command that carries it");
  h.sockets[0].recv("key_save_result", { id: sent.id, provider: "anthropic", ok: true, key: { set: true, source: "saved", last4: "ijkl" } });
  const r = await p;
  assert.equal(r.ok, true);
  assert.ok(!holds(h.wire, KEY));
  assert.ok(!holds(r, KEY), "the answer carries the last 4 at most");
  assert.equal(h.wire.waiting, 0);
});

test("while the core is away nothing is queued: a key is never sent later", async () => {
  const h = harness();
  h.wire.start();
  await h.tick();
  const r = await h.wire.request("key_save", { provider: "anthropic", key: KEY });
  assert.deepEqual(r, { ok: false, error: "offline" });
  h.sockets[0].open();
  assert.deepEqual(h.sockets[0].sent, []);
  assert.ok(!holds(h.wire, KEY));
});

test("the core going away answers every waiting command, and the window connects again with a fresh ask for its ticket", async () => {
  const h = await opened();
  const p = h.wire.request("models_list", { provider: "gemini" });
  h.sockets[0].close();
  assert.deepEqual(await p, { id: "r1", ok: false, error: "disconnected" });
  assert.equal(h.wire.status, "connecting");
  await h.run();
  assert.equal(h.tickets(), 2, "the shell is asked again (it tells a restarted core the ticket)");
  assert.equal(h.sockets.length, 2);
  h.sockets[1].open();
  assert.equal(h.wire.status, "open");
});

test("a command the core never answers ends as a timeout", async () => {
  const h = await opened();
  const p = h.wire.request("key_test", { provider: "gemini" });
  await h.run(30_000);
  assert.equal((await p).error, "timeout");
  assert.equal(h.wire.waiting, 0);
});

test("the hello, brains, problems and state reach their listeners; stop ends it all", async () => {
  const h = await opened();
  const got: string[] = [];
  h.wire.on("hello", (d) => got.push(`hello:${d.brains?.ready}`));
  h.wire.on("brains", (d) => got.push(`brains:${d.ready}`));
  h.wire.on("problems", (d) => got.push(`problems:${d.length}`));
  const off = h.wire.on("state", (d) => got.push(`state:${d}`));
  h.sockets[0].recv("hello", { state: "idle", problems: [], brains: { ready: false } });
  h.sockets[0].recv("brains", { ready: true });
  h.sockets[0].recv("problems", { items: ["a", "b"] });
  h.sockets[0].recv("state", { state: "thinking" });
  off();
  h.sockets[0].recv("state", { state: "idle" });
  h.sockets[0].onmessage?.({ data: "not json" });
  assert.deepEqual(got, ["hello:false", "brains:true", "problems:2", "state:thinking"]);
  const p = h.wire.request("brains_get");
  h.wire.stop();
  assert.equal((await p).error, "disconnected");
  assert.equal(h.wire.status, "stopped");
  await h.run();
  assert.equal(h.sockets.length, 1, "no reconnect after stop");
});

// ---- what the window sends -------------------------------------------------------------------------

const DETAIL: BrainsDetail = {
  catalog: [
    { id: "gemini", label: "Google Gemini", transport: "gemini", cost: "free_tier", key_url: "", key_hint: "AIza", roles: ["brain", "quick", "eyes", "camera", "voice"], env: "GEMINI_API_KEY" },
    { id: "anthropic", label: "Anthropic (Claude)", transport: "anthropic", cost: "paid", key_url: "", key_hint: "sk-ant-", roles: ["brain", "quick", "eyes", "camera"], env: "ANTHROPIC_API_KEY" },
    { id: "openai", label: "OpenAI", transport: "openai_responses", cost: "paid", key_url: "", key_hint: "sk-proj-", roles: ["brain", "quick", "eyes", "camera"], env: "OPENAI_API_KEY" },
    { id: "groq", label: "Groq", transport: "openai_compat", cost: "free", key_url: "", key_hint: "gsk_", roles: ["brain", "quick", "eyes"], env: "GROQ_API_KEY" },
    { id: "local", label: "Gemma", transport: "local", cost: "this_mac", key_url: "", key_hint: "", roles: ["brain", "quick"], env: null },
  ],
  keys: {
    gemini: { set: true, source: "dotenv", last4: "efgh" },
    anthropic: { set: true, source: "saved", last4: "ijkl" },
    openai: { set: false, source: "none", last4: null },
    groq: { set: true, source: "env", last4: "ghij" },
    "custom:ollama": { set: false, source: "none", last4: null },
  },
  roles: {
    brain: { ref: "claude@claude-opus-5-5", valid: true, source: "saved" },
    fast_brain: { ref: "claude@claude-haiku-5-5", valid: true, source: "saved" },
    eyes_brain: { ref: "", valid: true, source: "default" },
    light_brain: { ref: "", valid: true, source: "saved" },
    offline_brain: { ref: "local", valid: true, source: "default" },
    backup_brains: { refs: null, source: "default" },
  },
  custom: [{ id: "custom:ollama", label: "Ollama", template: "ollama", base_url: "http://localhost:11434/v1", model: "qwen3:14b", tools: true, vision: false, context: 32768, on_device: true, reachable: true, saved: true }],
  voice: { value: "auto", source: "default", live_possible: true },
  cloud_eyes: { value: "on", source: "default" },
  daily_budgets: { value: { anthropic: 2 }, source: "saved", prefill: {} },
  store: { path: "~/.neo/api_keys.json", ok: true },
};

test("brains_set gets only what changed; one-step commands follow Quick replies off Gemini", () => {
  const was = formFrom(DETAIL);
  assert.deepEqual(rolesChanges(was, DETAIL), {});
  assert.deepEqual(rolesChanges({ ...was, brain: "openai@gpt-6.1-sol" }, DETAIL), { brain: "openai@gpt-6.1-sol" });
  assert.deepEqual(rolesChanges({ ...was, fast_brain: "gemini_lite" }, DETAIL), { fast_brain: "gemini_lite", light_brain: "gemini_lite" });
  assert.deepEqual(rolesChanges({ ...was, fast_brain: "groq" }, DETAIL), { fast_brain: "groq", light_brain: "" });
  assert.deepEqual(rolesChanges({ ...was, cloud_eyes: "auto" }, DETAIL), {}, "on and auto both let others look");
  assert.deepEqual(rolesChanges({ ...was, cloud_eyes: "off" }, DETAIL), { cloud_eyes: "off" });
  assert.deepEqual(rolesChanges({ ...was, backup_brains: ["groq"] }, DETAIL), { backup_brains: ["groq"] });
  assert.equal(lightFor("gemini"), "gemini_lite");
  assert.equal(lightFor("claude@claude-haiku-5-5"), "");
});

test("the first run sets each role from the provider's defaults, and never touches the voice", () => {
  assert.deepEqual(quickSetup("gemini", { brain: "gemini", quick: "gemini_lite", eyes: "gemini" }), { brain: "gemini", fast_brain: "gemini_lite", light_brain: "gemini_lite", eyes_brain: "gemini" });
  assert.deepEqual(
    quickSetup("anthropic", { brain: "claude@claude-opus-5-5", quick: "claude@claude-haiku-5-5", eyes: "claude@claude-opus-5-5" }),
    { brain: "claude@claude-opus-5-5", fast_brain: "claude@claude-haiku-5-5", light_brain: "", eyes_brain: "claude@claude-opus-5-5" },
  );
  // A live list without Haiku: Quick replies falls back to the Brain's model (R8: a default the key can't use is dropped).
  assert.deepEqual(quickSetup("anthropic", { brain: "claude@claude-opus-5-5", eyes: "claude@claude-opus-5-5" })?.fast_brain, "claude@claude-opus-5-5");
  assert.equal(quickSetup("openrouter", {}), null, "OpenRouter: the user picks a model");
  assert.equal(quickSetup("local", { brain: "local", quick: "local" })?.eyes_brain, undefined, "NEO's own model can't see");
  for (const plan of [quickSetup("gemini", { brain: "gemini" }), quickSetupWith("openrouter", { id: "x/y", sees: true, tools: true })]) assert.ok(!("voice" in (plan ?? {})));
  assert.deepEqual(quickSetupWith("openrouter", { id: "deepseek/deepseek-v4-pro", sees: false, tools: true }), { fast_brain: "openrouter@deepseek/deepseek-v4-pro", light_brain: "", brain: "openrouter@deepseek/deepseek-v4-pro" });
  assert.deepEqual(quickSetupWith("custom:ollama", { id: "llama3", sees: false, tools: false }), { fast_brain: "custom:ollama@llama3", light_brain: "" }, "no tools: never the Brain");
});

test("the menus offer only what a role can use", () => {
  assert.deepEqual(providerChoices("brain", DETAIL).map((c) => c.id), ["gemini", "anthropic", "groq", "local", "custom:ollama"]);
  assert.deepEqual(providerChoices("eyes", DETAIL).map((c) => c.id), ["gemini", "anthropic", "groq"], "no key (OpenAI), can't see (local, the endpoint) — not offered");
  const groq = { provider: "groq", ok: true, models: [{ id: "openai/gpt-oss-120b", label: "GPT-OSS 120B", sees: false, tools: true, context: null }], defaults: {} };
  assert.deepEqual(modelOptions("groq", "brain", groq, null, DETAIL).map((o) => o.ref), ["groq", "groq@openai/gpt-oss-120b"], "the eyes-only model isn't a Brain");
  assert.deepEqual(modelOptions("groq", "eyes", groq, null, DETAIL).map((o) => o.ref), ["groq_vision"]);
  const claude = { provider: "anthropic", ok: true, models: [{ id: "claude-opus-5-5", label: "Claude Opus 5.5", sees: true, tools: true, context: 1_000_000 }], defaults: {} };
  const opts = modelOptions("anthropic", "brain", claude, "claude@claude-sonnet-5-5", DETAIL);
  assert.deepEqual(opts.map((o) => o.ref), ["claude@claude-opus-5-5", "claude@claude-sonnet-5-5"], "the current brain stays in its menu");
  const ollama = { provider: "custom:ollama", ok: true, models: [{ id: "llava", label: "llava", sees: true, tools: true, context: null }], defaults: {} };
  assert.deepEqual(modelOptions("custom:ollama", "eyes", ollama, null, DETAIL), [], "an endpoint that doesn't see never looks");
});

test("where a key comes from, in words; a value never", () => {
  assert.equal(sourceText(DETAIL.keys.anthropic), "Saved ••••ijkl");
  assert.equal(sourceText(DETAIL.keys.gemini), "From ~/.neo/.env");
  assert.equal(sourceText(DETAIL.keys.groq), "From environment");
  assert.equal(sourceText(DETAIL.keys.openai), "Not set");
  assert.equal(sourceText(undefined), "Not set");
  assert.equal(sourceText({ set: true, source: "checkout_env", last4: null }), "From the checkout's .env");
  assert.equal(shadowNote(DETAIL.keys.groq, "GROQ_API_KEY"), "Set by environment variable GROQ_API_KEY — it wins over what you save here.");
  assert.match(shadowNote(DETAIL.keys.gemini, "GEMINI_API_KEY") ?? "", /the line in ~\/\.neo\/\.env stays/);
  assert.equal(shadowNote(DETAIL.keys.anthropic, "ANTHROPIC_API_KEY"), null);
  assert.equal(testText({ ok: true, status: "ok", http: 200, message: "The key works (api.openai.com).", models: 12, credit: null, provider: "openai" }), "The key works (api.openai.com) · 12 models.");
  assert.equal(testText({ ok: true, status: "ok", http: 200, message: "The key works (openrouter.ai).", models: null, credit: { limit: 10, usage: 1.25 }, provider: "openrouter" }), "The key works (openrouter.ai) · $8.75 of $10.00 left.");
  for (const code of ["invalid", "bad_key", "store_failed", "offline", "whatever"]) assert.ok(!errorText(code).includes("sk-"));
});

test("daily caps: the paid providers with a key, the whole map sent back, a figure that can't be read refused", () => {
  assert.deepEqual(budgetRows(DETAIL).map((r) => [r.pid, r.value]), [["anthropic", 2]], "Gemini and Groq are free; OpenAI has no key; the endpoint is on this Mac");
  assert.deepEqual(budgetsAfter(DETAIL, { anthropic: "3.5" }), { budgets: { anthropic: 3.5 } });
  assert.deepEqual(budgetsAfter(DETAIL, { anthropic: "" }), { budgets: {} }, "empty: no cap");
  assert.deepEqual(budgetsAfter(DETAIL, { openai: "$1" }), { budgets: { anthropic: 2, openai: 1 } });
  for (const bad of ["-1", "abc", "1001", "NaN"]) assert.deepEqual(budgetsAfter(DETAIL, { anthropic: bad }), { bad: "anthropic" }, bad);
  const withPrefill = { ...DETAIL, keys: { ...DETAIL.keys, openai: { set: true, source: "saved", last4: "wxyz" } }, daily_budgets: { value: {}, source: "default", prefill: { openai: 2 } } };
  assert.deepEqual(budgetRows(withPrefill).map((r) => [r.pid, r.value, r.suggested]), [["anthropic", null, null], ["openai", null, 2]]);
});

test("an endpoint goes out with exactly the core's fields — never a key among them", () => {
  const py = readFileSync(new URL("../../neo2/core/brainref.py", import.meta.url), "utf8");
  const cls = py.split("class CustomProvider(BaseModel):")[1].split("\n    @")[0];
  const fields = [...cls.matchAll(/^ {4}([a-z_]+): /gm)].map((m) => m[1]);
  assert.ok(fields.includes("base_url") && fields.includes("short_tool_ids"), fields.join(","));
  const t = { id: "deepseek", label: "DeepSeek", base_url: "https://api.deepseek.com", env: "DEEPSEEK_API_KEY", key_url: "", key_hint: "", models: { brain: "deepseek-v4-pro", quick: "deepseek-flash" }, tools: true, vision: false, short_tool_ids: false, needs_key: true };
  const f = formFromTemplate(t);
  assert.equal(f.model, "deepseek-v4-pro");
  const body = endpointBody(f);
  for (const k of Object.keys(body)) assert.ok(fields.includes(k), `${k} isn't a CustomProvider field (extra=forbid)`);
  assert.ok(!("key" in body));
  assert.equal(endpointProblem(f), null);
  assert.equal(endpointBody({ ...f, base_url: "https://api.deepseek.com/v1///" }).base_url, "https://api.deepseek.com/v1");
  assert.equal(formFromTemplate({ ...t, id: "ollama", label: "Ollama (on this Mac)", base_url: "http://localhost:11434/v1", models: {} }).label, "Ollama");
  assert.equal(endpointBody({ ...f, model: "" }, { forProbe: true }).model, "probe");
  for (const [patch, says] of [
    [{ label: " " }, /name/], [{ base_url: "http://api.example.com/v1" }, /https/], [{ base_url: "https://u:p@api.example.com" }, /password/],
    [{ base_url: "https://api.example.com/v1?x=1" }, /query/], [{ model: "" }, /model/], [{ context: 100 }, /context/], [{ base_url: "nope" }, /isn't a URL/],
  ] as const) {
    assert.match(endpointProblem({ ...f, ...patch }) ?? "", says, JSON.stringify(patch));
  }
  assert.equal(endpointProblem({ ...f, base_url: "http://127.0.0.1:1234/v1" }), null, "http on this Mac");
});

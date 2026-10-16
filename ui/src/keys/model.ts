// What the Brains & Keys window shows and sends, worked out from what the core said — kept apart
// from the components so it can be tested on its own (tests/keys-wire.test.ts). The core checks
// every rule again (brains_set: neo2/core/brains.py); this only keeps the page from offering what
// would be refused.

import { LEGACY, PROVIDERS, brainName, modelOf, providerOf, refFor, show } from "./catalog.ts";
import { keyish } from "./keyshape.ts";
import type { BrainsDetail, CatalogTemplate, CustomEndpoint, KeyEntry, KeyTestResult, ModelEntry, ModelsResult, Result } from "./types.ts";

export type Role = "brain" | "quick" | "eyes";
/** The Settings field each role is. */
export const FIELD: Readonly<Record<Role, string>> = { brain: "brain", quick: "fast_brain", eyes: "eyes_brain" };
export const ROLE_NAME: Readonly<Record<string, string>> = {
  brain: "Brain", fast_brain: "Quick replies", eyes_brain: "Eyes", backup_brains: "Backups", light_brain: "One-step commands",
  offline_brain: "Offline brain", cloud_eyes: "Screenshots", voice: "Voice", daily_budgets: "Spending",
};

const DOTS = "••••";

/** "Saved ••••x9Qa", "From environment", "From ~/.neo/.env", "Not set". */
export function sourceText(e: KeyEntry | undefined): string {
  if (!e || !e.set) return "Not set";
  switch (e.source) {
    case "saved": return `Saved ${DOTS}${e.last4 ?? ""}`;
    case "env": return "From environment";
    case "dotenv": return "From ~/.neo/.env";
    case "checkout_env": return "From the checkout's .env";
    default: return "Set";
  }
}

/** What a key set somewhere else means for one saved here (null: nothing to say). */
export function shadowNote(e: KeyEntry | undefined, envName: string | null): string | null {
  if (!e?.set) return null;
  if (e.source === "env") return `Set by environment variable ${envName ?? "in NEO's environment"} — it wins over what you save here.`;
  if (e.source === "dotenv" || e.source === "checkout_env") {
    const where = e.source === "dotenv" ? "~/.neo/.env" : "the checkout's .env";
    return `Replace saves a key in NEO, which wins; the line in ${where} stays.`;
  }
  return null;
}

/** A key test's answer in a line: the core's own fixed words, then what the check could count. */
export function testText(r: KeyTestResult | Result): string {
  const t = r as Partial<KeyTestResult>;
  if (!t.message) return r.ok ? "The key works." : errorText(String((r as Result).error ?? t.status ?? "failed"));
  const parts = [t.message.replace(/\.$/, "")];
  if (r.ok && typeof t.models === "number") parts.push(`${t.models} model${t.models === 1 ? "" : "s"}`);
  const credit = t.credit as Record<string, unknown> | null | undefined;
  if (r.ok && credit && typeof credit.limit === "number" && typeof credit.usage === "number") {
    parts.push(`$${(credit.limit - credit.usage).toFixed(2)} of $${credit.limit.toFixed(2)} left`);
  } else if (r.ok && credit && typeof credit.balance === "string") {
    parts.push(`balance ${credit.balance}${typeof credit.currency === "string" ? " " + credit.currency : ""}`);
  }
  return parts.join(" · ") + ".";
}

/** A command that failed, in words (never anything it carried). */
export function errorText(code: string, message?: string): string {
  if (message && (code === "bad_key" || code === "no_access" || code === "no_credit" || code === "wrong_provider")) return message;
  switch (code) {
    case "invalid": return "NEO can't use that as it is.";
    case "bad_key": return "The provider didn't accept that key.";
    case "no_access": return "That key can't use the API from here (region, project or permission).";
    case "no_credit": return "That account has no credit left.";
    case "store_failed": return "NEO couldn't save it. Check that ~/.neo is yours and not full.";
    case "offline": case "disconnected": return "NEO's core isn't connected right now. Try again in a moment.";
    case "timeout": return "NEO's core didn't answer in time. Try again.";
    case "forbidden": return "This window isn't allowed to change NEO's keys.";
    case "in_use": return "A role still uses it: give that role another brain first.";
    case "not_found": return "NEO has no such endpoint saved.";
    case "too_many": return "NEO keeps at most 10 endpoints.";
    case "no_app": return "Only NEO.app can restart NEO. Restart it yourself.";
    case "wrong_provider": return "That key looks like another provider's, so NEO didn't send it here.";
    case "needs_key": return "The base URL moved to another host: paste the endpoint's key again (NEO never sends a saved key somewhere new).";
    default: return "That didn't work. Try again.";
  }
}

/** brains_set's refusal, said for the role it was about (the core's own words when it gave some). */
export function rolesErrorText(code: string, field?: string, message?: string): string {
  if (message) return message;
  const who = ROLE_NAME[field ?? ""] ?? "That";
  switch (code) {
    case "invalid": return `${who}: that isn't a brain NEO knows.`;
    case "not_configured": return `${who}: that provider isn't set up yet. Add its key first (or start it, for a model on this Mac).`;
    case "no_tools": return field === "backup_brains"
      ? "A Backup answers tasks too, so it has to call tools. Pick another, or use this one for Quick replies."
      : "The Brain has to call tools. Pick another model, or use this one for Quick replies.";
    case "cant_see": return "Eyes has to see pictures. Pick a model that sees.";
    default: return errorText(code);
  }
}

/** Whether `provider` can be asked now, by what brains_detail says (a key set, an endpoint added). */
export function configured(provider: string, d: BrainsDetail | null): boolean {
  if (!d) return false;
  if (provider === "local") return true; // (whether it answers is the core's to say: brains_set probes it)
  if (provider.startsWith("custom:")) {
    const cp = d.custom.find((c) => c.id === provider);
    return !!cp && (!!cp.on_device || !!d.keys[provider]?.set);
  }
  return !!d.keys[provider]?.set;
}

export function endpoint(d: BrainsDetail | null, id: string): CustomEndpoint | undefined {
  return d?.custom.find((c) => c.id === id);
}

export interface Choice { id: string; label: string }

/** The providers a role may use: set up, and able to do the role's job. */
export function providerChoices(role: Role, d: BrainsDetail | null): Choice[] {
  if (!d) return [];
  const out: Choice[] = [];
  for (const p of PROVIDERS) {
    if (!configured(p.id, d)) continue;
    const entry = d.catalog.find((c) => c.id === p.id);
    const roles = entry?.roles ?? [];
    if (!roles.includes(role)) continue;
    out.push({ id: p.id, label: p.id === "local" ? "On this Mac (NEO's model)" : p.short });
  }
  for (const cp of d.custom) {
    if (!configured(cp.id, d)) continue;
    if (role === "brain" && !cp.tools) continue;
    if (role === "eyes" && !cp.vision) continue;
    out.push({ id: cp.id, label: cp.label });
  }
  return out;
}

export interface ModelOption { ref: string; label: string; sees: boolean; tools: boolean; eyesOnly: boolean; preview?: boolean; tier?: string; note?: string; current?: boolean }

/** The model menu for `provider` in `role`: NEO's own names first (gemini, groq_vision…), then the
 *  provider's live list (models_list), each as a brain reference; the role's current brain is
 *  always there, even when the list doesn't have it. */
export function modelOptions(provider: string, role: Role, list: ModelsResult | undefined, current: string | null, d: BrainsDetail | null): ModelOption[] {
  const fits = (o: ModelOption) => (role === "brain" ? o.tools && !o.eyesOnly : role === "eyes" ? o.sees : !o.eyesOnly);
  const out: ModelOption[] = [];
  const add = (o: ModelOption) => { if (fits(o) && !out.some((x) => x.ref === o.ref)) out.push(o); };
  for (const l of LEGACY[provider] ?? []) add({ ref: l.ref, label: l.label, sees: l.sees, tools: true, eyesOnly: !!l.eyesOnly, preview: l.preview });
  const cp = provider.startsWith("custom:") ? endpoint(d, provider) : undefined;
  if (cp) add({ ref: cp.id, label: `${cp.model} (as saved)`, sees: cp.vision, tools: cp.tools, eyesOnly: false });
  for (const m of list?.models ?? []) add(fromEntry(provider, m, cp));
  if (current && providerOf(current) === provider && !out.some((o) => o.ref === current)) {
    out.push({ ref: current, label: `${modelOf(current) ?? current} (now)`, sees: true, tools: true, eyesOnly: false, current: true });
  }
  return out;
}

function fromEntry(provider: string, m: ModelEntry, cp?: CustomEndpoint): ModelOption {
  return {
    ref: refFor(provider, m.id),
    label: m.label && m.label !== m.id ? m.label : m.id,
    sees: cp ? (m.sees || cp.vision) && cp.vision : m.sees,
    tools: cp ? m.tools && cp.tools : m.tools,
    eyesOnly: false,
    tier: m.tier,
    note: m.note,
  };
}

/** The badges a brain gets: Free, Paid, On this Mac, Text only, Preview (and the list's own note). */
export function badges(provider: string, o: ModelOption | undefined, d: BrainsDetail | null): string[] {
  const out: string[] = [];
  const cp = provider.startsWith("custom:") ? endpoint(d, provider) : undefined;
  const cost = cp ? (cp.on_device ? "this_mac" : "paid") : show(provider)?.cost;
  if (cost === "free" || cost === "free_tier") out.push("Free");
  else if (cost === "this_mac") out.push("On this Mac");
  else if (cost === "paid") out.push(provider === "openrouter" && o?.ref.endsWith(":free") ? "Free" : "Paid");
  if (o && !o.sees) out.push("Text only");
  if (o?.eyesOnly) out.push("Eyes only");
  if (o?.preview) out.push("Preview");
  if (o?.note) out.push(o.note);
  return out;
}

/** One-step commands: Gemini Flash-Lite while Quick replies is Gemini; otherwise Quick replies' own
 *  brain ("") — a Claude user's commands don't quietly go to Gemini. (A Quick replies brain that
 *  can't call tools is passed over for them by the core's chain: the Brain answers them.) */
export function lightFor(fastRef: string): string {
  return providerOf(fastRef) === "gemini" ? "gemini_lite" : "";
}

export interface RoleForm { brain: string; fast_brain: string; eyes_brain: string; backup_brains: string[] | null; cloud_eyes: string }

export function formFrom(d: BrainsDetail): RoleForm {
  const ref = (f: string) => d.roles[f]?.ref ?? "";
  const b = d.roles.backup_brains?.refs;
  return {
    brain: ref("brain"),
    fast_brain: ref("fast_brain"),
    eyes_brain: ref("eyes_brain"),
    backup_brains: b == null ? null : b.map((x) => x.ref ?? "").filter(Boolean),
    cloud_eyes: d.cloud_eyes?.value ?? "auto",
  };
}

/** brains_set's fields for what the user changed (nothing else is sent). */
export function rolesChanges(form: RoleForm, d: BrainsDetail): Record<string, unknown> {
  const was = formFrom(d);
  const out: Record<string, unknown> = {};
  if (form.brain !== was.brain) out.brain = form.brain;
  if (form.fast_brain !== was.fast_brain) {
    out.fast_brain = form.fast_brain;
    out.light_brain = lightFor(form.fast_brain);
  }
  if (form.eyes_brain !== was.eyes_brain) out.eyes_brain = form.eyes_brain;
  if (JSON.stringify(form.backup_brains) !== JSON.stringify(was.backup_brains)) out.backup_brains = form.backup_brains;
  // "Others may help" is auto unless it was already on (both let others look); "only mine" is off.
  const others = (v: string) => v !== "off";
  if (others(form.cloud_eyes) !== others(was.cloud_eyes)) out.cloud_eyes = form.cloud_eyes;
  return out;
}

/** The first run's brains_set for a provider, from its live list's defaults: Brain, Quick replies
 *  and Eyes; one-step commands on Gemini only when Gemini answers quick replies. The voice isn't
 *  touched (Gemini Live stays the voice whenever there's a Gemini key). Null: the user picks a
 *  model first (OpenRouter has no default). */
export function quickSetup(provider: string, defaults: Record<string, string> | undefined): Record<string, unknown> | null {
  const brain = defaults?.brain;
  const quick = defaults?.quick ?? brain;
  if (!brain || !quick) return null;
  const out: Record<string, unknown> = { brain, fast_brain: quick, light_brain: lightFor(quick) };
  if (defaults?.eyes) out.eyes_brain = defaults.eyes;
  else if (provider !== "local" && !provider.startsWith("custom:")) out.eyes_brain = "";
  return out;
}

/** The first run with one model picked (OpenRouter, an endpoint on this Mac): it does every role it
 *  can — Brain only when it calls tools, Eyes only when it sees. */
export function quickSetupWith(provider: string, model: { id: string; sees: boolean; tools: boolean } | null): Record<string, unknown> {
  const ref = provider.startsWith("custom:") && !model ? provider : refFor(provider, model?.id ?? null);
  const out: Record<string, unknown> = { fast_brain: ref, light_brain: "" };
  if (!model || model.tools) out.brain = ref;
  if (model?.sees) out.eyes_brain = ref;
  return out;
}

export interface BudgetRow { pid: string; label: string; value: number | null; suggested: number | null }

/** The paid providers whose key is set, each with its daily cap (and the $2 NEO suggests for a key
 *  saved here). Cerebras keeps its own budget (NEO_CEREBRAS_DAILY_BUDGET); Gemini and Groq are free. */
export function budgetRows(d: BrainsDetail | null): BudgetRow[] {
  if (!d) return [];
  const value = d.daily_budgets?.value ?? {};
  const prefill = d.daily_budgets?.prefill ?? {};
  const rows: BudgetRow[] = [];
  for (const pid of ["anthropic", "openai", "openrouter"]) {
    if (!d.keys[pid]?.set) continue;
    rows.push({ pid, label: show(pid)?.short ?? pid, value: value[pid] ?? null, suggested: prefill[pid] ?? null });
  }
  for (const cp of d.custom) {
    if (cp.on_device || !d.keys[cp.id]?.set) continue;
    rows.push({ pid: cp.id, label: cp.label, value: value[cp.id] ?? null, suggested: prefill[cp.id] ?? null });
  }
  return rows;
}

export const MAX_BUDGET = 1000;

/** The whole daily_budgets map after the user's edits (by provider: "" means no cap), or which
 *  provider's figure can't be read. The core replaces the map it has with this one. */
export function budgetsAfter(d: BrainsDetail, edits: Record<string, string>): { budgets: Record<string, number> } | { bad: string } {
  const out: Record<string, number> = { ...(d.daily_budgets?.value ?? {}) };
  for (const [pid, raw] of Object.entries(edits)) {
    const s = raw.trim().replace(/^\$/, "");
    if (!s) { delete out[pid]; continue; }
    const x = Number(s);
    if (!Number.isFinite(x) || x < 0 || x > MAX_BUDGET) return { bad: pid };
    out[pid] = Math.round(x * 100) / 100;
  }
  return { budgets: out };
}

/** The voice as the window offers it: Gemini Live (auto or live) or on this Mac. */
export function voiceChoice(value: string): "live" | "local" | "off" {
  return value === "local" ? "local" : value === "off" ? "off" : "live";
}

/** Copies of the key file NEO set aside (it couldn't read them) are still on disk, and may hold keys
 *  that Remove here never touches: said, so the user can delete them. */
export function oldCopiesText(n: number, path: string): string {
  const what = n === 1 ? "An older copy" : `${n} older copies`;
  return `${what} of ${path} (named …bad-<date>) ${n === 1 ? "is" : "are"} still in that folder and may hold keys. Delete ${n === 1 ? "it" : "them"} in Finder if you don't need ${n === 1 ? "it" : "them"}: removing a key here doesn't touch ${n === 1 ? "it" : "them"}.`;
}

/** What a terminal user runs instead, when no window can reach the core. */
export function cliLines(provider = "gemini"): string[] {
  return [`python -m neo2 keys set ${provider}`, "python -m neo2 brain set --brain REF --quick REF --eyes REF"];
}

/** A short line for the header: what answers, or that nothing does yet. */
export function headline(d: BrainsDetail | null, summaryReady: boolean | undefined, brainLabel: string | undefined): string {
  if (!d && summaryReady === undefined) return "Connecting to NEO…";
  if (!summaryReady) return "NEO has no AI model yet";
  return brainLabel ? `NEO's brain: ${brainLabel}` : "NEO is ready";
}

/** The roles' current brains by provider, for picking which models_list to ask first. */
export function providersInUse(d: BrainsDetail | null): string[] {
  if (!d) return [];
  const out = new Set<string>();
  for (const f of ["brain", "fast_brain", "eyes_brain"]) {
    const p = providerOf(d.roles[f]?.ref ?? "");
    if (p) out.add(p);
  }
  for (const p of PROVIDERS) if (configured(p.id, d) && p.id !== "local") out.add(p.id);
  for (const cp of d.custom) if (configured(cp.id, d)) out.add(cp.id);
  return [...out];
}

/** The "Other endpoint" form. */
export interface EndpointForm {
  id: string | null; label: string; template: string; base_url: string; model: string;
  tools: boolean; vision: boolean; context: number; short_tool_ids: boolean; token_param: "max_tokens" | "max_completion_tokens";
}

export function formFromTemplate(t: CatalogTemplate | undefined): EndpointForm {
  return {
    id: null,
    label: t && t.id !== "other" ? t.label.replace(/ \(on this Mac\)$/, "") : "",
    template: t?.id ?? "other",
    base_url: t?.base_url ?? "",
    model: t?.models?.brain ?? "",
    tools: t?.tools ?? true,
    vision: t?.vision ?? false,
    context: 32768,
    short_tool_ids: t?.short_tool_ids ?? false,
    token_param: "max_tokens",
  };
}

export function formFromEndpoint(cp: CustomEndpoint): EndpointForm {
  return {
    id: cp.id, label: cp.label, template: cp.template, base_url: cp.base_url, model: cp.model, tools: cp.tools,
    vision: cp.vision, context: cp.context, short_tool_ids: !!cp.short_tool_ids,
    token_param: cp.token_param === "max_completion_tokens" ? "max_completion_tokens" : "max_tokens",
  };
}

const LOOPBACK = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

/** Whether a base URL is on this Mac (no key needed there). */
export function onThisMac(baseUrl: string): boolean {
  try { return LOOPBACK.has(new URL(baseUrl).hostname); } catch { return false; }
}

/** Why the form can't be saved yet (null: it can) — the core's rules (brainref.CustomProvider), said
 *  before it refuses. */
export function endpointProblem(f: EndpointForm): string | null {
  if (!f.label.trim()) return "Give it a name.";
  if (f.label.trim().length > 40) return "The name can be 40 characters at most.";
  let u: URL;
  try { u = new URL(f.base_url.trim()); } catch { return "The base URL isn't a URL (https://…/v1)."; }
  if (u.protocol !== "https:" && !(u.protocol === "http:" && LOOPBACK.has(u.hostname))) return "The base URL must start with https:// (http:// only on this Mac).";
  if (u.username || u.password) return "The base URL can't hold a user name or password.";
  if (u.search || u.hash || f.base_url.includes("?") || f.base_url.includes("#")) return "The base URL can't have a query.";
  if (f.base_url.trim().length > 300) return "The base URL is too long.";
  if (keyish(f.model) || keyish(f.label)) return "That looks like an API key: paste it in the Key field, not as the name or the model.";
  if (!/^[A-Za-z0-9._:/+-]{1,160}$/.test(f.model.trim())) return "Pick or type a model id.";
  if (!Number.isInteger(f.context) || f.context < 4096 || f.context > 2_000_000) return "The context size is 4096 to 2,000,000 tokens.";
  return null;
}

/** Where a key sent to a base URL goes: its scheme, host and port (null: not a URL). */
export function originOf(baseUrl: string): string | null {
  try {
    const u = new URL(baseUrl.trim());
    return `${u.protocol}//${u.hostname.toLowerCase()}:${u.port || (u.protocol === "https:" ? "443" : "80")}`;
  } catch { return null; }
}

/** An endpoint being edited whose base URL now points somewhere else than where it was saved: its
 *  saved key isn't sent there (the core refuses), so the form asks for the key again. */
export function keyMustBeRetyped(f: EndpointForm, d: BrainsDetail | null): boolean {
  if (!f.id) return false;
  const was = endpoint(d, f.id);
  if (!was || !d?.keys[f.id]?.set || d.keys[f.id]?.source !== "saved") return false;
  const a = originOf(was.base_url), b = originOf(f.base_url);
  return !!a && !!b && a !== b && !onThisMac(f.base_url);
}

/** The endpoint as custom_save, key_test and models_list take it: exactly the core's fields. */
export function endpointBody(f: EndpointForm, opts: { forProbe?: boolean } = {}): Record<string, unknown> {
  return {
    id: f.id,
    label: f.label.trim() || "Endpoint",
    template: f.template || "other",
    base_url: f.base_url.trim().replace(/\/+$/, ""),
    model: f.model.trim() || (opts.forProbe ? "probe" : ""),
    tools: f.tools,
    vision: f.vision,
    context: f.context,
    short_tool_ids: f.short_tool_ids,
    token_param: f.token_param,
  };
}

/** The endpoint templates brains_detail carries (the "custom" entry's). */
export function templates(d: BrainsDetail | null): CatalogTemplate[] {
  return d?.catalog.find((c) => c.id === "custom")?.templates ?? [];
}

export { brainName, providerOf, refFor };

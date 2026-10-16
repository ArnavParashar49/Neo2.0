// How the Brains & Keys window shows each provider: names, one-line blurbs, cost badges and where
// to get a key. Display only — what NEO can do with a provider, its models and its defaults come
// from the core (neo2/core/providers/catalog.py, through brains_detail and models_list). The key
// pages listed here are the only ones the window may open (ui/src-tauri/capabilities/keys.json;
// tests/keys-shape.test.ts holds the two together).

export type Cost = "free" | "free_tier" | "paid" | "this_mac";

export interface ProviderShow {
  id: string;
  /** As a heading: "Anthropic (Claude)". */
  label: string;
  /** In a sentence or a menu: "Claude". */
  short: string;
  /** The brain name its references start with ("claude@claude-opus-5-5"). */
  brain: string;
  cost: Cost;
  keyUrl: string;
  keyHint: string;
  blurb: string;
}

export const PROVIDERS: readonly ProviderShow[] = [
  { id: "gemini", label: "Google Gemini", short: "Gemini", brain: "gemini", cost: "free_tier", keyUrl: "https://aistudio.google.com/apikey", keyHint: "AIza", blurb: "Free tier, no card. Sees the screen, runs the camera and Gemini Live's voice." },
  { id: "anthropic", label: "Anthropic (Claude)", short: "Claude", brain: "claude", cost: "paid", keyUrl: "https://console.anthropic.com/settings/keys", keyHint: "sk-ant-", blurb: "Paid. Strong at long tasks; sees the screen." },
  { id: "openai", label: "OpenAI", short: "OpenAI", brain: "openai", cost: "paid", keyUrl: "https://platform.openai.com/api-keys", keyHint: "sk-proj-", blurb: "Paid. GPT models; they see the screen." },
  { id: "groq", label: "Groq", short: "Groq", brain: "groq", cost: "free", keyUrl: "https://console.groq.com/keys", keyHint: "gsk_", blurb: "Free. Very fast quick replies; text only, with a small model that sees." },
  { id: "cerebras", label: "Cerebras", short: "Cerebras", brain: "cerebras", cost: "paid", keyUrl: "https://cloud.cerebras.ai", keyHint: "csk-", blurb: "Paid, capped by its own daily budget. A very fast brain, tried ahead of Groq." },
  { id: "openrouter", label: "OpenRouter", short: "OpenRouter", brain: "openrouter", cost: "paid", keyUrl: "https://openrouter.ai/keys", keyHint: "sk-or-", blurb: "Pay as you go, a few models free. Hundreds of models behind one key." },
  { id: "local", label: "NEO's model on this Mac (Gemma)", short: "the model on this Mac", brain: "local", cost: "this_mac", keyUrl: "", keyHint: "", blurb: "Free and offline. Runs while python -m neo2 --local is running." },
];

/** The "Other endpoint" templates' key pages and hints (their base URLs and models come from the core). */
export const TEMPLATE_KEY_URLS: Readonly<Record<string, string>> = {
  xai: "https://console.x.ai",
  deepseek: "https://platform.deepseek.com/api_keys",
  mistral: "https://console.mistral.ai/api-keys",
  together: "https://api.together.ai/settings/api-keys",
};

/** Every page the window may open (capabilities/keys.json allows exactly these). */
export function keyPages(): string[] {
  return [...PROVIDERS.map((p) => p.keyUrl).filter(Boolean), ...Object.values(TEMPLATE_KEY_URLS)];
}

/** A key page NEO shows a link to: one of the catalogue's, or nothing (never a URL from elsewhere). */
export function keyPage(url: string | null | undefined): string | null {
  return url && keyPages().includes(url) ? url : null;
}

export function show(id: string): ProviderShow | undefined {
  return PROVIDERS.find((p) => p.id === id);
}

/** The providers whose key the window takes (NEO's own model has none). */
export const KEYED: readonly string[] = PROVIDERS.filter((p) => p.cost !== "this_mac").map((p) => p.id);

/** The first-run cards, in order: Gemini first and preselected ("free, no card"). */
export const QUICK: readonly { id: string; title: string; line: string }[] = [
  { id: "gemini", title: "Google Gemini", line: "Free, no card" },
  { id: "anthropic", title: "Claude", line: "Paid · Anthropic" },
  { id: "openai", title: "OpenAI", line: "Paid" },
  { id: "groq", title: "Groq", line: "Free · text only" },
  { id: "openrouter", title: "OpenRouter", line: "Many models, one key" },
  { id: "this_mac", title: "On this Mac", line: "Ollama, LM Studio or NEO's own" },
];

export function costBadge(cost: string): { text: string; tone: "free" | "paid" | "mac" } {
  if (cost === "free") return { text: "Free", tone: "free" };
  if (cost === "free_tier") return { text: "Free tier", tone: "free" };
  if (cost === "this_mac") return { text: "On this Mac", tone: "mac" };
  return { text: "Paid", tone: "paid" };
}

/** The provider a brain reference belongs to, the way the core counts them (catalog.provider_of). */
export function providerOf(ref: string | null | undefined): string {
  const r = (ref ?? "").trim().toLowerCase();
  const brain = r.split("@", 1)[0];
  if (brain.startsWith("custom:")) return brain;
  switch (brain) {
    case "gemini": case "gemini_lite": case "google": return "gemini";
    case "groq": case "groq_vision": return "groq";
    case "cerebras": case "cerebras_vision": return "cerebras";
    case "claude": case "anthropic": return "anthropic";
    case "openai": case "openrouter": case "local": return brain;
    default: return "";
  }
}

/** The model a reference names after "@" (null: the brain's own default). */
export function modelOf(ref: string | null | undefined): string | null {
  const r = ref ?? "";
  const at = r.indexOf("@");
  return at < 0 ? null : r.slice(at + 1) || null;
}

/** The brain name a provider's references start with ("anthropic" → "claude"). */
export function brainName(provider: string): string {
  return provider.startsWith("custom:") ? provider : show(provider)?.brain ?? provider;
}

/** A reference for `provider`'s `model` ("claude@claude-haiku-5-5"). */
export function refFor(provider: string, model: string | null): string {
  return model ? `${brainName(provider)}@${model}` : brainName(provider);
}

/** The bare brain names NEO always had, by provider: offered first in the model menus. */
export const LEGACY: Readonly<Record<string, readonly { ref: string; label: string; sees: boolean; eyesOnly?: boolean; preview?: boolean }[]>> = {
  gemini: [
    { ref: "gemini", label: "Gemini Flash (NEO's pick, newest first)", sees: true },
    { ref: "gemini_lite", label: "Gemini Flash-Lite (fastest)", sees: true },
  ],
  groq: [
    { ref: "groq", label: "GPT-OSS 120B on Groq", sees: false },
    { ref: "groq_vision", label: "Qwen 3.8 27B on Groq (sees; preview)", sees: true, eyesOnly: true, preview: true },
  ],
  cerebras: [
    { ref: "cerebras", label: "GPT-OSS 120B on Cerebras", sees: false },
    { ref: "cerebras_vision", label: "Qwen 3.8 27B on Cerebras (sees)", sees: true },
  ],
  local: [{ ref: "local", label: "Gemma on this Mac", sees: false }],
};

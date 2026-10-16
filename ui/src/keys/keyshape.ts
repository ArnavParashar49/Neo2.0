// What a pasted API key looks like: soft hints for the Brains & Keys window, and the composer's
// key guard. The twin of neo2/core/keyshape.py (and keystore.clean_key): a hint never stops a save,
// and nothing here keeps, logs or sends a key.

/** Prefixes that say which provider a key is for, longest first. "sk-" alone names no provider
 *  (OpenAI's older keys start with it, and so do DeepSeek's and others'). */
export const PREFIXES: ReadonlyArray<readonly [string, string]> = [
  ["sk-ant-", "anthropic"],
  ["sk-or-", "openrouter"],
  ["sk-proj-", "openai"],
  ["AIza", "gemini"],
  ["gsk_", "groq"],
  ["csk-", "cerebras"],
  ["xai-", "xai"],
];
/** What a provider's keys start with (the first is the one to suggest). */
export const EXPECTED: Readonly<Record<string, readonly string[]>> = {
  anthropic: ["sk-ant-"],
  openrouter: ["sk-or-"],
  openai: ["sk-proj-", "sk-"],
  gemini: ["AIza"],
  groq: ["gsk_"],
  cerebras: ["csk-"],
  xai: ["xai-"],
};
const LABELS: Readonly<Record<string, string>> = {
  anthropic: "Anthropic", openrouter: "OpenRouter", openai: "OpenAI", gemini: "Gemini", groq: "Groq", cerebras: "Cerebras", xai: "xAI",
};
const AN = new Set(["anthropic", "openrouter", "openai", "xai"]); // "an Anthropic key"
export const MAX_KEY = 512;
/** What NEO says when typed text starts with a key (neo2/core/onboarding.py's KEY_TYPED). */
export const KEY_TYPED = "That looks like an API key, so I didn't send it anywhere. Save it in Brains & Keys.";

const KEY_BODY = /^[A-Za-z0-9_-]{16,}$/;
const OLD_OPENAI_BODY = /^[A-Za-z0-9_-]{32,}$/;
// A line break, a space or any other control or separator character (Python's not isprintable(), or isspace()).
const UNKEEPABLE = /[\s\p{C}\p{Z}]/u;

function bare(key: string): string {
  return (key ?? "").trim().replace(/^["']+|["']+$/g, "").trim();
}

/** The provider a key is for: an endpoint the user added from the xAI template is xAI's. */
export function family(provider: string): string {
  const p = (provider ?? "").trim().toLowerCase();
  return p.startsWith("custom:") ? p.slice("custom:".length) : p;
}

/** The provider `key`'s prefix names (null: none does). */
export function guess(key: string): string | null {
  const k = bare(key);
  return PREFIXES.find(([prefix]) => k.startsWith(prefix))?.[1] ?? null;
}

/** A soft hint about `key` for `provider` (null: nothing to say). Never a reason to refuse it. */
export function hint(provider: string, key: string): string | null {
  const k = bare(key);
  if (!k) return null;
  const fam = family(provider);
  const g = guess(k);
  if (g !== null && g !== fam) {
    const name = LABELS[g];
    return `This looks like ${AN.has(g) ? "an" : "a"} ${name} key — did you mean ${name}?`;
  }
  const want = EXPECTED[fam];
  if (want && !want.some((p) => k.startsWith(p))) return `${LABELS[fam]} keys usually start with ${want[0]}.`;
  return null;
}

/** A pasted key as the core keeps it (keystore.clean_key): spaces and one pair of quotes around it
 *  go. A key with a line break or another control character in it, over 512 characters, or empty
 *  can't be kept — the words say why, never the key. */
export function cleanKey(value: string): { ok: true; key: string } | { ok: false; problem: string } {
  let s = String(value ?? "").trim();
  if (s.length >= 2 && s[0] === s[s.length - 1] && (s[0] === '"' || s[0] === "'")) s = s.slice(1, -1).trim();
  if (!s) return { ok: false, problem: "Paste a key first." };
  if ([...s].length > MAX_KEY) return { ok: false, problem: `That's longer than any key (${MAX_KEY} characters at most).` };
  if (UNKEEPABLE.test(s)) return { ok: false, problem: "That has a line break, a space or a control character in it: paste the key alone." };
  return { ok: true, key: s };
}

/** Typed text that starts with an API key (the composer's guard: such a message goes to no model).
 *  Narrow on purpose — only the known prefixes, each followed by at least 16 key characters — so
 *  "sk-learn" or a long hash is still just text. */
export function looksLikeKey(text: string): boolean {
  const first = (text ?? "").trim().split(/\s+/, 1)[0] ?? "";
  if (!first) return false;
  const word = first.replace(/^["'`]+|["'`]+$/g, "");
  for (const [prefix] of PREFIXES) {
    if (word.startsWith(prefix)) return KEY_BODY.test(word.slice(prefix.length));
  }
  return word.startsWith("sk-") && OLD_OPENAI_BODY.test(word.slice(3));
}

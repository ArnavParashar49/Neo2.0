// The results sheet's chips: what can be done with the area, in one order whatever arrives first —
// what this Mac found (copy the text, translate it, open a link or a code's address, mail, call,
// a Wi-Fi password), then Search with Google (or the provider the user chose), then a search for
// what the answer named, then Brains & Keys when no brain could see. At most six, then More.

import type { Action, AnswerData, LensHello, LocalData } from "./wire.ts";

export type ChipKind = Action["kind"] | "translate";
export interface Chip {
  key: string;
  kind: ChipKind;
  label: string;
  /** The core's action id (every chip but Translate). */
  id?: string;
  /** Shown beside the label: a link's full host, a number, a network's name. */
  detail?: string;
  warn?: "userinfo" | "punycode";
  /** Translate: into which language. */
  to?: string;
  /** Choosing it opens something, and Search Screen closes. */
  closes: boolean;
}

export const ORDER: readonly ChipKind[] = [
  "copy_text", "translate", "open", "mail", "call", "wifi", "code", "copy_image", "image_search", "text_search", "brains",
];
const CLOSES: ReadonlySet<ChipKind> = new Set(["open", "mail", "call", "image_search", "text_search", "brains"]);
export const MAX_CHIPS = 6;

/** The base language of a BCP-47 code ("en-GB" → "en", "zh-Hans" → "zh"). */
export const baseLang = (code: string) => code.toLowerCase().split(/[-_]/)[0] ?? "";

/** A language's name for the chip: the core's list, then the system's names, then the code. */
export function langName(code: string, langs: LensHello["langs"] = []): string {
  const hit = langs.find((l) => l.code === code) ?? langs.find((l) => baseLang(l.code) === baseLang(code));
  if (hit) return hit.name;
  try {
    const dn = new Intl.DisplayNames(["en"], { type: "language" });
    return dn.of(code) ?? code;
  } catch {
    return code;
  }
}

/** Into which language Translate goes: the user's last choice, else the core's — never the text's own. */
export function translateTarget(local: LocalData | null, chosen: string | null): string | null {
  if (!local || !local.text.trim()) return null;
  const from = local.lang ? baseLang(local.lang.code) : "";
  for (const to of [chosen, local.translateTo]) {
    if (to && baseLang(to) && baseLang(to) !== from) return to;
  }
  return null;
}

export function chipsFor(
  local: LocalData | null,
  answer: AnswerData | null,
  hello: Pick<LensHello, "search" | "langs"> | null,
  chosen: string | null = null,
): { shown: Chip[]; more: Chip[] } {
  const all: Chip[] = [];
  const seen = new Set<string>();
  for (const a of [...(local?.actions ?? []), ...(answer?.actions ?? [])]) {
    if (seen.has(a.id)) continue;
    if (a.kind === "image_search" && hello?.search && !hello.search.image) continue; // that provider takes no pictures
    seen.add(a.id);
    all.push({ key: a.id, kind: a.kind, label: a.label, id: a.id, ...(a.detail ? { detail: a.detail } : {}), ...(a.warn ? { warn: a.warn } : {}), closes: CLOSES.has(a.kind) });
  }
  const to = translateTarget(local, chosen);
  if (to) all.push({ key: `translate:${to}`, kind: "translate", label: `Translate to ${langName(to, hello?.langs)}`, to, closes: false });
  const rank = (c: Chip) => ORDER.indexOf(c.kind);
  const sorted = all.map((c, i) => ({ c, i })).sort((a, b) => rank(a.c) - rank(b.c) || a.i - b.i).map((x) => x.c);
  return { shown: sorted.slice(0, MAX_CHIPS), more: sorted.slice(MAX_CHIPS) };
}

/** What a finished action says on its chip for a moment (lens_done's note). */
export function doneText(note: string, ok: boolean): string {
  switch (note) {
    case "copied": return "Copied";
    case "opened": return "Opened";
    case "no_browser": return "No browser to open it in";
    case "no_image": return "Nothing to copy";
    case "off": return "Search is off in settings";
    default: return ok ? "Done" : "Couldn't do that";
  }
}

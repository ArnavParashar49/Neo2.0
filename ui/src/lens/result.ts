// Search Screen's results for one area, as both places that show them keep them: the sheet beside
// the area (the overlay's reducer, ./state.ts) and the card in the notch (./card.ts). What the core
// says about the area is merged in here, and what the user's clicks ask for comes out as ops each
// place turns into its own commands (the overlay's lens_* with `sel`, the card's lens_card_* with
// its `cid`) — so "everything the sheet has" is true of the card by construction.

import { doneText } from "./chips.ts";
import type { AnswerData, Brain, CoreMsg, Lang, LinkRow, LocalData, SearchProvider, Sent } from "./wire.ts";

export interface Turn { qid: number; q: string; state: "thinking" | "done" | "failed"; text?: string; error?: string; brain?: string }
export interface Translation { to: string; text?: string; error?: string; pending: boolean }

export interface Result {
  /** The area's number (`sel`): every message about it carries it. */
  n: number;
  local: LocalData | null;
  answer: AnswerData | null;
  links: LinkRow[];
  thread: Turn[];
  translation: Translation | null;
  /** What a chip says for a moment after its action (by action id; "…" while it runs). */
  notes: Record<string, string>;
  /** "search my screen for X": X, asked with this area. */
  question: string | null;
  /** The area went to the search provider too (the privacy line says so). */
  searchedWeb: boolean;
  /** Every brain (and the web search) the core says the area or what it holds went to. */
  sentTo: Sent[];
}

/** What the results need of the core's hello: the brain an area goes to first, Ask, the search
 *  provider, the Translate menu. (The overlay's lens_hello has all of it; a card's snapshot too.) */
export interface ResultHello {
  eyes: Brain | null;
  eyesWhy: "" | "no_brain" | "local_only";
  ask: boolean;
  search: SearchProvider | null;
  langs: Lang[];
}

export const NOTE_MS = 1200;
export const MAX_THREAD = 12;

export function emptyResult(n: number, question: string | null = null): Result {
  return { n, local: null, answer: null, links: [], thread: [], translation: null, notes: {}, question, searchedWeb: false, sentTo: [] };
}

/** What the user did with the results (the sheet's and the card's clicks and keys). */
export type ResultAction =
  | { t: "chip"; kind: string; id?: string; to?: string }
  | { t: "translate_to"; to: string }
  | { t: "ask"; q: string }
  | { t: "ask_again" }
  | { t: "continue"; text: string }
  | { t: "listen" }
  | { t: "note_clear"; id: string };

/** What it asks the core for (each place sends it as its own command). */
export type ResultOp =
  | { op: "ask"; qid: number; q: string }
  | { op: "ask_again" }
  | { op: "translate"; to: string }
  | { op: "act"; id: string }
  | { op: "continue"; text: string }
  | { op: "listen" };

/** Where the area went, as the core said in a message, added to what it said before. */
function went<R extends Result>(r: R, more: Sent[] | undefined): R {
  if (!more?.length) return r;
  const all = [...r.sentTo];
  for (const w of more) if (!all.some((x) => x.label === w.label && x.what === w.what)) all.push(w);
  return { ...r, sentTo: all.slice(0, 24) };
}

/** One of the core's messages about an area, merged in (another area's is left alone). A finished
 *  action's note clears after a moment: `noteClear` says which and when. */
export function resultMsg<R extends Result>(r: R, m: CoreMsg): { r: R; noteClear?: { id: string; ms: number } } {
  switch (m.t) {
    case "lens_local":
      return m.data.sel === r.n ? { r: { ...r, local: m.data } } : { r };
    case "lens_answer":
      return m.data.sel === r.n ? { r: went({ ...r, answer: m.data }, m.data.sentTo) } : { r };
    case "lens_links":
      return m.data.sel === r.n ? { r: went({ ...r, links: m.data.rows }, m.data.sentTo) } : { r };
    case "lens_reply": {
      const d = m.data;
      if (d.sel !== r.n) return { r };
      const turn: Turn = { qid: d.qid, q: "", state: d.state, text: d.text, error: d.error, brain: d.brain?.label };
      const i = r.thread.findIndex((t) => t.qid === d.qid);
      // A question asked by voice arrives with its answer only (its words went to the voice).
      const thread = i >= 0 ? r.thread.map((t, k) => (k === i ? { ...t, ...turn, q: t.q } : t)) : [...r.thread, turn];
      return { r: went({ ...r, thread: thread.slice(-MAX_THREAD) }, d.sentTo) };
    }
    case "lens_translation": {
      const d = m.data;
      return d.sel === r.n ? { r: went({ ...r, translation: { to: d.to, text: d.text, error: d.error, pending: false } }, d.sentTo) } : { r };
    }
    case "lens_done": {
      const d = m.data;
      if (d.sel !== r.n) return { r };
      const next = { ...r, notes: { ...r.notes, [d.id]: doneText(d.note, d.ok) } };
      return d.close ? { r: next } : { r: next, noteClear: { id: d.id, ms: NOTE_MS } };
    }
  }
  return { r };
}

/** A click or key on the results: the next results, what to ask the core for, the next question's
 *  number — and, for a language picked from the Translate menu, the user's language from now on. */
export function resultAction<R extends Result>(r: R, a: ResultAction, nextQid: number): { r: R; ops: ResultOp[]; nextQid: number; translateTo?: string } {
  switch (a.t) {
    case "chip": {
      if (a.kind === "translate" && a.to) {
        return { r: { ...r, translation: { to: a.to, pending: true } }, ops: [{ op: "translate", to: a.to }], nextQid };
      }
      if (!a.id) return { r, ops: [], nextQid };
      const id = a.id;
      const next = { ...r, notes: { ...r.notes, [id]: "…" }, searchedWeb: r.searchedWeb || a.kind === "image_search" };
      return { r: next, ops: [{ op: "act", id }], nextQid };
    }
    case "translate_to": {
      const out = resultAction(r, { t: "chip", kind: "translate", to: a.to }, nextQid);
      return { ...out, translateTo: a.to };
    }
    case "ask": {
      const q = a.q.replace(/\s+/g, " ").trim().slice(0, 500);
      if (!q) return { r, ops: [], nextQid };
      const thread = [...r.thread, { qid: nextQid, q, state: "thinking" as const }].slice(-MAX_THREAD);
      return { r: { ...r, thread }, ops: [{ op: "ask", qid: nextQid, q }], nextQid: nextQid + 1 };
    }
    case "ask_again":
      return { r: { ...r, answer: { sel: r.n, state: "thinking", actions: [] } }, ops: [{ op: "ask_again" }], nextQid };
    case "continue":
      return { r, ops: [{ op: "continue", text: a.text }], nextQid };
    case "listen":
      return { r, ops: [{ op: "listen" }], nextQid };
    case "note_clear": {
      const notes = { ...r.notes };
      delete notes[a.id];
      return { r: { ...r, notes }, ops: [], nextQid };
    }
  }
  return { r, ops: [], nextQid };
}

/** "A", "A and B", "A, B and C". */
const listed = (xs: string[]) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

/** The privacy line under the results: every place the area, its text or a search phrase from it
 *  went (the core says each brain as it is asked, the ones that lost the race or failed too) —
 *  "Stayed on this Mac" only when nothing left it. */
export function privacyLine(r: Result | null, hello: Pick<ResultHello, "eyes" | "search"> | null): string {
  if (!r) return "Stayed on this Mac";
  const all: Sent[] = [...r.sentTo];
  const add = (label: string | undefined, local: boolean, what: Sent["what"]) => {
    if (label && !all.some((w) => w.label === label && w.what === what)) all.push({ label, local, what });
  };
  const a = r.answer;
  if (a?.brain && a.state !== "thinking") add(a.brain.label, a.brain.local, "picture");
  // Asked, not answered yet: the brain it goes to first (its own message names it once it's asked).
  if (a?.state === "thinking" && hello?.eyes) add(hello.eyes.label, hello.eyes.local, "picture");
  if (r.searchedWeb) add(hello?.search?.label ?? "the search provider", false, "picture");
  const cloud = all.filter((w) => !w.local);
  const names = (what: Sent["what"]) => [...new Set(cloud.filter((w) => w.what === what).map((w) => w.label))];
  const pics = names("picture");
  const texts = names("text").filter((l) => !pics.includes(l));
  const parts: string[] = [];
  if (pics.length) parts.push(`Sent to ${listed(pics)}: only this area`);
  if (texts.length) parts.push(parts.length ? `its text to ${listed(texts)}` : `Its text went to ${listed(texts)}`);
  if (cloud.some((w) => w.what === "query")) parts.push(parts.length ? "a search phrase to the web" : "A search phrase went to the web");
  return parts.length ? parts.join(" · ") : "Stayed on this Mac";
}

/** A follow-up's words when it got no answer (lens_reply's error). */
export function replyProblem(error = ""): string {
  switch (error) {
    case "key": return "A brain's key was refused — Brains & Keys can fix that.";
    case "busy": return "Busy — try again in a minute.";
    case "offline": return "You're offline.";
    case "no_brain":
    case "local_only": return "No brain here can see pictures.";
    default: return "No answer this time.";
  }
}

/** The answer slot's words for a failed answer (design §13). */
export function answerProblem(a: AnswerData | null, eyesWhy = ""): string | null {
  if (!a || a.state !== "failed") return null;
  switch (a.error) {
    case "no_brain": return "On-device results only. To get answers, give NEO a brain that can see pictures.";
    case "local_only": return "Your brains stay on this Mac and none of them can see pictures.";
    case "offline": return "You're offline — here's what NEO found on this Mac.";
    case "dark": return "This area came out black — protected video can't be captured.";
    case "refused": return `${a.brain?.label ?? "The brain"} wouldn't answer about this.`;
    case "key": return `${a.brain?.label ?? "A brain"}'s key was refused — Brains & Keys can fix that.`;
    case "busy": return a.brain ? `${a.brain.label} is busy — try again in a minute.` : "The brains are busy — try again in a minute.";
    case "off": return eyesWhy ? "On-device results only." : "Ask NEO about this area when you're ready.";
    default: return "No answer this time.";
  }
}

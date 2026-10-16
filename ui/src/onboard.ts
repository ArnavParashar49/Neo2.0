// "Give NEO a brain": the card the chat shows (in NEO's window and in the notch) while NEO has no
// AI model to ask (hello.brains.ready false, or a `setup_needed` from a request that needed one).
// Its button asks the core to open Brains & Keys (`open_keys`: the core asks NEO.app's shell, so
// the page gets no door of its own to the shell); with no NEO.app at the other end — a core started
// by hand — it shows the terminal command instead.

import type { BrainsSummary, SetupNeeded } from "./keys/types.ts";

export const TITLE = "Give NEO a brain";
export const BODY = "NEO already does simple things on this Mac (volume, apps, timers). To chat and do tasks, connect an AI model. Gemini has a free tier, no card needed.";
export const ALSO = "Also works with Claude, OpenAI, Groq (free), OpenRouter, or a model on this Mac.";
export const SET_UP = "Set up a brain";
export const GET_KEY = "Get a free Gemini key";
export const GEMINI_KEY_URL = "https://aistudio.google.com/apikey";
export const COMMAND = "python -m neo2 keys set gemini";

/** The card shows while connected and the core says NEO has no model; an older core (no "brains"
 *  in its hello) never shows it. */
export function needsBrain(connected: boolean, brains: BrainsSummary | null | undefined): boolean {
  return connected && !!brains && brains.ready === false;
}

/** The health line the card says better: hidden while the card is up. */
export function isNoModelProblem(text: string): boolean {
  return /^NEO has no AI model yet\b/.test(text);
}

export type CardState = "idle" | "opening" | "opened" | "terminal" | "busy";

/** What the card shows after `open_keys` answered (or a setup_needed came): the window opened, the
 *  terminal command (no NEO.app), or "already opening" (asked in the last few seconds). */
export function afterOpenKeys(result: { ok?: boolean; error?: string } | null | undefined): CardState {
  if (!result) return "idle";
  if (result.ok) return "opened";
  if (result.error === "busy") return "busy";
  return "terminal"; // no_app (a core started by hand), forbidden, or anything else: the CLI works
}

export function afterSetupNeeded(s: SetupNeeded | null | undefined): CardState {
  if (!s) return "idle";
  return s.shell ? (s.opened ? "opened" : "idle") : "terminal";
}

/** The card's one-line status under its buttons. */
export function cardNote(state: CardState): string | null {
  switch (state) {
    case "opening": return "Opening Brains & Keys…";
    case "opened": return "Brains & Keys is open — paste a key there.";
    case "busy": return "Brains & Keys is opening.";
    case "terminal": return `In a terminal, run: ${COMMAND}`;
    default: return null;
  }
}

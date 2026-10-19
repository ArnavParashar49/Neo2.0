// The on-device voice (Parakeet listens, Kokoro speaks): what the core says of it (the `brains`
// summary's "local_voice", then each `local_voice` event — neo2/core/fetch.py) and the words Brains &
// Keys and the chat's "Give NEO a brain" card show for it. NEO never downloads it by itself: the
// card's and the window's button sends `voice_download` (neo2/core/server/ws.py), and the progress
// comes back as events to every window.

import type { BrainsSummary, LocalVoice } from "./types.ts";

export const DOWNLOAD_CMD = "voice_download";

/** Bytes as Finder counts them: "2.9 GB", "160 MB". */
export function sizeText(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`;
}

export function etaText(s: number | null | undefined): string {
  if (typeof s !== "number" || !Number.isFinite(s) || s <= 0) return "";
  if (s < 60) return "less than a minute left";
  const m = Math.round(s / 60);
  return m === 1 ? "about a minute left" : `about ${m} minutes left`;
}

/** What the core said of the on-device voice, or null (an older core that doesn't say). */
export function localVoiceOf(b: BrainsSummary | null | undefined): LocalVoice | null {
  const lv = b?.local_voice;
  return lv && typeof lv === "object" && typeof lv.state === "string" ? lv : null;
}

/** The button: "Download on-device voice (2.9 GB)". */
export function downloadLabel(lv: LocalVoice | null): string {
  const size = sizeText(lv?.size ?? 0);
  return size ? `Download on-device voice (${size})` : "Download on-device voice";
}

/** 0–100 while downloading, else null. */
export function percent(lv: LocalVoice | null): number | null {
  if (!lv || lv.state !== "downloading") return null;
  const total = lv.total || lv.size || 0;
  return total > 0 ? Math.max(0, Math.min(100, Math.floor((100 * (lv.done ?? 0)) / total))) : 0;
}

const FAILED: Readonly<Record<string, string>> = {
  offline: "NEO couldn't reach Hugging Face. Check the internet connection, then try again.",
  disk_full: "There isn't enough free space on this Mac for it. Free some up, then try again.",
  checksum: "The download was damaged. Try again.",
  not_found: "Its files aren't on Hugging Face any more. An update of NEO fixes this.",
};

/** One line under the button: how far the download is, or why it stopped. */
export function statusLine(lv: LocalVoice | null): string | null {
  if (!lv) return null;
  switch (lv.state) {
    case "downloading": {
      const pct = percent(lv) ?? 0;
      const total = lv.total || lv.size || 0;
      const eta = etaText(lv.eta_s);
      return `Downloading the on-device voice… ${pct}%${total ? ` of ${sizeText(total)}` : ""}${eta ? ` · ${eta}` : ""}`;
    }
    case "failed": return FAILED[lv.code ?? ""] ?? "The download didn't finish. Try again.";
    case "ready": return "Downloaded. NEO's voice works on this Mac without a key or the internet.";
    default: return null;
  }
}

/** The chat's card offers the on-device voice while NEO has no voice it could use: no Gemini key
 *  (Gemini Live needs one) and the voice not downloaded. Never for an older core that doesn't say. */
export function offerOnCard(b: BrainsSummary | null | undefined): boolean {
  const lv = localVoiceOf(b);
  if (!lv || lv.state === "ready") return false;
  return b?.providers?.gemini !== true;
}

/** What answers a `voice_download` says of it (`local_voice`), folded into the summary. */
export function withLocalVoice(b: BrainsSummary | null, lv: unknown): BrainsSummary | null {
  if (!b || !lv || typeof lv !== "object" || typeof (lv as LocalVoice).state !== "string") return b;
  return { ...b, local_voice: lv as LocalVoice };
}

/** A `voice_download` the core didn't start, in words ("ready" needs none: the line says it). */
export function downloadErrorText(error: string): string {
  switch (error) {
    case "busy": return "It's downloading already.";
    case "not_running": return "Nothing is downloading.";
    case "forbidden": return "This window can't start downloads.";
    case "offline": case "disconnected": return "NEO's core isn't answering. This window reconnects by itself.";
    case "timeout": return "NEO's core didn't answer in time. Try again.";
    default: return "NEO couldn't start the download. Try again.";
  }
}

// Words the card shows: hosts (with the registrable domain picked out), captions built from
// NEO's actions (present tense, §9), times. Page-controlled strings only ever render as text.
import type { OrbState } from "../ws";
import type { BrowserAction, BrowserState } from "./types";

const MULTI_SUFFIX = new Set([
  "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "ltd.uk", "plc.uk", "com.au", "net.au", "org.au", "edu.au", "gov.au",
  "co.nz", "org.nz", "co.jp", "ne.jp", "or.jp", "ac.jp", "co.in", "net.in", "org.in", "firm.in", "gen.in", "ind.in",
  "com.br", "net.br", "org.br", "com.mx", "com.ar", "com.co", "com.tr", "com.cn", "net.cn", "org.cn", "com.hk", "com.sg",
  "com.tw", "co.kr", "or.kr", "co.za", "co.il", "com.my", "com.ph", "com.pk", "com.sa", "com.eg", "com.ng", "co.id",
  "github.io", "gitlab.io", "vercel.app", "netlify.app", "pages.dev", "herokuapp.com", "blogspot.com", "azurewebsites.net",
]);

export function hostOf(url: string): string {
  try { return new URL(url).hostname; } catch { return ""; }
}

const IP = /^(\d{1,3}\.){3}\d{1,3}$|^\[?[0-9a-f:]+\]?$/i;

/** "smile.amazon.com" → { sub: "smile.", reg: "amazon.com" }. Lookalikes ("amazon.com.evil.io")
 *  put the real owner in `reg` ("evil.io"), which the address capsule emphasises. */
export function splitHost(host: string): { sub: string; reg: string } {
  const h = host.replace(/\.$/, "").toLowerCase();
  if (!h || IP.test(h) || !h.includes(".")) return { sub: "", reg: h };
  const parts = h.split(".");
  const last2 = parts.slice(-2).join(".");
  const n = MULTI_SUFFIX.has(last2) ? 3 : 2;
  if (parts.length <= n) return { sub: "", reg: h };
  const reg = parts.slice(-n).join(".");
  return { sub: h.slice(0, h.length - reg.length), reg };
}

/** "www.amazon.com" → "amazon.com" (for prose: "Browsing · amazon.com"). */
export function displayHost(url: string): string {
  const h = hostOf(url);
  return h.replace(/^www\d?\./, "") || url;
}

export function isLocalHost(host: string): boolean {
  return host === "localhost" || host.endsWith(".localhost") || host === "127.0.0.1" || host === "[::1]" || host === "::1"
    || /^10\./.test(host) || /^192\.168\./.test(host) || /^172\.(1[6-9]|2\d|3[01])\./.test(host) || host.endsWith(".local");
}

/** Quoted page label: curly quotes, cut to 32 chars + "…". */
export function q(s: string | undefined, n = 32): string {
  const t = (s ?? "").replace(/\s+/g, " ").trim();
  return `“${t.length > n ? t.slice(0, n).trimEnd() + "…" : t}”`;
}

export interface Caption { text: string; ctx?: string; orb: OrbState }

const ORB_FOR: Record<string, OrbState> = {
  click: "working", type: "working", key: "working", select: "working", move: "working", scroll: "working",
  read: "searching", navigate: "connecting", wait: "connecting",
};

/** Present-tense caption for an action (the core's `caption` wins when present). */
export function captionFor(a: BrowserAction): Caption {
  const orb = ORB_FOR[a.kind] ?? "working";
  if (a.gated && a.phase === "intent") {
    return { text: "Waiting for your OK", ctx: `· Click ${q(a.label)}`, orb: "shaping" };
  }
  if (a.caption) return { text: a.caption, orb };
  const lab = a.label ? q(a.label) : "";
  switch (a.kind) {
    case "navigate":
      return { text: a.url ? `Opening ${displayHost(a.url)}` : "Opening the page", orb };
    case "click": {
      const what = lab || `the ${a.role || "element"}`;
      if ((a.clicks ?? 1) >= 2) return { text: `Double-clicking ${what}`, orb };
      if (a.button === "right") return { text: `Right-clicking ${what}`, orb };
      return { text: `Clicking ${what}`, orb };
    }
    case "type":
      if (a.secret || !a.text) return { text: `Typing in ${a.field || "the field"}`, orb };
      return a.field ? { text: `Typing ${q(a.text)}`, ctx: `in ${a.field}`, orb } : { text: `Typing ${q(a.text)}`, orb };
    case "key":
      return { text: `Pressing ${a.key || "a key"}`, orb };
    case "select":
      return a.field ? { text: `Choosing ${q(a.label || a.text)}`, ctx: `in ${a.field}`, orb } : { text: `Choosing ${q(a.label || a.text)}`, orb };
    case "scroll":
      if (a.label) return { text: `Scrolling to ${lab}`, orb };
      return { text: `Scrolling ${a.dir ?? "down"}`, orb };
    case "read":
      return { text: a.label ? `Reading ${lab}` : "Reading the page", orb };
    case "wait":
      return { text: a.label ? `Waiting for ${lab}` : "Waiting for the page", orb };
    case "move":
      return { text: `Pointing at ${lab || "the page"}`, orb };
  }
  return { text: "Working", orb };
}

export function fmtDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Epoch seconds or ms → seconds. */
export function toSec(t: number): number { return t > 1e12 ? t / 1000 : t; }

/** "Amazon" from "smile.amazon.com" — from the host only, never from page text. */
export function siteName(url: string): string {
  const { reg } = splitHost(hostOf(url));
  const first = reg.split(".")[0] ?? "";
  return first ? first[0].toUpperCase() + first.slice(1) : "this site";
}

/** The status line under NEO's name while browsing (compact header). */
export function browsingLine(s: BrowserState | null): string {
  if (!s) return "";
  const host = displayHost(s.url);
  switch (s.status) {
    case "starting": return "Starting the browser…";
    case "waiting": return "Waiting for your OK";
    case "need_user": return `Your turn on ${host}`;
    case "user": return "You're in control";
    case "error": return `Browser problem · ${host}`;
    default: return `Browsing · ${host}`;
  }
}

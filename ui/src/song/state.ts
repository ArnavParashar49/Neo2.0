// The song card's state as one pure reducer: every server event and every local step (Stop pressed,
// a reconnect, a session that turned out to be over) goes through `reduce`, so the card's logic is
// tested without React, a socket or a microphone (ui/tests/song-state.test.ts). What arrives is
// checked here once — a cover is only ever a data URL, a link only ever https — so the components
// can trust what they draw. Nothing in this file imports at run time.
import type { SongCloseReason, SongLinkKind, SongResult, SongSource, SongStatus } from "./types";

export const STATUSES: readonly SongStatus[] = ["listening", "matching", "found", "not_found", "error"];
const SOURCES: readonly SongSource[] = ["mic", "mac", "both"];
const CLOSE_REASONS: readonly SongCloseReason[] = ["done", "stopped", "error"];
/** Ended cards kept for the transcript; older ones are forgotten, covers and all (memory stays bounded). */
export const MAX_ENDED = 24;
/** A cover bigger than this (about 1 MB of JPEG) is dropped rather than kept. */
export const MAX_COVER_CHARS = 1_500_000;
/** The core's default listening window (song_listen_s), for a session this client never saw open. */
export const DEFAULT_LISTEN_S = 12;

export const NOT_FOUND_TEXT = "Couldn't recognise it";
export const ERROR_TEXT = "Song recognition isn't working right now";
export const STOPPED_TEXT = "Stopped listening";

export interface SongSession {
  sid: string;
  /** null: joined late (a reload), so what it listens to isn't known. */
  source: SongSource | null;
  listenS: number;
  status: SongStatus;
  level: number;
  elapsedS: number;
  message: string | null;
  result: SongResult | null;
  /** Date.now() of the latest state: the card's clock runs on from `elapsedS` between updates. */
  at: number;
  /** Stop was pressed; waiting for the core's song_close. */
  stopping: boolean;
  /** Live before a reconnect: kept until song_hello's answer says whether it still is. */
  unconfirmed: boolean;
  /** Why it ended (null while live). "gone": live before a reconnect, and not any more. */
  closed: SongCloseReason | "gone" | null;
}

export interface SongSnap {
  connected: boolean;
  live: SongSession | null;
  ended: Record<string, SongSession>;
  /** Ended sids, oldest first (at most MAX_ENDED). */
  order: string[];
  /** The newest session's sid, live or ended: the one result whose links the core still opens. */
  latest: string | null;
}

export const EMPTY: SongSnap = { connected: false, live: null, ended: {}, order: [], latest: null };

export type SongAction =
  | { t: "event"; type: string; data: unknown }
  | { t: "connected"; on: boolean }
  /** A (re)connect's replay has landed: a session still live here must be confirmed by the core. */
  | { t: "hello_end" }
  /** No answer to song_hello for this session: it ended while this client was away. */
  | { t: "gone"; sid: string }
  | { t: "stop" }
  /** The transcript was cleared or erased: ended cards go, a live one stays. */
  | { t: "clear" };

// ---- what arrives ------------------------------------------------------------------------------

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Obj) : null);

function num(v: unknown, lo: number, hi: number, dflt: number): number {
  const n = typeof v === "number" ? v : typeof v === "string" && v.trim() ? Number(v) : NaN;
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt;
}

function text(v: unknown, max = 200): string | null {
  if (typeof v === "number" && Number.isFinite(v)) v = String(v);
  if (typeof v !== "string") return null;
  const s = v.replace(/\s+/g, " ").trim();
  if (!s) return null;
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}

// Raster images only (an SVG can carry more than a picture), base64, nothing that could end a CSS
// string or an attribute.
const COVER = /^data:image\/(?:jpeg|png|webp|gif);base64,[A-Za-z0-9+/]+={0,2}$/;

/** The cover as the core sent it, if it is a data URL of a raster image — never a remote URL. */
export function safeCover(v: unknown): string | null {
  return typeof v === "string" && v.length <= MAX_COVER_CHARS && COVER.test(v) ? v : null;
}

/** An https link with a host and no credentials in it, or null. */
export function safeLink(v: unknown): string | null {
  if (typeof v !== "string" || v.length > 2048) return null;
  try {
    const u = new URL(v);
    return u.protocol === "https:" && u.hostname && !u.username && !u.password ? u.href : null;
  } catch {
    return null;
  }
}

/** A result the card can show (a title at least), everything in it checked; null otherwise. */
export function normResult(v: unknown): SongResult | null {
  const r = obj(v);
  const title = r ? text(r.title) : null;
  if (!r || !title) return null;
  const links = obj(r.links) ?? {};
  return {
    title,
    artist: text(r.artist) ?? "",
    album: text(r.album),
    year: text(r.year, 12),
    genre: text(r.genre, 60),
    cover: safeCover(r.cover),
    links: { apple_music: safeLink(links.apple_music), shazam: safeLink(links.shazam) },
  };
}

// ---- the reducer ---------------------------------------------------------------------------------

function fresh(sid: string, source: SongSource | null, listenS: number, now: number): SongSession {
  return {
    sid, source, listenS, status: "listening", level: 0, elapsedS: 0, message: null, result: null,
    at: now, stopping: false, unconfirmed: false, closed: null,
  };
}

function merge(cur: SongSession, d: Obj, now: number): SongSession {
  const status = STATUSES.includes(d.status as SongStatus) ? (d.status as SongStatus) : cur.status;
  const hearing = status === "listening" || status === "matching";
  return {
    ...cur,
    status,
    level: hearing ? num(d.level, 0, 1, cur.level) : 0,
    elapsedS: num(d.elapsed_s, 0, 3600, cur.elapsedS),
    message: "message" in d ? text(d.message, 300) : cur.message,
    result: "result" in d ? normResult(d.result) : cur.result,
    at: now,
    unconfirmed: false,
  };
}

/** Keep an ended session (newest last), forgetting the oldest beyond MAX_ENDED. */
function keep(s: SongSnap, e: SongSession): SongSnap {
  const order = [...s.order.filter((x) => x !== e.sid), e.sid];
  const ended = { ...s.ended, [e.sid]: e };
  while (order.length > MAX_ENDED) delete ended[order.shift()!];
  return { ...s, ended, order };
}

function drop(s: SongSnap, sid: string): SongSnap {
  if (!s.ended[sid]) return s;
  const ended = { ...s.ended };
  delete ended[sid];
  return { ...s, ended, order: s.order.filter((x) => x !== sid) };
}

function end(s: SongSnap, reason: SongSession["closed"]): SongSnap {
  const l = s.live;
  if (!l) return s;
  return keep({ ...s, live: null }, { ...l, closed: reason, stopping: false, unconfirmed: false, level: 0 });
}

function onEvent(s: SongSnap, type: string, d: Obj, now: number): SongSnap {
  const sid = typeof d.sid === "string" ? d.sid : "";
  if (!sid) return s;
  switch (type) {
    case "song_open": {
      if (s.live?.sid === sid) return s;
      const old = s.ended[sid];
      if (old && old.closed !== "gone") return s; // a replay of one that already ended here
      // One session at a time: an open while another shows live means that one is over.
      const next = drop(s.live ? end(s, "gone") : s, sid);
      const source = SOURCES.includes(d.source as SongSource) ? (d.source as SongSource) : null;
      return { ...next, live: fresh(sid, source, num(d.listen_s, 1, 120, DEFAULT_LISTEN_S), now), latest: sid };
    }
    case "song_state": {
      const cur = s.live;
      if (cur && cur.sid === sid) return { ...s, live: merge(cur, d, now) };
      const old = s.ended[sid];
      if (old && old.closed !== "gone") return s;
      if (cur && !cur.unconfirmed) return s; // another session is live and confirmed
      // Joined late (a reload), or wrongly taken for over across a reconnect: it is live now.
      const base = old ? { ...old, closed: null, stopping: false } : fresh(sid, null, DEFAULT_LISTEN_S, now);
      const next = drop(cur ? end(s, "gone") : s, sid);
      return { ...next, live: merge(base, d, now), latest: sid };
    }
    case "song_close": {
      if (d.reason === "erased") return { ...EMPTY, connected: s.connected };
      const reason = CLOSE_REASONS.includes(d.reason as SongCloseReason) ? (d.reason as SongCloseReason) : "done";
      if (s.live?.sid === sid) return end(s, reason);
      const old = s.ended[sid];
      return old && old.closed === "gone" ? keep(s, { ...old, closed: reason }) : s;
    }
  }
  return s;
}

/** The next snapshot (the same object when nothing changed, so React skips the render). */
export function reduce(s: SongSnap, a: SongAction, now: number): SongSnap {
  switch (a.t) {
    case "event":
      return onEvent(s, a.type, obj(a.data) ?? {}, now);
    case "connected":
      if (a.on) return s.connected ? s : { ...s, connected: true };
      return { ...s, connected: false, live: s.live?.stopping ? { ...s.live, stopping: false } : s.live };
    case "hello_end":
      return s.live && !s.live.unconfirmed ? { ...s, live: { ...s.live, unconfirmed: true } } : s;
    case "gone":
      return s.live?.sid === a.sid && s.live.unconfirmed ? end(s, "gone") : s;
    case "stop":
      return s.live && !s.live.stopping && hearing(s.live) ? { ...s, live: { ...s.live, stopping: true } } : s;
    case "clear":
      return { ...s, ended: {}, order: [], latest: s.live ? s.live.sid : null };
  }
  return s;
}

// ---- what the overlay needs (not the ticking level) ---------------------------------------------

/** The part of the snapshot the overlay itself reacts to: which card is live and what face it
 *  shows, and which ended cards exist. It changes a few times a session, not ten times a second,
 *  so the whole chat doesn't re-render with every loudness update (each card follows its own). */
export interface SongShape {
  liveSid: string | null;
  liveView: SongView | null;
  /** NEO is listening now (the overlay stays open; the orb shows it). */
  hearing: boolean;
  ended: Record<string, SongSession>;
}

export const EMPTY_SHAPE: SongShape = { liveSid: null, liveView: null, hearing: false, ended: {} };

/** The shape of `s`, reusing `prev` when nothing in it changed. */
export function shapeOf(s: SongSnap, prev: SongShape): SongShape {
  const l = s.live;
  const next: SongShape = { liveSid: l ? l.sid : null, liveView: l ? viewOf(l) : null, hearing: !!l && hearing(l), ended: s.ended };
  return next.liveSid === prev.liveSid && next.liveView === prev.liveView && next.hearing === prev.hearing && next.ended === prev.ended ? prev : next;
}

/** One session by sid, live or ended. */
export function sessionOf(s: SongSnap, sid: string): SongSession | null {
  return s.live?.sid === sid ? s.live : s.ended[sid] ?? null;
}

// ---- what the card shows ------------------------------------------------------------------------

export type SongView = "listening" | "matching" | "found" | "not_found" | "error" | "stopped";

/** Still taking in sound (Stop means something). */
export function hearing(x: SongSession): boolean {
  return x.closed === null && (x.status === "listening" || x.status === "matching");
}

/** Which face of the card a session shows, live or ended. */
export function viewOf(x: SongSession): SongView {
  if (x.status === "found") return x.result ? "found" : "not_found";
  if (x.status === "not_found" || x.status === "error") return x.status;
  if (x.closed === null) return x.status;
  if (x.closed === "error") return "error";
  if (x.closed === "done") return "not_found"; // over without a match
  return "stopped";
}

export function sourceLabel(src: SongSource | null): string {
  switch (src) {
    case "mic": return "to the room";
    case "mac": return "to your Mac";
    case "both": return "to the room and your Mac";
  }
  return "for a song";
}

/** "4 s"; with `precise`, "4.2 s" under ten seconds. */
export function fmtSecs(s: number, precise = false): string {
  const v = Number.isFinite(s) ? Math.max(0, s) : 0;
  return (precise && v < 10 ? v.toFixed(1) : String(Math.round(v))) + " s";
}

/** Seconds listened so far: the core's count, run on locally between its updates, never past the window. */
export function clockAt(x: SongSession, now: number): number {
  if (!hearing(x)) return x.elapsedS;
  return Math.min(Math.max(x.listenS, x.elapsedS), x.elapsedS + Math.max(0, now - x.at) / 1000);
}

/** "Album · 2020 · Pop", whatever of it is known. */
export function extraLine(r: SongResult): string {
  return [r.album, r.year, r.genre].filter(Boolean).join(" · ");
}

/** The line a screen reader hears when the card changes (never the ticking seconds). */
export function announce(x: SongSession): string {
  switch (viewOf(x)) {
    case "listening": return `Listening ${sourceLabel(x.source)}`;
    case "matching": return "Matching the song";
    case "found": return `Found: ${x.result!.title}${x.result!.artist ? ` by ${x.result!.artist}` : ""}`;
    case "not_found": return x.message ?? NOT_FOUND_TEXT;
    case "error": return x.message ?? ERROR_TEXT;
    case "stopped": return STOPPED_TEXT;
  }
}

/** How loud it looks (0..1): a square-root curve, so ordinary room loudness (0.2–0.5) fills the
 *  meter visibly instead of hugging the floor. */
export function shownLevel(level: number): number {
  return Number.isFinite(level) ? Math.sqrt(Math.min(1, Math.max(0, level))) : 0;
}

const MIN_BAR = 0.12;
/** Equaliser bar heights (0.12..1) for a loudness: tallest in the middle, with a little variety
 *  from `seed` (the state's timestamp) so it looks alive without inventing any sound. */
export function eqBars(level: number, seed: number, n = 5): number[] {
  const lv = shownLevel(level);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const w = 0.55 + 0.45 * Math.sin((Math.PI * (i + 0.5)) / n);
    const r = Math.sin((seed % 100_000) * 12.9898 + i * 78.233) * 43758.5453;
    const j = r - Math.floor(r);
    out.push(Math.min(1, Math.max(MIN_BAR, lv * w * (0.7 + 0.6 * j))));
  }
  return out;
}

/** The link the core will open for this card, if any: only the newest session's, only https. */
export function linkFor(s: SongSnap, sid: string, which: SongLinkKind): string | null {
  if (sid !== s.latest) return null;
  const x = s.live?.sid === sid ? s.live : s.ended[sid];
  return x?.result?.links[which] ?? null;
}

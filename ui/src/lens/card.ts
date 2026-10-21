// Search Screen's card in the notch, on NEO's page there: when NotchHub hosts NEO, the overlay
// hands an area to the core instead of showing its sheet, and the core sends this page the area's
// results (`lens_card`, then every result tagged with the card's `cid`). This hub keeps the one
// live card — the same results as the sheet (./result.ts) — and sends the card's own commands
// (`lens_card_*`, each naming its cid). React reads it with useSyncExternalStore (useLensCard).
//
// Only in the notch: the page says `lens_card_hello` after each hello there, and nowhere else (NEO's
// own window never shows a card). A chat turn takes the panel back: the user speaking to NEO (but not
// the one sentence the card's mic asked for), a browser or a song card opening — and so does anything
// of the chat's own that must be seen (`needChat`: NEO asking for an OK, a browser's turn for the
// user), which no card ever hides.
//
// A card NotchHub's panel folds away (Esc, the ×, a click elsewhere, an action that opened something —
// any close but the ones after which the panel stays, as the chat) stays on the page, as it was, until
// the page is off screen: the island's content fades out with the card in it, as it fades the chat
// out — never the chat swapped in under the fold, never a new size asked for on the way down.

import { useSyncExternalStore } from "react";
import { emptyResult, resultAction, resultMsg, type Result, type ResultAction, type ResultHello, type ResultOp } from "./result.ts";
import { cardSend, CID, parseCard, parseLens, type CardData } from "./wire.ts";

export interface CardSnap {
  cid: string;
  r: Result;
  hello: ResultHello;
  thumb: string | null;
  translateTo: string | null;
  /** The card's mic is armed: the next sentence is a question about the area. */
  listening: boolean;
  nextQid: number;
  /** Bumped by every change (the card measures itself again). */
  v: number;
}

type Send = (cmd: Record<string, unknown>) => boolean;
export interface Clock { now(): number; set(cb: () => void, ms: number): number; clear(id: number): void }
const realClock: Clock = {
  now: () => Date.now(),
  set: (cb, ms) => setTimeout(cb, ms) as unknown as number,
  clear: (id) => clearTimeout(id),
};

/** A user's sentence within this long after the mic stopped listening is still the card's own. */
export const LISTEN_GRACE_MS = 2000;

/** At most this long a card the panel folds away stays on the page. NotchHub's island lets its content
 *  go within 0.15 s and takes the page off screen (`visibilitychange`) by about 0.2 s; this is only
 *  the fallback for a host that never does. */
export const FOLD_HOLD_MS = 600;

/** Closes after which NotchHub keeps its panel, as the chat (Continue in NEO, a chat turn), or never
 *  opened it for the card: the chat comes back at once. Every other close folds the panel. */
export const CLOSE_KEEPS_PANEL: ReadonlySet<string> = new Set(["continue", "chat", "unshown"]);

/** Whether the page is on screen (in the notch: NotchHub's panel is open with it). */
const pageVisible = () => typeof document !== "undefined" && document.visibilityState === "visible";

export class LensCardHub {
  private s: CardSnap | null = null;
  private listeners = new Set<() => void>();
  private sendFn: Send = () => false;
  private replaying = false;
  private listenOffAt = -Infinity;
  private notes = new Map<string, number>();
  private chatNeeded = false;
  /** The card closed and NotchHub's panel is folding it away: the fallback that drops it. */
  private foldHold: number | null = null;
  readonly inNotch: boolean;
  private clock: Clock;
  private visible: () => boolean;

  constructor(inNotch: boolean, clock: Clock = realClock, visible: () => boolean = pageVisible) {
    this.inNotch = inNotch;
    this.clock = clock;
    this.visible = visible;
  }

  // ---- store ----------------------------------------------------------------------------------
  snap = (): CardSnap | null => this.s;
  getSnapshot = (): CardSnap | null => this.s;
  subscribe = (fn: () => void) => { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; };
  private set(next: CardSnap | null) {
    this.endHold();
    this.s = next ? { ...next, v: (this.s?.v ?? 0) + 1 } : null;
    if (!next) this.clearNotes();
    for (const l of this.listeners) l();
  }
  private clearNotes() {
    for (const id of this.notes.values()) this.clock.clear(id);
    this.notes.clear();
  }

  get cid(): string | null { return this.s?.cid ?? null; }

  /** The card on the page is closed, and stays only while NotchHub's panel folds it away: nothing on
   *  it acts, nothing it hears changes it. */
  get folding(): boolean { return this.foldHold !== null; }

  // ---- the fold ---------------------------------------------------------------------------------
  /** The card closed (`why`): in the notch, unless the panel stays (or never opened), it stays as it
   *  is until the page is off screen — the panel folding takes it away with it. */
  private gone(why: string) {
    if (!this.inNotch || CLOSE_KEEPS_PANEL.has(why) || !this.visible()) { this.set(null); return; }
    if (this.foldHold !== null) return;
    this.clearNotes(); // (nothing on it changes now)
    this.foldHold = this.clock.set(() => { this.foldHold = null; this.set(null); }, FOLD_HOLD_MS);
  }

  private endHold() {
    if (this.foldHold === null) return;
    this.clock.clear(this.foldHold);
    this.foldHold = null;
  }

  /** The page went off screen (or came back): a card the panel was folding away is gone now. */
  visibilityChanged() {
    if (this.foldHold !== null && !this.visible()) this.set(null);
  }

  /** NotchHub opened its panel (`neo://opened`, whichever way; `neo://show`, at the user's hand): a
   *  card it was folding away gives the page back to the chat (opened again at once — before the page
   *  ever went off screen, so nothing else would). A new card has replaced it already. */
  panelShown() {
    if (this.foldHold !== null) this.set(null);
  }

  // ---- the connection ---------------------------------------------------------------------------
  setSender(fn: Send) { this.sendFn = fn; }

  /** The line to the core went: whatever card was up is gone with it (a reconnect asks again). */
  setConnected(on: boolean) {
    if (!on && this.s) this.set(null);
  }

  /** The hello's replay of old events is running: nothing in it is a chat turn now. */
  beginHello() { this.replaying = true; }
  /** …and it is over: in the notch, the page says it shows cards (the core answers with the live
   *  one, if there is one). */
  endHello() {
    this.replaying = false;
    if (this.inNotch) this.sendFn(cardSend.hello());
  }

  // ---- what the core says ------------------------------------------------------------------------
  /** `lens_card`, or a lens_* message carrying a cid. Returns whether it was the card's. */
  handle(type: string, data: unknown): boolean {
    if (type === "lens_card") {
      const m = parseCard(data);
      if (!m) return true;
      if (!m.open) {
        if (this.s?.cid === m.cid && this.foldHold === null) this.gone(m.why);
        return true;
      }
      if (!m.card) return true;
      if (this.chatNeeded) {
        // The chat must be seen now (a question waits for an OK): this card never shows — the core
        // hears it closed as a chat turn, and NotchHub keeps (or opens) nothing for it.
        this.sendFn(cardSend.close(m.cid, "chat"));
        if (this.s) this.set(null);
        return true;
      }
      this.set(this.fromCard(m.cid, m.card));
      return true;
    }
    const d = data && typeof data === "object" ? (data as Record<string, unknown>) : {};
    const s = this.s;
    if (!this.inNotch || typeof d.cid !== "string" || !CID.test(d.cid)) return false;
    if (!s || d.cid !== s.cid) return true; // (another card's: dropped)
    if (this.foldHold !== null) return true; // (closed, folding away: it stays as it was)
    const m = parseLens(type, data);
    if (!m) return true;
    if (m.t === "lens_listening") {
      if (!m.data.on && s.listening) this.listenOffAt = this.clock.now();
      this.set({ ...s, listening: m.data.on });
      return true;
    }
    const { r, noteClear } = resultMsg(s.r, m);
    if (r !== s.r) this.set({ ...s, r });
    if (noteClear) this.later(noteClear.id, noteClear.ms);
    return true;
  }

  private fromCard(cid: string, c: CardData): CardSnap {
    let r: Result = { ...emptyResult(c.sel, c.question), local: c.local, answer: c.answer, links: c.links, sentTo: c.sentTo };
    if (c.translation) r = { ...r, translation: { to: c.translation.to, text: c.translation.text, error: c.translation.error, pending: false } };
    r = { ...r, thread: c.thread.map((t) => ({ qid: t.qid, q: t.q, state: t.state, text: t.text, error: t.error, brain: t.brain?.label })) };
    const keep = this.s?.cid === cid ? this.s : null; // (asked again: the questions in flight stay numbered)
    return {
      cid, r, hello: c.hello, thumb: c.thumb ?? c.local?.thumb ?? null,
      translateTo: keep?.translateTo ?? (c.translateTo || null),
      listening: keep?.listening ?? false,
      nextQid: Math.max(keep?.nextQid ?? 1, ...c.thread.map((t) => (t.qid < 100_000 ? t.qid + 1 : 1))),
      v: 0,
    };
  }

  private later(id: string, ms: number) {
    const old = this.notes.get(id);
    if (old !== undefined) this.clock.clear(old);
    this.notes.set(id, this.clock.set(() => {
      this.notes.delete(id);
      const s = this.s;
      if (s) this.set({ ...s, r: resultAction(s.r, { t: "note_clear", id }, s.nextQid).r });
    }, ms));
  }

  // ---- what the user does --------------------------------------------------------------------------
  /** A click or key on the card's results: as the card's own commands. */
  op(a: ResultAction) {
    const s = this.s;
    if (!s || this.foldHold !== null) return;
    const out = resultAction(s.r, a, s.nextQid);
    this.set({ ...s, r: out.r, nextQid: out.nextQid, translateTo: out.translateTo ?? s.translateTo });
    for (const o of out.ops) this.sendFn(cardCmd(s.cid, o));
  }

  /** Esc, the ×, or a chat turn: the card closes now (the core tells NotchHub) — and, but for a chat
   *  turn, stays on the page while the panel folds it away. Returns whether it closed (once). */
  close(why: "esc" | "x" | "chat"): boolean {
    const s = this.s;
    if (!s || this.foldHold !== null) return false;
    this.sendFn(cardSend.close(s.cid, why));
    this.gone(why);
    return true;
  }

  /** Something of the chat's own must be seen while `on` — NEO asking for an OK (its confirm card),
   *  a browser session's turn for the user: a live card gives the panel back now, as a chat turn
   *  (NotchHub keeps the panel, as the chat, which shows it), and a card offered meanwhile closes the
   *  moment it comes. The card's keys (Esc, ⌘Return) never act on a question the user can't see. */
  needChat(on: boolean) {
    this.chatNeeded = on;
    if (on && this.s) this.close("chat");
  }

  /** An event of the chat: a turn of the user's own, a browser or a song card, takes the panel back. */
  chat(type: string, data: Record<string, unknown> | undefined) {
    if (!this.s || this.replaying) return;
    if (type === "browser_open" || type === "song_open") { this.close("chat"); return; }
    if (type !== "transcript" || data?.role !== "user" || data?.final !== true || data?.attach === "screen") return;
    // The sentence the card's mic asked for (a voice that also says it aloud as a transcript).
    if (this.s.listening || this.clock.now() - this.listenOffAt < LISTEN_GRACE_MS) return;
    this.close("chat");
  }
}

/** The card's command for one of the results' ops (the overlay sends its own: ./state.ts). */
export function cardCmd(cid: string, o: ResultOp): Record<string, unknown> {
  switch (o.op) {
    case "ask": return cardSend.ask(cid, o.qid, o.q);
    case "ask_again": return cardSend.ask(cid, 0, "");
    case "translate": return cardSend.translate(cid, o.to);
    case "act": return cardSend.act(cid, o.id);
    case "continue": return cardSend.cont(cid, o.text);
    case "listen": return cardSend.listen(cid);
  }
}

export function useLensCard(hub: LensCardHub): CardSnap | null {
  return useSyncExternalStore(hub.subscribe, hub.getSnapshot);
}

// ---- the card's size in the notch ---------------------------------------------------------------------

/** 372 wide (320 on a narrow screen), as tall as its content within 160–560 (and the host's room),
 *  in steps of 8; it grows at once, and shrinks only by more than 32 pt held for 300 ms, so the
 *  island never bounces. */
export const CARD_SIZE = { w: 372, narrowW: 320, minH: 160, maxH: 560, step: 8, shrinkPt: 32, shrinkMs: 300 } as const;

export function cardWidth(areaW?: number): number {
  if (!areaW || !Number.isFinite(areaW) || areaW >= CARD_SIZE.w) return CARD_SIZE.w;
  return Math.max(1, Math.min(CARD_SIZE.narrowW, Math.floor(areaW)));
}

export function cardHeight(natural: number, areaH?: number): number {
  const top = Math.min(CARD_SIZE.maxH, areaH && Number.isFinite(areaH) ? areaH : CARD_SIZE.maxH);
  const h = Math.ceil((Number.isFinite(natural) ? natural : 0) / CARD_SIZE.step) * CARD_SIZE.step;
  return Math.max(Math.min(CARD_SIZE.minH, top), Math.min(top, h));
}

/** What LensCard reads off the page to size itself (client px): the card's height and its scrolling
 *  column's, the column's top edge, how far it is scrolled, its padding, and the lowest bottom edge of
 *  the column's children (null: none). */
export interface CardMetrics { rootH: number; scrollH: number; scrollTop: number; scrolled: number; padTop: number; padBottom: number; contentBottom: number | null }

/** How tall the card would be with room for all of it: the fixed rows (the question box, the footer)
 *  plus the column's own content — never the room the panel gives the column now (it flexes to fill
 *  the panel, so its scrollHeight is at least the panel's height, and a card measured that way could
 *  only ever grow). */
export function naturalHeight(m: CardMetrics): number {
  const n = (v: number) => (Number.isFinite(v) ? v : 0);
  const fixed = Math.max(0, n(m.rootH) - n(m.scrollH));
  const content = m.contentBottom === null || !Number.isFinite(m.contentBottom)
    ? n(m.padTop) + n(m.padBottom)
    : Math.max(n(m.padTop), m.contentBottom - n(m.scrollTop) + n(m.scrolled)) + n(m.padBottom);
  return fixed + content;
}

/** The height to show now: `cur` (null: none yet) for content that wants `want`; `since` is when a
 *  shrink began waiting. `recheck`: come back in this many ms to apply it. */
export function settleHeight(cur: number | null, want: number, since: number | null, now: number): { h: number; since: number | null; recheck?: number } {
  if (cur === null || want >= cur) return { h: want, since: null };
  if (cur - want <= CARD_SIZE.shrinkPt) return { h: cur, since: null };
  if (since === null) return { h: cur, since: now, recheck: CARD_SIZE.shrinkMs };
  if (now - since >= CARD_SIZE.shrinkMs) return { h: want, since: null };
  return { h: cur, since, recheck: CARD_SIZE.shrinkMs - (now - since) };
}

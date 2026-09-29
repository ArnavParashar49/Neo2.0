// The song half of the websocket client: the session NEO is listening in (its level, clock and
// result), the ended ones as cards in the transcript, and the two commands the card sends back.
// The logic is the pure reducer in ./state; this only holds the snapshot, sends, and keeps the one
// timer a reconnect needs. React reads it via useSyncExternalStore: the overlay reads the coarse
// shape (which card, which face), each card its own session — so a level update ten times a
// second re-renders the card that shows it, not the chat around it.
import { useSyncExternalStore } from "react";
import { EMPTY, EMPTY_SHAPE, linkFor, reduce, sessionOf, shapeOf, type SongAction, type SongSession, type SongShape, type SongSnap } from "./state";
import type { SongLinkKind } from "./types";

type Send = (cmd: Record<string, unknown>) => boolean;

/** How long a session that was live before a reconnect waits for song_hello's answer. */
const CONFIRM_MS = 2000;

export class SongHub {
  private snap: SongSnap = EMPTY;
  private shape: SongShape = EMPTY_SHAPE;
  private listeners = new Set<() => void>();
  private sendFn: Send = () => false;
  private confirmTimer: number | undefined;

  // ---- store --------------------------------------------------------------------------------
  getSnapshot = () => this.snap;
  getShape = () => this.shape;
  subscribe = (fn: () => void) => { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; };
  private dispatch(a: SongAction) {
    const next = reduce(this.snap, a, Date.now());
    if (next === this.snap) return;
    this.snap = next;
    this.shape = shapeOf(next, this.shape);
    for (const l of this.listeners) l();
  }

  /** The live session's sid (listening, matching, or showing its answer before the close). */
  get sid(): string | null { return this.snap.live ? this.snap.live.sid : null; }
  session(sid: string): SongSession | null { return sessionOf(this.snap, sid); }

  // ---- connection -------------------------------------------------------------------------------
  setSender(fn: Send) { this.sendFn = fn; }
  setConnected(on: boolean) { this.dispatch({ t: "connected", on }); }

  /** After a (re)connect's replay: ask for the live card's state. A session this client still shows
   *  as live that the core doesn't answer for ended while it was away. */
  endHello() {
    this.dispatch({ t: "hello_end" });
    this.sendFn({ cmd: "song_hello" });
    if (this.confirmTimer) clearTimeout(this.confirmTimer);
    const sid = this.sid;
    if (sid) this.confirmTimer = window.setTimeout(() => { this.confirmTimer = undefined; this.dispatch({ t: "gone", sid }); }, CONFIRM_MS);
  }

  clearEnded() { this.dispatch({ t: "clear" }); }

  // ---- inbound ------------------------------------------------------------------------------------
  /** Handles a `song_*` event. Returns the sid of a session it just made live (so the transcript
   *  can give it a place), null otherwise. */
  handle(type: string, data: unknown): string | null {
    const before = this.sid;
    this.dispatch({ t: "event", type, data });
    const sid = this.sid;
    return sid && sid !== before ? sid : null;
  }

  // ---- outbound -----------------------------------------------------------------------------------
  /** Stop listening now. */
  stop(): boolean {
    const sid = this.sid;
    if (!sid || !this.sendFn({ cmd: "song_stop", sid })) return false;
    this.dispatch({ t: "stop" });
    return true;
  }

  /** Ask the core to open one of this card's links (it opens only the newest result's, only https). */
  openLink(sid: string, which: SongLinkKind): boolean {
    if (!linkFor(this.snap, sid, which)) return false;
    return this.sendFn({ cmd: "song_link", sid, which });
  }
}

/** What the overlay reacts to: which card is live, its face, the ended cards. */
export function useSongShape(hub: SongHub): SongShape {
  return useSyncExternalStore(hub.subscribe, hub.getShape);
}

/** One card's session, whether it is the newest (its links still open), and the connection. */
export function useSongSession(hub: SongHub, sid: string): { x: SongSession | null; latest: boolean; connected: boolean } {
  const x = useSyncExternalStore(hub.subscribe, () => hub.session(sid));
  const latest = useSyncExternalStore(hub.subscribe, () => hub.getSnapshot().latest === sid);
  const connected = useSyncExternalStore(hub.subscribe, () => hub.getSnapshot().connected);
  return { x, latest, connected };
}

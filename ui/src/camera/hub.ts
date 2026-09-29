// The camera half of the websocket client: the one live camera session (status, labels, boxes,
// answer), the ended ones as rows (their answer only — never an image), the preview frame pipe,
// and the commands the card sends back. React reads a snapshot via useSyncExternalStore; frames
// bypass React entirely and are never stored anywhere but the canvas they are painted on.
import { useSyncExternalStore } from "react";
import { FramePipe, parseFrame } from "../browser/frames";
import { normBoxes, normLabels } from "./frames";
import type { CamClose, CamEnded, CamHelloSnapshot, CamLive, CamOpen, CamState, CamStatus } from "./types";

export interface CamSnap {
  connected: boolean;
  live: CamLive | null;
  ended: Record<string, CamEnded>;
  /** This overlay showed the session in its own window (the card here is a row). */
  popped: boolean;
  /** Stop was pressed; waiting for the core's camera_close. */
  stopping: boolean;
}

const EMPTY: CamSnap = { connected: false, live: null, ended: {}, popped: false, stopping: false };
const STATUSES: CamStatus[] = ["starting", "live", "analysing", "answered", "off"];
type Send = (cmd: Record<string, unknown>) => boolean;

export class CameraHub {
  private snap: CamSnap = EMPTY;
  private listeners = new Set<() => void>();
  private sendFn: Send = () => false;
  private replaying = false;
  private prevLive: CamLive | null = null;
  /** The frame sid the live session's preview arrives under (bound on its first frame). */
  private boundSid = "";
  private views = new Map<string, boolean>();
  private viewTimer: number | undefined;
  private lastViewSent = "";
  readonly frames: FramePipe;
  /** performance.now() of the last preview frame (for "Camera is starting…"). */
  lastFrameAt = 0;

  constructor(readonly surface: "overlay" | "window") {
    this.frames = new FramePipe((sid, fid) => this.sendFn({ cmd: "camera_ack", sid, fid }));
  }

  // ---- store --------------------------------------------------------------------------
  getSnapshot = () => this.snap;
  subscribe = (fn: () => void) => { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; };
  private set(patch: Partial<CamSnap>) {
    this.snap = { ...this.snap, ...patch };
    for (const l of this.listeners) l();
  }

  get sid(): string | null { return this.snap.live ? this.snap.live.sid : null; }

  // ---- connection -----------------------------------------------------------------------
  setSender(fn: Send) { this.sendFn = fn; }
  setConnected(on: boolean) { this.set(on ? { connected: true } : { connected: false, stopping: false }); }

  beginHello() {
    this.replaying = true;
    this.prevLive = this.snap.live;
    this.set({ live: null, ended: {}, stopping: false });
  }

  endHello(c: CamHelloSnapshot | undefined) {
    this.replaying = false;
    const s = c?.live;
    if (s && s.sid && s.status !== "off") {
      const prior = this.snap.live?.sid === s.sid ? this.snap.live : this.prevLive?.sid === s.sid ? this.prevLive : null;
      const ended = { ...this.snap.ended };
      delete ended[s.sid];
      if (!prior) this.unbind();
      this.set({ live: this.merge(prior ?? this.fresh(s.sid, s.reason ?? ""), s), ended });
    } else if (c === undefined && this.snap.live) {
      // A core that sends no camera snapshot: the replay's open session stands until it closes.
    } else {
      // Opened in the replay (or live before the reconnect) but not live now: it ended meanwhile.
      // ("gone" rows come back to life if the core says otherwise — see camera_state.)
      const gone = this.snap.live ?? this.prevLive;
      if (gone && !this.snap.ended[gone.sid]) this.endLocal(gone, "gone");
      else this.set({ live: null });
      this.unbind();
    }
    this.prevLive = null;
    this.sendFn({ cmd: "camera_hello", v: 1, surface: this.surface });
    this.lastViewSent = "";
    this.flushView();
  }

  private fresh(sid: string, reason: string): CamLive {
    return { sid, reason, status: "starting", labels: [], boxes: [], answer: "", startedAt: Date.now() };
  }

  private merge(cur: CamLive, s: CamState): CamLive {
    return {
      ...cur,
      status: STATUSES.includes(s.status) ? s.status : cur.status,
      labels: s.labels !== undefined ? normLabels(s.labels) : cur.labels,
      boxes: s.boxes !== undefined ? normBoxes(s.boxes) : cur.boxes,
      answer: typeof s.answer === "string" ? s.answer : cur.answer,
    };
  }

  private endLocal(l: CamLive, reason: string) {
    const e: CamEnded = { sid: l.sid, answer: l.answer, ms: Math.max(0, Date.now() - l.startedAt), reason };
    this.set({ live: null, ended: { ...this.snap.ended, [l.sid]: e }, popped: false, stopping: false });
    this.unbind();
  }

  /** Forget the preview: the last frame is dropped from memory as soon as the camera is off. */
  private unbind() {
    this.boundSid = "";
    this.frames.setSid("");
  }

  clearEnded() { this.set({ ended: {} }); }

  // ---- inbound --------------------------------------------------------------------------
  /** Handles a `camera_*` JSON event. Returns false for anything else. */
  handle(type: string, d: any): boolean {
    switch (type) {
      case "camera_open": {
        const o = d as CamOpen;
        if (!o?.sid) return true;
        const cur = this.snap.live;
        if (cur && cur.sid === o.sid) return true;
        if (cur) this.endLocal(cur, "replaced");
        const ended = { ...this.snap.ended };
        delete ended[o.sid];
        if (!this.replaying) this.unbind();
        this.set({ live: this.fresh(o.sid, String(o.reason ?? "")), ended, stopping: false, popped: false });
        this.lastViewSent = "";
        this.flushView();
        return true;
      }
      case "camera_state": {
        const s = d as CamState;
        if (!s?.sid) return true;
        if (s.status === "off") return this.handle("camera_close", { sid: s.sid, reason: "off" });
        const cur = this.snap.live;
        if (cur && cur.sid === s.sid) { this.set({ live: this.merge(cur, s) }); return true; }
        // A state for a session this client never saw open (joined late), or one it wrongly took
        // for ended across a reconnect: it is live now.
        const e = this.snap.ended[s.sid];
        if (!cur && (!e || e.reason === "gone")) {
          const ended = { ...this.snap.ended };
          delete ended[s.sid];
          this.set({ live: this.merge(this.fresh(s.sid, ""), s), ended });
        }
        return true;
      }
      case "camera_close": {
        const c = d as CamClose;
        if (c?.reason === "erased") {
          this.unbind();
          this.set({ live: null, ended: {}, popped: false, stopping: false });
          return true;
        }
        const cur = this.snap.live;
        if (cur && cur.sid === c?.sid) this.endLocal(cur, String(c.reason ?? "done"));
        return true;
      }
    }
    return type.startsWith("camera_");
  }

  /** A binary NBF1 camera frame. Frames only paint while a session is live here. */
  frame(buf: ArrayBuffer) {
    const p = parseFrame(buf);
    if (!p) return;
    const live = this.snap.live;
    if (!live) { this.sendFn({ cmd: "camera_ack", sid: p.meta.sid, fid: p.meta.fid }); return; }
    // The frame header carries the session id as 16 raw bytes; bind the first one after the
    // open to this session (one camera session at a time), so frames never depend on encoding.
    if (!this.boundSid || (p.meta.sid === live.sid && this.boundSid !== live.sid)) { this.boundSid = p.meta.sid; this.frames.setSid(p.meta.sid); }
    this.lastFrameAt = performance.now();
    this.frames.push(p.meta, new Blob([p.image as BlobPart], { type: p.meta.mime }));
  }

  // ---- outbound -------------------------------------------------------------------------
  /** Turn the camera off now. */
  stop(): boolean {
    const sid = this.sid;
    if (!sid) return false;
    const ok = this.sendFn({ cmd: "camera_stop", sid });
    if (ok) this.set({ stopping: true });
    return ok;
  }

  /** The card moved into its own window (true) or came back to the chat (false). */
  setPopped(popped: boolean) {
    const sid = this.sid;
    this.set({ popped: popped && !!sid });
    if (sid) this.sendFn({ cmd: "camera_popout", sid, popped });
  }

  /** Register whether a view shows the preview; the core hears one `camera_view` (debounced). */
  setView(key: string, visible: boolean | null) {
    if (visible === null) this.views.delete(key); else this.views.set(key, visible);
    if (this.viewTimer) clearTimeout(this.viewTimer);
    this.viewTimer = window.setTimeout(() => this.flushView(), 120);
  }

  private flushView() {
    this.viewTimer = undefined;
    const sid = this.sid;
    if (!sid) return;
    const visible = [...this.views.values()].some(Boolean) && !(typeof document !== "undefined" && document.hidden);
    const msg = { cmd: "camera_view", sid, visible, surface: this.surface };
    const key = JSON.stringify(msg);
    if (key === this.lastViewSent) return;
    if (this.sendFn(msg)) this.lastViewSent = key;
  }

  /** The window was hidden or shown: re-report the view. */
  visibilityChanged() { this.lastViewSent = ""; this.flushView(); }
}

export function useCamera(hub: CameraHub): CamSnap {
  return useSyncExternalStore(hub.subscribe, hub.getSnapshot);
}

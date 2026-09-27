// The browser half of the websocket client: one live session's chrome state, its steps, the
// ended sessions shown as rows, NEO's actions as an event stream (for the cursor/highlight
// overlay), the frame pipe, and the commands the card sends back. React reads a snapshot via
// useSyncExternalStore; frames and actions bypass React entirely.
import { useSyncExternalStore } from "react";
import { FramePipe, parseFrame, b64urlToBytes } from "./frames";
import type {
  BrowserAction, BrowserClose, BrowserDialog, BrowserHelloSnapshot, BrowserHint, BrowserOpen, BrowserResult,
  BrowserSelect, BrowserState, BrowserStep, ControlAction, FrameMeta, NeedUser, Surface,
} from "./types";

export interface EndedSession { sid: string; close: BrowserClose; open: BrowserOpen | null; steps: BrowserStep[] }

export interface BrowserSnap {
  connected: boolean;
  open: BrowserOpen | null;       // the live session's opening message
  live: BrowserState | null;      // its latest chrome state (null = no live session)
  steps: BrowserStep[];
  action: BrowserAction | null;   // NEO's latest action (for captions and late joiners)
  dialog: BrowserDialog | null;
  select: BrowserSelect | null;
  hint: BrowserHint | null;
  ended: Record<string, EndedSession>;
  pending: Partial<Record<ControlAction, string>>; // control in flight → its req id
}

const EMPTY: BrowserSnap = {
  connected: false, open: null, live: null, steps: [], action: null, dialog: null, select: null, hint: null, ended: {}, pending: {},
};

type Send = (cmd: Record<string, unknown>) => boolean;
type ViewReg = { surface: Surface; w: number; h: number; visible: boolean };
const SURFACE_RANK: Record<Surface, number> = { window: 4, focus: 3, docked: 2, mini: 1, hidden: 0 };
const PHASE_RANK = { intent: 0, done: 1, fail: 1 } as const;

function synthState(o: BrowserOpen): BrowserState {
  return {
    sid: o.sid, seq: -1, url: o.url, title: o.title, favicon: null,
    secure: o.url.startsWith("https:") ? "secure" : o.url.startsWith("http:") ? "insecure" : "internal",
    loading: o.status === "starting", progress: null, can_back: false, can_forward: false,
    controller: o.controller, status: o.status, viewport: o.viewport, tabs: { count: 1, active: 0 },
    popped: false, job: o.job || null, need_user: null, error: null, caption: null,
  };
}

export class BrowserHub {
  private snap: BrowserSnap = EMPTY;
  private listeners = new Set<() => void>();
  private actionListeners = new Set<(a: BrowserAction) => void>();
  private toastListeners = new Set<(t: string, ms?: number) => void>();
  private sendFn: Send = () => false;
  private lastAid = -1;
  private lastPhase = -1;
  private inputSeq = 0;
  private reqN = 0;
  private waiters = new Map<string, (r: BrowserResult) => void>();
  private views = new Map<string, ViewReg>();
  private viewTimer: number | undefined;
  private lastViewSent = "";
  private helloSent = false;
  readonly frames: FramePipe;
  /** performance.now() of the last frame that arrived (for "Live view is catching up…"). */
  lastFrameAt = 0;
  /** Dev harness only: keep frames flowing while the tab is in the background. */
  ignoreHidden = false;

  constructor(readonly surface: "overlay" | "window") {
    this.frames = new FramePipe((sid, fid) => this.sendFn({ cmd: "browser_ack", sid, fid }));
  }

  // ---- store --------------------------------------------------------------------------
  getSnapshot = () => this.snap;
  subscribe = (fn: () => void) => { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; };
  private set(patch: Partial<BrowserSnap>) {
    this.snap = { ...this.snap, ...patch };
    for (const l of this.listeners) l();
  }
  onAction(fn: (a: BrowserAction) => void) { this.actionListeners.add(fn); return () => { this.actionListeners.delete(fn); }; }
  onToast(fn: (t: string, ms?: number) => void) { this.toastListeners.add(fn); return () => { this.toastListeners.delete(fn); }; }
  toast(text: string, ms?: number) { for (const l of this.toastListeners) l(text, ms); }

  get sid(): string | null { return this.snap.live ? this.snap.live.sid : null; }

  // ---- connection -----------------------------------------------------------------------
  setSender(fn: Send) { this.sendFn = fn; }
  setConnected(on: boolean) {
    if (!on) {
      this.helloSent = false;
      for (const [, w] of this.waiters) w({ req: "", ok: false, code: "timeout" });
      this.waiters.clear();
      this.set({ connected: false, pending: {} });
    } else this.set({ connected: true });
  }

  /** `hello` arrived: transcript replay follows (browser_open/close from `_recent`), then the snapshot. */
  beginHello() {
    this.replaying = true;
    this.set({ ended: {}, open: null, steps: [], action: null, dialog: null, select: null, hint: null, pending: {} });
    this.prevLive = this.snap.live;
    this.set({ live: null });
  }
  private prevLive: BrowserState | null = null;
  /** Replaying `_recent`: sessions come and go, but the frame on screen stays until the snapshot says. */
  private replaying = false;

  endHello(b: BrowserHelloSnapshot | undefined) {
    this.replaying = false;
    const live = b?.live ?? null;
    if (live) {
      const open = this.snap.open && this.snap.open.sid === live.sid ? this.snap.open : b?.open ?? {
        sid: live.sid, job: live.job ?? "", reason: "task" as const, url: live.url, title: live.title,
        viewport: live.viewport, controller: live.controller, status: live.status, started_at: Date.now() / 1000,
      };
      const ended = { ...this.snap.ended };
      delete ended[live.sid];
      this.frames.setSid(live.sid);
      this.lastAid = b?.action?.aid ?? -1;
      this.lastPhase = b?.action ? PHASE_RANK[b.action.phase] : -1;
      this.set({ live, open, ended, steps: b?.steps ?? [], action: b?.action ?? null, dialog: b?.dialog ?? null, select: b?.select ?? null });
      if (b?.action) for (const l of this.actionListeners) l(b.action);
    } else {
      if (this.snap.open && !this.snap.ended[this.snap.open.sid]) {
        // Opened in the replay but not live any more: the core restarted and the session is gone.
        this.endLocal(this.snap.open, this.prevLive, "error");
      } else if (this.prevLive && !this.snap.ended[this.prevLive.sid]) {
        // Live before the reconnect, absent now (and out of the replay): it ended while we were away.
        const p = this.prevLive;
        this.endLocal({ sid: p.sid, job: p.job ?? "", reason: "task", url: p.url, title: p.title, viewport: p.viewport, controller: p.controller, status: p.status, started_at: Date.now() / 1000 }, p, "error");
      }
      this.frames.setSid("");
    }
    this.prevLive = null;
    this.helloSent = true;
    this.sendFn({ cmd: "browser_hello", v: 1, binary: true, formats: ["jpeg"], surface: this.surface });
    this.lastViewSent = "";
    this.flushView();
  }

  private endLocal(open: BrowserOpen, st: BrowserState | null, reason: BrowserClose["reason"]) {
    const close: BrowserClose = {
      sid: open.sid, reason, alive: false, url: st?.url ?? open.url, title: st?.title ?? open.title, favicon: st?.favicon ?? null,
      poster: null, steps: this.snap.steps.length, duration_ms: Math.max(0, Date.now() - open.started_at * 1000), summary: "",
    };
    this.set({ ended: { ...this.snap.ended, [open.sid]: { sid: open.sid, close, open, steps: this.snap.steps } }, open: null, live: null, steps: [], action: null });
  }

  /** The user answered a dialog / picked an option: close it here without waiting for the core. */
  dismiss(kind: "dialog" | "select", id: string) {
    if (kind === "dialog" && this.snap.dialog?.dialog_id === id) this.set({ dialog: null });
    if (kind === "select" && this.snap.select?.select_id === id) this.set({ select: null });
  }

  /** Clear transcript / erase: rows, posters and thumbnails go with it. */
  clearEnded() { this.set({ ended: {} }); }

  // ---- inbound --------------------------------------------------------------------------
  /** Handles a `browser_*` JSON event. Returns false for anything else. */
  handle(type: string, d: any): boolean {
    switch (type) {
      case "browser_open": {
        const o = d as BrowserOpen;
        const prior = this.snap.ended[o.sid];
        if (this.snap.open && this.snap.open.sid !== o.sid && this.snap.live) this.endLocal(this.snap.open, this.snap.live, "replaced");
        const ended = { ...this.snap.ended };
        delete ended[o.sid];
        if (!this.replaying) this.frames.setSid(o.sid);
        this.lastAid = -1; this.lastPhase = -1; this.inputSeq = 0;
        this.set({ open: o, live: synthState(o), ended, steps: prior?.steps ?? [], action: null, dialog: null, select: null, hint: null });
        this.lastViewSent = "";
        this.flushView();
        return true;
      }
      case "browser_state": {
        const s = d as BrowserState;
        const cur = this.snap.live;
        if (!cur || s.sid !== cur.sid || s.seq <= cur.seq) return true;
        const next: BrowserState = { ...s, favicon: s.favicon === undefined ? cur.favicon : s.favicon };
        this.set(s.controller === "neo" ? { live: next, hint: null } : { live: next });
        return true;
      }
      case "browser_action": {
        const a = d as BrowserAction;
        if (!this.snap.live || a.sid !== this.snap.live.sid) return true;
        const rank = PHASE_RANK[a.phase] ?? 0;
        if (a.aid < this.lastAid || (a.aid === this.lastAid && rank <= this.lastPhase)) return true;
        this.lastAid = a.aid; this.lastPhase = rank;
        this.set({ action: a });
        for (const l of this.actionListeners) l(a);
        return true;
      }
      case "browser_step": {
        const st = d as BrowserStep;
        const targetLive = this.snap.live && st.sid === this.snap.live.sid;
        const list = targetLive ? this.snap.steps : this.snap.ended[st.sid]?.steps;
        if (!list) return true;
        const i = list.findIndex((x) => x.step_id === st.step_id);
        const merged = i >= 0 ? { ...list[i], ...st, thumb: st.thumb === undefined ? list[i].thumb : st.thumb } : st;
        const next = i >= 0 ? list.map((x, j) => (j === i ? merged : x)) : [...list, merged].slice(-200);
        if (targetLive) this.set({ steps: next });
        else this.set({ ended: { ...this.snap.ended, [st.sid]: { ...this.snap.ended[st.sid], steps: next } } });
        return true;
      }
      case "browser_need_user": {
        const nu = d as NeedUser & { sid: string };
        const cur = this.snap.live;
        if (!cur || nu.sid !== cur.sid) return true;
        const confirm = nu.reason === "confirm";
        this.set({ live: { ...cur, need_user: nu, status: confirm ? "waiting" : "need_user", controller: confirm ? cur.controller : "user" } });
        return true;
      }
      case "browser_close": {
        const c = d as BrowserClose;
        if (c.reason === "erased") {
          this.frames.setSid("");
          this.set({ ended: {}, open: null, live: null, steps: [], action: null, dialog: null, select: null, hint: null });
          return true;
        }
        const isLive = this.snap.open && this.snap.open.sid === c.sid;
        const open = isLive ? this.snap.open : this.snap.ended[c.sid]?.open ?? null;
        const steps = isLive ? this.snap.steps : this.snap.ended[c.sid]?.steps ?? [];
        const ended = { ...this.snap.ended, [c.sid]: { sid: c.sid, close: c, open, steps } };
        if (isLive) this.set({ ended, open: null, live: null, steps: [], action: null, dialog: null, select: null, hint: null });
        else this.set({ ended });
        return true;
      }
      case "browser_hint": {
        const h = d as BrowserHint;
        if (!this.snap.live || h.sid !== this.snap.live.sid) return true;
        this.set({ hint: { ...(this.snap.hint ?? { sid: h.sid }), ...h } });
        return true;
      }
      case "browser_select": {
        const s = d as BrowserSelect;
        if (this.snap.live && s.sid === this.snap.live.sid) this.set({ select: s });
        return true;
      }
      case "browser_dialog": {
        const g = d as BrowserDialog | { sid: string; dialog_id: string; closed: true };
        if (!this.snap.live || g.sid !== this.snap.live.sid) return true;
        if ("closed" in g) { if (this.snap.dialog?.dialog_id === g.dialog_id) this.set({ dialog: null }); }
        else this.set({ dialog: g });
        return true;
      }
      case "browser_clipboard": {
        const text = String(d.text ?? "").slice(0, 100_000);
        if (text) void writeClipboard(text).then((ok) => this.toast(ok ? "Copied" : "Couldn't copy", 1600));
        return true;
      }
      case "browser_result": {
        const r = d as BrowserResult;
        const w = this.waiters.get(r.req);
        if (w) { this.waiters.delete(r.req); w(r); }
        else if (r.req === "input" && !r.ok && r.code === "not_user") this.toast("NEO has control", 1800);
        return true;
      }
      case "browser_frame": {
        // JSON fallback frames (clients without binary support).
        const f = d as Omit<FrameMeta, "kind"> & { kind: string | number; b64: string };
        const meta: FrameMeta = { ...f, kind: f.kind === "still" || f.kind === 2 ? 2 : 1 };
        this.lastFrameAt = performance.now();
        const bytes = b64urlToBytes(String(f.b64).replace(/\+/g, "-").replace(/\//g, "_"));
        this.frames.push(meta, new Blob([bytes as BlobPart], { type: meta.mime }));
        return true;
      }
    }
    return type.startsWith("browser_");
  }

  /** A binary websocket message: one NBF1 frame. */
  frame(buf: ArrayBuffer) {
    const p = parseFrame(buf);
    if (!p) return;
    this.lastFrameAt = performance.now();
    this.frames.push(p.meta, new Blob([p.image as BlobPart], { type: p.meta.mime }));
  }

  // ---- outbound -------------------------------------------------------------------------
  /** One `browser_control`; resolves with its `browser_result` (or a timeout after 8 s). */
  control(action: ControlAction, extra: Record<string, unknown> = {}): Promise<BrowserResult> {
    const sid = this.snap.live?.sid ?? (extra.sid as string | undefined);
    const req = `c${++this.reqN}`;
    if (!sid) return Promise.resolve({ req, ok: false, code: "no_session" });
    if (!this.sendFn({ cmd: "browser_control", sid, req, action, ...extra })) return Promise.resolve({ req, ok: false, code: "timeout" });
    this.set({ pending: { ...this.snap.pending, [action]: req } });
    return new Promise((resolve) => {
      const done = (r: BrowserResult) => {
        clearTimeout(t);
        if (this.snap.pending[action] === req) { const p = { ...this.snap.pending }; delete p[action]; this.set({ pending: p }); }
        resolve(r);
      };
      const t = window.setTimeout(() => { this.waiters.delete(req); done({ req, ok: false, code: "timeout" }); }, 8000);
      this.waiters.set(req, done);
    });
  }

  /** Controls that target an ended session (Reopen) carry its sid explicitly. */
  controlSid(sid: string, action: ControlAction, extra: Record<string, unknown> = {}) {
    const req = `c${++this.reqN}`;
    this.sendFn({ cmd: "browser_control", sid, req, action, ...extra });
  }

  input(type: string, fields: Record<string, unknown>) {
    const live = this.snap.live;
    if (!live) return;
    this.sendFn({ cmd: "browser_input", sid: live.sid, seq: ++this.inputSeq, type, ...fields });
  }

  /** Register what a view shows; the largest visible surface is reported (debounced 150 ms). */
  setView(key: string, v: ViewReg | null) {
    if (v) this.views.set(key, v); else this.views.delete(key);
    if (this.viewTimer) clearTimeout(this.viewTimer);
    this.viewTimer = window.setTimeout(() => this.flushView(), 150);
  }

  private flushView() {
    this.viewTimer = undefined;
    const sid = this.snap.live?.sid;
    if (!sid || !this.helloSent) return;
    let best: ViewReg | null = null;
    for (const v of this.views.values()) {
      if (!v.visible || v.surface === "hidden") continue;
      if (!best || SURFACE_RANK[v.surface] > SURFACE_RANK[best.surface]) best = v;
    }
    const hidden = !best || (!this.ignoreHidden && typeof document !== "undefined" && document.hidden);
    const msg = hidden
      ? { cmd: "browser_view", sid, surface: "hidden", css_w: 0, css_h: 0, dpr: devicePixelRatio || 1, visible: false }
      : { cmd: "browser_view", sid, surface: best!.surface, css_w: Math.round(best!.w), css_h: Math.round(best!.h), dpr: Math.min(devicePixelRatio || 1, 2), visible: true };
    const key = JSON.stringify(msg);
    if (key === this.lastViewSent) return;
    this.lastViewSent = key;
    this.sendFn(msg);
  }

  /** The window was hidden or shown: re-report the view. */
  visibilityChanged() { this.lastViewSent = ""; this.flushView(); }
}

async function writeClipboard(text: string): Promise<boolean> {
  try { await navigator.clipboard.writeText(text); return true; } catch { /* outside a user gesture: fall back */ }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.cssText = "position:fixed;opacity:0;left:-9999px";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch { return false; }
}

export function useBrowser(hub: BrowserHub): BrowserSnap {
  return useSyncExternalStore(hub.subscribe, hub.getSnapshot);
}

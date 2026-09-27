// The mock core (dev only, ?mock=browser): speaks NEO's real websocket protocol — hello, state,
// transcripts, tool events, confirm cards, and the whole browser_* protocol with binary NBF1
// frames, credits and acks — and plays a scripted shopping task on a fictional store. When the
// user takes control, their input really drives the mock page (type into the sign-in form, click
// buttons, scroll), so every state in the spec can be seen and felt without a core or network.
import type {
  BrowserAction, BrowserClose, BrowserDialog, BrowserOpen, BrowserSelect, BrowserState, BrowserStep, Controller, NeedUser, Rect, Status, Surface,
} from "../types";
import { b64url, packFrame } from "../frames";
import { pageViewport } from "../sizes";
import { build, render, hit, urlOf, titleOf, keyForUrl, type Built, type El, type PState, type PageKey } from "./pages";
import { makeFavicon } from "./draw";
import type { MockSocket } from "./socket";

const CANCEL = Symbol("cancelled");
/** Seconds of history a jumped-to state (?state=…) pretends to have behind it. */
const SEED_SPAN = 42;
const DPR = 2;
const now = () => Date.now() / 1000;
const RECENT_TYPES = new Set(["transcript", "tool_start", "tool_end", "turn", "browser_open", "browser_close"]);
const SURFACE_RANK: Record<Surface, number> = { window: 4, focus: 3, docked: 2, mini: 1, hidden: 0 };
const FPS: Record<Surface, number> = { window: 30, focus: 30, docked: 24, mini: 2, hidden: 0 };

export class MockConn {
  hello = false;
  view: { surface: Surface; w: number; h: number; dpr: number; visible: boolean } | null = null;
  credits = 2;
  slot: { buf: ArrayBuffer; fid: number } | null = null;
  lastSent = 0;
  lastAck = performance.now();
  timer: number | undefined;
  constructor(private core: MockCore, public sock: MockSocket) {}
  receive(data: string) { let m: any; try { m = JSON.parse(data); } catch { return; } this.core.command(this, m); }
  closed() { this.core.drop(this); }
  json(type: string, data: unknown) { this.sock.deliver(JSON.stringify({ type, data, ts: now() })); }
  get visible() { return !!this.view && this.view.visible && this.view.surface !== "hidden"; }
  get fps() { return this.view ? FPS[this.view.surface] : 0; }
}

interface Session {
  sid: string; sidBytes: Uint8Array; job: string | null; startedAt: number;
  controller: Controller; status: Status; seq: number; aid: number; stepId: number;
  steps: BrowserStep[]; needUser: NeedUser | null; error: { code: string; message: string } | null;
  dialog: BrowserDialog | null; select: BrowserSelect | null; action: BrowserAction | null; caption: string | null;
  popped: boolean; page: PState; history: { key: PageKey; genericUrl: string }[]; hIndex: number;
  loading: boolean; progress: number | null; alive: boolean; ended: boolean; userStep: number | null; userSince: number;
}

type Gate = { action_id: string; resolve: (yes: boolean) => void };

function randomSid(): { sid: string; bytes: Uint8Array } {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return { sid: b64url(bytes), bytes };
}

function freshPage(vw = 1024, vh = 590): PState {
  return {
    key: "home", vw, vh, scrollY: 0, hover: null, pressed: null, focus: null, values: {}, caret: true, loaded: 1,
    cart: 0, colour: "Midnight", qty: "1", banner: false, signedIn: false, keepSigned: false, genericUrl: "", signinError: false,
  };
}

export class MockCore {
  accepting = true;
  speed = 1;
  private conns = new Set<MockConn>();
  private state = "idle";
  private recent: { type: string; data: any; ts: number }[] = [];
  private confirm: { action_id: string; tool: string; summary: string; detail: string } | null = null;
  private s: Session | null = null;
  private gen = 0;
  private running = false;
  private gate: Gate | null = null;
  private releaseWaiters: (() => void)[] = [];
  private lock = false;
  /** "Take control" is waiting for the current action to finish: no new action may start. */
  private wantTake = false;
  private favicon = "";
  private canvas: OffscreenCanvas | HTMLCanvasElement | null = null;
  private dirty = false;
  private rendering = false;
  private renderTimer: number | undefined;
  private lastRender = 0;
  private fid = 0;
  private stateTimer: number | undefined;
  private caretTimer: number | undefined;
  private viewTimer: number | undefined;
  /** When a view first asked for a size the page doesn't have yet (0 = nothing pending). */
  private viewAskedAt = 0;
  /** NEO's action in progress: re-aimed if the page is resized under it. */
  private inflight: { aid: number; base: Omit<BrowserAction, "sid" | "aid" | "phase">; id?: string } | null = null;
  private built: Built | null = null;
  private stillNext = false;
  private closeTimer: number | undefined;
  listeners = new Set<() => void>();

  constructor() {
    window.setInterval(() => this.watchdog(), 1000);
  }

  // ---- connections -------------------------------------------------------------------------
  connect(sock: MockSocket): MockConn {
    const c = new MockConn(this, sock);
    this.conns.add(c);
    c.json("hello", {
      state: this.state,
      recent: this.recent,
      confirm: this.confirm,
      problems: [],
      browser: this.s && !this.s.ended ? {
        v: 1, live: this.stateMsg(false), open: this.openMsg("task"), steps: this.s.steps.slice(-60), action: this.s.action, dialog: this.s.dialog, select: null,
      } : { v: 1, live: null, steps: [], action: null, dialog: null, select: null },
    });
    return c;
  }

  drop(c: MockConn) { this.conns.delete(c); clearTimeout(c.timer); this.pickViewport(); }

  /** Simulate the core dying: every socket closes, reconnects are refused for `ms`. */
  kill(ms = 6000) {
    this.accepting = false;
    for (const c of [...this.conns]) { c.sock.drop(); this.conns.delete(c); }
    window.setTimeout(() => { this.accepting = true; }, ms);
  }

  private bcast(type: string, data: any) {
    if (RECENT_TYPES.has(type)) { this.recent.push({ type, data, ts: now() }); if (this.recent.length > 200) this.recent.shift(); }
    for (const c of this.conns) c.json(type, data);
  }
  private bcastBrowser(type: string, data: any) {
    if (type === "browser_open" || type === "browser_close") { this.bcast(type, data); return; }
    for (const c of this.conns) if (c.hello) c.json(type, data);
  }

  private setNeo(state: string) { this.state = state; this.bcast("state", { state }); }
  private say(role: "user" | "assistant", text: string, job?: string, final = true) { this.bcast("transcript", { role, text, final, job }); }

  // ---- commands --------------------------------------------------------------------------------
  command(c: MockConn, m: any) {
    switch (m.cmd) {
      case "browser_hello": c.hello = true; this.invalidate(true); return;
      case "browser_view": {
        if (!this.s || m.sid !== this.s.sid) return;
        const wasVisible = c.visible;
        c.view = { surface: m.surface, w: m.css_w, h: m.css_h, dpr: m.dpr, visible: !!m.visible };
        if (c.visible && !wasVisible) { this.stillNext = true; this.invalidate(true); }
        if (!this.viewAskedAt) this.viewAskedAt = performance.now();
        clearTimeout(this.viewTimer);
        this.viewTimer = window.setTimeout(() => this.pickViewport(), 300);
        return;
      }
      case "browser_ack":
        c.credits = Math.min(2, c.credits + 1);
        c.lastAck = performance.now();
        this.trySend(c);
        return;
      case "browser_input": this.input(m); return;
      case "browser_control": this.control(c, m); return;
      case "confirm":
        if (this.gate && this.gate.action_id === m.action_id) this.gate.resolve(!!m.yes);
        return;
      case "text": this.onText(String(m.text ?? "")); return;
      case "stop": this.stopJob(); return;
      case "forget_all":
        this.stopJob();
        if (this.s) { this.bcastBrowser("browser_close", { sid: this.s.sid, reason: "erased", alive: false, url: "", title: "", poster: null, steps: 0, duration_ms: 0, summary: "" }); this.s = null; }
        this.recent = [];
        this.bcast("erased", {});
        return;
    }
  }

  private result(c: MockConn | null, req: string, ok: boolean, code?: string) { c?.json("browser_result", { req, ok, ...(code ? { code } : {}) }); }

  private async control(c: MockConn, m: any) {
    const s = this.s;
    const req = String(m.req ?? "");
    if (m.action === "reopen") {
      if (s && s.ended && s.alive && m.sid === s.sid) { this.reopen(); this.result(c, req, true); } else this.result(c, req, false, "no_session");
      return;
    }
    if (!s || s.ended || m.sid !== s.sid) { this.result(c, req, false, "bad_sid"); return; }
    const neoDrives = s.controller === "neo";
    switch (m.action) {
      case "take": await this.take(); break;
      case "release": this.release(); break;
      case "stop":
        if (!s.job) { this.result(c, req, false, "no_session"); return; }
        this.stopJob(); break;
      case "back": case "forward": case "reload": case "hard_reload": case "stop_loading":
        if (neoDrives) await this.take();
        if (m.action === "back" && s.hIndex > 0) { s.hIndex--; void this.go(s.history[s.hIndex].key, { push: false, genericUrl: s.history[s.hIndex].genericUrl }); }
        else if (m.action === "forward" && s.hIndex < s.history.length - 1) { s.hIndex++; void this.go(s.history[s.hIndex].key, { push: false, genericUrl: s.history[s.hIndex].genericUrl }); }
        else if (m.action === "reload" || m.action === "hard_reload") void this.go(s.page.key, { push: false, genericUrl: s.page.genericUrl });
        else if (m.action === "stop_loading") { s.loading = false; s.progress = null; s.page.loaded = 1; this.invalidate(); this.emitState(); }
        break;
      case "navigate": {
        if (s.controller !== "user") { this.result(c, req, false, "not_user"); return; }
        const url = String(m.url ?? "");
        if (!/^https?:\/\//.test(url)) { this.result(c, req, false, "blocked_url"); return; }
        const key = keyForUrl(url);
        void this.go(key ?? "generic", { genericUrl: key ? "" : url });
        break;
      }
      case "select":
        if (s.select && s.select.select_id === m.select_id) { s.page.qty = String(m.value); s.select = null; this.invalidate(); this.emitState(); }
        break;
      case "dialog":
        if (s.dialog && s.dialog.dialog_id === m.dialog_id) {
          if (m.accept && s.dialog.message.startsWith("Remove")) { s.page.cart = 0; s.page.banner = false; }
          const id = s.dialog.dialog_id;
          s.dialog = null;
          this.bcastBrowser("browser_dialog", { sid: s.sid, dialog_id: id, closed: true });
          this.invalidate();
        }
        break;
      case "copy": case "cut": {
        const f = s.page.focus;
        const el = f ? this.built?.els.find((e) => e.id === f) : null;
        const t = el && !el.secret ? s.page.values[f!] ?? "" : "";
        c.json("browser_clipboard", { sid: s.sid, req, text: t });
        if (m.action === "cut" && el && !el.secret) { s.page.values[f!] = ""; this.invalidate(); }
        break;
      }
      case "popout": s.popped = true; this.emitState(); break;
      case "popin": s.popped = false; this.emitState(); break;
      case "close":
        if (s.job && s.controller === "neo" && (s.status === "working" || s.status === "thinking" || s.status === "waiting") && !m.force) { this.result(c, req, false, "busy"); return; }
        if (s.job) this.stopJob(true);
        this.endSession("closed");
        break;
      case "restart":
        s.error = null; s.status = s.controller === "user" ? "user" : "working"; void this.go(s.page.key, { push: false }); break;
      case "tab": case "expand": case "collapse": break;
      default: this.result(c, req, false, "unsupported"); return;
    }
    this.result(c, req, true);
  }

  // ---- control handoff ------------------------------------------------------------------------------
  private async take() {
    const s = this.s;
    if (!s || s.controller === "user") return;
    this.wantTake = true;
    for (let i = 0; i < 160 && this.lock; i++) await new Promise((r) => setTimeout(r, 50));
    this.wantTake = false;
    if (this.gate) { const g = this.gate; this.gate = null; g.resolve(false); }
    s.controller = "user";
    s.status = "user";
    s.userSince = now();
    s.userStep = this.step({ kind: "user", label: "You took over", status: "running", by: "user" });
    this.setNeo("idle");
    this.emitState();
    this.hintFocus(); // the core's 10 Hz hint: which field (if any) has the page's focus
  }

  private release() {
    const s = this.s;
    if (!s || s.controller !== "user") return;
    if (!s.job && !s.needUser) { this.endSession("closed"); return; }
    const secs = Math.round(now() - s.userSince);
    if (s.userStep != null) {
      const label = s.needUser?.reason === "login" ? "You logged in" : s.needUser?.reason === "2fa" ? "You entered the code" : s.needUser?.reason === "captcha" ? "You solved the check" : "You took over";
      this.step({ step_id: s.userStep, kind: "user", label: `${label} · ${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`, status: "ok", by: "user", duration_ms: secs * 1000 });
      s.userStep = null;
    }
    s.needUser = null;
    s.controller = "neo";
    s.status = s.job ? "thinking" : "user";
    s.page.hover = null;
    this.setNeo(s.job ? "working" : "idle");
    this.emitState();
    const w = this.releaseWaiters;
    this.releaseWaiters = [];
    for (const f of w) f();
  }

  private waitRelease(): Promise<void> { return new Promise((r) => this.releaseWaiters.push(r)); }

  private stopJob(silent = false) {
    if (!this.running) return;
    this.gen++;
    this.running = false;
    if (this.gate) { const g = this.gate; this.gate = null; g.resolve(false); }
    if (this.confirm) { this.bcast("confirm_done", { action_id: this.confirm.action_id }); this.confirm = null; }
    const s = this.s;
    if (s && !s.ended) {
      s.job = null; s.controller = "user"; s.status = "user"; s.needUser = null;
      if (s.action && s.action.phase === "intent") this.action({ ...s.action, phase: "fail" }, false);
      this.emitState();
    }
    this.lock = false;
    this.releaseWaiters = [];
    this.setNeo("idle");
    if (!silent) this.say("assistant", "Stopped. The page is yours — say the word if you want me to pick it up again.", "j" + this.gen);
  }

  // ---- session ------------------------------------------------------------------------------------
  private openMsg(reason: BrowserOpen["reason"]): BrowserOpen {
    const s = this.s!;
    return { sid: s.sid, job: s.job ?? "", reason, url: urlOf(s.page), title: titleOf(s.page), viewport: { w: s.page.vw, h: s.page.vh, dpr: DPR }, controller: s.controller, status: s.status, started_at: s.startedAt };
  }

  private open(job: string, key: PageKey, status: Status = "starting", controller: Controller = "neo", startedAt = now()) {
    clearTimeout(this.closeTimer);
    if (!this.favicon) this.favicon = makeFavicon();
    const { sid, bytes } = randomSid();
    const vp = this.largestView();
    const page = freshPage(vp.w, vp.h);
    page.key = key;
    this.s = {
      sid, sidBytes: bytes, job, startedAt, controller, status, seq: 0, aid: 0, stepId: 0, steps: [], needUser: null, error: null,
      dialog: null, select: null, action: null, caption: null, popped: false, page, history: [{ key, genericUrl: "" }], hIndex: 0,
      loading: status === "starting", progress: null, alive: true, ended: false, userStep: null, userSince: now(),
    };
    this.fid = 0;
    for (const c of this.conns) { c.view = null; c.credits = 2; c.slot = null; }
    this.bcastBrowser("browser_open", this.openMsg("task"));
    this.emitState(true);
  }

  private reopen() {
    const s = this.s!;
    s.ended = false; s.controller = "user"; s.status = "user"; s.job = null; s.action = null;
    this.bcastBrowser("browser_open", this.openMsg("reopen"));
    this.emitState(true);
    this.invalidate(true);
  }

  private async endSession(reason: BrowserClose["reason"]) {
    const s = this.s;
    if (!s || s.ended) return;
    s.ended = true;
    const poster = await this.snapshot(640, 0.6);
    const lastOk = [...s.steps].reverse().find((x) => x.status === "ok" && x.by === "neo");
    const summary = s.page.key === "confirmed" ? "Ordered Aero Buds Pro" : lastOk?.label ?? "";
    this.bcastBrowser("browser_close", {
      sid: s.sid, reason, alive: true, url: urlOf(s.page), title: titleOf(s.page), favicon: this.favicon, poster,
      steps: s.steps.length, duration_ms: Math.round((now() - s.startedAt) * 1000), summary: summary.slice(0, 60),
    } satisfies BrowserClose);
    clearInterval(this.caretTimer); this.caretTimer = undefined;
    // The page stays warm for Reopen (10 min in the real core; the mock keeps it until the next task).
  }

  private stateMsg(bump = true): BrowserState {
    const s = this.s!;
    if (bump) s.seq++;
    return {
      sid: s.sid, seq: s.seq, url: urlOf(s.page), title: s.status === "starting" ? "" : titleOf(s.page), favicon: s.status === "starting" ? null : this.favicon,
      secure: "secure", loading: s.loading, progress: s.progress, can_back: s.hIndex > 0, can_forward: s.hIndex < s.history.length - 1,
      controller: s.controller, status: s.status, viewport: { w: s.page.vw, h: s.page.vh, dpr: DPR }, tabs: { count: 1, active: 0 },
      popped: s.popped, job: s.job, need_user: s.needUser, error: s.error, caption: s.caption,
    };
  }

  /** Coalesced to ≤ 10/s, like the core. */
  private emitState(now_ = false) {
    if (!this.s || this.s.ended) return;
    const fire = () => { this.stateTimer = undefined; if (this.s && !this.s.ended) this.bcastBrowser("browser_state", this.stateMsg()); this.listeners.forEach((l) => l()); };
    if (now_) { clearTimeout(this.stateTimer); fire(); return; }
    if (this.stateTimer) return;
    this.stateTimer = window.setTimeout(fire, 60);
  }

  private step(p: Partial<BrowserStep> & { kind: BrowserStep["kind"]; label: string; status: BrowserStep["status"]; by: "neo" | "user" }): number {
    const s = this.s!;
    const id = p.step_id ?? ++s.stepId;
    const i = s.steps.findIndex((x) => x.step_id === id);
    const st: BrowserStep = { ...(i >= 0 ? s.steps[i] : {}), ...p, sid: s.sid, step_id: id, ts: i >= 0 ? s.steps[i].ts : now(), url: p.url ?? urlOf(s.page) } as BrowserStep;
    if (i >= 0) s.steps[i] = st; else s.steps.push(st);
    this.bcastBrowser("browser_step", st);
    return id;
  }

  private action(a: Omit<BrowserAction, "sid" | "aid"> & { aid?: number }, bump = true): BrowserAction {
    const s = this.s!;
    // The page the target was measured on (size + scroll), so the UI can keep boxes on the pixels.
    const full: BrowserAction = { vw: s.page.vw, vh: s.page.vh, sy: Math.round(s.page.scrollY * 10) / 10, ...a, sid: s.sid, aid: a.aid ?? (bump ? ++s.aid : s.aid) } as BrowserAction;
    s.action = full;
    this.bcastBrowser("browser_action", full);
    return full;
  }

  // ---- viewport ---------------------------------------------------------------------------------
  private largestView(): { w: number; h: number } {
    let best: MockConn["view"] = null;
    for (const c of this.conns) {
      const v = c.view;
      if (!v || !v.visible || v.surface === "mini" || v.surface === "hidden") continue;
      if (!best || SURFACE_RANK[v.surface] > SURFACE_RANK[best.surface]) best = v;
    }
    return best ? pageViewport(best.w, best.h) : this.s ? { w: this.s.page.vw, h: this.s.page.vh } : { w: 1024, h: 590 };
  }

  private pickViewport() {
    const s = this.s;
    if (!s || s.ended) return;
    const vp = this.largestView();
    if (vp.w === s.page.vw && vp.h === s.page.vh) { this.viewAskedAt = 0; return; }
    // Never while an approval is pending: the highlighted target must stay put. Mid-action, wait
    // for the action to finish — but not past 600 ms (a long type action would leave bars on
    // screen for seconds): resize, then re-aim the action at where its target now is.
    const waited = this.viewAskedAt ? performance.now() - this.viewAskedAt : 0;
    if (s.status === "waiting" || (this.lock && waited < 600)) { clearTimeout(this.viewTimer); this.viewTimer = window.setTimeout(() => this.pickViewport(), this.lock ? 100 : 250); return; }
    this.viewAskedAt = 0;
    s.page.vw = vp.w; s.page.vh = vp.h;
    this.clampScroll();
    this.stillNext = true;
    this.invalidate(true);
    this.emitState();
    const f = this.inflight;
    const e = f?.id ? this.el(f.id) : null;
    if (f && e && this.lock) {
      const rect = this.vrect(e);
      f.base = { ...f.base, rect, x: rect.x + rect.w / 2, y: rect.y + rect.h / 2 };
      f.aid = this.action({ ...f.base, phase: "intent" }).aid;
    }
  }

  private clampScroll() {
    const s = this.s!;
    const b = build(s.page);
    s.page.scrollY = Math.max(0, Math.min(s.page.scrollY, b.docH - s.page.vh));
  }

  // ---- frames -------------------------------------------------------------------------------------
  /** Something on the page changed: paint and send (Chromium only emits frames on repaint). */
  invalidate(soon = false) {
    this.dirty = true;
    if (this.rendering || this.renderTimer) return;
    const wait = soon ? 0 : Math.max(0, 1000 / 30 - (performance.now() - this.lastRender));
    this.renderTimer = window.setTimeout(() => { this.renderTimer = undefined; void this.renderFrame(); }, wait);
  }

  private anyVisible() { for (const c of this.conns) if (c.hello && c.visible) return true; return false; }

  private ensureCanvas(w: number, h: number): OffscreenCanvas | HTMLCanvasElement {
    if (!this.canvas || this.canvas.width !== w || this.canvas.height !== h) {
      if (typeof OffscreenCanvas !== "undefined") this.canvas = new OffscreenCanvas(w, h);
      else { const c = document.createElement("canvas"); c.width = w; c.height = h; this.canvas = c; }
    }
    return this.canvas;
  }

  private async encode(canvas: OffscreenCanvas | HTMLCanvasElement, q: number): Promise<Blob | null> {
    if ("convertToBlob" in canvas) return canvas.convertToBlob({ type: "image/jpeg", quality: q });
    return new Promise((r) => (canvas as HTMLCanvasElement).toBlob((b) => r(b), "image/jpeg", q));
  }

  private async renderFrame() {
    const s = this.s;
    if (!s || s.ended || !this.dirty) return;
    if (!this.anyVisible()) return; // stay dirty: the next viewer gets a fresh frame at once
    this.dirty = false;
    this.rendering = true;
    this.lastRender = performance.now();
    try {
      const st = s.page;
      // Capped like the core's screencast (≤ 2560 × 3200 device px), keeping the page's aspect.
      const k = Math.min(DPR, 2560 / st.vw, 3200 / st.vh);
      const w = Math.round(st.vw * k), h = Math.round(st.vh * k);
      const canvas = this.ensureCanvas(w, h);
      const ctx = canvas.getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
      this.built = build(st);
      if (s.status === "starting") { ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, w, h); }
      else render(ctx, st, w / st.vw, this.built);
      const blob = await this.encode(canvas, 0.8);
      if (!blob || this.s !== s || s.ended) return;
      const img = new Uint8Array(await blob.arrayBuffer());
      const still = this.stillNext;
      this.stillNext = false;
      const fid = ++this.fid;
      const buf = packFrame({
        kind: still ? 2 : 1, fid, ts: Date.now(), vw: st.vw, vh: st.vh, iw: w, ih: h, dpr: w / st.vw, sx: 0, sy: st.scrollY,
        flags: (s.controller === "user" ? 1 : 0) | (s.loading ? 2 : 0) | (still ? 4 : 0), mime: "image/jpeg",
      }, s.sidBytes, img);
      for (const c of this.conns) {
        if (!c.hello || !c.visible) continue;
        c.slot = { buf, fid }; // replaces any unsent frame: latency stays at one frame
        this.trySend(c);
      }
    } finally {
      this.rendering = false;
      if (this.dirty) this.invalidate();
    }
  }

  private trySend(c: MockConn) {
    if (!c.slot || c.credits <= 0 || !c.visible) return;
    const gap = 1000 / Math.max(1, c.fps) - (performance.now() - c.lastSent);
    if (gap > 0) { clearTimeout(c.timer); c.timer = window.setTimeout(() => this.trySend(c), gap); return; }
    const f = c.slot;
    c.slot = null;
    c.credits--;
    c.lastSent = performance.now();
    c.sock.deliver(f.buf);
  }

  private watchdog() {
    for (const c of this.conns) if (c.credits <= 0 && performance.now() - c.lastAck > 2000) { c.credits = 1; this.trySend(c); }
  }

  private async snapshot(width: number, q: number, page?: PState): Promise<string | null> {
    const s = this.s;
    if (!s) return null;
    const st = page ?? s.page;
    const h = Math.round((width * st.vh) / st.vw);
    const c = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(width, h) : Object.assign(document.createElement("canvas"), { width, height: h });
    const ctx = c.getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
    render(ctx, st, width / st.vw);
    const blob = await this.encode(c, q);
    if (!blob) return null;
    return await new Promise((r) => { const fr = new FileReader(); fr.onload = () => r(String(fr.result)); fr.onerror = () => r(null); fr.readAsDataURL(blob); });
  }

  private thumb(stepId: number, seeded = false) {
    window.setTimeout(async () => {
      const s = this.s;
      if (!s || (s.controller === "user" && !seeded)) return; // never while the user drives
      const t = await this.snapshot(400, 0.55);
      if (this.s !== s) return;
      const st = s.steps.find((x) => x.step_id === stepId);
      if (st && t) this.step({ ...st, thumb: t });
    }, 350 / this.speed);
  }

  // ---- page mechanics ---------------------------------------------------------------------------
  private async go(key: PageKey, o: { push?: boolean; genericUrl?: string; keepLoading?: boolean } = {}) {
    const s = this.s;
    if (!s) return;
    const gen = this.gen;
    s.loading = true; s.progress = 0.12;
    this.emitState(true);
    await this.sleep(260);
    if (this.s !== s) return;
    if (o.push !== false) { s.history = s.history.slice(0, s.hIndex + 1); s.history.push({ key, genericUrl: o.genericUrl ?? "" }); s.hIndex = s.history.length - 1; }
    const p = s.page;
    p.key = key; p.genericUrl = o.genericUrl ?? ""; p.scrollY = 0; p.hover = null; p.pressed = null; p.focus = null; p.loaded = 0.3; p.signinError = false;
    if (key !== "cart") p.banner = false;
    // The user finished signing in (the page moved on): the prompt says so, from NEO's own words.
    if (s.needUser && s.needUser.reason === "login" && key !== "signin" && s.status === "need_user") {
      s.needUser = { ...s.needUser, title: "Signed in? Hand back to NEO.", sub: "NEO picks up from here.", rect: undefined };
      this.bcastBrowser("browser_need_user", { ...s.needUser, sid: s.sid });
    }
    s.progress = 0.55;
    this.invalidate(true);
    this.emitState();
    await this.sleep(380);
    if (this.s !== s) return;
    p.loaded = 1;
    s.progress = 1;
    this.invalidate(true);
    if (!o.keepLoading) { s.loading = false; s.progress = null; }
    this.emitState();
    void gen;
    this.caret();
  }

  private caret() {
    const s = this.s;
    clearInterval(this.caretTimer);
    this.caretTimer = undefined;
    if (!s || !s.page.focus) return;
    s.page.caret = true;
    this.caretTimer = window.setInterval(() => {
      if (!this.s || !this.s.page.focus) { clearInterval(this.caretTimer); this.caretTimer = undefined; return; }
      this.s.page.caret = !this.s.page.caret;
      this.invalidate();
    }, 530);
  }

  private async scrollTo(target: number, ms = 480) {
    const s = this.s!;
    const b = build(s.page);
    const to = Math.max(0, Math.min(target, b.docH - s.page.vh));
    const from = s.page.scrollY;
    if (Math.abs(to - from) < 1) return;
    const t0 = performance.now();
    const dur = ms / this.speed;
    await new Promise<void>((resolve) => {
      const tick = () => {
        if (!this.s) { resolve(); return; }
        const k = Math.min(1, (performance.now() - t0) / dur);
        const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
        s.page.scrollY = from + (to - from) * e;
        this.invalidate();
        if (k < 1) window.setTimeout(tick, 16); else resolve();
      };
      tick();
    });
  }

  private el(id: string): El | null { return build(this.s!.page).els.find((e) => e.id === id) ?? null; }
  private vrect(e: El): Rect { const s = this.s!; return { x: e.x, y: e.y - s.page.scrollY, w: e.w, h: e.h }; }

  private activate(e: El, by: "neo" | "user", conn?: MockConn | null) {
    const s = this.s!;
    const p = s.page;
    if (e.kind === "input") { p.focus = e.id; this.caret(); this.invalidate(); this.hintFocus(); return; }
    if (e.kind === "select" && by === "user") {
      s.select = { sid: s.sid, select_id: e.id, rect: this.vrect(e), multiple: false, options: (e.options ?? []).map((o) => ({ value: o, label: o, selected: o === p.qty, disabled: false })) };
      this.bcastBrowser("browser_select", s.select);
      void conn;
      return;
    }
    if (e.act?.startsWith("colour:")) { p.colour = e.act.slice(7); this.invalidate(); return; }
    switch (e.act) {
      case "search": void this.go("search"); return;
      case "add": p.cart = Math.max(1, p.cart + 1); p.banner = true; void this.go("cart"); return;
      case "keep": p.keepSigned = !p.keepSigned; this.invalidate(); return;
      case "delete":
        s.dialog = { sid: s.sid, dialog_id: "d" + Math.random().toString(36).slice(2, 8), kind: "confirm", message: "Remove Aero Buds Pro from your cart?", origin: "www.northwind.store" };
        this.bcastBrowser("browser_dialog", s.dialog);
        return;
      case "signin":
        if ((p.values.email ?? "").includes("@") && (p.values.password ?? "").length >= 4) { p.signedIn = true; p.values.password = ""; void this.go("checkout"); }
        else { p.signinError = true; this.invalidate(); }
        return;
      case "place": void this.go("confirmed"); return;
    }
    if (e.href) void this.go(e.href);
  }

  // ---- user input (only while the user drives) -------------------------------------------------------
  private lastHint = "";
  private input(m: any) {
    const s = this.s;
    if (!s || s.ended || m.sid !== s.sid || s.controller !== "user" || (s.status !== "user" && s.status !== "need_user")) return;
    const p = s.page;
    const b = build(p);
    const x = Number(m.x), y = Number(m.y);
    switch (m.type) {
      case "mousemove": {
        const e = hit(b, x, y + p.scrollY);
        const id = e?.id ?? null;
        if (id !== p.hover) { p.hover = id; this.invalidate(); }
        const cursor = !e ? "default" : e.kind === "input" ? "text" : "pointer";
        if (cursor !== this.lastHint) { this.lastHint = cursor; this.bcastBrowser("browser_hint", { sid: s.sid, cursor }); }
        return;
      }
      case "mousedown": {
        const e = hit(b, x, y + p.scrollY);
        p.pressed = e?.id ?? null;
        if (!e || e.kind !== "input") { if (p.focus) { p.focus = null; this.caret(); this.hintFocus(); } }
        else { p.focus = e.id; this.caret(); this.hintFocus(); }
        this.invalidate();
        return;
      }
      case "mouseup": {
        const e = hit(b, x, y + p.scrollY);
        const was = p.pressed;
        p.pressed = null;
        this.invalidate();
        if (e && e.id === was && e.kind !== "input") this.activate(e, "user");
        return;
      }
      case "wheel": {
        const to = Math.max(0, Math.min(p.scrollY + Number(m.dy || 0), b.docH - p.vh));
        if (to !== p.scrollY) { p.scrollY = to; this.invalidate(); this.hintFocus(); }
        return;
      }
      case "keydown": {
        const f = p.focus;
        const key = String(m.key ?? "");
        const meta = (Number(m.mods) & 4) !== 0;
        if (!f) {
          if (key === "PageDown" || key === " ") void this.scrollTo(p.scrollY + p.vh * 0.85, 300);
          else if (key === "PageUp") void this.scrollTo(p.scrollY - p.vh * 0.85, 300);
          else if (key === "ArrowDown") void this.scrollTo(p.scrollY + 60, 120);
          else if (key === "ArrowUp") void this.scrollTo(p.scrollY - 60, 120);
          else if (key === "Tab") { const first = b.els.find((e) => e.kind === "input"); if (first) { p.focus = first.id; this.caret(); this.invalidate(); this.hintFocus(); } }
          return;
        }
        const v = p.values[f] ?? "";
        if (meta && key.toLowerCase() === "a") { this.selAll = true; return; }
        if (key === "Backspace") { p.values[f] = this.selAll || (Number(m.mods) & 1) ? "" : v.slice(0, -1); this.selAll = false; }
        else if (key === "Enter") {
          this.selAll = false;
          if (f === "q") void this.go("search");
          else if (f === "email" || f === "password") { const go = b.els.find((e) => e.id === "signin-go"); if (go) this.activate(go, "user"); }
          return;
        } else if (key === "Tab") {
          const inputs = b.els.filter((e) => e.kind === "input");
          const i = inputs.findIndex((e) => e.id === f);
          const n = inputs[(i + ((Number(m.mods) & 8) ? -1 : 1) + inputs.length) % inputs.length];
          p.focus = n.id; this.caret(); this.hintFocus();
        } else if (key === "Escape") { p.focus = null; this.hintFocus(); }
        else if (typeof m.text === "string" && m.text && !meta) { p.values[f] = (this.selAll ? "" : v) + m.text; this.selAll = false; }
        else return;
        s.page.caret = true;
        this.invalidate();
        return;
      }
      case "text": case "paste": {
        const f = p.focus;
        if (!f) return;
        p.values[f] = (this.selAll ? "" : p.values[f] ?? "") + String(m.text ?? "").replace(/\n/g, " ").slice(0, 400);
        this.selAll = false;
        this.invalidate();
        return;
      }
    }
  }
  private selAll = false;

  /** Like the core before a login / 2FA handover: the first empty field has the page's focus. */
  private focusField(id: string) {
    const s = this.s!;
    if (s.page.values[id]) return;
    s.page.focus = id;
    this.caret();
    this.invalidate(true);
    this.hintFocus();
  }

  private hintFocus() {
    const s = this.s!;
    const f = s.page.focus;
    const e = f ? build(s.page).els.find((x) => x.id === f) : null;
    this.bcastBrowser("browser_hint", { sid: s.sid, focus: e ? { rect: this.vrect(e), kind: e.secret ? "password" : "text" } : null });
  }

  // ---- NEO's scripted work --------------------------------------------------------------------------------
  private sleep(ms: number) { return new Promise((r) => setTimeout(r, ms / this.speed)); }
  private async wait(gen: number, ms: number) { await this.sleep(ms); if (gen !== this.gen) throw CANCEL; }

  private async hold(gen: number) {
    while (this.s && this.s.controller === "user") {
      this.setNeo("idle");
      await this.waitRelease();
      if (gen !== this.gen) throw CANCEL;
    }
  }

  /** One instrumented NEO action (§12.3): scroll into view → intent → presentation delay →
   *  (approval) → act → done → step with a thumbnail. */
  private async act(gen: number, o: {
    kind: BrowserAction["kind"]; id?: string; label?: string; text?: string; key?: string; field?: string; secret?: boolean;
    dir?: "up" | "down"; amount?: number; gated?: boolean; step: string; rect?: Rect; caption?: string; tool?: string; perform?: () => Promise<void> | void;
  }): Promise<boolean> {
    // A pending "Take control" wins over NEO's next action (§6.4: at most the current one finishes).
    while (this.wantTake && gen === this.gen) await new Promise((r) => setTimeout(r, 30));
    await this.hold(gen);
    const s = this.s!;
    // A resize that came in during the last action lands now, between actions (not whenever the
    // lock happens to be free), so the page matches the view before NEO measures its next target.
    clearTimeout(this.viewTimer);
    this.pickViewport();
    this.lock = true;
    s.status = "working";
    this.setNeo("working");
    this.emitState();
    const job = s.job ?? undefined;
    const tool = o.tool ?? `browser_${o.kind === "key" ? "press" : o.kind}`;
    this.bcast("tool_start", { name: tool, args: {}, job });
    try {
      let e = o.id ? this.el(o.id) : null;
      if (e && (e.y - s.page.scrollY < 110 || e.y + e.h - s.page.scrollY > s.page.vh - 90)) {
        const target = e.y - s.page.vh * 0.35;
        const sa = this.action({ kind: "scroll", phase: "intent", dir: target > s.page.scrollY ? "down" : "up", label: o.label });
        await this.scrollTo(target);
        this.action({ ...sa, phase: "done" }, false);
        e = this.el(o.id!);
        await this.wait(gen, 200);
      }
      const rect = e ? this.vrect(e) : o.rect;
      const x = rect ? rect.x + rect.w / 2 : undefined, y = rect ? rect.y + rect.h / 2 : undefined;
      const base: Omit<BrowserAction, "sid" | "aid" | "phase"> = {
        kind: o.kind, x, y, rect, label: o.label, role: e?.role, field: o.field ?? e?.field, text: o.secret ? undefined : o.text, secret: o.secret,
        key: o.key, dir: o.dir, amount: o.amount, button: o.kind === "click" ? "left" : undefined, clicks: o.kind === "click" ? 1 : undefined, caption: o.caption,
      };
      const a = this.action({ ...base, phase: "intent", gated: o.gated });
      const live = { aid: a.aid, base: { ...base, gated: o.gated }, id: o.id };
      this.inflight = live;
      const stepId = this.step({ kind: o.kind, label: o.step, status: "running", by: "neo" });
      // Presentation delay: long enough for the cursor to arrive.
      await this.wait(gen, o.kind === "read" || o.kind === "scroll" || o.kind === "key" ? 120 : 520);
      if (o.gated) {
        const action_id = "act_" + Math.random().toString(36).slice(2, 10);
        this.confirm = { action_id, tool: "browser_click", summary: `click “${o.label}” on northwind.store`, detail: "Order total: $76.20 — Aero Buds Pro (" + s.page.colour + ")\nPay with: Visa ending in 4242\nDeliver to: Alex Rivera, 221 Market Street, San Francisco" };
        s.needUser = { reason: "confirm", site: "Northwind", title: "Waiting for your OK", sub: "", action_id, rect };
        s.status = "waiting";
        this.step({ step_id: stepId, kind: "confirm", label: `Asked to click “${o.label}”`, status: "waiting", by: "neo" });
        this.bcast("confirm", this.confirm);
        this.bcastBrowser("browser_need_user", { ...s.needUser, sid: s.sid });
        this.setNeo("working");
        this.emitState(true);
        this.lock = false;
        const yes = await new Promise<boolean>((resolve) => { this.gate = { action_id, resolve }; });
        this.gate = null;
        if (this.confirm) { this.bcast("confirm_done", { action_id }); this.confirm = null; }
        if (gen !== this.gen) throw CANCEL;
        this.lock = true;
        s.needUser = null;
        if (!yes) {
          this.action({ ...live.base, aid: live.aid, phase: "fail", reason: "declined" }, false);
          this.step({ step_id: stepId, kind: "confirm", label: `Didn't click “${o.label}”`, status: "declined", by: "neo" });
          if (s.controller !== "user") { s.status = "thinking"; }
          this.emitState(true);
          this.bcast("tool_end", { name: tool, ok: false, summary: "declined", job });
          return false;
        }
        s.status = "working";
        live.base = { ...live.base, gated: false };
        this.emitState(true);
        this.step({ step_id: stepId, kind: "click", label: `Approved · Clicked “${o.label}”`, status: "running", by: "neo" });
        await this.wait(gen, 160);
      }
      if (o.kind === "click" && e) { s.page.pressed = e.id; this.invalidate(true); await this.wait(gen, 110); s.page.pressed = null; }
      if (o.perform) await o.perform();
      else if (e && o.kind === "click") this.activate(e, "neo");
      this.action({ ...live.base, aid: live.aid, phase: "done" }, false);
      this.step({ step_id: stepId, kind: o.kind, label: o.gated ? `Approved · Clicked “${o.label}”` : o.step, status: "ok", by: "neo" });
      this.thumb(stepId);
      this.bcast("tool_end", { name: tool, ok: true, summary: o.step, job });
      return true;
    } finally {
      this.lock = false;
      this.inflight = null;
      if (this.s && this.s.controller === "neo" && gen === this.gen) { this.s.status = "thinking"; this.emitState(); }
    }
  }

  private async typeInto(gen: number, id: string, value: string) {
    const s = this.s!;
    s.page.focus = id;
    s.page.values[id] = "";
    this.caret();
    for (const ch of value) {
      s.page.values[id] += ch;
      s.page.caret = true;
      this.invalidate();
      await this.wait(gen, 42 + Math.random() * 36);
    }
  }

  private onText(t: string) {
    if (this.running) {
      this.say("user", t);
      window.setTimeout(() => this.say("assistant", "Got it — I'll keep that in mind as I go."), 600);
      return;
    }
    void this.shop(t);
  }

  /** The demo task: find earbuds under $80, pick the best-rated, check out (with a login and an approval on the way). */
  async shop(request = "Find wireless earbuds under $80 on Northwind with great reviews, and order the best pair") {
    this.stopJob(true);
    const gen = ++this.gen;
    this.running = true;
    const job = "j" + gen;
    const t0 = performance.now();
    try {
      this.say("user", request, job);
      this.setNeo("thinking");
      await this.wait(gen, 700);
      this.say("assistant", "On it — opening Northwind to look for earbuds under $80.", job);
      this.open(job, "home");
      await this.wait(gen, 1500);
      const s = this.s!;
      s.status = "working";
      s.loading = true; s.progress = 0.2;
      s.page.loaded = 0.3;
      this.invalidate(true);
      this.emitState(true);
      this.action({ kind: "navigate", phase: "intent", url: urlOf(s.page) });
      const nav = this.step({ kind: "navigate", label: "Opened northwind.store", status: "running", by: "neo" });
      await this.wait(gen, 500);
      s.page.loaded = 1; s.loading = false; s.progress = null;
      this.invalidate(true);
      this.emitState(true);
      this.step({ step_id: nav, kind: "navigate", label: "Opened northwind.store", status: "ok", by: "neo" });
      this.thumb(nav);
      s.status = "thinking"; this.emitState();
      await this.wait(gen, 700);

      await this.act(gen, { kind: "type", id: "q", label: "Search Northwind", field: "Search", text: "wireless earbuds", step: "Typed “wireless earbuds” in Search", perform: () => this.typeInto(gen, "q", "wireless earbuds") });
      await this.act(gen, { kind: "key", key: "Enter", step: "Pressed Enter", rect: this.s!.page.focus ? this.vrect(this.el("q")!) : undefined, perform: () => this.go("search") });
      await this.wait(gen, 900);
      await this.act(gen, { kind: "read", step: "Read the results", perform: () => this.wait(gen, 900) });
      await this.wait(gen, 500);
      await this.act(gen, { kind: "scroll", dir: "down", amount: 420, step: "Scrolled down", perform: async () => { await this.scrollTo(this.s!.page.scrollY + 420); } });
      await this.wait(gen, 400);
      const card = this.el("img-aero")!;
      await this.act(gen, { kind: "read", rect: this.vrect(card), label: "Aero Buds Pro", step: "Compared ratings and prices", perform: () => this.wait(gen, 900) });
      await this.wait(gen, 600);
      await this.act(gen, { kind: "click", id: "p-aero", label: "Aero Buds Pro", step: "Opened “Aero Buds Pro”" });
      await this.wait(gen, 1100);
      await this.act(gen, { kind: "read", step: "Read the product page", perform: () => this.wait(gen, 800) });
      await this.act(gen, { kind: "select", id: "sw-Midnight", label: "Midnight", field: "Colour", step: "Chose “Midnight” in Colour", perform: () => { this.s!.page.colour = "Midnight"; this.invalidate(); } });
      await this.wait(gen, 400);
      await this.act(gen, { kind: "click", id: "add", label: "Add to cart", step: "Clicked “Add to cart”" });
      this.say("assistant", "Found them — the Aero Buds Pro are $69.99 with a 4.7★ rating. Added to your cart.", job);
      await this.wait(gen, 1200);
      await this.act(gen, { kind: "click", id: "checkout", label: "Proceed to checkout", step: "Clicked “Proceed to checkout”" });
      await this.wait(gen, 900);

      // A login page: NEO never types passwords. The user signs in, then hands back.
      await this.hold(gen);
      const ss = this.s!;
      if (ss.page.key === "signin" && !ss.page.signedIn) {
        // NEO never types passwords: it reads the page, sees a sign-in form, and hands over.
        await this.act(gen, { kind: "read", step: "Found a sign-in form", perform: () => this.wait(gen, 700) });
        ss.controller = "user";
        ss.status = "need_user";
        ss.userSince = now();
        this.focusField("email");
        ss.needUser = { reason: "login", site: "Northwind", title: "Log in to Northwind here, then hand back.", sub: "NEO won't see what you type.", rect: this.vrect(this.el("email")!) };
        ss.userStep = this.step({ kind: "user", label: "Your turn to log in", status: "waiting", by: "user" });
        this.bcastBrowser("browser_need_user", { ...ss.needUser, sid: ss.sid });
        this.setNeo("idle");
        this.emitState(true);
        this.say("assistant", "Northwind wants you to sign in. Log in right there in the browser card, then hand it back — I won't see what you type.", job);
        await this.waitRelease();
        if (gen !== this.gen) throw CANCEL;
        if (!ss.page.signedIn) { ss.page.signedIn = true; await this.go("checkout"); }
        else if ((ss.page.key as PageKey) !== "checkout") await this.go("checkout");
        await this.wait(gen, 700);
      }
      await this.act(gen, { kind: "read", step: "Checked the order summary", perform: () => this.wait(gen, 900) });
      const placed = await this.act(gen, { kind: "click", id: "place", label: "Place your order", gated: true, step: "Asked to click “Place your order”" });
      await this.wait(gen, 1300);
      this.setNeo("speaking");
      const ms = Math.round(performance.now() - t0);
      if (placed) {
        this.say("assistant", "Done — I ordered the **Aero Buds Pro** in Midnight for **$76.20** with tax. They're the best-rated pair under $80 (4.7★ from 12,408 ratings) and arrive **tomorrow**. Order #NW-208-4417.", job);
      } else {
        this.say("assistant", "Okay — I didn't place the order. The Aero Buds Pro are still in your cart at $69.99 whenever you want them.", job);
      }
      this.bcast("turn", { route: "agent", brain: "gemini", model: "gemini-2.5-flash", ms, tools: this.s?.steps.length ?? 0, job });
      this.running = false;
      if (this.s) { this.s.job = null; if (this.s.controller === "neo") this.s.status = "thinking"; this.emitState(true); }
      await this.sleep(700);
      this.setNeo("idle");
      // Task end: the card folds into a row 1.5 s after the final reply (not while the user drives).
      this.closeTimer = window.setTimeout(() => { if (this.s && this.s.controller === "neo" && !this.s.needUser && !this.running) void this.endSession("done"); }, 1500);
    } catch (err) {
      if (err !== CANCEL) console.error("[mock]", err);
    } finally {
      if (gen === this.gen) this.running = false;
    }
  }

  // ---- jump straight to a state (?state=… or the dev panel) -------------------------------------------
  async jump(state: string) {
    this.stopJob(true);
    clearTimeout(this.closeTimer);
    const gen = ++this.gen;
    const job = "j" + gen;
    const setup = async (key: PageKey, status: Status, controller: Controller = "neo", jobbed = true) => {
      this.say("user", "(mock) Show me the “" + state + "” state", job);
      this.open(jobbed ? job : "", key, status, controller, now() - SEED_SPAN);
      const s = this.s!;
      if (!jobbed) s.job = null;
      s.page.loaded = 1; s.loading = false;
      s.page.cart = key === "cart" || key === "checkout" ? 1 : 0;
      this.invalidate(true);
      this.emitState(true);
      this.seedSteps();
      return s;
    };
    try {
      switch (state) {
        case "starting": { await setup("home", "starting"); this.s!.loading = true; this.emitState(true); this.running = true; break; }
        case "loading": {
          const s = await setup("search", "working");
          this.running = true;
          s.loading = true; s.progress = 0.4; s.page.loaded = 0.3;
          this.action({ kind: "navigate", phase: "intent", url: urlOf(s.page) });
          this.invalidate(true); this.emitState(true);
          break;
        }
        case "acting": case "thinking": {
          const s = await setup("search", state === "thinking" ? "thinking" : "working");
          this.running = true;
          if (state === "thinking") { this.action({ kind: "read", phase: "done", x: 700, y: 360 }); break; }
          for (let i = 0; ; i++) {
            const ids = ["img-pulse", "img-aero", "img-halo", "img-studio"];
            await this.act(gen, { kind: "read", rect: this.vrect(this.el(ids[i % 4])!), label: "Product card", step: "Looked at a product" , perform: () => this.wait(gen, 900) });
            await this.act(gen, { kind: "type", id: "q", label: "Search Northwind", field: "Search", text: "earbuds noise cancelling", step: "Typed “earbuds noise cancelling” in Search", perform: () => this.typeInto(gen, "q", "earbuds noise cancelling") });
            await this.act(gen, { kind: "scroll", dir: i % 2 ? "up" : "down", amount: 360, step: "Scrolled", perform: () => this.scrollTo(s.page.scrollY + (i % 2 ? -360 : 360)) });
            await this.act(gen, { kind: "click", id: "q-go", label: "Go", step: "Clicked “Go”", perform: () => {} });
            await this.wait(gen, 700);
          }
        }
        case "waiting": {
          const s = await setup("checkout", "working");
          s.page.signedIn = true;
          this.running = true;
          const placed = await this.act(gen, { kind: "click", id: "place", label: "Place your order", gated: true, step: "Asked to click “Place your order”" });
          await this.wait(gen, 900);
          this.say("assistant", placed ? "Done — your order is placed." : "Okay — I didn't place the order.", job);
          // The job is over while the page is still NEO's: the card says so, then folds (as in the full task).
          this.running = false;
          if (this.s) { this.s.job = null; if (this.s.controller === "neo") this.s.status = "thinking"; this.emitState(true); }
          this.setNeo("idle");
          this.closeTimer = window.setTimeout(() => { if (this.s && this.s.controller === "neo" && !this.s.needUser && !this.running) void this.endSession("done"); }, 1500);
          break;
        }
        case "user": { const s = await setup("product", "user", "user"); s.userStep = this.step({ kind: "user", label: "You took over", status: "running", by: "user" }); this.hintFocus(); this.running = true; await this.waitRelease(); break; }
        case "login": case "captcha": case "2fa": case "other": {
          const s = await setup("signin", "need_user", "user");
          this.running = true;
          const texts: Record<string, [NeedUser["reason"], string, string, string?]> = {
            login: ["login", "Log in to Northwind here, then hand back.", "NEO won't see what you type."],
            captcha: ["captcha", "Solve the check on Northwind, then hand back.", "It's there to confirm a person is here."],
            "2fa": ["2fa", "Enter your verification code here, then hand back.", "Codes stay between you and Northwind."],
            other: ["other", "NEO needs you to finish a step on this page, then hand back.", "", "Pick a delivery slot that works for you"],
          };
          const [reason, title, sub, detail] = texts[state];
          if (reason === "login" || reason === "2fa") this.focusField("email");
          s.needUser = { reason, site: "Northwind", title, sub, detail, rect: this.vrect(this.el("email")!) };
          s.userStep = this.step({ kind: "user", label: "Your turn", status: "waiting", by: "user" });
          this.bcastBrowser("browser_need_user", { ...s.needUser, sid: s.sid });
          this.emitState(true);
          await this.waitRelease();
          break;
        }
        case "dialog": {
          const s = await setup("cart", "user", "user");
          const del = this.el("del");
          if (del) this.activate(del, "user");
          void s;
          break;
        }
        case "select": {
          const s = await setup("product", "user", "user");
          const q = this.el("qty");
          if (q) {
            if (q.y + q.h > s.page.vh - 40) { s.page.scrollY = Math.min(q.y - 200, build(s.page).docH - s.page.vh); this.invalidate(true); }
            window.setTimeout(() => this.activate(this.el("qty")!, "user"), 500);
          }
          break;
        }
        case "error": {
          const s = await setup("product", "error");
          s.error = { code: "crashed", message: "This page crashed" };
          this.emitState(true);
          break;
        }
        case "tabs": {
          const s = await setup("search", "user", "user");
          void s;
          break;
        }
        case "ended": {
          await setup("confirmed", "thinking");
          await this.sleep(900);
          await this.endSession("done");
          break;
        }
        case "offline": {
          await setup("product", "working");
          this.running = true;
          window.setTimeout(() => this.kill(6000), 900);
          break;
        }
        default: void this.shop();
      }
    } catch (err) {
      if (err !== CANCEL) console.error("[mock]", err);
    }
  }

  /** The steps a jumped-to state would have behind it, timed from the (backdated) start, each
   *  with a thumbnail of the page it happened on — not of the page the jump lands on. */
  private seedSteps() {
    const s = this.s!;
    const p = s.page;
    const at = (key: PageKey, extra: Partial<PState> = {}): PState => ({ ...p, key, scrollY: 0, hover: null, pressed: null, focus: null, values: {}, loaded: 1, banner: false, ...extra });
    const seed: [BrowserStep["kind"], string, PState][] = [
      ["navigate", "Opened northwind.store", at("home")],
      ["type", "Typed “wireless earbuds” in Search and pressed Enter", at("home", { values: { q: "wireless earbuds" }, focus: "q" })],
      ["read", "Read the results", at("search")],
      ["scroll", "Scrolled down", at("search", { scrollY: 420 })],
    ];
    const gap = SEED_SPAN / (seed.length + 1);
    seed.forEach(([kind, label, page], i) => {
      const id = ++s.stepId;
      const st: BrowserStep = { sid: s.sid, step_id: id, ts: s.startedAt + gap * (i + 1), kind, label, status: "ok", by: "neo", url: urlOf(page) };
      s.steps.push(st);
      this.bcastBrowser("browser_step", st);
      void this.snapshot(400, 0.55, page).then((thumb) => { if (this.s === s && thumb) this.step({ ...st, thumb }); });
    });
  }

  setSpeed(x: number) { this.speed = x; }
}

let core: MockCore | null = null;
export function mockCore(): MockCore {
  if (!core) {
    core = new MockCore();
    const st = new URLSearchParams(location.search).get("state");
    window.setTimeout(() => { if (st) void core!.jump(st); else void core!.shop(); }, 900);
    (window as unknown as { neoMock: MockCore }).neoMock = core;
  }
  return core;
}

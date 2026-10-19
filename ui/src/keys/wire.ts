// The Brains & Keys window's line to the core: the one websocket client of kind `keys`
// (neo2/core/server/ws.py, "Brains & Keys"). It connects with the token and the ticket NEO.app's
// shell minted for this window alone (ui/src-tauri/src/keys.rs: `keys_ticket`), sends the window's
// commands with a request id each, and hands each answer back to whoever asked.
//
// A key travels once: in the command that carries it, sent at once and not kept. Nothing is queued
// while the core is away (a command then fails as "offline" — a key is never sent later, to a core
// the user didn't see), no command is remembered after it is sent, and nothing goes to storage.
// The connection's URL carries the ticket: it is never logged.

import type { BrainsSummary, LocalVoice, Result } from "./types.ts";

export interface SocketLike {
  readyState: number;
  onopen: ((ev?: unknown) => void) | null;
  onmessage: ((m: { data: unknown }) => void) | null;
  onclose: ((ev?: unknown) => void) | null;
  onerror: ((ev?: unknown) => void) | null;
  send(data: string): void;
  close(): void;
}

export interface WireDeps {
  /** The window's ticket from NEO.app's shell. Rejects with "no_link" (a core started outside the
   *  app), "not_ready" (one is starting), "no_core" (none runs) or "forbidden" (not that window). */
  ticket: () => Promise<string>;
  /** The core's websocket token. */
  token: () => Promise<string>;
  socket: (url: string) => SocketLike;
  url?: string;
  later?: (fn: () => void, ms: number) => unknown;
  cancel?: (handle: unknown) => void;
}

/** Where the line is: connecting (or reconnecting), open, or why it can't be. */
export type WireStatus = "connecting" | "open" | "no_link" | "no_core" | "forbidden" | "stopped";

/** Each command's answer, by the command's name (neo2/core/brains.py: RESULTS). */
export const RESULT_OF: Readonly<Record<string, string>> = {
  brains_get: "brains_detail",
  key_test: "key_test_result",
  key_save: "key_save_result",
  key_remove: "key_remove_result",
  models_list: "models_result",
  brains_set: "brains_set_result",
  custom_save: "custom_save_result",
  custom_remove: "custom_remove_result",
  brains_restart: "brains_restart_result",
  voice_download: "voice_download_result", // (not one of brains.py's: the core's download, ws.py)
};

export interface HelloData { state?: string; problems?: string[]; brains?: BrainsSummary | null; shell?: unknown; ui_host?: string }
type Events = {
  hello: HelloData;
  brains: BrainsSummary;
  problems: string[];
  state: string;
  status: WireStatus;
  local_voice: LocalVoice;
};
type Handler<K extends keyof Events> = (data: Events[K]) => void;

const OPEN = 1;
const TICKET = /^[0-9a-f]{64}$/;
const ID = /^[A-Za-z0-9_.:-]{1,32}$/;
/** How long to wait before trying again, by why the last try didn't connect. */
const RETRY_MS: Readonly<Record<string, number>> = { closed: 1500, not_ready: 1500, no_core: 4000, no_link: 8000 };
const TIMEOUT_MS = 30_000; // a key check is 10 s at most (keycheck.TIMEOUT_S); a save tests first

interface Pending { expect: string; done: (r: Result) => void; timer: unknown }

export class KeysWire {
  status: WireStatus = "connecting";
  private readonly deps: Required<Pick<WireDeps, "url" | "later" | "cancel">> & WireDeps;
  private sock: SocketLike | null = null;
  private seq = 0;
  private readonly pending = new Map<string, Pending>();
  private readonly handlers: { [K in keyof Events]: Set<Handler<K>> } = {
    hello: new Set(), brains: new Set(), problems: new Set(), state: new Set(), status: new Set(), local_voice: new Set(),
  };
  private retry: unknown = null;
  private stopped = false;

  constructor(deps: WireDeps) {
    this.deps = {
      ...deps,
      url: deps.url ?? "ws://127.0.0.1:8765",
      later: deps.later ?? ((fn, ms) => setTimeout(fn, ms)),
      cancel: deps.cancel ?? ((h) => clearTimeout(h as ReturnType<typeof setTimeout>)),
    };
  }

  on<K extends keyof Events>(event: K, cb: Handler<K>): () => void {
    (this.handlers[event] as Set<Handler<K>>).add(cb);
    return () => { (this.handlers[event] as Set<Handler<K>>).delete(cb); };
  }

  private emit<K extends keyof Events>(event: K, data: Events[K]): void {
    for (const h of [...(this.handlers[event] as Set<Handler<K>>)]) {
      try { h(data); } catch { /* one listener's error must not stop the rest */ }
    }
  }

  private setStatus(s: WireStatus): void {
    if (this.status === s) return;
    this.status = s;
    this.emit("status", s);
  }

  start(): void {
    this.stopped = false;
    void this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.retry !== null) { this.deps.cancel(this.retry); this.retry = null; }
    const s = this.sock;
    this.sock = null;
    this.failAll("disconnected");
    try { s?.close(); } catch { /* already closed */ }
    this.setStatus("stopped");
  }

  private again(why: string): void {
    if (this.stopped) return;
    if (this.retry !== null) this.deps.cancel(this.retry);
    this.retry = this.deps.later(() => { this.retry = null; void this.connect(); }, RETRY_MS[why] ?? RETRY_MS.closed);
  }

  private async connect(): Promise<void> {
    if (this.stopped || this.sock) return;
    let ticket: string;
    let token: string;
    try {
      ticket = await this.deps.ticket();
    } catch (e) {
      const why = typeof e === "string" ? e : e instanceof Error ? e.message : "";
      if (why === "no_link" || why === "no_core") { this.setStatus(why); this.again(why); return; }
      if (why === "not_ready") { this.setStatus("connecting"); this.again(why); return; }
      this.setStatus("forbidden"); // not NEO.app's own Brains & Keys window: nothing to ask for, ever
      return;
    }
    if (!TICKET.test(ticket)) { this.setStatus("forbidden"); return; }
    try { token = await this.deps.token(); } catch { this.setStatus("connecting"); this.again("not_ready"); return; }
    if (this.stopped || this.sock) return;
    const ws = this.deps.socket(`${this.deps.url}/?token=${encodeURIComponent(token)}&ticket=${ticket}`);
    this.sock = ws;
    this.setStatus("connecting");
    ws.onopen = () => { if (this.sock === ws) this.setStatus("open"); };
    ws.onmessage = (m) => { if (this.sock === ws && typeof m.data === "string") this.receive(m.data); };
    ws.onclose = () => {
      if (this.sock !== ws) return;
      this.sock = null;
      this.failAll("disconnected");
      if (!this.stopped) { this.setStatus("connecting"); this.again("closed"); }
    };
    ws.onerror = () => { try { ws.close(); } catch { /* closing */ } };
  }

  private receive(text: string): void {
    let msg: { type?: unknown; data?: unknown };
    try { msg = JSON.parse(text); } catch { return; }
    const type = typeof msg?.type === "string" ? msg.type : "";
    const data = (msg?.data && typeof msg.data === "object" ? msg.data : {}) as Record<string, unknown>;
    switch (type) {
      case "hello": this.emit("hello", data as HelloData); return;
      case "brains": this.emit("brains", data as unknown as BrainsSummary); return;
      case "problems": this.emit("problems", Array.isArray(data.items) ? (data.items as string[]) : []); return;
      case "state": if (typeof data.state === "string") this.emit("state", data.state); return;
      case "local_voice": if (typeof data.state === "string") this.emit("local_voice", data as unknown as LocalVoice); return;
    }
    const id = typeof data.id === "string" ? data.id : "";
    const p = id ? this.pending.get(id) : undefined;
    if (!p) return;
    if (type === "keys_refused") { this.finish(id, { ...data, id, ok: false, error: "forbidden" }); return; }
    if (type === p.expect) this.finish(id, data as Result);
  }

  private finish(id: string, r: Result): void {
    const p = this.pending.get(id);
    if (!p) return;
    this.pending.delete(id);
    this.deps.cancel(p.timer);
    p.done(r);
  }

  private failAll(error: string): void {
    for (const id of [...this.pending.keys()]) this.finish(id, { id, ok: false, error });
  }

  /** How many answers are still awaited (none holds what was sent). */
  get waiting(): number { return this.pending.size; }

  /** Send one of the window's commands; resolves with its answer (never rejects). Not connected:
   *  {ok: false, error: "offline"} at once — nothing is kept to send later. */
  request(cmd: string, body: Record<string, unknown> = {}, timeoutMs = TIMEOUT_MS): Promise<Result> {
    const expect = RESULT_OF[cmd];
    if (!expect) return Promise.resolve({ ok: false, error: "unknown" });
    const ws = this.sock;
    if (!ws || ws.readyState !== OPEN || this.status !== "open") return Promise.resolve({ ok: false, error: "offline" });
    const id = `r${++this.seq}`;
    if (!ID.test(id)) this.seq = 0; // (never: 2^53 requests)
    return new Promise<Result>((done) => {
      const timer = this.deps.later(() => this.finish(id, { id, ok: false, error: "timeout" }), timeoutMs);
      this.pending.set(id, { expect, done, timer });
      try {
        ws.send(JSON.stringify({ ...body, cmd, id }));
      } catch {
        this.finish(id, { id, ok: false, error: "offline" });
      }
    });
  }
}

/** NEO.app's shell, for the real window. Outside NEO.app (a plain dev tab) there is no ticket. */
export function tauriDeps(): WireDeps {
  const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
  return {
    async ticket() {
      if (!inTauri) throw "forbidden";
      const { invoke } = await import("@tauri-apps/api/core");
      try { return await invoke<string>("keys_ticket"); } catch (e) { throw typeof e === "string" ? e : "forbidden"; }
    },
    async token() {
      const { invoke } = await import("@tauri-apps/api/core");
      return invoke<string>("ws_token");
    },
    socket: (url) => new WebSocket(url) as unknown as SocketLike,
  };
}

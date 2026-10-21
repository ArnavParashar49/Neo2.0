// Search Screen's own line to the core: the websocket client of kind `lens`, admitted with the
// session's ticket (neo2/core/server/ws.py, "Search Screen"). It sends the overlay's commands and
// hands back what arrives — text messages checked by ./wire.ts, and the still (NBF1 kind 4).
// A dropped line is tried again at most three times within five seconds (a core that restarted
// knows nothing of the ticket: the overlay then says so). The URL carries the ticket: never logged.

import { parseMsg, parseStill, type CoreMsg, type Still } from "./wire.ts";

export interface SocketLike {
  readyState: number;
  binaryType?: string;
  onopen: ((ev?: unknown) => void) | null;
  onmessage: ((m: { data: unknown }) => void) | null;
  onclose: ((ev?: unknown) => void) | null;
  onerror: ((ev?: unknown) => void) | null;
  send(data: string): void;
  close(): void;
}

export interface LensSocketDeps {
  token: () => Promise<string>;
  socket: (url: string) => SocketLike;
  url?: string;
  later?: (fn: () => void, ms: number) => unknown;
  cancel?: (h: unknown) => void;
  now?: () => number;
}

export interface LensSocketHandlers {
  msg: (m: CoreMsg) => void;
  still: (s: Still) => void;
  status: (s: "connecting" | "open" | "failed") => void;
}

const OPEN = 1;
const TICKET = /^[0-9a-f]{64}$/;
export const RETRIES = 3;
export const RETRY_WINDOW_MS = 5000;
const RETRY_MS = 600;
/** Commands are tiny; anything bigger is not one of the overlay's. */
const MAX_SEND = 8192;

export class LensSocket {
  private ws: SocketLike | null = null;
  private ticket = "";
  private tries: number[] = [];
  private timer: unknown = null;
  private stopped = true;
  private readonly d: Required<Pick<LensSocketDeps, "url" | "later" | "cancel" | "now">> & LensSocketDeps;
  private readonly on: LensSocketHandlers;

  constructor(deps: LensSocketDeps, on: LensSocketHandlers) {
    this.on = on;
    this.d = {
      ...deps,
      url: deps.url ?? "ws://127.0.0.1:8765",
      later: deps.later ?? ((fn, ms) => setTimeout(fn, ms)),
      cancel: deps.cancel ?? ((h) => clearTimeout(h as ReturnType<typeof setTimeout>)),
      now: deps.now ?? (() => Date.now()),
    };
  }

  get open(): boolean { return !!this.ws && this.ws.readyState === OPEN; }

  connect(ticket: string): void {
    if (!TICKET.test(ticket)) { this.on.status("failed"); return; }
    if (!this.stopped && this.ticket === ticket) return;
    this.close();
    this.stopped = false;
    this.ticket = ticket;
    this.tries = [];
    void this.dial();
  }

  close(): void {
    this.stopped = true;
    if (this.timer !== null) { this.d.cancel(this.timer); this.timer = null; }
    const ws = this.ws;
    this.ws = null;
    try { ws?.close(); } catch { /* closed */ }
  }

  /** Whether the command went out (nothing is kept to send later). */
  send(cmd: Record<string, unknown>): boolean {
    const ws = this.ws;
    if (!ws || ws.readyState !== OPEN) return false;
    const text = JSON.stringify(cmd);
    if (text.length > MAX_SEND) return false;
    try { ws.send(text); return true; } catch { return false; }
  }

  private async dial(): Promise<void> {
    if (this.stopped) return;
    const now = this.d.now();
    this.tries = this.tries.filter((t) => now - t < RETRY_WINDOW_MS);
    if (this.tries.length >= RETRIES) { this.stopped = true; this.on.status("failed"); return; }
    this.tries.push(now);
    this.on.status("connecting");
    let token = "";
    try { token = await this.d.token(); } catch { /* none: the core says no */ }
    if (this.stopped) return;
    const ws = this.d.socket(`${this.d.url}/?token=${encodeURIComponent(token)}&lens=${this.ticket}`);
    this.ws = ws;
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { if (this.ws === ws) this.on.status("open"); };
    ws.onmessage = (m) => {
      if (this.ws !== ws) return;
      if (typeof m.data === "string") {
        const msg = parseMsg(m.data);
        if (msg) this.on.msg(msg);
      } else if (m.data instanceof ArrayBuffer) {
        const still = parseStill(m.data);
        if (still) this.on.still(still);
      }
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      if (this.stopped) return;
      this.timer = this.d.later(() => { this.timer = null; void this.dial(); }, RETRY_MS);
    };
    ws.onerror = () => { try { ws.close(); } catch { /* closing */ } };
  }
}

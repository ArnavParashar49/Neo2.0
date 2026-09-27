// A WebSocket look-alike wired to the in-page mock core (dev only, ?mock=browser). Messages are
// delivered asynchronously and in order, like a real socket; frames arrive as ArrayBuffers.
import { mockCore, type MockConn } from "./server";

type Handler = ((ev: any) => void) | null;

export class MockSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  readyState = 0;
  binaryType = "arraybuffer";
  onopen: Handler = null;
  onmessage: Handler = null;
  onclose: Handler = null;
  onerror: Handler = null;
  private conn: MockConn | null = null;

  constructor() {
    setTimeout(() => {
      const core = mockCore();
      if (!core.accepting) { this.readyState = 3; this.onerror?.({}); this.onclose?.({}); return; }
      this.readyState = 1;
      this.onopen?.({});
      this.conn = core.connect(this);
    }, 40);
  }

  send(data: string) {
    if (this.readyState !== 1) return;
    const c = this.conn;
    queueMicrotask(() => c?.receive(data));
  }

  close() {
    if (this.readyState >= 2) return;
    this.readyState = 3;
    this.conn?.closed();
    this.conn = null;
    queueMicrotask(() => this.onclose?.({}));
  }

  /** Server → client. */
  deliver(data: string | ArrayBuffer) {
    if (this.readyState !== 1) return;
    queueMicrotask(() => { if (this.readyState === 1) this.onmessage?.({ data }); });
  }

  /** The mock core went away (simulated crash / restart). */
  drop() {
    if (this.readyState !== 1) return;
    this.readyState = 3;
    this.conn = null;
    queueMicrotask(() => this.onclose?.({}));
  }
}

// While the user has control: pointer, wheel and keys on the view become page input (§6.1).
// Every coordinate is converted to page viewport CSS px through the frame rect.

export interface PageMap { left: number; top: number; s: number; vw: number; vh: number }

/** CDP modifier order: Alt 1, Ctrl 2, Meta 4, Shift 8. */
export function modsOf(e: { altKey: boolean; ctrlKey: boolean; metaKey: boolean; shiftKey: boolean }): number {
  return (e.altKey ? 1 : 0) | (e.ctrlKey ? 2 : 0) | (e.metaKey ? 4 : 0) | (e.shiftKey ? 8 : 0);
}

export function buttonName(b: number): "left" | "right" | "middle" {
  return b === 2 ? "right" : b === 1 ? "middle" : "left";
}

/** DOM `buttons` bitmask is already Left 1, Right 2, Middle 4. */
export function buttonsMask(b: number): number { return b & 7; }

const r1 = (v: number) => Math.round(v * 10) / 10;

export function toPage(clientX: number, clientY: number, m: PageMap): { x: number; y: number } {
  const x = (clientX - m.left) / m.s, y = (clientY - m.top) / m.s;
  return { x: r1(Math.min(m.vw, Math.max(0, x))), y: r1(Math.min(m.vh, Math.max(0, y))) };
}

/** Wheel deltas in CSS px of the page (so content moves under the fingers 1:1 on screen). */
export function wheelDelta(e: WheelEvent, viewH: number, s: number): { dx: number; dy: number } {
  const k = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? viewH : 1;
  return { dx: (e.deltaX * k) / s, dy: (e.deltaY * k) / s };
}

type Seg = { segment(s: string): Iterable<unknown> };
const SegCtor = (Intl as unknown as { Segmenter?: new () => Seg }).Segmenter;
const seg: Seg | null = SegCtor ? new SegCtor() : null;

/** One grapheme with no ⌘/⌃: text the page should receive as typed. */
export function isPrintable(e: KeyboardEvent): boolean {
  if (e.metaKey || e.ctrlKey) return false;
  const k = e.key;
  if (!k || k === "Unidentified" || k === "Dead" || k === "Process") return false;
  if (k.length === 1) return true;
  if (!seg) return false;
  let n = 0;
  for (const _ of seg.segment(k)) { if (++n > 1) return false; }
  return n === 1 && !/^[A-Z][a-zA-Z0-9]+$/.test(k); // named keys ("Enter", "F5") are multi-char words
}

/** Shortcuts the card keeps for itself; never forwarded to the page (§6.3). */
export function reservedShortcut(e: KeyboardEvent): string | null {
  const k = e.key.toLowerCase();
  if (e.metaKey && e.shiftKey && !e.altKey && !e.ctrlKey) {
    if (k === "u") return "control";
    if (k === "e") return "expand";
    if (k === "m") return "minimize";
    if (k === "o") return "popout";
    if (k === "r") return "hard_reload";
  }
  if (e.metaKey && !e.shiftKey && !e.altKey && !e.ctrlKey) {
    if (k === ".") return "stop";
    if (k === "l") return "address";
    if (k === "[") return "back";
    if (k === "]") return "forward";
    if (k === "r") return "reload";
    if (k === "w") return "close_window";
  }
  if (e.ctrlKey && e.key === "Tab") return e.shiftKey ? "focus_prev" : "focus_next";
  return null;
}

/** Coalesce to one message per animation frame (the latest position wins; wheel deltas sum). */
export class FrameCoalescer<T> {
  private pending: T | null = null;
  private raf = 0;
  constructor(private flushFn: (v: T) => void, private merge: (a: T, b: T) => T = (_a, b) => b) {}
  push(v: T) {
    this.pending = this.pending ? this.merge(this.pending, v) : v;
    if (!this.raf) this.raf = requestAnimationFrame(() => this.flush());
  }
  flush() {
    if (this.raf) { cancelAnimationFrame(this.raf); this.raf = 0; }
    const v = this.pending;
    this.pending = null;
    if (v) this.flushFn(v);
  }
  cancel() { if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; this.pending = null; }
}

/** Click count for pointerdown (PointerEvent.detail is 0 in WebKit). */
export class ClickCounter {
  private last = { t: 0, x: 0, y: 0, n: 0, b: -1 };
  next(e: PointerEvent): number {
    const now = performance.now();
    const l = this.last;
    const same = now - l.t < 500 && Math.hypot(e.clientX - l.x, e.clientY - l.y) < 5 && e.button === l.b;
    this.last = { t: now, x: e.clientX, y: e.clientY, n: same ? Math.min(l.n + 1, 3) : 1, b: e.button };
    return this.last.n;
  }
}

// Live page frames: the NBF1 binary header (§10.4.1), decode off the main thread, paint on the
// next animation frame, ack so the core sends the next one. Frames never touch React state.
import type { FrameMeta } from "./types";

export const HEADER_LEN = 64;
const MAGIC = 0x4e424631; // "NBF1", read big-endian
const MIME: Record<number, string> = { 1: "image/jpeg", 2: "image/webp", 3: "image/png" };
const MIME_CODE: Record<string, number> = { "image/jpeg": 1, "image/webp": 2, "image/png": 3 };

export function b64url(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function b64urlToBytes(s: string): Uint8Array {
  const b = atob(s.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((s.length + 3) % 4));
  const out = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) out[i] = b.charCodeAt(i);
  return out;
}

/** Parse one binary frame. Returns null for anything that isn't a v1 NBF1 frame. */
export function parseFrame(buf: ArrayBuffer): { meta: FrameMeta; image: Uint8Array } | null {
  if (buf.byteLength < HEADER_LEN) return null;
  const dv = new DataView(buf);
  if (dv.getUint32(0, false) !== MAGIC || dv.getUint8(4) !== 1) return null;
  const hl = dv.getUint16(6, true);
  if (hl < HEADER_LEN || hl > buf.byteLength) return null;
  const kind = dv.getUint8(5) === 2 ? 2 : 1;
  const meta: FrameMeta = {
    sid: b64url(new Uint8Array(buf, 8, 16)),
    fid: dv.getUint32(24, true),
    kind,
    ts: dv.getFloat64(28, true),
    vw: dv.getUint16(36, true),
    vh: dv.getUint16(38, true),
    iw: dv.getUint16(40, true),
    ih: dv.getUint16(42, true),
    dpr: dv.getUint16(44, true) / 100,
    sx: dv.getFloat32(46, true),
    sy: dv.getFloat32(50, true),
    flags: dv.getUint16(54, true),
    mime: MIME[dv.getUint8(56)] ?? "image/jpeg",
  };
  return { meta, image: new Uint8Array(buf, hl) };
}

/** Build one binary frame (used by the dev mock and by tests; the core does this in Python). */
export function packFrame(meta: Omit<FrameMeta, "sid">, sid: Uint8Array, image: Uint8Array): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_LEN + image.byteLength);
  const dv = new DataView(buf);
  dv.setUint32(0, MAGIC, false);
  dv.setUint8(4, 1);
  dv.setUint8(5, meta.kind);
  dv.setUint16(6, HEADER_LEN, true);
  new Uint8Array(buf, 8, 16).set(sid.subarray(0, 16));
  dv.setUint32(24, meta.fid, true);
  dv.setFloat64(28, meta.ts, true);
  dv.setUint16(36, meta.vw, true);
  dv.setUint16(38, meta.vh, true);
  dv.setUint16(40, meta.iw, true);
  dv.setUint16(42, meta.ih, true);
  dv.setUint16(44, Math.round(meta.dpr * 100), true);
  dv.setFloat32(46, meta.sx, true);
  dv.setFloat32(50, meta.sy, true);
  dv.setUint16(54, meta.flags, true);
  dv.setUint8(56, MIME_CODE[meta.mime] ?? 1);
  new Uint8Array(buf, HEADER_LEN).set(image);
  return buf;
}

export interface Frame { meta: FrameMeta; bmp: ImageBitmap }
export type Painter = (f: Frame) => void;

/**
 * One decode in flight, one decoded frame waiting for paint, the rest dropped (never queued):
 * latency stays at one frame. Every frame is acked exactly once — painted or dropped — so the
 * core's credits always come back.
 */
export class FramePipe {
  private decoding = false;
  private queued: { meta: FrameMeta; blob: Blob } | null = null;
  private ready: Frame | null = null;
  private raf = 0;
  private lastFid = -1;
  private sid = "";
  private painters = new Set<Painter>();
  /** The frame on screen now; kept so a view that mounts later paints at once. */
  latest: Frame | null = null;

  constructor(private ack: (sid: string, fid: number) => void) {}

  /** A new session: forget the old one's frames. */
  setSid(sid: string) {
    if (sid === this.sid) return;
    this.sid = sid;
    this.lastFid = -1;
    if (this.queued) { this.ack(this.queued.meta.sid, this.queued.meta.fid); this.queued = null; }
    if (this.ready) { this.ready.bmp.close(); this.ack(this.ready.meta.sid, this.ready.meta.fid); this.ready = null; }
    if (this.latest) { this.latest.bmp.close(); this.latest = null; }
  }

  push(meta: FrameMeta, blob: Blob) {
    if (meta.sid !== this.sid || meta.fid <= this.lastFid) { this.ack(meta.sid, meta.fid); return; }
    this.lastFid = meta.fid;
    if (this.decoding) {
      if (this.queued) this.ack(this.queued.meta.sid, this.queued.meta.fid); // superseded
      this.queued = { meta, blob };
      return;
    }
    void this.decode(meta, blob);
  }

  private async decode(meta: FrameMeta, blob: Blob) {
    this.decoding = true;
    try {
      const bmp = await createImageBitmap(blob);
      if (meta.sid !== this.sid) { bmp.close(); this.ack(meta.sid, meta.fid); }
      else {
        if (this.ready) { this.ready.bmp.close(); this.ack(this.ready.meta.sid, this.ready.meta.fid); }
        this.ready = { meta, bmp };
        this.schedule();
      }
    } catch {
      this.ack(meta.sid, meta.fid);
    }
    this.decoding = false;
    const q = this.queued;
    if (q) { this.queued = null; void this.decode(q.meta, q.blob); }
  }

  private schedule() {
    if (this.raf) return;
    // rAF stops while the window is hidden; paint (and ack) anyway so credits keep flowing.
    if (document.hidden || typeof requestAnimationFrame === "undefined") { this.raf = window.setTimeout(() => this.paint(), 0) as unknown as number; return; }
    this.raf = requestAnimationFrame(() => this.paint());
  }

  private paint() {
    this.raf = 0;
    const f = this.ready;
    if (!f) return;
    this.ready = null;
    const prev = this.latest;
    this.latest = f;
    for (const p of this.painters) {
      try { p(f); } catch { /* one broken view must not stall the rest */ }
    }
    if (prev && prev !== f) prev.bmp.close();
    this.ack(f.meta.sid, f.meta.fid);
  }

  onPaint(p: Painter): () => void {
    this.painters.add(p);
    return () => { this.painters.delete(p); };
  }
}

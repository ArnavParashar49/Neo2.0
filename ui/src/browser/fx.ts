// What NEO is doing, drawn over the live page: its cursor gliding to each target, a highlight on
// the element it is about to use, a ripple where a click lands, a sweep while it reads, a pill
// while it scrolls. Imperative on purpose — Web Animations on transform/opacity only, no React
// renders per frame. Everything is placed in page viewport CSS px × the view scale.
import type { BrowserAction, Rect } from "./types";

export type FxMode = "acting" | "thinking" | "waiting" | "off";
type HlKind = "click" | "type" | "read" | "select" | "waiting";
type Pt = { x: number; y: number };
/** The page a highlight's rect was measured on: its viewport size and (when the core says) scroll. */
type Base = { vw: number; vh: number; sy?: number };
type Hl = { el: HTMLDivElement; rect: Rect; kind: HlKind; timer?: number; base: Base | null; dy: number };
export interface FrameGeo { vw: number; vh: number; sy: number }

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const CURSOR_SVG =
  '<svg viewBox="0 0 18 22" width="18" height="22" aria-hidden="true"><path d="M2 1 L2 17.5 L6.2 13.6 L9 20 L11.6 18.9 L8.9 12.6 L14.6 12.4 Z" /></svg>';

function div(cls: string, parent: HTMLElement): HTMLDivElement {
  const e = document.createElement("div");
  e.className = cls;
  parent.appendChild(e);
  return e;
}

function readXY(ox: HTMLElement, oy: HTMLElement): Pt {
  const m = (el: HTMLElement) => {
    const t = getComputedStyle(el).transform;
    return !t || t === "none" ? new DOMMatrixReadOnly() : new DOMMatrixReadOnly(t);
  };
  return { x: m(ox).m41, y: m(oy).m42 };
}

export class Fx {
  private s = 1;
  private w = 0;
  private h = 0;
  private mode: FxMode = "off";
  private reduced: boolean;
  private hlLayer: HTMLDivElement;
  private fxLayer: HTMLDivElement;
  private ox: HTMLDivElement;
  private oy: HTMLDivElement;
  private body: HTMLDivElement;
  private glyph: HTMLDivElement;
  private scrollPill: HTMLDivElement;
  private shown = false;
  private target: Pt | null = null;
  private anims: Animation[] = [];
  private travelEnd = 0;
  private hls = new Map<number, Hl>();
  private page: FrameGeo | null = null;
  private last: BrowserAction | null = null;
  private tagTimer: number | undefined;
  private restTimer: number | undefined;
  private scrollTimer: number | undefined;
  private alive = true;

  constructor(private root: HTMLElement, reduced: boolean) {
    this.reduced = reduced;
    div("fx-scrim", root);
    this.fxLayer = div("fx-layer", root);
    this.hlLayer = div("fx-layer", root);
    this.ox = div("fx-cursor", root);
    this.oy = div("fx-cursor-y", this.ox);
    this.body = div("fx-cursor-body", this.oy);
    this.glyph = div("fx-cursor-glyph", this.body);
    this.glyph.innerHTML = CURSOR_SVG;
    const tag = div("fx-cursor-tag", this.body);
    tag.textContent = "NEO";
    div("fx-cursor-dot", this.body);
    this.scrollPill = div("fx-scroll", root);
    this.scrollPill.innerHTML =
      '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path d="M7 10l5 5 5-5" /></svg><svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path d="M7 10l5 5 5-5" /></svg>';
  }

  destroy() {
    this.alive = false;
    clearTimeout(this.tagTimer); clearTimeout(this.restTimer); clearTimeout(this.scrollTimer);
    for (const h of this.hls.values()) clearTimeout(h.timer);
    this.root.replaceChildren();
  }

  setReduced(r: boolean) { this.reduced = r; }

  /** The frame rect changed: scale page px → view px again, without animating. */
  setGeometry(s: number, w: number, h: number) {
    if (Math.abs(s - this.s) < 1e-4 && Math.abs(w - this.w) < 0.5 && Math.abs(h - this.h) < 0.5) return;
    this.s = s; this.w = w; this.h = h;
    if (this.target && this.shown) {
      for (const a of this.anims) a.cancel();
      this.anims = [];
      this.ox.style.transform = `translateX(${this.target.x * s}px)`;
      this.oy.style.transform = `translateY(${this.target.y * s}px)`;
    }
    for (const hl of this.hls.values()) this.placeHl(hl.el, hl.rect, hl.dy);
  }

  /** Every painted frame. A highlight is a viewport rect on the page as it was when NEO measured
   *  it: when the page on screen is a different size it points at nothing (hide it); when it
   *  has only scrolled, the element moved with it (follow). Never trust a stale box on an approval. */
  frame(g: FrameGeo) {
    this.page = g;
    for (const h of this.hls.values()) this.syncHl(h);
  }

  private baseOf(a: BrowserAction): Base | null {
    if (a.vw && a.vh) return { vw: a.vw, vh: a.vh, sy: a.sy };
    // Older cores don't say: assume the frame on screen (size only — its scroll may lag the action).
    return this.page ? { vw: this.page.vw, vh: this.page.vh } : null;
  }

  private syncHl(h: Hl) {
    const b = h.base, g = this.page;
    if (!b || !g) return;
    // A different page size: the box points at nothing on screen. Hide it until a frame of the
    // size it was measured on arrives (a lagging frame catches up; a real resize never does).
    const stale = Math.abs(b.vw - g.vw) > 0.5 || Math.abs(b.vh - g.vh) > 0.5;
    h.el.classList.toggle("stale", stale);
    if (stale || b.sy == null) return;
    const dy = b.sy - g.sy;
    if (Math.abs(dy - h.dy) < 0.25) return;
    h.dy = dy;
    this.placeHl(h.el, h.rect, dy);
  }

  /** "Take control" was pressed: NEO finishes its step, and the cursor steps back meanwhile. */
  setTaking(on: boolean) { this.root.classList.toggle("taking", on); }

  setMode(m: FxMode) {
    if (m === this.mode) return;
    const was = this.mode;
    this.mode = m;
    this.root.classList.toggle("waiting", m === "waiting");
    this.root.classList.toggle("thinking", m === "thinking");
    if (m === "off") {
      this.hideCursor();
      for (const id of [...this.hls.keys()]) this.hideHl(id, 0);
      this.fxLayer.replaceChildren();
    } else if (was === "off") {
      // Turned on late (first frame, reconnect, hand back): pick the story up where it is.
      if (this.last) this.place(this.last);
      else if (this.target) this.showCursorAt(this.target);
    }
    const l = this.last;
    if (m === "waiting" && l && l.gated && l.phase === "intent" && l.rect && this.hls.get(l.aid)?.kind !== "waiting") this.showHl(l.aid, l.rect, "waiting", this.baseOf(l));
    if (m === "thinking") this.rest(true);
    else if (m === "acting") this.rest(false);
    if (m !== "waiting") for (const [id, h] of this.hls) if (h.kind === "waiting") this.hideHl(id, 0);
  }

  /** Put the cursor somewhere without a story (late joiner: the last known action). */
  place(a: BrowserAction) {
    this.last = a;
    const pt = pointOf(a);
    if (!pt) return;
    this.target = pt;
    if (this.mode === "off") return;
    this.showCursorAt(pt);
    if (a.phase === "intent" && a.rect && a.kind !== "move" && a.kind !== "navigate" && a.kind !== "scroll") {
      this.showHl(a.aid, a.rect, a.gated ? "waiting" : a.kind === "type" ? "type" : a.kind === "read" ? "read" : "click", this.baseOf(a));
    }
  }

  action(a: BrowserAction) {
    if (!this.alive) return;
    this.last = a.kind === "scroll" && this.last ? this.last : a;
    const pt = pointOf(a);
    if (this.mode === "off") { if (pt) this.target = pt; return; }
    this.wake();
    switch (a.kind) {
      case "click":
      case "select":
      case "move":
        if (a.phase === "intent") {
          this.clearHls();
          if (a.rect && a.kind !== "move") this.showHl(a.aid, a.rect, a.gated ? "waiting" : a.kind === "select" ? "select" : "click", this.baseOf(a));
          if (pt) this.moveTo(pt);
        } else if (a.phase === "done") {
          const hl = this.hls.get(a.aid);
          if (hl && hl.kind === "waiting") { hl.el.classList.remove("waiting"); hl.kind = "click"; hl.el.querySelector(".fx-hl-chip")?.remove(); }
          if (pt && a.kind !== "move") {
            const wait = this.arriveSoon(pt);
            window.setTimeout(() => { if (this.alive && this.mode !== "off") this.click(a.button === "right", (a.clicks ?? 1) >= 2); }, wait);
            this.hideHl(a.aid, wait + 300);
          } else this.hideHl(a.aid, 300);
        } else this.hideHl(a.aid, 0);
        break;
      case "type":
        if (a.phase === "intent") {
          this.clearHls();
          if (a.rect) this.showHl(a.aid, a.rect, "type", this.baseOf(a));
          if (pt) this.moveTo(pt);
        } else this.hideHl(a.aid, a.phase === "done" ? 1200 : 0);
        break;
      case "read":
        if (a.phase === "intent") {
          this.clearHls();
          if (a.rect) this.showHl(a.aid, a.rect, "read", this.baseOf(a));
          this.sweep(a.rect);
        } else this.hideHl(a.aid, 250);
        break;
      case "scroll":
        // Highlights are viewport rects: once the page moves under them they point at the wrong thing.
        if (a.phase === "intent") { this.clearHls(); this.showScroll(a.dir ?? "down"); }
        break;
      case "navigate":
        if (a.phase === "intent") this.clearHls();
        break;
      case "key":
        // Typing then Enter in the same field reads as one gesture; the field lets go shortly after.
        if (a.phase !== "intent") for (const [id, h] of this.hls) if (h.kind !== "waiting") this.hideHl(id, 600);
        break;
      default:
        break;
    }
  }

  // ---- cursor ---------------------------------------------------------------------------
  private showCursorAt(pt: Pt) {
    for (const a of this.anims) a.cancel();
    this.anims = [];
    this.ox.style.transform = `translateX(${pt.x * this.s}px)`;
    this.oy.style.transform = `translateY(${pt.y * this.s}px)`;
    if (!this.shown) {
      this.shown = true;
      this.ox.classList.add("on");
      this.body.animate([{ opacity: 0, transform: "scale(.6)" }, { opacity: 1, transform: "scale(1)" }], { duration: this.reduced ? 120 : 140, easing: "cubic-bezier(0.2, 0.8, 0.2, 1)" });
    }
  }

  private hideCursor() {
    if (!this.shown) return;
    this.shown = false;
    for (const a of this.anims) a.cancel();
    this.anims = [];
    this.ox.classList.remove("on");
  }

  private moveTo(pt: Pt) {
    this.target = pt;
    const to = { x: pt.x * this.s, y: pt.y * this.s };
    if (!this.shown) {
      // First appearance: bottom centre of the view, then travel.
      this.shown = true;
      this.ox.classList.add("on");
      const start = { x: this.w * 0.5, y: this.h * 0.78 };
      this.ox.style.transform = `translateX(${start.x}px)`;
      this.oy.style.transform = `translateY(${start.y}px)`;
      this.body.animate([{ opacity: 0, transform: "scale(.6)" }, { opacity: 1, transform: "scale(1)" }], { duration: 140, easing: "cubic-bezier(0.2, 0.8, 0.2, 1)" });
    }
    const from = readXY(this.ox, this.oy);
    for (const a of this.anims) a.cancel();
    this.anims = [];
    this.ox.style.transform = `translateX(${to.x}px)`;
    this.oy.style.transform = `translateY(${to.y}px)`;
    const d = Math.hypot(to.x - from.x, to.y - from.y);
    if (d < 0.5) { this.travelEnd = 0; return; }
    if (this.reduced) {
      // No travel: fade out, reappear at the target.
      this.anims = [this.body.animate([{ opacity: 1 }, { opacity: 0, offset: 0.5 }, { opacity: 1 }], { duration: 120 })];
      this.travelEnd = performance.now() + 60;
      return;
    }
    const dur = clamp(160 + d * 0.35, 180, 560);
    this.anims = [
      this.ox.animate([{ transform: `translateX(${from.x}px)` }, { transform: `translateX(${to.x}px)` }], { duration: dur, easing: "cubic-bezier(.3,.7,.2,1)" }),
      this.oy.animate([{ transform: `translateY(${from.y}px)` }, { transform: `translateY(${to.y}px)` }], { duration: dur, easing: "cubic-bezier(.5,.1,.3,1)" }),
    ];
    this.travelEnd = performance.now() + dur;
    this.ox.classList.add("moving");
  }

  /** ms until the cursor is at `pt` — hurrying a travel in progress to land within 80 ms. */
  private arriveSoon(pt: Pt): number {
    if (!this.target || Math.hypot(this.target.x - pt.x, this.target.y - pt.y) > 2) this.moveTo(pt);
    const left = this.travelEnd - performance.now();
    if (left <= 0) return 0;
    if (left > 80 && !this.reduced) {
      for (const a of this.anims) a.updatePlaybackRate((a.playbackRate || 1) * (left / 80));
      return 80;
    }
    return left;
  }

  private click(right: boolean, double: boolean) {
    const p = this.target;
    if (!p) return;
    const x = p.x * this.s, y = p.y * this.s;
    if (!this.reduced) {
      this.glyph.animate(
        [{ transform: "scale(1)", easing: "ease-in" }, { transform: "scale(.86)", offset: 0.36, easing: "cubic-bezier(0.34, 1.4, 0.64, 1)" }, { transform: "scale(1)" }],
        { duration: 250 },
      );
    }
    this.ripple(x, y, right);
    if (double) window.setTimeout(() => this.alive && this.ripple(x, y, right), 120);
  }

  private ripple(x: number, y: number, right: boolean) {
    const mk = (cls: string, size: number) => {
      const e = div(cls + (right ? " right" : ""), this.fxLayer);
      e.style.left = `${x - size / 2}px`;
      e.style.top = `${y - size / 2}px`;
      return e;
    };
    if (this.reduced) {
      const r = mk("fx-ripple static", 44);
      r.animate([{ opacity: 0.9 }, { opacity: 0 }], { duration: 300 }).onfinish = () => r.remove();
      return;
    }
    const ring = (delay: number, peak: number) => {
      const r = mk("fx-ripple", 44);
      const an = r.animate([{ transform: "scale(.2)", opacity: peak }, { transform: "scale(1)", opacity: 0 }], { duration: 520, delay, easing: "cubic-bezier(.1,.7,.3,1)", fill: "backwards" });
      an.onfinish = () => r.remove();
    };
    ring(0, 0.9);
    ring(80, 0.55);
    const dot = mk("fx-ripple-dot", 10);
    dot.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 300, easing: "ease-out" }).onfinish = () => dot.remove();
  }

  /** Tag shows while moving and 1.2 s after the last action, then shrinks to a dot; after 2.5 s
   *  idle the whole cursor rests at 55 %. */
  private wake() {
    this.ox.classList.add("moving");
    this.rest(false);
    clearTimeout(this.tagTimer);
    clearTimeout(this.restTimer);
    this.tagTimer = window.setTimeout(() => this.ox.classList.remove("moving"), 1200 + Math.max(0, this.travelEnd - performance.now()));
    this.restTimer = window.setTimeout(() => this.rest(true), 2500);
  }

  private rest(on: boolean) { this.ox.classList.toggle("rest", on); }

  // ---- highlights -----------------------------------------------------------------------
  private placeHl(el: HTMLDivElement, r: Rect, dy = 0) {
    const pad = 4, s = this.s;
    el.style.left = `${r.x * s - pad}px`;
    el.style.top = `${(r.y + dy) * s - pad}px`;
    el.style.width = `${r.w * s + pad * 2}px`;
    el.style.height = `${r.h * s + pad * 2}px`;
    el.style.borderRadius = `${clamp(Math.min((r.h * s) / 2, 10), 4, 10)}px`;
    const chip = el.querySelector<HTMLElement>(".fx-hl-chip");
    if (chip) chip.classList.toggle("below", (r.y + dy) * s < 28);
  }

  private showHl(id: number, rect: Rect, kind: HlKind, base: Base | null) {
    const old = this.hls.get(id);
    if (old) { clearTimeout(old.timer); old.el.remove(); }
    const el = div("fx-hl " + kind, this.hlLayer);
    if (kind === "waiting") {
      const chip = div("fx-hl-chip", el);
      chip.textContent = "Needs your OK";
    }
    const h: Hl = { el, rect, kind, base, dy: 0 };
    if (base?.sy != null && this.page) h.dy = base.sy - this.page.sy;
    this.placeHl(el, rect, h.dy);
    el.animate([{ opacity: 0, transform: this.reduced ? "none" : "scale(1.03)" }, { opacity: 1, transform: "none" }], { duration: 160, easing: "cubic-bezier(0.2, 0.8, 0.2, 1)" });
    this.hls.set(id, h);
    this.syncHl(h);
  }

  private hideHl(id: number, delay: number) {
    const h = this.hls.get(id);
    if (!h) return;
    clearTimeout(h.timer);
    const go = () => {
      if (this.hls.get(id) !== h) return;
      this.hls.delete(id);
      const an = h.el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 220, easing: "cubic-bezier(0.2, 0.8, 0.2, 1)", fill: "forwards" });
      an.onfinish = () => h.el.remove();
    };
    if (delay <= 0) go(); else h.timer = window.setTimeout(go, delay);
  }

  private clearHls() { for (const [id, h] of this.hls) if (h.kind !== "waiting") this.hideHl(id, 0); }

  // ---- read sweep and scroll pill ---------------------------------------------------------
  private sweep(r?: Rect) {
    if (this.reduced) return; // the caption carries it
    const box = div("fx-sweep", this.fxLayer);
    const s = this.s;
    const x = r ? r.x * s : 0, y = r ? r.y * s : 0, w = r ? r.w * s : this.w, h = r ? r.h * s : this.h;
    Object.assign(box.style, { left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${h}px` });
    const band = div("fx-sweep-band", box);
    const bh = Math.max(24, h * 0.28);
    band.style.height = `${bh}px`;
    const an = band.animate([{ transform: `translateY(${-bh}px)` }, { transform: `translateY(${h}px)` }], { duration: 900, easing: "cubic-bezier(.4,0,.2,1)" });
    an.onfinish = () => box.remove();
  }

  private showScroll(dir: "up" | "down" | "left" | "right") {
    const p = this.scrollPill;
    p.dataset.dir = dir;
    p.classList.remove("on");
    void p.offsetWidth; // restart the chevron animation
    p.classList.add("on");
    clearTimeout(this.scrollTimer);
    this.scrollTimer = window.setTimeout(() => p.classList.remove("on"), 700);
  }
}

export function pointOf(a: BrowserAction): Pt | null {
  if (a.x != null && a.y != null) return { x: a.x, y: a.y };
  if (a.rect) return { x: a.rect.x + a.rect.w / 2, y: a.rect.y + a.rect.h / 2 };
  return null;
}

// The live page: frames painted into a canvas fitted into the view, NEO's overlay (cursor,
// highlights, ripples, sweep, caption), and — while the user drives — their pointer, wheel and
// keys forwarded to the page in page coordinates.
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import { ThinkingOrb } from "thinking-orbs";
import { getCurrentWindow } from "@tauri-apps/api/window";
import type { BrowserHub, BrowserSnap } from "./hub";
import type { Frame } from "./frames";
import type { Rect, Surface } from "./types";
import { Fx, pointOf, type FxMode } from "./fx";
import { fitRect } from "./sizes";
import { captionFor, displayHost, failCaption, type Caption } from "./text";
import type { OrbState } from "../ws";
import { ClickCounter, FrameCoalescer, buttonName, buttonsMask, isPrintable, modsOf, reservedShortcut, toPage, wheelDelta, type PageMap } from "./input";
import { I } from "./icons";
import { inTauri } from "../win";

export interface ViewProps {
  hub: BrowserHub;
  snap: BrowserSnap;
  surface: Exclude<Surface, "hidden" | "mini">;
  connected: boolean;
  reduced: boolean;
  onTakeControl: () => void;
  /** Bumped when the view should take keyboard focus (need_user). */
  focusSignal: number;
  /** A drawer covers the bottom of the view: keep the caption clear of it. */
  drawerOpen?: boolean;
  /** "Take control" is pending: NEO is finishing its current step. */
  taking?: boolean;
  /** NEO's job is over but the page is still its (the card is about to fold). */
  finished?: boolean;
}

const CAPTION_DWELL = 700;
const SOFT_HOLD = 1000;
const FAIL_HOLD = 1500;
const PILL_H = 32;
const PILL_GAP = 14;
const EASE_OUT = "cubic-bezier(0.2, 0.8, 0.2, 1)";

function useToast(hub: BrowserHub) {
  const [t, setT] = useState<{ text: string; id: number; ms: number } | null>(null);
  useEffect(() => hub.onToast((text, ms) => setT((p) => ({ text, id: (p?.id ?? 0) + 1, ms: ms ?? 2500 }))), [hub]);
  useEffect(() => {
    if (!t) return;
    const h = window.setTimeout(() => setT((cur) => (cur && cur.id === t.id ? null : cur)), t.ms);
    return () => clearTimeout(h);
  }, [t]);
  return t;
}

/** Captions dwell ≥ 700 ms; newer ones wait and only the latest shows, so they never strobe. A
 *  lead-in ("Scrolling to “Buy”") waits 1 s and is dropped if the action on its target follows. */
function useCaption(want: Caption | null) {
  const [shown, setShown] = useState<{ cap: Caption | null; key: number }>({ cap: want && !want.soft ? want : null, key: 0 });
  const at = useRef(performance.now());
  const timer = useRef<number | undefined>(undefined);
  const wantKey = want ? `${want.text}|${want.ctx ?? ""}|${want.orb}` : "";
  useEffect(() => {
    const curKey = shown.cap ? `${shown.cap.text}|${shown.cap.ctx ?? ""}|${shown.cap.orb}` : "";
    clearTimeout(timer.current);
    if (wantKey === curKey) return;
    const apply = () => { at.current = performance.now(); setShown((s) => ({ cap: want, key: s.key + 1 })); };
    let wait = CAPTION_DWELL - (performance.now() - at.current);
    if (want?.soft) wait = Math.max(wait, SOFT_HOLD);
    else if (!shown.cap || !want || want.now) wait = 0;
    if (wait <= 0) apply(); else timer.current = window.setTimeout(apply, wait);
    return () => clearTimeout(timer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wantKey]);
  return shown;
}

/** Screen readers hear captions that last ≥ 1.2 s, at most one every 2 s. */
function useAnnounce(text: string) {
  const [said, setSaid] = useState("");
  const last = useRef(0);
  useEffect(() => {
    if (!text) return;
    const t = window.setTimeout(() => {
      if (performance.now() - last.current < 2000) return;
      last.current = performance.now();
      setSaid(text);
    }, 1200);
    return () => clearTimeout(t);
  }, [text]);
  return said;
}

/** One line of narration: what NEO is doing, or a short notice (toast). One pill stays mounted
 *  while there is something to say; its width glides between lines and only the words crossfade,
 *  so it never blinks. It sits bottom-centre, or top-centre when that would cover NEO's target. */
type Narr = { key: string; text: string; ctx?: string; orb?: OrbState };

function NarrationPill({ item, y, reduced }: { item: Narr | null; y: number; reduced: boolean }) {
  const [cur, setCur] = useState<Narr | null>(item);
  const [prev, setPrev] = useState<Narr | null>(null);
  const [leaving, setLeaving] = useState(false);
  const curRef = useRef<Narr | null>(item);
  const ref = useRef<HTMLDivElement>(null);
  const lastW = useRef(0);
  const widthAnim = useRef<Animation | null>(null);
  const itemKey = item?.key ?? "";

  useEffect(() => {
    if (item) {
      setLeaving(false);
      const c = curRef.current;
      if (!c || c.key !== item.key) {
        setPrev(c && !reduced ? c : null);
        curRef.current = item;
        setCur(item);
      }
      return;
    }
    // Nothing to say: linger 300 ms (the next line often follows), then leave.
    const t = window.setTimeout(() => setLeaving(true), 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [itemKey]);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) { lastW.current = 0; return; }
    const running = widthAnim.current && widthAnim.current.playState === "running";
    const from = running ? el.getBoundingClientRect().width : lastW.current;
    widthAnim.current?.cancel();
    const to = el.getBoundingClientRect().width;
    lastW.current = to;
    delete el.dataset.growing;
    if (from > 0 && Math.abs(from - to) > 1 && !reduced) {
      const an = el.animate([{ width: `${from}px` }, { width: `${to}px` }], { duration: 200, easing: EASE_OUT });
      widthAnim.current = an;
      // While the pill widens to fit a longer line, the words fade at the edge instead of
      // showing "…" for a few frames.
      if (to > from) {
        el.dataset.growing = "";
        an.finished.then(() => { if (widthAnim.current === an) delete el.dataset.growing; }, () => {});
      }
    }
  }, [cur?.key, reduced]);

  if (!cur) return null;
  const body = (n: Narr) => <>{n.text}{n.ctx && <span className="br-caption-ctx"> {n.ctx}</span>}</>;
  return (
    <div
      ref={ref}
      className={"br-caption" + (cur.orb ? "" : " no-orb") + (leaving ? " leaving" : "")}
      style={{ transform: `translate(-50%, ${Math.round(y)}px)` }}
      onAnimationEnd={(e) => {
        if (e.target !== e.currentTarget || !e.animationName.startsWith("br-cap-leave")) return;
        curRef.current = null; setCur(null); setPrev(null);
      }}
      aria-hidden
    >
      {cur.orb && <span className="br-caption-orb"><ThinkingOrb state={cur.orb} size={20} theme="auto" /></span>}
      <span className="br-caption-slot">
        {prev && <span key={"o" + prev.key} className="br-caption-text out" dir="auto" onAnimationEnd={() => setPrev(null)}>{body(prev)}</span>}
        <span key={cur.key} className="br-caption-text in" dir="auto">{body(cur)}</span>
      </span>
    </div>
  );
}

export function BrowserView({ hub, snap, surface, connected, reduced, onTakeControl, focusSignal, drawerOpen, taking = false, finished = false }: ViewProps) {
  const st = snap.live!;
  const viewRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bgRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const fxRef = useRef<HTMLDivElement>(null);
  const sinkRef = useRef<HTMLTextAreaElement>(null);
  const fx = useRef<Fx | null>(null);
  const viewKey = useId();
  const geo = useRef({ W: 0, H: 0, pw: st.viewport.w, ph: st.viewport.h, r: fitRect(1, 1, 1, 1), bgAt: 0 });
  const resizeTimer = useRef<number | undefined>(undefined);
  const [geoTick, setGeoTick] = useState(0);
  const [hasFrame, setHasFrame] = useState(() => !!hub.frames.latest);
  const [stalled, setStalled] = useState(false);
  const [composing, setComposing] = useState<string | null>(null);
  const [hint, setHint] = useState<{ x: number; y: number; pulse: number } | null>(null);
  const [keepClear, setKeepClear] = useState<Rect | null>(null);
  const [failCap, setFailCap] = useState<Caption | null>(null);
  const [slowLoad, setSlowLoad] = useState(false);
  const [keysChip, setKeysChip] = useState(0);
  const toast = useToast(hub);

  const userDriving = connected && st.controller === "user" && (st.status === "user" || st.status === "need_user");
  const neoDriving = !finished && connected && st.controller === "neo" && (st.status === "working" || st.status === "thinking" || st.status === "waiting");
  const fxMode: FxMode = !connected || !hasFrame || finished ? "off"
    : st.status === "waiting" ? "waiting"
    : st.status === "thinking" ? "thinking"
    : st.status === "working" ? "acting" : "off";
  const host = displayHost(st.url);

  // ---- geometry: frame rect, canvas box, overlay scale ----------------------------------
  // The blurred page behind a letterbox: bars read as the page's own light, not a broken view.
  const drawBackdrop = useCallback((force = false) => {
    const c = bgRef.current, f = hub.frames.latest, g = geo.current;
    if (!c) return;
    const boxed = g.r.x > 1 || g.r.y > 1;
    c.classList.toggle("on", boxed && !!f);
    const v = viewRef.current;
    if (v) { if (boxed && f) v.dataset.boxed = ""; else delete v.dataset.boxed; }
    if (!boxed || !f) return;
    const now = performance.now();
    if (!force && now - g.bgAt < 250) return;
    g.bgAt = now;
    const w = 64, h = Math.max(8, Math.round((64 * g.H) / Math.max(1, g.W)));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const k = Math.max(w / f.bmp.width, h / f.bmp.height);
    const dw = f.bmp.width * k, dh = f.bmp.height * k;
    ctx.imageSmoothingQuality = "low";
    ctx.drawImage(f.bmp, (w - dw) / 2, (h - dh) / 2, dw, dh);
  }, [hub]);

  const drawLatest = useCallback(() => {
    const c = canvasRef.current, f = hub.frames.latest;
    if (!c || !f) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(f.bmp, 0, 0, c.width, c.height);
  }, [hub]);

  const resizeBacking = useCallback(() => {
    const c = canvasRef.current;
    if (!c) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const { r } = geo.current;
    const bw = Math.max(1, Math.round(r.w * dpr)), bh = Math.max(1, Math.round(r.h * dpr));
    if (c.width !== bw || c.height !== bh) { c.width = bw; c.height = bh; drawLatest(); }
  }, [drawLatest]);

  const layout = useCallback((immediate = false) => {
    const v = viewRef.current, c = canvasRef.current, stg = stageRef.current;
    if (!v || !c || !stg) return;
    const W = v.clientWidth, H = v.clientHeight;
    const g = geo.current;
    // Contain-fit, then snapped to device pixels (in page coordinates, so a view that sits on a
    // half pixel still lands the frame on whole ones): the page is drawn 1:1, never smeared.
    const f = fitRect(W, H, g.pw, g.ph);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const vb = v.getBoundingClientRect();
    const snapX = (x: number) => Math.round((vb.left + x) * dpr) / dpr - vb.left;
    const snapY = (y: number) => Math.round((vb.top + y) * dpr) / dpr - vb.top;
    const x = snapX(f.x), y = snapY(f.y);
    const w = Math.round(f.w * dpr) / dpr, h = Math.round(f.h * dpr) / dpr;
    const r = { x, y, w, h, s: g.pw > 0 ? w / g.pw : 1 };
    const changed = Math.abs(r.x - g.r.x) + Math.abs(r.y - g.r.y) + Math.abs(r.w - g.r.w) + Math.abs(r.h - g.r.h) + Math.abs(H - g.H) > 0.5;
    g.W = W; g.H = H; g.r = r;
    for (const el of [c, stg]) {
      el.style.left = `${r.x}px`; el.style.top = `${r.y}px`;
      el.style.width = `${r.w}px`; el.style.height = `${r.h}px`;
    }
    fx.current?.setGeometry(r.s, r.w, r.h);
    // The backing store follows once the size settles (a CSS-scaled canvas keeps its aspect).
    clearTimeout(resizeTimer.current);
    if (immediate || !c.width || c.width <= 2) resizeBacking();
    else resizeTimer.current = window.setTimeout(resizeBacking, 120);
    drawBackdrop(true);
    if (changed) setGeoTick((n) => n + 1);
    hub.setView(viewKey, { surface, w: W, h: H, visible: true });
  }, [hub, surface, viewKey, resizeBacking, drawBackdrop]);

  useLayoutEffect(() => {
    const v = viewRef.current;
    if (!v) return;
    layout(true);
    const ro = new ResizeObserver(() => layout());
    ro.observe(v);
    return () => { ro.disconnect(); clearTimeout(resizeTimer.current); hub.setView(viewKey, null); };
  }, [layout, hub, viewKey]);

  // Page viewport known before any frame arrives (starting): size the rect to it.
  useEffect(() => {
    const g = geo.current;
    if (hub.frames.latest) return;
    if (g.pw !== st.viewport.w || g.ph !== st.viewport.h) { g.pw = st.viewport.w; g.ph = st.viewport.h; layout(true); }
  }, [st.viewport.w, st.viewport.h, hub, layout]);

  // ---- frames -----------------------------------------------------------------------------
  useEffect(() => {
    const paint = (f: Frame) => {
      const g = geo.current;
      if (f.meta.vw !== g.pw || f.meta.vh !== g.ph) { g.pw = f.meta.vw; g.ph = f.meta.vh; layout(true); }
      fx.current?.frame({ vw: f.meta.vw, vh: f.meta.vh, sy: f.meta.sy });
      const c = canvasRef.current;
      if (!c) return;
      if (!c.width || c.width <= 2) resizeBacking();
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(f.bmp, 0, 0, c.width, c.height);
      drawBackdrop();
      setHasFrame(true);
    };
    const f = hub.frames.latest;
    if (f) paint(f);
    return hub.frames.onPaint(paint);
  }, [hub, layout, resizeBacking, drawBackdrop]);

  useEffect(() => { if (!hub.frames.latest) setHasFrame(false); }, [st.sid, hub]);

  // "Live view is catching up…" — loading, and nothing painted for 5 s.
  useEffect(() => {
    if (!st.loading) { setStalled(false); return; }
    const t = window.setInterval(() => setStalled(hub.lastFrameAt > 0 && performance.now() - hub.lastFrameAt > 5000), 1000);
    return () => clearInterval(t);
  }, [st.loading, hub]);

  useEffect(() => {
    setSlowLoad(false);
    if (!st.loading) return;
    const t = window.setTimeout(() => setSlowLoad(true), 8000);
    return () => clearTimeout(t);
  }, [st.loading, st.url]);

  // ---- NEO's overlay ------------------------------------------------------------------------
  useLayoutEffect(() => {
    const el = fxRef.current;
    if (!el) return;
    const f = new Fx(el, reduced);
    fx.current = f;
    const g = geo.current;
    f.setGeometry(g.r.s, g.r.w, g.r.h);
    const latest = hub.frames.latest;
    if (latest) f.frame({ vw: latest.meta.vw, vh: latest.meta.vh, sy: latest.meta.sy });
    return () => { f.destroy(); fx.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => { fx.current?.setReduced(reduced); }, [reduced]);
  useEffect(() => { fx.current?.setMode(fxMode); }, [fxMode]);
  useEffect(() => { fx.current?.setTaking(taking && neoDriving); }, [taking, neoDriving]);
  const failTimer = useRef<number | undefined>(undefined);
  useEffect(() => {
    // A late joiner (reopened card, reconnect): put the cursor where NEO last acted.
    if (snap.action) fx.current?.place(snap.action);
    const unsub = hub.onAction((a) => {
      fx.current?.action(a);
      if (a.phase === "intent" && a.kind !== "scroll") {
        const pt = pointOf(a);
        // What the narration must not cover: the target, and the cursor with its tag.
        setKeepClear(a.rect ?? (pt ? { x: pt.x - 6, y: pt.y - 6, w: 64, h: 48 } : null));
      }
      if (a.phase === "fail" && a.reason !== "stopped" && a.reason !== "cancelled") {
        setFailCap(failCaption(a));
        clearTimeout(failTimer.current);
        failTimer.current = window.setTimeout(() => setFailCap(null), FAIL_HOLD);
      } else if (a.phase === "intent") setFailCap(null);
    });
    return () => { unsub(); clearTimeout(failTimer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hub]);

  // ---- narration: caption (NEO's action) or a notice ------------------------------------------
  const want = useMemo<Caption | null>(() => {
    if (!connected || !hasFrame || finished) return null;
    if (taking && neoDriving) return { text: "Finishing this step, then it's yours", orb: "shaping", now: true };
    if (failCap && neoDriving) return failCap;
    const a = snap.action;
    if (st.status === "waiting") return a && a.gated ? captionFor(a) : { text: "Waiting for your OK", orb: "shaping" };
    if (st.status === "thinking") return { text: "Thinking…", orb: "solving" };
    if (st.status !== "working") return null;
    if (st.loading && (!a || a.kind === "navigate")) return { text: slowLoad ? `Still loading ${host}…` : `Opening ${host}`, orb: "connecting" };
    if (a) return captionFor(a);
    if (st.caption) return { text: st.caption, orb: "working" };
    return null;
  }, [connected, hasFrame, finished, taking, neoDriving, failCap, snap.action, st.status, st.loading, st.caption, slowLoad, host]);
  const caption = useCaption(want);
  const said = useAnnounce(caption.cap ? caption.cap.text + (caption.cap.ctx ? " " + caption.cap.ctx : "") : "");
  const narr: Narr | null = toast
    ? { key: `t${toast.id}`, text: toast.text }
    : caption.cap ? { key: `c${caption.key}`, text: caption.cap.text, ctx: caption.cap.ctx, orb: caption.cap.orb } : null;

  // Where the pill sits (stage px): bottom-centre, above the steps drawer when it is open, or
  // top-centre when the bottom would cover what NEO is working on (or the field the user is in).
  const clearRect = userDriving ? (snap.hint?.focus?.rect ?? st.need_user?.rect ?? null) : keepClear;
  const pillY = useMemo(() => {
    const g = geo.current, r = g.r;
    const drawerH = drawerOpen ? Math.min(0.56 * g.H, 320) : 0;
    const bottomEdge = drawerOpen ? g.H - drawerH - 12 : r.y + r.h - PILL_GAP; // view px
    const bottom = Math.max(PILL_GAP, bottomEdge - PILL_H - r.y);
    const top = PILL_GAP;
    if (!clearRect) return bottom;
    const k0 = clearRect.y * r.s, k1 = (clearRect.y + clearRect.h) * r.s;
    const hits = (y: number) => k1 > y - 12 && k0 < y + PILL_H + 12;
    if (!hits(bottom)) return bottom;
    if (!hits(top)) return top;
    const mid = (k0 + k1) / 2;
    return Math.abs(mid - (bottom + PILL_H / 2)) >= Math.abs(mid - (top + PILL_H / 2)) ? bottom : top;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clearRect, drawerOpen, geoTick]);

  // ---- input (user in control) -------------------------------------------------------------------
  const clicks = useRef(new ClickCounter());
  const pressed = useRef(new Set<string>());
  const mapNow = (): PageMap | null => {
    const c = canvasRef.current;
    if (!c) return null;
    const b = c.getBoundingClientRect();
    const g = geo.current;
    if (!b.width) return null;
    return { left: b.left, top: b.top, s: b.width / g.pw, vw: g.pw, vh: g.ph };
  };
  const moves = useMemo(() => new FrameCoalescer<{ x: number; y: number; buttons: number; mods: number }>((v) => hub.input("mousemove", v)), [hub]);
  const wheels = useMemo(() => new FrameCoalescer<{ x: number; y: number; dx: number; dy: number; mods: number }>(
    (v) => hub.input("wheel", { ...v, dx: Math.round(v.dx * 10) / 10, dy: Math.round(v.dy * 10) / 10 }),
    (a, b) => ({ ...b, dx: a.dx + b.dx, dy: a.dy + b.dy }),
  ), [hub]);
  useEffect(() => () => { moves.cancel(); wheels.cancel(); }, [moves, wheels]);

  const drivingRef = useRef(userDriving);
  drivingRef.current = userDriving;
  const hintTimer = useRef<number | undefined>(undefined);
  const hintLeave = useRef<number | undefined>(undefined);
  const lastPtr = useRef({ x: 0, y: 0 });
  /** The sink is being focused by a click on the page (not by Tab or a prompt). */
  const ptrFocus = useRef(false);
  /** "Click a field on the page first" has been said this turn. */
  const nudged = useRef(false);
  /** The sink lost focus because the whole window did. */
  const winBlur = useRef(false);
  const hintable = neoDriving && !taking;

  const isChrome = (t: EventTarget | null) => !!(t as HTMLElement | null)?.closest?.("[data-noinput]");
  const localXY = (e: { clientX: number; clientY: number }) => {
    const b = viewRef.current!.getBoundingClientRect();
    return { x: e.clientX - b.left, y: e.clientY - b.top };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (isChrome(e.target)) return;
    if (!drivingRef.current) {
      // NEO drives: presses never reach the page. The hint says why, and offers the handover.
      if (neoDriving) {
        e.preventDefault();
        if (!hintable) return;
        const p = localXY(e);
        clearTimeout(hintTimer.current);
        setHint((h) => ({ x: h?.x ?? p.x, y: h?.y ?? p.y, pulse: (h?.pulse ?? 0) + 1 }));
      }
      return;
    }
    if (e.button > 2) return;
    e.preventDefault();
    const m = mapNow();
    if (!m) return;
    viewRef.current?.setPointerCapture(e.pointerId);
    ptrFocus.current = true;
    sinkRef.current?.focus({ preventScroll: true });
    if (inTauri && !document.hasFocus()) getCurrentWindow().setFocus().catch(() => {});
    moves.flush();
    const p = toPage(e.clientX, e.clientY, m);
    hub.input("mousedown", { ...p, button: buttonName(e.button), buttons: buttonsMask(e.buttons), clicks: clicks.current.next(e.nativeEvent), mods: modsOf(e) });
  };

  const onPointerMove = (e: React.PointerEvent) => {
    lastPtr.current = { x: e.clientX, y: e.clientY };
    if (!drivingRef.current) {
      if (!hintable || isChrome(e.target)) return;
      clearTimeout(hintLeave.current);
      if (!hint) {
        clearTimeout(hintTimer.current);
        const p = localXY(e);
        hintTimer.current = window.setTimeout(() => setHint({ x: p.x, y: p.y, pulse: 0 }), 500);
      }
      return;
    }
    const m = mapNow();
    if (!m) return;
    moves.push({ ...toPage(e.clientX, e.clientY, m), buttons: buttonsMask(e.buttons), mods: modsOf(e) });
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (!drivingRef.current || isChrome(e.target) && !viewRef.current?.hasPointerCapture(e.pointerId)) return;
    const m = mapNow();
    if (!m) return;
    moves.flush();
    hub.input("mouseup", { ...toPage(e.clientX, e.clientY, m), button: buttonName(e.button), buttons: buttonsMask(e.buttons), clicks: 1, mods: modsOf(e) });
    if (viewRef.current?.hasPointerCapture(e.pointerId)) viewRef.current.releasePointerCapture(e.pointerId);
  };

  const onPointerLeave = () => {
    clearTimeout(hintTimer.current);
    hintLeave.current = window.setTimeout(() => setHint(null), 120);
  };

  useEffect(() => { if (!hintable) { clearTimeout(hintTimer.current); setHint(null); } }, [hintable]);

  // Wheel: native, non-passive, so it can be kept from the page when needed. Pinch never passes.
  useEffect(() => {
    const v = viewRef.current;
    if (!v) return;
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey) { e.preventDefault(); return; }
      if (!drivingRef.current || isChrome(e.target)) return;
      e.preventDefault();
      const m = mapNow();
      if (!m) return;
      const d = wheelDelta(e, geo.current.H, m.s);
      wheels.push({ ...toPage(e.clientX, e.clientY, m), dx: d.dx, dy: d.dy, mods: modsOf(e) });
    };
    v.addEventListener("wheel", onWheel, { passive: false });
    return () => v.removeEventListener("wheel", onWheel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wheels]);

  // Keys: a hidden textarea holds focus so the IME, dead keys, emoji and dictation all work.
  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const ne = e.nativeEvent;
    if (reservedShortcut(ne)) return; // the card handles these in the capture phase
    if (!drivingRef.current) return;
    if (ne.isComposing || e.key === "Process" || e.key === "Dead" || e.keyCode === 229) return;
    const k = e.key.toLowerCase();
    if (e.metaKey && !e.ctrlKey && !e.altKey && k === "v") return; // native paste on the sink
    if (e.metaKey && !e.ctrlKey && !e.altKey && (k === "c" || k === "x")) {
      e.preventDefault();
      void hub.control(k === "c" ? "copy" : "cut");
      return;
    }
    e.preventDefault();
    const printable = isPrintable(ne);
    hub.input("keydown", { key: e.key, code: e.code, mods: modsOf(e), repeat: e.repeat, location: ne.location, ...(printable ? { text: e.key } : {}) });
    pressed.current.add(e.code);
    // Letters with no field focused on the page usually vanish (pages may still use them as
    // shortcuts, so they are sent anyway). Say so once per turn at the wheel — only when the core
    // has said nothing is focused, and never for Space (it scrolls).
    if (printable && e.key.trim() && !e.repeat && snap.hint?.focus === null && !nudged.current) {
      nudged.current = true;
      hub.toast("Click a field on the page first", 2200);
    }
  };
  const onKeyUp = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (!pressed.current.has(e.code)) return;
    pressed.current.delete(e.code);
    e.preventDefault();
    hub.input("keyup", { key: e.key, code: e.code, mods: modsOf(e), repeat: false, location: e.nativeEvent.location });
  };
  const releaseKeys = () => {
    for (const code of pressed.current) hub.input("keyup", { key: "", code, mods: 0, repeat: false, location: 0 });
    pressed.current.clear();
  };
  useEffect(() => {
    if (!userDriving) { releaseKeys(); setComposing(null); composingRef.current = false; }
    nudged.current = false;
  }, [userDriving]); // eslint-disable-line react-hooks/exhaustive-deps

  // IME, dead keys, emoji, dictation. Engines disagree on order: Chromium commits in
  // compositionend after its last input event; WebKit fires the committing input event (with
  // isComposing false) before compositionend. Either way the text is sent exactly once — at
  // compositionend — and the sink is emptied after the engine has finished with it.
  const composingRef = useRef(false);
  const committed = useRef<{ text: string; at: number } | null>(null);
  const sendText = (text: string) => { if (text && drivingRef.current) hub.input("text", { text: text.slice(0, 2000) }); };
  const clearSink = (sink: HTMLTextAreaElement) => { window.setTimeout(() => { if (!composingRef.current) sink.value = ""; }, 0); };
  const onInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
    const ne = e.nativeEvent as InputEvent;
    const sink = e.currentTarget;
    if (ne.isComposing || composingRef.current || ne.inputType === "insertCompositionText" || ne.inputType === "insertFromComposition") return;
    if (ne.inputType === "insertText" || ne.inputType === "insertReplacementText" || ne.inputType === "insertFromDictation") {
      // The same text a composition just committed (an engine that echoes it after
      // compositionend): already sent.
      const c = committed.current;
      if (!(c && c.text === sink.value && performance.now() - c.at < 80)) sendText(sink.value);
    }
    sink.value = "";
  };
  const onCompositionStart = () => { composingRef.current = true; setComposing(""); };
  const onCompositionEnd = (e: React.CompositionEvent<HTMLTextAreaElement>) => {
    composingRef.current = false;
    setComposing(null);
    committed.current = { text: e.data, at: performance.now() };
    sendText(e.data);
    clearSink(e.currentTarget);
  };
  const onPaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    e.preventDefault();
    if (!drivingRef.current) return;
    let text = e.clipboardData.getData("text/plain");
    if (!text) return;
    if (text.length > 20_000) { text = text.slice(0, 20_000); hub.toast("Pasted the first 20,000 characters", 2400); }
    hub.input("paste", { text });
  };

  // The sink sits at the page's caret so the IME candidate window opens next to it.
  const caret = useMemo(() => {
    const f = snap.hint?.focus;
    const g = geo.current;
    if (!f || !userDriving) return null;
    return { x: g.r.x + f.rect.x * g.r.s, y: g.r.y + (f.rect.y + f.rect.h) * g.r.s, h: f.rect.h * g.r.s };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snap.hint, userDriving, geoTick]);

  useEffect(() => {
    if (!focusSignal) return;
    const t = window.setTimeout(() => (userDriving ? sinkRef.current : viewRef.current)?.focus({ preventScroll: true }), 60);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusSignal]);

  // Keys typed while the page has the keyboard go to the page, Tab included. Whenever the page
  // takes the keyboard other than by a click (Tab into it, a login prompt, ⌘⇧U), say so and how
  // to get out, so keyboard users never feel stuck.
  const onSinkFocus = () => {
    const byPointer = ptrFocus.current, returning = winBlur.current;
    ptrFocus.current = false; winBlur.current = false;
    // Not when focus merely comes back with the window (the user switched apps and back).
    if (!byPointer && !returning && drivingRef.current) setKeysChip((n) => n + 1);
  };
  const onSinkBlur = () => { winBlur.current = !document.hasFocus(); releaseKeys(); };
  useEffect(() => {
    if (!keysChip) return;
    const t = window.setTimeout(() => setKeysChip(0), 2600);
    return () => clearTimeout(t);
  }, [keysChip]);
  useEffect(() => { if (!userDriving) setKeysChip(0); }, [userDriving]);

  // Keep keys with the page when the user takes control from the keyboard.
  useEffect(() => {
    if (userDriving && document.activeElement === viewRef.current) sinkRef.current?.focus({ preventScroll: true });
  }, [userDriving]);

  const pointerCursor = userDriving ? cssCursor(snap.hint?.cursor) : "default";
  const errored = st.status === "error";
  const crashed = errored && st.error && (st.error.code === "crashed" || st.error.code === "browser_closed");
  const starting = connected && !errored && (st.status === "starting" || !hasFrame);

  const label = userDriving
    ? `${st.title || host}, ${host}. You're in control; keys go to the page. Control-Tab to leave.`
    : `${st.title || host}, ${host}. NEO is in control. Press Command-Shift-U to take control.`;

  return (
    <div
      ref={viewRef}
      className={"br-view" + (userDriving ? " user" : "") + (neoDriving ? " neo" : "") + (!connected ? " offline" : "") + (errored ? " errored" : "")}
      tabIndex={userDriving ? -1 : 0}
      role="application"
      aria-roledescription="live web page"
      aria-label={label}
      style={{ cursor: pointerCursor }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onPointerLeave={onPointerLeave}
      onContextMenu={(e) => e.preventDefault()}
      onFocus={(e) => { if (e.target === viewRef.current && userDriving) sinkRef.current?.focus({ preventScroll: true }); }}
    >
      <canvas ref={bgRef} className="br-canvas-bg" aria-hidden />
      <canvas ref={canvasRef} className="br-canvas" aria-hidden />
      <div ref={stageRef} className="br-stage">
        <div className="br-vignette" aria-hidden />
        <div ref={fxRef} className="br-fx" aria-hidden />
        <NarrationPill item={narr} y={pillY} reduced={reduced} />
      </div>
      <div className="sr-only" role="status" aria-live="polite">{said}</div>
      <div className="sr-only" role="status" aria-live="polite">{toast?.text ?? ""}</div>

      {stalled && hasFrame && <div className="br-chip-tr" role="status"><span className="br-spin sm" />Live view is catching up…</div>}
      {keysChip > 0 && userDriving && (
        <div key={keysChip} className="br-chip-top" aria-hidden>
          {I.keyboard(13)}<span>Keys go to the page</span><span className="br-chip-sep">·</span><kbd>⌃⇥</kbd><span className="br-chip-muted">to leave</span>
        </div>
      )}

      {starting && (
        <div className="br-panel starting" data-noinput>
          <ThinkingOrb state="connecting" size={64} theme="auto" />
          <div className="br-panel-title">Starting NEO's browser…</div>
          <div className="br-panel-sub delayed">The first launch takes a few seconds.</div>
        </div>
      )}
      {crashed && (
        <div className="br-panel error" data-noinput role="alert">
          <div className="br-panel-card">
            <div className="br-panel-badge">{I.warn(18)}</div>
            <div className="br-panel-title">{st.error!.code === "crashed" ? "This page crashed" : "NEO's browser closed unexpectedly"}</div>
            <div className="br-panel-sub">{st.error!.code === "crashed" ? "NEO can load it again." : "Your logins are safe."}</div>
            <button className="br-pill primary" onClick={() => void hub.control(st.error!.code === "crashed" ? "reload" : "restart")}>
              {st.error!.code === "crashed" ? "Reload page" : "Restart browser"}
            </button>
          </div>
        </div>
      )}
      {!connected && hasFrame && (
        <div className="br-chip-center" role="status" data-noinput><span className="br-spin sm" />Reconnecting to NEO…</div>
      )}

      {hint && hintable && (
        <div key={hint.pulse} className={"br-hint" + (hint.pulse ? " pulse" : "")} style={{ left: Math.min(hint.x + 12, (geo.current.W || 400) - 260), top: Math.min(hint.y + 16, (geo.current.H || 300) - 44) }}
          data-noinput onPointerDown={(e) => e.stopPropagation()}>
          <span>NEO is using this page</span>
          <button className="br-hint-btn" onClick={() => { setHint(null); onTakeControl(); }}>Take control</button>
        </div>
      )}

      {composing !== null && caret && (
        <div className="br-compose" style={{ left: caret.x, top: caret.y + 4 }} aria-hidden>{composing}</div>
      )}

      <textarea
        ref={sinkRef}
        className="br-sink"
        tabIndex={userDriving ? 0 : -1}
        aria-label={userDriving ? `Page keyboard: keys go to ${host}. Control-Tab to leave.` : "Page keyboard (NEO is in control)"}
        autoCapitalize="off" autoComplete="off" autoCorrect="off" spellCheck={false}
        style={caret ? { left: caret.x, top: caret.y - Math.min(caret.h, 20) } : undefined}
        onKeyDown={onKeyDown}
        onKeyUp={onKeyUp}
        onFocus={onSinkFocus}
        onBlur={onSinkBlur}
        onInput={onInput}
        onPaste={onPaste}
        onCompositionStart={onCompositionStart}
        onCompositionUpdate={(e) => setComposing(e.data)}
        onCompositionEnd={onCompositionEnd}
      />
    </div>
  );
}

const CURSORS = new Set(["default", "pointer", "text", "vertical-text", "crosshair", "move", "grab", "grabbing", "not-allowed", "wait", "progress", "help", "col-resize", "row-resize", "ew-resize", "ns-resize", "nesw-resize", "nwse-resize", "zoom-in", "zoom-out", "context-menu", "cell", "copy", "alias", "none"]);
function cssCursor(c?: string): string {
  return c && CURSORS.has(c) ? c : "default";
}

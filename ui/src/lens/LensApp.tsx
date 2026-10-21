// index.html#/lens — Search Screen's overlay. NEO.app's shell alone opens this window, full-screen
// on the display under the pointer and above everything (ui/src-tauri/src/lens.rs); it talks to
// the core as its one `lens` client (./socket.ts) with the ticket the shell hands it. Everything it
// decides is ./state.ts's reducer; this file wires the reducer to the shell, the socket, the
// pointer and the keys, and draws: the still, the glow, the trail, the brackets, the sheet.
//
// The pointer never touches React state: a move only updates refs, and its work — the cursor, the
// trail, a corner being dragged, smart pick's hover outline (./hover.ts) — runs once per animation
// frame and writes the few elements it moves directly. Nothing display-sized repaints while the
// mouse moves: the still sits on its own layer and is never rendered again once it is in.
//
// Open, the window covers the display and takes every click, so a page that broke must never stay
// up: a render error (the boundary) or an uncaught one closes it, and the page tells the shell it is
// alive every second (`painted`) — the shell closes one that falls silent, and the panel's own Esc
// works with no page at all (lens.rs).

import { Component, memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { EdgeGlow } from "./EdgeGlow";
import { Trail, type TrailHandle } from "./Trail";
import { SelectionView, type SelectionHandle } from "./Selection";
import { Sheet } from "./Sheet";
import { makeHover, type Hover } from "./hover";
import { flyFrames, type FlyPlan } from "./fly";
import { Card } from "./Cards";
import { tauriDoor, type Door } from "./door";
import { LensSocket, type SocketLike } from "./socket";
import { CURSOR, classify, dragBox, hitTest, type Box, type Hit, type Pt } from "./geometry";
import { isDouble, regionAt, tabStops } from "./pick";
import type { PhaseEvent } from "./door";
import { handingOff, hint, initial, reduce, showsWebGlow, type Action, type Effect, type LensState } from "./state";
import type { Still } from "./wire";
import { wsToken } from "../shell";
import "./lens.css";

const IDLE_MS = 3 * 60_000;
const ALIVE_MS = 1000; // open: "still here" to the shell this often (it closes a page silent for 5 s)
const LIVE_SETTLE_MS = 70;
const LIVE_GIVE_BACK_MS = 2500;
const TRANSLATE_KEY = "neo.lens.translateTo";

function readTranslateTo(): string | null {
  try { const v = localStorage.getItem(TRANSLATE_KEY); return v && /^[A-Za-z]{2,3}([-_][A-Za-z0-9]{2,8})*$/.test(v) ? v : null; } catch { return null; }
}
function saveTranslateTo(v: string) {
  try { localStorage.setItem(TRANSLATE_KEY, v); } catch { /* a private window: just not remembered */ }
}

type Drag =
  | { mode: "draw"; pts: Pt[]; last: Pt }
  | { mode: "adjust"; hit: Hit; start: Box; p0: Pt; box: Box };

function useMedia(q: string): boolean {
  const [on, setOn] = useState(() => (typeof matchMedia === "function" ? matchMedia(q).matches : false));
  useEffect(() => {
    if (typeof matchMedia !== "function") return;
    const m = matchMedia(q);
    const f = () => setOn(m.matches);
    m.addEventListener?.("change", f);
    return () => m.removeEventListener?.("change", f);
  }, [q]);
  return on;
}

/** A render error unmounts the overlay's page, and with it every way to close it from inside: the
 *  overlay closes instead (the shell's own door, asked afresh). */
class LensBoundary extends Component<{ children: ReactNode }, { broken: boolean }> {
  state = { broken: false };
  static getDerivedStateFromError() {
    return { broken: true };
  }
  componentDidCatch() {
    void tauriDoor().then((d) => d?.close("cancel")).catch(() => { /* no shell: nothing to close */ });
  }
  render() {
    return this.state.broken ? null : this.props.children;
  }
}

export function LensApp() {
  return <LensBoundary><Overlay /></LensBoundary>;
}

/** The screen as it was: on its own composited layer, and never rendered again for the same props
 *  (whatever else the overlay renders). Mounted, at opacity 0, as soon as its URL is known, so it is
 *  decoded and uploaded before it shows. */
const Still = memo(function Still({ src, shown, leaving }: { src: string; shown: boolean; leaving: boolean }) {
  return <img className={"lx-still" + (shown ? " on" : "") + (leaving ? " leaving" : "")} src={src} alt="" draggable={false} decoding="sync" />;
});

/** Room round the crop for its shadow (points): the shadow is drawn into the crop's canvas, once, as
 *  it is cut — no CSS shadow on the flying layer (a blurred box-shadow there makes WebKit paint a
 *  layer the area's size, the whole display for ⌘A, as the flight starts). */
const CROP_SHADOW_PT = 48;

/** The area as a small canvas cut from the still (at most 960 px on its long side), with the hole's
 *  rounded corners, a hairline and its shadow: what flies into the notch. Cut when the hand-off is
 *  asked for. */
function cropOf(img: HTMLImageElement | null, b: Box, W: number, H: number): HTMLCanvasElement | null {
  if (!img || !img.naturalWidth || !img.naturalHeight || !(W > 0) || !(H > 0) || !(b.w > 0) || !(b.h > 0)) return null;
  const kx = img.naturalWidth / W, ky = img.naturalHeight / H;
  const f = Math.min(1, 960 / Math.max(b.w * kx, b.h * ky));
  const iw = Math.max(1, Math.round(b.w * kx * f));
  const ih = Math.max(1, Math.round(b.h * ky * f));
  const k = iw / b.w, ky2 = ih / b.h; // canvas pixels per point, across and down
  const m = Math.ceil(CROP_SHADOW_PT * k);
  const c = document.createElement("canvas");
  c.width = iw + 2 * m;
  c.height = ih + 2 * m;
  // Placed so the area itself lies exactly over the flying layer; the shadow's room hangs outside it.
  const mx = m / k, my = m / ky2;
  Object.assign(c.style, { left: `${-mx}px`, top: `${-my}px`, width: `${b.w + 2 * mx}px`, height: `${b.h + 2 * my}px` });
  const ctx = c.getContext("2d");
  if (!ctx) return null;
  const round = (inset: number) => {
    ctx.beginPath();
    const r = Math.max(0, Math.min(14 * k, iw / 2, ih / 2) - inset);
    if (typeof ctx.roundRect === "function") ctx.roundRect(m + inset, m + inset, iw - 2 * inset, ih - 2 * inset, r);
    else ctx.rect(m + inset, m + inset, iw - 2 * inset, ih - 2 * inset);
  };
  ctx.save();
  ctx.shadowColor = "rgba(0, 0, 0, 0.35)";
  ctx.shadowBlur = 36 * k;
  ctx.shadowOffsetY = 12 * k;
  round(0);
  ctx.fillStyle = "#000";
  ctx.fill(); // (under the picture: only its shadow shows)
  ctx.restore();
  round(0);
  ctx.save();
  ctx.clip();
  ctx.drawImage(img, b.x * kx, b.y * ky, b.w * kx, b.h * ky, m, m, iw, ih);
  ctx.restore();
  ctx.lineWidth = Math.max(1, k);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
  round(ctx.lineWidth / 2);
  ctx.stroke();
  return c;
}

/** The area flying into the notch (./fly.ts): three nested layers the size of the area — sideways,
 *  up, and smaller and fainter — so the path is an arc. */
function Fly({ plan, crop }: { plan: FlyPlan; crop: HTMLCanvasElement | null }) {
  const x = useRef<HTMLDivElement>(null);
  const y = useRef<HTMLDivElement>(null);
  const sc = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const f = flyFrames(plan);
    const anims: Animation[] = [];
    const run = (el: HTMLDivElement | null, frames: Keyframe[], easing: string) => {
      if (el && frames.length && typeof el.animate === "function") anims.push(el.animate(frames, { duration: plan.ms, easing, fill: "forwards" }));
    };
    if (crop && sc.current && crop.parentNode !== sc.current) sc.current.appendChild(crop);
    run(x.current, f.x, f.easing.x);
    run(y.current, f.y, f.easing.y);
    run(sc.current, f.s, f.easing.s);
    run(sc.current, f.o, f.easing.o);
    return () => anims.forEach((a) => a.cancel());
  }, [plan, crop]);
  const b = plan.box;
  return (
    <div className="lx-fly" ref={x} style={{ left: b.x, top: b.y, width: b.w, height: b.h }} aria-hidden="true">
      <div className="lx-fly-y" ref={y}>
        <div className="lx-fly-s" ref={sc} />
      </div>
    </div>
  );
}

/** The hint pill's top: below the notch, and below the glow's band (36 pt open) on any display. */
const pillTop = (safeTop: number) => Math.max(safeTop + 14, 48);

/** Whether smart pick's hover outline may show now. */
const hovers = (st: LensState) => st.phase === "open" && !!st.regions && !st.sel && !st.card && st.candidate < 0 && !st.pendingPick;

function Overlay() {
  const [s, setS] = useState<LensState>(() => initial(window.innerWidth, window.innerHeight, readTranslateTo()));
  const sref = useRef(s);
  const door = useRef<Door | null>(null);
  const sock = useRef<LensSocket | null>(null);
  const timers = useRef(new Set<number>());
  const trail = useRef<TrailHandle>(null);
  const surface = useRef<HTMLDivElement>(null);
  const selView = useRef<SelectionHandle>(null);
  const hoverEl = useRef<HTMLDivElement>(null);
  const hover = useRef<Hover | null>(null);
  const composer = useRef<HTMLInputElement>(null);
  /** The crop that flies into the notch (cut when the hand-off was asked for). */
  const crop = useRef<HTMLCanvasElement | null>(null);
  const drag = useRef<Drag | null>(null);
  const space = useRef(false);
  const pointer = useRef<Pt>({ x: window.innerWidth / 2, y: window.innerHeight / 2 });
  /** The session (g) the pointer was last seeded for, and whether it has moved since. */
  const seeded = useRef<{ g: number; moved: boolean }>({ g: -1, moved: false });
  const lastClick = useRef<{ t: number; p: Pt } | null>(null);
  /** The pointer's work for the next animation frame is pending. */
  const frame = useRef<number | undefined>(undefined);
  const cursor = useRef("");
  const active = useRef(Date.now());
  const [q, setQ] = useState("");
  const mqReduced = useMedia("(prefers-reduced-motion: reduce)");
  const mqContrast = useMedia("(prefers-contrast: more)");
  const light = useMedia("(prefers-color-scheme: light)");

  const runRef = useRef<(e: Effect) => void>(() => {});
  const dispatch = useCallback((a: Action) => {
    const [next, fx] = reduce(sref.current, a);
    sref.current = next;
    setS(next);
    for (const e of fx) runRef.current(e);
  }, []);

  const later = (fn: () => void, ms: number) => {
    const id = window.setTimeout(() => { timers.current.delete(id); fn(); }, ms);
    timers.current.add(id);
  };

  /** Where the shell saw the pointer (until it moves over the page): Return's "window under the pointer". */
  const seed = useCallback((ev: PhaseEvent) => {
    if (seeded.current.g !== ev.g) seeded.current = { g: ev.g, moved: false };
    const p = ev.pointer;
    if (!seeded.current.moved && p && Number.isFinite(p.x) && Number.isFinite(p.y)) pointer.current = { x: p.x, y: p.y };
  }, []);

  const askShell = useCallback(async () => {
    const h = await door.current?.hello();
    if (h) {
      seed(h);
      dispatch({ t: "shell", hello: h, now: Date.now() });
    }
  }, [dispatch, seed]);

  /** Live mode (no still): out of the way, a moment for the screen to show what's under, then the area. */
  const liveSend = useCallback(async (cmd: Record<string, unknown>) => {
    const d = door.current;
    if (!d) return;
    await d.peek(true);
    await new Promise((r) => setTimeout(r, LIVE_SETTLE_MS)); // (timers, not frames: a hidden page gets none)
    if (!sock.current?.send(cmd)) { await d.peek(false); return; }
    later(() => { if (sref.current.sel?.captured === "pending") void d.peek(false); }, LIVE_GIVE_BACK_MS);
  }, []);

  runRef.current = (e: Effect) => {
    switch (e.t) {
      case "send": sock.current?.send(e.cmd); break;
      case "send_live": void liveSend(e.cmd); break;
      case "peek": void door.current?.peek(e.hidden); break;
      case "close": void door.current?.close(e.why); break;
      case "allow_sc": void door.current?.allowSc(); break;
      case "connect": sock.current?.connect(e.ticket); break;
      case "disconnect": sock.current?.close(); break;
      case "shell_hello": void askShell(); break;
      case "later": later(() => dispatch(e.action), e.ms); break;
      case "revoke": try { URL.revokeObjectURL(e.url); } catch { /* gone */ } break;
      case "glow": void door.current?.glow(e.selected); break;
      case "fly_prep": {
        const st = sref.current;
        crop.current = cropOf(document.querySelector<HTMLImageElement>("img.lx-still"), e.box, st.W, st.H);
        break;
      }
      case "focus_composer": {
        const el = composer.current;
        if (!el) break;
        el.focus();
        setQ((v) => (v + e.text).slice(0, 500));
        break;
      }
    }
  };

  const onStill = useCallback((st: Still) => {
    const g = sref.current.g;
    const url = URL.createObjectURL(new Blob([st.image as BlobPart], { type: st.mime }));
    const img = new Image();
    img.src = url;
    img.decode().then(() => dispatch({ t: "still", g, url })).catch(() => URL.revokeObjectURL(url));
  }, [dispatch]);

  // The shell's door and the core's line, once.
  useEffect(() => {
    let alive = true;
    let off: (() => void) | undefined;
    (async () => {
      const d = await tauriDoor();
      if (!d || !alive) return;
      door.current = d;
      sock.current = new LensSocket(
        { token: wsToken, socket: (url) => new WebSocket(url) as unknown as SocketLike },
        { msg: (m) => dispatch({ t: "msg", msg: m }), still: onStill, status: (status) => dispatch({ t: "socket", status }) },
      );
      off = await d.onPhase((ev) => {
        seed(ev);
        dispatch({ t: "phase", ev, now: Date.now() });
      });
      await askShell();
      // Loaded, drawing nothing yet: the shell may bring the window up. (A timer, not a frame: the
      // window is still hidden, and a hidden page gets no frames.)
      later(() => { if (alive) void d.painted(); }, 30);
    })();
    // An error nothing caught, while open: the overlay closes rather than stay up half-working.
    const broke = () => { if (sref.current.phase === "open") void door.current?.close("cancel"); };
    window.addEventListener("error", broke);
    const t = timers.current;
    return () => {
      alive = false;
      off?.();
      window.removeEventListener("error", broke);
      sock.current?.close();
      t.forEach((id) => clearTimeout(id));
      t.clear();
    };
  }, [askShell, dispatch, onStill, seed]);

  // Open: alive, every second (the shell closes an overlay whose page fell silent).
  useEffect(() => {
    if (s.phase !== "open") return;
    const id = window.setInterval(() => { void door.current?.painted(); }, ALIVE_MS);
    return () => clearInterval(id);
  }, [s.phase]);

  // The window is the display: its size, kept.
  useEffect(() => {
    const f = () => dispatch({ t: "size", W: window.innerWidth, H: window.innerHeight });
    f();
    window.addEventListener("resize", f);
    return () => window.removeEventListener("resize", f);
  }, [dispatch]);

  // The Mac's settings, on the page: Reduce Transparency, Increase Contrast, Reduce Motion, light or dark.
  const reduced = s.flags.reduceMotion || mqReduced;
  useEffect(() => {
    const el = document.documentElement;
    el.toggleAttribute("data-rt", s.flags.reduceTransparency);
    el.toggleAttribute("data-hc", s.flags.increaseContrast || mqContrast);
    el.toggleAttribute("data-rm", reduced);
    el.dataset.theme = light ? "light" : "dark";
  }, [s.flags.reduceTransparency, s.flags.increaseContrast, mqContrast, reduced, light]);

  // Remember the language the user translates into.
  useEffect(() => { if (s.translateTo) saveTranslateTo(s.translateTo); }, [s.translateTo]);

  // A new session starts with an empty question box.
  useEffect(() => { setQ(""); }, [s.g]);
  // The flown crop goes with the session.
  useEffect(() => { if (s.phase === "idle") crop.current = null; }, [s.phase]);

  // Three minutes untouched: it closes by itself.
  useEffect(() => {
    if (s.phase !== "open") return;
    active.current = Date.now();
    const id = window.setInterval(() => {
      if (sref.current.phase === "open" && Date.now() - active.current > IDLE_MS) dispatch({ t: "close", why: "idle" });
    }, 15_000);
    return () => clearInterval(id);
  }, [s.phase, dispatch]);

  // ---- the keys ------------------------------------------------------------------------------
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      active.current = Date.now();
      const st = sref.current;
      if (e.key === " " && drag.current?.mode === "draw") { space.current = true; e.preventDefault(); return; }
      const inComposer = document.activeElement === composer.current && !!composer.current;
      const inSheet = !!(document.activeElement && document.activeElement !== document.body && (document.activeElement as HTMLElement).closest?.(".lx-sheet, .lx-card, .lx-pill"));
      const composing = e.isComposing || e.keyCode === 229;
      if (st.phase === "open" && !composing) {
        const arrows = e.key.startsWith("Arrow");
        if (e.key === "Escape") e.preventDefault();
        if (!inComposer && !inSheet && ((e.key === "Tab" && !st.sel) || (e.key === "Enter" && !e.metaKey) || (e.metaKey && e.key.toLowerCase() === "a") || (arrows && st.sel))) e.preventDefault();
        if (!inComposer && st.sel && e.key.length === 1 && !e.metaKey && !e.ctrlKey && e.key !== " ") e.preventDefault(); // typed into the question box instead
        if (inSheet && !inComposer && (e.key === "Enter" || e.key === " " || e.key === "Tab")) return; // the sheet's own buttons
      }
      if (inComposer && e.key === "Enter") return; // the question box sends it
      dispatch({
        t: "key",
        key: {
          key: e.key, meta: e.metaKey, shift: e.shiftKey, alt: e.altKey, ctrl: e.ctrlKey, composing,
          inComposer, domSelection: !!window.getSelection()?.toString(), pointer: pointer.current,
        },
      });
    };
    const up = (e: KeyboardEvent) => { if (e.key === " ") space.current = false; };
    const menu = (e: MouseEvent) => e.preventDefault(); // no "Reload" menu in here
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("contextmenu", menu);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("contextmenu", menu);
    };
  }, [dispatch]);

  // ---- the pointer ---------------------------------------------------------------------------
  // Smart pick's hover outline: one element, written here and nowhere else (never through React).
  useEffect(() => {
    let size = "";
    const write = (b: Box | null) => {
      const el = hoverEl.current;
      if (!el) return;
      if (!b) { el.classList.remove("on"); return; }
      el.style.transform = `translate3d(${b.x}px, ${b.y}px, 0)`;
      const wh = `${b.w}x${b.h}`;
      if (wh !== size) { el.style.width = `${b.w}px`; el.style.height = `${b.h}px`; size = wh; }
      el.classList.add("on");
    };
    const h = makeHover({
      raf: (cb) => requestAnimationFrame(cb),
      cancelRaf: (id) => cancelAnimationFrame(id),
      setTimeout: (cb, ms) => window.setTimeout(cb, ms),
      clearTimeout: (id) => window.clearTimeout(id),
      regionAt: (p) => { const st = sref.current; return hovers(st) ? regionAt(st.regions, p, st.W, st.H) : null; },
      size: () => ({ W: sref.current.W, H: sref.current.H }),
      write,
    });
    hover.current = h;
    return () => { h.dispose(); hover.current = null; if (frame.current !== undefined) cancelAnimationFrame(frame.current); };
  }, []);
  // A selection, a Tab candidate, a card, a pending pick or the end of the session: no hover.
  const hoverAllowed = hovers(s);
  useEffect(() => { if (!hoverAllowed) hover.current?.hide(); }, [hoverAllowed]);

  const setCursor = (c: string) => {
    if (c === cursor.current) return;
    cursor.current = c;
    if (surface.current) surface.current.style.cursor = c;
  };

  /** The pointer's work, once per animation frame: the cursor, the trail, a corner being dragged. */
  const pointerFrame = () => {
    frame.current = undefined;
    const d = drag.current;
    const st = sref.current;
    if (!d) { setCursor(st.sel && !handingOff(st.sel) ? CURSOR[hitTest(st.sel.box, pointer.current)] : "crosshair"); return; }
    if (d.mode === "draw") { trail.current?.draw(d.pts); return; }
    selView.current?.place(d.box); // (the veil and brackets follow; React hears of it on release)
  };
  const nextFrame = () => { if (frame.current === undefined) frame.current = requestAnimationFrame(pointerFrame); };

  const onDown = (e: React.PointerEvent<HTMLDivElement>) => {
    active.current = Date.now();
    const st = sref.current;
    if (e.button !== 0 || st.phase !== "open" || st.card) return;
    const p = { x: e.clientX, y: e.clientY };
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* not capturable */ }
    hover.current?.hide();
    composer.current?.blur();
    const sel = st.sel;
    // The second click of a double-click on what a click just picked (a line): its paragraph —
    // never an adjustment of the line's box.
    if (sel && sel.how === "pick" && isDouble(lastClick.current, performance.now(), p)) {
      lastClick.current = null;
      drag.current = null;
      dispatch({ t: "click", at: p, double: true });
      return;
    }
    if (sel) {
      const hit = hitTest(sel.box, p);
      // On its way to the notch: the area takes no adjusting (a new one may still be drawn outside it).
      if (hit !== "outside" && (handingOff(sel) || sel.handoff === "ok")) return;
      if (hit !== "outside") {
        drag.current = { mode: "adjust", hit, start: sel.box, p0: p, box: sel.box };
        if (hit === "inside") setCursor("grabbing");
        return;
      }
    }
    drag.current = { mode: "draw", pts: [p], last: p };
    trail.current?.draw([p]);
  };

  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const p = { x: e.clientX, y: e.clientY };
    if (Math.hypot(p.x - pointer.current.x, p.y - pointer.current.y) > 2) active.current = Date.now();
    pointer.current = p;
    seeded.current.moved = true;
    const d = drag.current;
    if (!d) {
      hover.current?.move(p);
      nextFrame();
      return;
    }
    if (d.mode === "draw") {
      const evs = typeof e.nativeEvent.getCoalescedEvents === "function" ? e.nativeEvent.getCoalescedEvents() : [];
      const pts = evs.length ? evs.map((c) => ({ x: c.clientX, y: c.clientY })) : [p];
      if (space.current) {
        // Space held: the selection being drawn moves with the pointer (⌘⇧4's way).
        const dx = p.x - d.last.x, dy = p.y - d.last.y;
        d.pts = d.pts.map((q0) => ({ x: q0.x + dx, y: q0.y + dy }));
      } else {
        d.pts.push(...pts);
      }
      d.last = p;
      nextFrame();
      return;
    }
    const st = sref.current;
    d.box = dragBox(d.start, d.hit, p.x - d.p0.x, p.y - d.p0.y, st.W, st.H);
    nextFrame();
  };

  const onUp = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    drag.current = null;
    space.current = false;
    if (!d) return;
    const st = sref.current;
    if (d.mode === "adjust") {
      dispatch({ t: "adjusted", box: d.box });
      setCursor(CURSOR[hitTest(d.box, { x: e.clientX, y: e.clientY })]);
      return;
    }
    trail.current?.fade();
    const g = classify(d.pts, st.W, st.H);
    if (g.kind === "box") { dispatch({ t: "select", box: g.box, how: g.how }); return; }
    const now = performance.now();
    const double = isDouble(lastClick.current, now, g.at);
    lastClick.current = double ? null : { t: now, p: g.at };
    dispatch({ t: "click", at: g.at, double });
  };

  const onContext = (e: React.MouseEvent) => {
    e.preventDefault();
    dispatch({ t: "close", why: "cancel" }); // right-click closes, as Esc does
  };

  // ---- drawing ---------------------------------------------------------------------------------
  const stops = useMemo(() => (s.candidate >= 0 ? tabStops(s.regions, s.W, s.H) : []), [s.candidate, s.regions, s.W, s.H]);
  const candidate = s.candidate >= 0 ? stops[s.candidate] : undefined;
  const open = s.phase === "open";
  const showStill = !!s.still && (open || s.phase === "closing");
  const pill = hint(s);
  const glowPhase = s.phase === "idle" ? "idle" : s.phase;

  return (
    <div className={"lx" + (s.phase === "closing" ? " closing" : "")} data-phase={s.phase}>
      {s.still && <Still src={s.still} shown={showStill} leaving={!!s.fly} />}
      <div className={"lx-layer lx-scrim" + (open && !s.sel && !s.card ? " on" : "")} />
      {showsWebGlow(s) && (
        <EdgeGlow phase={glowPhase} phaseAt={s.phaseAt} toArmMs={s.toArmMs} via={s.via} reduced={reduced} selected={!!s.sel} notch={s.why === "notch"} W={s.W} H={s.H} />
      )}
      {open && (
        <div
          ref={surface}
          className="lx-surface"
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onContextMenu={onContext}
          aria-label="Circle or drag over anything to search it"
        />
      )}
      <Trail ref={trail} reduced={reduced} />
      {s.sel && <SelectionView ref={selView} key={s.sel.n} box={s.sel.box} W={s.W} H={s.H} />}
      {candidate && !s.sel && <SelectionView key={`c${s.candidate}`} box={candidate} W={s.W} H={s.H} candidate label={`Region ${s.candidate + 1} of ${stops.length}. Return selects it.`} />}
      <div className="lx-hover" ref={hoverEl} aria-hidden="true" />
      {s.pendingPick && <div className="lx-dot" style={{ left: s.pendingPick.at.x, top: s.pendingPick.at.y }} aria-hidden="true" />}
      {pill && (
        <div className="lx-pill" style={{ top: pillTop(s.safeTop) }} role="status">
          <span>{pill}</span>
          <button className="lx-x" aria-label="Close Search Screen" onClick={() => dispatch({ t: "close", why: "cancel" })}>
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"><path d="M6 6l12 12M18 6L6 18" /></svg>
          </button>
        </div>
      )}
      {s.sel && (open || s.phase === "closing") && !s.card && !handingOff(s.sel) && s.sel.handoff !== "ok" && (
        <Sheet s={s} sel={s.sel} dispatch={dispatch} composer={composer} q={q} setQ={setQ} />
      )}
      {s.fly && <Fly key={`fly${s.sel?.n ?? 0}`} plan={s.fly} crop={crop.current} />}
      {s.card && (open || s.phase === "closing") && <Card kind={s.card} dispatch={dispatch} />}
    </div>
  );
}

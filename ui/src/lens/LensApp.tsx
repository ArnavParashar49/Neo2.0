// index.html#/lens — Search Screen's overlay. NEO.app's shell alone opens this window, full-screen
// on the display under the pointer and above everything (ui/src-tauri/src/lens.rs); it talks to
// the core as its one `lens` client (./socket.ts) with the ticket the shell hands it. Everything it
// decides is ./state.ts's reducer; this file wires the reducer to the shell, the socket, the
// pointer and the keys, and draws: the still, the glow, the trail, the brackets, the sheet.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { EdgeGlow } from "./EdgeGlow";
import { Trail, type TrailHandle } from "./Trail";
import { Hover, Selection } from "./Selection";
import { Sheet } from "./Sheet";
import { Card } from "./Cards";
import { tauriDoor, type Door } from "./door";
import { LensSocket, type SocketLike } from "./socket";
import { CURSOR, classify, dragBox, hitTest, type Box, type Hit, type Pt } from "./geometry";
import { regionAt, tabStops } from "./pick";
import { hint, initial, reduce, type Action, type Effect, type LensState } from "./state";
import type { Still } from "./wire";
import { wsToken } from "../shell";
import "./lens.css";

const IDLE_MS = 3 * 60_000;
const HOVER_REST_MS = 150;
const DOUBLE_MS = 350;
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

export function LensApp() {
  const [s, setS] = useState<LensState>(() => initial(window.innerWidth, window.innerHeight, readTranslateTo()));
  const sref = useRef(s);
  const door = useRef<Door | null>(null);
  const sock = useRef<LensSocket | null>(null);
  const timers = useRef(new Set<number>());
  const trail = useRef<TrailHandle>(null);
  const surface = useRef<HTMLDivElement>(null);
  const stillImg = useRef<HTMLImageElement>(null);
  const composer = useRef<HTMLInputElement>(null);
  const drag = useRef<Drag | null>(null);
  const space = useRef(false);
  const pointer = useRef<Pt>({ x: window.innerWidth / 2, y: window.innerHeight / 2 });
  const lastClick = useRef<{ t: number; p: Pt } | null>(null);
  const hoverTimer = useRef<number | undefined>(undefined);
  const raf = useRef<number | undefined>(undefined);
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

  const askShell = useCallback(async () => {
    const h = await door.current?.hello();
    if (h) dispatch({ t: "shell", hello: h, now: Date.now() });
  }, [dispatch]);

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
      off = await d.onPhase((ev) => dispatch({ t: "phase", ev, now: Date.now() }));
      await askShell();
      // Painted once (nothing to see yet): the shell may bring the window up.
      requestAnimationFrame(() => requestAnimationFrame(() => void d.painted()));
    })();
    const t = timers.current;
    return () => {
      alive = false;
      off?.();
      sock.current?.close();
      t.forEach((id) => clearTimeout(id));
      t.clear();
    };
  }, [askShell, dispatch, onStill]);

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
        if (!inComposer && !inSheet && (e.key === "Tab" || (e.key === "Enter" && !e.metaKey) || (e.metaKey && e.key.toLowerCase() === "a") || (arrows && st.sel))) e.preventDefault();
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
  const setCursor = (c: string) => { if (surface.current && surface.current.style.cursor !== c) surface.current.style.cursor = c; };

  const onDown = (e: React.PointerEvent<HTMLDivElement>) => {
    active.current = Date.now();
    const st = sref.current;
    if (e.button !== 0 || st.phase !== "open" || st.card) return;
    const p = { x: e.clientX, y: e.clientY };
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* not capturable */ }
    window.clearTimeout(hoverTimer.current);
    if (st.hover) dispatch({ t: "hover", box: null });
    composer.current?.blur();
    const sel = st.sel;
    if (sel) {
      const hit = hitTest(sel.box, p);
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
    const moved = Math.hypot(p.x - pointer.current.x, p.y - pointer.current.y);
    pointer.current = p;
    const d = drag.current;
    const st = sref.current;
    if (!d) {
      if (moved > 2) active.current = Date.now();
      setCursor(st.sel ? CURSOR[hitTest(st.sel.box, p)] : "crosshair");
      // Smart pick's hover: once the pointer rests on something.
      if (!st.sel && st.regions && st.phase === "open" && !st.card && moved > 3) {
        window.clearTimeout(hoverTimer.current);
        if (st.hover) dispatch({ t: "hover", box: null });
        hoverTimer.current = window.setTimeout(() => {
          const now = sref.current;
          if (drag.current || now.sel) return;
          dispatch({ t: "hover", box: regionAt(now.regions, pointer.current, now.W, now.H)?.box ?? null });
        }, HOVER_REST_MS);
      }
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
      trail.current?.draw(d.pts);
      return;
    }
    d.box = dragBox(d.start, d.hit, p.x - d.p0.x, p.y - d.p0.y, st.W, st.H);
    if (raf.current === undefined) {
      raf.current = requestAnimationFrame(() => {
        raf.current = undefined;
        const cur = drag.current;
        if (cur?.mode === "adjust") dispatch({ t: "moving", box: cur.box });
      });
    }
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
    const prev = lastClick.current;
    const double = !!prev && now - prev.t < DOUBLE_MS && Math.hypot(prev.p.x - g.at.x, prev.p.y - g.at.y) < 6;
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
      {s.still && <img ref={stillImg} className={"lx-still" + (showStill ? " on" : "")} src={s.still} alt="" draggable={false} />}
      <div className={"lx-layer lx-scrim" + (open && !s.sel && !s.card ? " on" : "")} />
      <EdgeGlow phase={glowPhase} phaseAt={s.phaseAt} toArmMs={s.toArmMs} via={s.via} reduced={reduced} selected={!!s.sel} W={s.W} H={s.H} />
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
      {s.sel && <Selection key={s.sel.n} box={s.sel.box} />}
      {candidate && !s.sel && <Selection key={`c${s.candidate}`} box={candidate} candidate label={`Region ${s.candidate + 1} of ${stops.length}. Return selects it.`} />}
      {s.hover && !s.sel && !candidate && <Hover box={s.hover} />}
      {s.pendingPick && <div className="lx-dot" style={{ left: s.pendingPick.at.x, top: s.pendingPick.at.y }} aria-hidden="true" />}
      {pill && (
        <div className="lx-pill" style={{ top: s.safeTop + 14 }} role="status">
          <span>{pill}</span>
          <button className="lx-x" aria-label="Close Search Screen" onClick={() => dispatch({ t: "close", why: "cancel" })}>
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"><path d="M6 6l12 12M18 6L6 18" /></svg>
          </button>
        </div>
      )}
      {s.sel && (open || s.phase === "closing") && !s.card && (
        <Sheet s={s} sel={s.sel} dispatch={dispatch} still={stillImg.current} composer={composer} q={q} setQ={setQ} />
      )}
      {s.card && (open || s.phase === "closing") && <Card kind={s.card} dispatch={dispatch} />}
    </div>
  );
}

import { lazy, Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ThinkingOrb } from "thinking-orbs";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { hasShell, hideWindow, listen, openExternal, pinHost, showQuiet } from "./shell";
import { inNotch } from "./host";
import { MOCK, SONG_MOCK, useNeo, type ConfirmScope, type Item, type NeoState, type OrbState, type ToolRun, type TurnMeta } from "./ws";
import { useBrowser, type BrowserSnap } from "./browser/hub";
import { BrowserCard } from "./browser/BrowserCard";
import { EndedRow, MiniRow, PoppedRow } from "./browser/BrowserRow";
import { CARD, PILL, browserSizes } from "./browser/sizes";
import { browsingLine, displayHost } from "./browser/text";
import { useReducedMotion } from "./browser/motion";
import { closeBrowserWindow, openBrowserWindow, showBrowserWindow } from "./browser/popout";
import { I } from "./browser/icons";
import { inTauri, placeInitial, resizeAnchored, workArea, type Area, type Size } from "./win";
import { useCamera } from "./camera/hub";
import { CameraCard, CameraEndedRow, CameraPoppedRow, camIcon } from "./camera/CameraCard";
import { CAMERA_POPIN_EVENT, closeCameraWindow, openCameraWindow, showCameraWindow } from "./camera/popout";
import { useSongShape } from "./song/hub";
import { cardHeight, cardWidth, settleHeight, useLensCard } from "./lens/card";
import { LensCard } from "./lens/LensCard";
import { SongItem, noteIcon } from "./song/SongCard";
import { Onboarding } from "./Onboarding";
import { isNoModelProblem, needsBrain } from "./onboard";
import { KEY_TYPED, looksLikeKey } from "./keys/keyshape";
import { awayEscFolds, escFolds, wantsAwayEsc } from "./esc";
import "./App.css";
import "./browser/browser.css";
import "./notch.css";

// Dev only (`?mock=browser`): the scripted core's control panel and the simulated pop-out window.
const MockHud = import.meta.env.DEV && MOCK ? lazy(() => import("./browser/mock/Hud")) : null;
const SimPopout = import.meta.env.DEV ? lazy(() => import("./browser/mock/SimWindow")) : null;
// Dev only (`?mock=song`): the scripted song core's control panel.
const SongHud = import.meta.env.DEV && SONG_MOCK ? lazy(() => import("./song/mock/Hud")) : null;

const LABEL: Record<NeoState, string> = {
  idle: "Ready",
  listening: "Listening…",
  thinking: "Thinking…",
  working: "Working…",
  searching: "Searching…",
  speaking: "Speaking…",
  connecting: "Connecting…",
  confirming: "Needs your OK",
};

/** Which thinking-orbs animation plays for each NEO state. */
const ORB: Record<NeoState, OrbState> = {
  idle: "breathing",
  listening: "listening",
  thinking: "solving",
  working: "solving",
  searching: "searching",
  speaking: "composing",
  connecting: "connecting",
  confirming: "shaping",
};
const ORB_SPEED = 1;

const SUGGESTIONS = [
  "What's on my screen?",
  "Read my unread mail",
  "Set a 10 minute timer",
  "Open GitHub and search for thinking-orbs",
];

const IDLE_COLLAPSE_MS = 40_000;
/** The chat's size in the notch: NotchHub's compact panel under the notch (`NeoPanelLayout.standard`
 *  there) — what a click or a question opens. A live camera or song card asks a little taller; a
 *  browser card asks for the host's room (`browserSizes`). The host clamps every request. */
const NOTCH_CARD: Size = { w: 340, h: 300 };
const NOTCH_CAMERA: Size = { w: 340, h: 440 };
const NOTCH_SONG: Size = { w: 340, h: 380 };
const MORPH_MS = 280;
type BMode = "docked" | "focus" | "mini" | "popped";

/** Which brain answered, as the core names it ("Claude Opus 5.5", "GPT-OSS 120B on Groq"). An
 *  older core sends no label: its model id as it is. */
function brainLabel(m: TurnMeta): string {
  if (m.route === "quick") return "quick action · no model";
  return m.label || m.model || m.brain || "local";
}

function Meta({ m }: { m: TurnMeta }) {
  const secs = (m.ms / 1000).toFixed(m.ms < 10_000 ? 1 : 0) + " s";
  const tools = m.route === "agent" && m.tools ? ` · ${m.tools} tool${m.tools === 1 ? "" : "s"}` : "";
  return <div className="meta">{brainLabel(m)}{tools} · {secs}</div>;
}

function Tools({ tools }: { tools: ToolRun[] }) {
  return (
    <div className="tools">
      {tools.map((t, i) => (
        <span key={i} className={"chip" + (t.running ? " running" : t.ok === false ? " fail" : " ok")} title={t.summary || JSON.stringify(t.args ?? {})}>
          <span className="dot" />{inNotch ? t.name.replace(/_/g, " ") : t.name}
        </span>
      ))}
    </div>
  );
}

function Md({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children }) => (
          <a href={href} onClick={(e) => { e.preventDefault(); if (href) (hasShell ? void openExternal(href) : window.open(href, "_blank")); }}>{children}</a>
        ),
      }}
    >
      {text}
    </ReactMarkdown>
  );
}

const Icon = {
  mic: <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></svg>,
  stop: <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>,
  send: <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19V5M5 12l7-7 7 7" /></svg>,
  collapse: <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M6 12h12" /></svg>,
  pin: <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 17v5M8 3h8l-1 7 3 3H6l3-3z" /></svg>,
  broom: <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M4 20h16M6 20l2-8h8l2 8M12 12V4" /></svg>,
};

export default function App() {
  const neo = useNeo();
  const hub = neo.hub;
  const bsnap = useBrowser(hub);
  const cam = neo.cam;
  const csnap = useCamera(cam);
  const camLive = csnap.live;
  const song = neo.song;
  const sshape = useSongShape(song); // which song card is live and its face — not its level
  const songHearing = sshape.hearing;
  // Search Screen's card: in the notch, an area's results take the panel in place of the chat —
  // never while the chat has something that must be seen (a question waiting for an OK, a browser
  // session's turn for the user): the chat shows it, and the card gives the panel back (below).
  const lcard = useLensCard(neo.lensCard);
  const chatMustShow = !!neo.ask || bsnap.live?.status === "need_user" || bsnap.live?.status === "waiting";
  const lensLive = inNotch && !!lcard && !chatMustShow;
  useEffect(() => { neo.lensCard.needChat(chatMustShow); }, [chatMustShow, neo.lensCard]);
  const reduced = useReducedMotion();
  // In the notch the page is always the open chat: NotchHub's panel is what expands and collapses.
  const [expanded, setExpanded] = useState(inNotch);
  const [pinned, setPinned] = useState(false);
  useEffect(() => { if (inNotch) pinHost(pinned); }, [pinned]);
  // A pending question outranks everything: the card shows exactly what will run.
  const confirming = !!neo.ask;
  const shown = confirming ? "confirming" : neo.state;
  const orb = ORB[shown];
  const [text, setText] = useState("");
  // Typed text that started with an API key: kept from every model (and from ↑), and said so here.
  const [keyGuard, setKeyGuard] = useState(false);
  const [hover, setHover] = useState(false);
  // "Erase NEO's Memory…" from the menu bar: "ask" shows the question, "sent" waits for the
  // core to confirm (the dialog only closes once it has actually erased).
  const [erasing, setErasing] = useState<false | "ask" | "sent">(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const lastInput = useRef("");
  const init = useRef<Promise<void> | null>(null);
  const press = useRef<{ x: number; y: number } | null>(null);
  // When the chat last came on screen (an Esc pressed elsewhere just then wasn't meant for it).
  const shownAt = useRef(0);

  // ---- the browser session's place in the overlay -------------------------------------------
  const [bmode, setBmode] = useState<BMode>("docked");
  const [area, setArea] = useState<Area | null>(null);
  const [simPopout, setSimPopout] = useState(false);
  const [viewFocus, setViewFocus] = useState(0);
  // Task end: the card lingers (frozen) while the pointer is on it, then folds away. Derived in
  // render, so there is never a frame between "session over" and "card lingering".
  const [folding, setFolding] = useState(false);
  const [foldedSid, setFoldedSid] = useState<string | null>(null);
  // The fold's flight: a copy of the page lifts off the card and lands on the session's row.
  const [foldFx, setFoldFx] = useState<null | "out" | "in">(null);
  const [landingSid, setLandingSid] = useState<string | null>(null);
  const ghost = useRef<{ el: HTMLCanvasElement; x: number; y: number; w: number; h: number; sid: string } | null>(null);
  const foldHold = useRef<Promise<void> | null>(null); // the window keeps its size until the flight lands
  const foldLand = useRef<(() => void) | null>(null);
  const lastLive = useRef<BrowserSnap | null>(null);
  const collapsedDuringJob = useRef(false);
  const live = bsnap.live ? bsnap : null;
  if (live) lastLive.current = live;
  const lingering = !live && lastLive.current?.live && foldedSid !== lastLive.current.live.sid ? lastLive.current : null;
  const linger = useMemo<BrowserSnap | null>(
    () => (lingering ? { ...lingering, live: { ...lingering.live!, status: "ended", job: null, loading: false, need_user: null }, action: null, dialog: null, select: null } : null),
    [lingering],
  );
  const card = live ?? linger; // what the stage shows
  const sizes = browserSizes(area);
  const stageMode: BMode | null = card ? (bmode === "popped" || bmode === "mini" ? (live ? bmode : null) : bmode) : null;
  const layout: "chat" | "docked" | "focus" = !card || stageMode === "mini" || stageMode === "popped" || !stageMode ? "chat" : stageMode === "focus" ? "focus" : "docked";
  const compact = sizes.compact && layout === "docked";
  // In the notch the chat is compact; a live camera or song card on it asks a little taller.
  const notchChat: Size = camLive && !csnap.popped ? NOTCH_CAMERA : sshape.liveSid ? NOTCH_SONG : NOTCH_CARD;
  // The card asks for 372 × its content (lens/card.ts: 160–560, in steps of 8, never bouncing) once
  // it has measured itself; until then the panel's size stays as it was.
  const [lensNatural, setLensNatural] = useState(0);
  const [lensH, setLensH] = useState<number | null>(null);
  const lensSince = useRef<number | null>(null);
  const [lensRecheck, setLensRecheck] = useState(0);
  useEffect(() => {
    if (!lensLive || lensNatural <= 0) {
      if (!lensLive) { setLensH(null); setLensNatural(0); lensSince.current = null; }
      return;
    }
    const out = settleHeight(lensH, cardHeight(lensNatural, area?.h), lensSince.current, Date.now());
    lensSince.current = out.since;
    if (out.h !== lensH) setLensH(out.h);
    if (out.recheck === undefined) return;
    const t = window.setTimeout(() => setLensRecheck((n) => n + 1), out.recheck);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lensLive, lensNatural, area?.h, lensRecheck]);
  const lensTarget: Size | null = lensLive && lensNatural > 0 ? { w: cardWidth(area?.w), h: lensH ?? cardHeight(lensNatural, area?.h) } : null;
  const target: Size = lensTarget ?? (!expanded ? PILL : layout === "docked" ? sizes.browse : layout === "focus" ? sizes.focus : inNotch ? notchChat : CARD);

  // A window drag ends with a click event whose client coords match the press (the window moved
  // with the cursor), so compare *screen* coords to tell a drag from a click.
  const onOrbDown = (e: React.MouseEvent) => { press.current = { x: e.screenX, y: e.screenY }; };
  const onOrbClick = (e: React.MouseEvent) => {
    const p = press.current;
    press.current = null;
    if (p && Math.hypot(e.screenX - p.x, e.screenY - p.y) > 4) return; // it was a drag
    expand();
  };

  const expand = useCallback(() => {
    shownAt.current = performance.now();
    setExpanded(true);
    if (inNotch) void showQuiet(); // the host opens its panel
    // Into the chat input — unless something in the card (a page prompt, a dialog) already has focus.
    setTimeout(() => { const a = document.activeElement; if (!a || a === document.body) inputRef.current?.focus(); }, 60);
  }, []);
  const collapse = useCallback(() => {
    if (inNotch) { void hideWindow(); return; } // the host folds its panel; the page stays the chat
    if (!["idle", "listening"].includes(neo.state)) collapsedDuringJob.current = true;
    setExpanded(false);
  }, [neo.state]);
  useEffect(() => { if (neo.state === "idle" && !live) collapsedDuringJob.current = false; }, [neo.state, live]);

  useEffect(() => {
    let alive = true;
    const upd = () => { void workArea().then((a) => { if (alive && a) setArea(a); }); };
    upd();
    if (!inTauri) window.addEventListener("resize", upd);
    return () => { alive = false; window.removeEventListener("resize", upd); };
  }, [expanded]);

  // ---- window frame + card morph (§1.3) --------------------------------------------------------
  // Growing: frame first (the transparent window is bigger, the card still its old size, anchored
  // bottom-right), then the card animates up. Shrinking: card first, frame on the way out.
  const [cardSize, setCardSize] = useState<Size>(CARD);
  const [snapSize, setSnapSize] = useState(true); // no transition for this change
  const winSize = useRef<Size>(PILL);
  const gen = useRef(0);
  useEffect(() => {
    // First mount: park the orb bottom-right. Every later change resizes around the orb's own
    // corner — after the initial placement has finished, so a StrictMode double-run can't race it.
    if (!init.current) {
      init.current = placeInitial(PILL).then(() => { winSize.current = PILL; });
      // The host hears what the chat needs from the start (its compact panel), not only on a change.
      if (inNotch) { void resizeAnchored(target); winSize.current = target; }
      return;
    }
    const my = ++gen.current;
    const run = async () => {
      await init.current;
      if (my !== gen.current) return;
      if (inNotch) {
        // No window to morph: the card takes its size and the host springs its panel to it.
        setSnapSize(true); setCardSize(target);
        await resizeAnchored(target);
        winSize.current = target;
        return;
      }
      const from = winSize.current;
      if (!expanded) { await resizeAnchored(PILL); winSize.current = PILL; return; }
      const hold = foldHold.current;
      foldHold.current = null;
      if (hold) {
        // Folding: the card is already its new size; the window waits for the flight to land.
        setSnapSize(true); setCardSize(target);
        await Promise.race([hold, new Promise((r) => setTimeout(r, 1200))]);
        if (my !== gen.current) return;
        await resizeAnchored(target);
        winSize.current = target;
        return;
      }
      const fromPill = from.w === PILL.w && from.h === PILL.h;
      if (fromPill || reduced) {
        setSnapSize(true); setCardSize(target);
        await resizeAnchored(target);
        winSize.current = target;
        return;
      }
      if (from.w === target.w && from.h === target.h) { setSnapSize(true); setCardSize(target); return; }
      const union = { w: Math.max(from.w, target.w), h: Math.max(from.h, target.h) };
      const anchored = await resizeAnchored(union);
      winSize.current = union;
      if (my !== gen.current) return;
      if (!anchored) { setSnapSize(true); setCardSize(target); await resizeAnchored(target); winSize.current = target; return; }
      setSnapSize(false);
      requestAnimationFrame(() => { if (my === gen.current) setCardSize(target); });
      await new Promise((r) => setTimeout(r, MORPH_MS + 40));
      if (my !== gen.current) return;
      if (union.w !== target.w || union.h !== target.h) { await resizeAnchored(target); winSize.current = target; }
    };
    void run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expanded, target.w, target.h]);

  // ---- browser session: automatic behaviour (§1.5) ----------------------------------------------
  const liveSid = live?.live?.sid ?? null;
  const liveStatus = live?.live?.status;
  const livePopped = !!live?.live?.popped;
  useEffect(() => {
    if (!liveSid) return;
    setFolding(false);
    setBmode(livePopped ? "popped" : "docked");
    if (!collapsedDuringJob.current) expand();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveSid]);

  // Login / CAPTCHA / 2FA always surface — shown without stealing focus from other apps (the
  // shell orders the window front without making it key; plain show() on older shells).
  const needUser = liveStatus === "need_user";
  useEffect(() => {
    if (!needUser) return;
    if (hasShell) void showQuiet();
    setExpanded(true);
    setBmode((m) => (m === "mini" ? "docked" : m));
    setViewFocus((n) => n + 1);
  }, [needUser]);

  // ---- Search Screen ---------------------------------------------------------------------------
  // "Continue in NEO": the area went to the chat — it comes forward (without taking the keys from the
  // app the user is in; in the notch the host opens its panel). Opening it from the voice or a host
  // in the notch: the host's panel folds away first, so it isn't in the still.
  useEffect(() => {
    if (!neo.lensContinueAt) return;
    if (hasShell) void showQuiet();
    shownAt.current = performance.now(); // (brought forward, though it may have been open already)
    setExpanded(true);
  }, [neo.lensContinueAt]);
  useEffect(() => {
    if (neo.lensOpeningAt && inNotch) void hideWindow();
  }, [neo.lensOpeningAt]);

  // ---- camera: while it is on, its preview is always in sight ----------------------------------
  // The camera only turns on at the user's own request; its card opens with it — shown without
  // stealing focus from other apps — and the overlay stays open until the camera is off.
  const camSid = camLive?.sid ?? null;
  useEffect(() => {
    if (!camSid) return;
    if (hasShell) void showQuiet();
    setExpanded(true);
  }, [camSid]);
  // Off (answered, stopped, erased): its pop-out window goes with it.
  useEffect(() => { if (!camSid) void closeCameraWindow(); }, [camSid]);
  useEffect(() => {
    if (!hasShell) return;
    const un = listen(CAMERA_POPIN_EVENT, () => { cam.setPopped(false); setExpanded(true); });
    return () => { un.then((f) => f()).catch(() => {}); };
  }, [cam]);
  const camPopOut = useCallback(async () => {
    if (!cam.sid || !inTauri) return;
    cam.setPopped(true);
    if (!(await openCameraWindow())) cam.setPopped(false);
  }, [cam]);
  const camBringBack = useCallback(() => { cam.setPopped(false); void closeCameraWindow(); }, [cam]);

  // ---- song: while NEO listens, its card is in sight ----------------------------------------------
  // It listens only when the user asked; the card opens with it (without taking focus from other
  // apps) and the overlay stays open until it has stopped listening.
  const songSid = sshape.liveSid;
  useEffect(() => {
    if (!songSid) return;
    if (hasShell) void showQuiet();
    setExpanded(true);
  }, [songSid]);

  // The pop-out window went away without "Bring back" (closed elsewhere): come home.
  const wasPopped = useRef(false);
  useEffect(() => {
    if (livePopped) { wasPopped.current = true; return; }
    if (wasPopped.current && bmode === "popped") { setBmode("docked"); setSimPopout(false); }
    wasPopped.current = false;
  }, [livePopped, bmode]);

  useEffect(() => {
    if (!hasShell) return;
    const un = listen("neo://browser-popin", () => { setBmode("docked"); setExpanded(true); });
    return () => { un.then((f) => f()).catch(() => {}); };
  }, []);

  // Session over: fold the card into its row. A finished task waits while the user is still
  // looking at the page: the pointer is on the browser card and has moved in the last 1.5 s (up
  // to 6 s). A pointer that is merely inside the overlay — say, on the Yes button it just
  // clicked — doesn't hold it.
  const stagePtr = useRef({ inside: false, moved: 0 });
  const onStagePointer = (e: React.PointerEvent) => {
    const p = stagePtr.current;
    if (e.type === "pointerleave") { p.inside = false; return; }
    p.inside = true;
    p.moved = performance.now();
  };
  useEffect(() => {
    const prev = lastLive.current?.live;
    if (liveSid || !prev || foldedSid === prev.sid) return;
    const sid = prev.sid;
    const reason = bsnap.ended[sid]?.close.reason;
    const done = () => { setFoldedSid(sid); setFolding(false); setBmode("docked"); };
    if (bmode === "popped" || bmode === "mini" || !expanded) { void closeBrowserWindow(); setSimPopout(false); done(); return; }
    const t0 = performance.now();
    let timer: number | undefined;
    const fade = () => { setFolding(true); timer = window.setTimeout(done, reduced ? 120 : 240); };
    const fly = () => {
      // FLIP (§7): copy the page as it is on screen, fade the card away around it (160 ms), snap
      // the card to its chat size, then fly the copy onto the row's thumbnail (360 ms).
      const root = document.getElementById("root");
      const canvas = root?.querySelector<HTMLCanvasElement>(".stage .br-canvas");
      // In the notch the host springs its panel down to the chat at the same time: a plain fade.
      if (!root || !canvas || !canvas.width || reduced || inNotch || !neo.items.some((i) => i.kind === "browser" && i.sid === sid)) { fade(); return; }
      const rb = root.getBoundingClientRect(), cb = canvas.getBoundingClientRect();
      const el = document.createElement("canvas");
      el.width = canvas.width; el.height = canvas.height;
      el.getContext("2d")?.drawImage(canvas, 0, 0);
      el.className = "fold-ghost";
      const g = { el, x: cb.left - rb.left, y: cb.top - rb.top, w: cb.width, h: cb.height, sid };
      Object.assign(el.style, { left: `${g.x}px`, top: `${g.y}px`, width: `${g.w}px`, height: `${g.h}px` });
      root.appendChild(el);
      ghost.current = g;
      setFoldFx("out");
      // Never leave a copy behind if the landing can't happen (the layout changed under it).
      window.setTimeout(() => {
        if (!el.isConnected) return;
        el.remove(); ghost.current = null; setFoldFx(null); setLandingSid(null); foldLand.current?.(); foldLand.current = null;
      }, 1600);
      timer = window.setTimeout(() => {
        // Snap the card to its chat size in the same render as the layout change, so the row the
        // copy flies to is measured where it will rest; the window keeps its size until it lands.
        foldHold.current = new Promise<void>((r) => { foldLand.current = r; });
        setSnapSize(true); setCardSize(CARD);
        setLandingSid(sid); setFoldFx("in"); done();
      }, 160);
    };
    const wait = () => {
      const p = stagePtr.current, now = performance.now();
      if (reason === "done" && p.inside && now - p.moved < 1500 && now - t0 < 6000) { timer = window.setTimeout(wait, 250); return; }
      fly();
    };
    wait();
    return () => {
      clearTimeout(timer);
      // A new session opened mid-fold: drop the flight rather than leave the card hidden.
      const g = ghost.current;
      if (g && g.sid === sid) { g.el.remove(); ghost.current = null; setFoldFx(null); setLandingSid(null); foldLand.current?.(); foldLand.current = null; }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveSid]);

  // Folded (the page's copy is on its way to the row): the last live frame isn't needed any more.
  useEffect(() => { if (foldedSid && !liveSid) hub.releaseFrames(); }, [foldedSid, liveSid, hub]);

  // The fold lands: the chat layout is on screen, so the row's thumbnail has a place to fly to.
  useLayoutEffect(() => {
    const g = ghost.current;
    if (!g || foldFx !== "in" || layout !== "chat") return;
    ghost.current = null;
    const root = document.getElementById("root"), log = logRef.current;
    if (log && stickToBottom.current) log.scrollTop = log.scrollHeight; // land where the chat will rest
    const row = root?.querySelector<HTMLElement>(`[data-ended-sid="${CSS.escape(g.sid)}"]`);
    const thumb = row?.querySelector<HTMLElement>(".br-row-thumb");
    // The row's own entrance (a small rise) would move the target mid-flight: settle it first.
    row?.getAnimations({ subtree: true }).forEach((a) => a.finish());
    // The card (now chat-sized) comes back around the flight. Imperative, so the card's own CSS
    // entrance animation is never restarted by a class change.
    root?.querySelector<HTMLElement>(".card")?.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 220, delay: 60, easing: "ease", fill: "backwards" });
    const finish = () => { g.el.remove(); setLandingSid(null); setFoldFx(null); foldLand.current?.(); foldLand.current = null; };
    if (!root || !thumb || !log) { g.el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 200, fill: "forwards" }).onfinish = finish; return; }
    const rb = root.getBoundingClientRect(), tb = thumb.getBoundingClientRect(), lb = log.getBoundingClientRect();
    const visible = tb.bottom > lb.top + 8 && tb.top < lb.bottom - 8;
    const to = { x: tb.left - rb.left, y: tb.top - rb.top, w: tb.width, h: tb.height };
    // Uniform scale by width; the bottom is clipped away, like the row's top-anchored cover fit.
    const k = to.w / g.w;
    const clipB = Math.max(0, g.h - to.h / k);
    const end = visible
      ? { transform: `translate(${to.x - g.x}px, ${to.y - g.y}px) scale(${k})`, clipPath: `inset(0px 0px ${clipB}px 0px round ${8 / k}px)`, opacity: 1 }
      : { transform: `translate(${to.x - g.x}px, ${Math.max(lb.top - rb.top, to.y) - g.y}px) scale(${k})`, clipPath: `inset(0px 0px ${clipB}px 0px round ${8 / k}px)`, opacity: 0 };
    const an = g.el.animate(
      [{ transform: "none", clipPath: "inset(0px 0px 0px 0px round 0px)", opacity: 1 }, end],
      { duration: 360, easing: "cubic-bezier(0.2, 0.8, 0.2, 1)", fill: "forwards" },
    );
    an.onfinish = finish;
  }, [foldFx, layout]);

  // ---- modes -------------------------------------------------------------------------------------
  const toggleFocus = useCallback(() => { if (live) { setBmode((m) => (m === "focus" ? "docked" : "focus")); setExpanded(true); } }, [live]);
  const toggleMini = useCallback(() => { if (live) { setBmode((m) => (m === "mini" ? "docked" : "mini")); setExpanded(true); } }, [live]);
  const popOut = useCallback(async () => {
    if (!live?.live) return;
    if (inNotch) { hub.toast("The notch has no pop-out window", 2200); return; }
    setBmode("popped");
    void hub.control("popout");
    if (inTauri) { if (!(await openBrowserWindow(displayHost(live.live.url)))) { hub.toast("Couldn't open a window", 2200); setBmode("docked"); void hub.control("popin"); } }
    else setSimPopout(true);
  }, [live, hub]);
  const bringBack = useCallback(() => {
    setBmode("docked");
    setExpanded(true);
    setSimPopout(false);
    void closeBrowserWindow();
    if (hub.sid) void hub.control("popin");
  }, [hub]);
  const showWindow = useCallback(() => { if (inTauri) void showBrowserWindow(); else setSimPopout(true); }, []);

  // Mode shortcuts work whether the card, the mini row or the popped row is showing.
  useEffect(() => {
    if (!live) return;
    const h = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      // ⌘W reads as "close tab" on a page; here it would close NEO's overlay window.
      if (k === "w" && e.metaKey && !e.shiftKey && !e.altKey && !e.ctrlKey) { e.preventDefault(); return; }
      if (!e.metaKey || !e.shiftKey || e.altKey || e.ctrlKey) return;
      if (k === "e") toggleFocus();
      else if (k === "m") toggleMini();
      else if (k === "o") { if (bmode === "popped") bringBack(); else void popOut(); }
      else return;
      e.preventDefault(); e.stopPropagation();
    };
    window.addEventListener("keydown", h, true);
    return () => window.removeEventListener("keydown", h, true);
  }, [live, bmode, toggleFocus, toggleMini, popOut, bringBack]);

  useEffect(() => {
    // (Search Screen's card folding away with NotchHub's panel goes once the page is off screen.)
    const v = () => { hub.visibilityChanged(); cam.visibilityChanged(); neo.lensCard.visibilityChanged(); };
    document.addEventListener("visibilitychange", v);
    return () => document.removeEventListener("visibilitychange", v);
  }, [hub, cam, neo.lensCard]);

  // Native summon (hotkey / tray) → expand + arm the voice session.
  useEffect(() => {
    if (!hasShell) return;
    const un = listen("neo://wake", () => { expand(); neo.send({ cmd: "wake" }); });
    return () => { un.then((f) => f()).catch(() => {}); };
  }, [neo.send, expand]);

  // The host opened its panel at the user's own hand (a click on the island, the hotkey): the
  // chat is on screen, so the input takes the keyboard as it does when the card expands here.
  useEffect(() => {
    if (!inNotch) return;
    // (A Search Screen card the panel was folding away gives way to the chat it opens on.)
    const un = listen("neo://show", () => { neo.lensCard.panelShown(); expand(); });
    return () => { un.then((f) => f()).catch(() => {}); };
  }, [expand, neo.lensCard]);

  // The host's panel opened any way at all (NEO summoned, the page's own show, a card) — no keys with
  // it: a Search Screen card it was still folding away (opened again before the page went off screen)
  // gives way to the chat now, not in the open panel FOLD_HOLD_MS later.
  useEffect(() => {
    if (!inNotch) return;
    const un = listen("neo://opened", () => neo.lensCard.panelShown());
    return () => { un.then((f) => f()).catch(() => {}); };
  }, [neo.lensCard]);

  // A confirmation request always surfaces the card.
  useEffect(() => { if (confirming) expand(); }, [confirming, expand]);

  useEffect(() => {
    if (erasing !== "ask") return;
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") { e.preventDefault(); setErasing(false); } };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [erasing]);

  // Menu bar → Erase NEO's Memory…: the overlay asks before anything is deleted.
  useEffect(() => {
    if (!hasShell) return;
    const un = listen("neo://erase", () => { expand(); setErasing("ask"); });
    return () => { un.then((f) => f()).catch(() => {}); };
  }, [expand]);

  // Erased: close the dialog, and forget the typed history too (↑ must not bring it back).
  useEffect(() => {
    if (!neo.erasedAt) return;
    setErasing(false);
    lastInput.current = "";
    setText("");
    setKeyGuard(false);
  }, [neo.erasedAt]);

  // Dismissed without an answer (collapsed, hidden): the question doesn't linger for next time.
  useEffect(() => { if (!expanded && erasing === "ask") setErasing(false); }, [expanded, erasing]);

  // NEO's UI lives in the notch, but "Erase NEO's Memory…" (the host's Settings ask NEO.app's shell
  // for it; the menu bar's item, when there is one) still opens this window for its dialog — the
  // one thing it shows there: once the dialog is over, hide it again.
  const wasErasing = useRef(false);
  useEffect(() => {
    if (inTauri && neo.uiHost === "notch" && wasErasing.current && erasing === false) void hideWindow();
    wasErasing.current = erasing !== false;
  }, [erasing, neo.uiHost]);

  // Auto-collapse after a quiet stretch unless pinned, hovered, being typed into — or browsing
  // on stage, the user driving, or NEO waiting on them.
  const browsingHolds = !!live && (bmode === "docked" || bmode === "focus" || live.live!.controller === "user" || live.live!.status === "need_user");
  useEffect(() => {
    if (inNotch) return; // the host times its own collapse
    if (!expanded || pinned || hover || neo.state !== "idle" || confirming || erasing || browsingHolds || camLive || songHearing) return;
    const t = window.setTimeout(() => {
      if (document.activeElement === inputRef.current && text) return;
      setExpanded(false);
    }, IDLE_COLLAPSE_MS);
    return () => clearTimeout(t);
  }, [expanded, pinned, hover, neo.state, neo.items, text, confirming, erasing, browsingHolds, camLive, songHearing]);

  // Follow new content only when the user hasn't scrolled up.
  useEffect(() => {
    const el = logRef.current;
    // A smooth scroll never arrives while the document counts as hidden: jump instead.
    if (el && stickToBottom.current) el.scrollTo({ top: el.scrollHeight, behavior: reduced || document.hidden ? "auto" : "smooth" });
  }, [neo.items, neo.state, bsnap.ended, csnap.ended, sshape.liveView, sshape.ended, reduced]);

  // A (re)connect replays the transcript, and the log mounts afresh each time the orb opens: land on
  // the newest line at once. A layout change (the card docking, the strip appearing) jumps too — a
  // smooth scroll there is cut off by the resize.
  useLayoutEffect(() => {
    const el = logRef.current;
    if (!el) return;
    stickToBottom.current = true;
    el.scrollTop = el.scrollHeight;
  }, [neo.helloAt, expanded]);
  useLayoutEffect(() => {
    const el = logRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [layout, compact, neo.ask, erasing]); // what decides whether the log is the docked strip

  // The log changing size under a reader at the bottom (the host springing its panel from a card's
  // size to the chat's, a confirm card taking the input's place) keeps the newest line in view.
  useEffect(() => {
    const el = logRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => { if (stickToBottom.current) el.scrollTop = el.scrollHeight; });
    ro.observe(el);
    return () => ro.disconnect();
  }, [layout, compact]); // the log remounts with the layout

  const onScroll = () => {
    const el = logRef.current;
    if (el) stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };

  const submit = (t = text) => {
    const v = t.trim();
    if (!v) return;
    // A pasted key goes nowhere: not to the core, not into ↑'s history (the core checks again).
    if (looksLikeKey(v)) { setText(""); setKeyGuard(true); return; }
    setKeyGuard(false);
    lastInput.current = v;
    stickToBottom.current = true;
    neo.send({ cmd: "text", text: v });
    setText("");
  };

  // (Esc in the input folds the chat as it does anywhere on the page: see "Esc" below.)
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") submit();
    else if (e.key === "ArrowUp" && !text) setText(lastInput.current);
  };

  // Keys while a question is up: ⌘⏎ approves (a stray Enter meant for something else must not),
  // ⌘⇧⏎ approves for the rest of the task (only when the card offers it), ⎋ declines. The answer
  // names the exact action on the card.
  const offersTask = !!neo.ask?.scopes?.includes("task");
  const answer = useCallback((yes: boolean, scope?: ConfirmScope) => {
    if (neo.ask) neo.send({ cmd: "confirm", yes, action_id: neo.ask.action_id, ...(scope ? { scope } : {}) });
  }, [neo.ask, neo.send]);
  // Only while the confirm card is on the page: never under Search Screen's card in the notch.
  useEffect(() => {
    if (!confirming || lensLive) return;
    const h = (e: KeyboardEvent) => {
      if (e.key === "Enter" && e.metaKey && e.shiftKey) {
        // Never a plain yes in its place: without the offer, this key does nothing.
        e.preventDefault();
        if (offersTask) answer(true, "task");
        return;
      }
      if (e.key === "Enter" && e.metaKey) { e.preventDefault(); answer(true); }
      if (e.key === "Escape") { e.preventDefault(); answer(false); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [confirming, lensLive, answer, offersTask]);

  // ---- Esc (src/esc.ts) ----------------------------------------------------------------------------
  // The page has the keys: Esc folds the chat wherever its focus is, as "−" does — once the confirm
  // card, the erase dialog, Search Screen's card, the browser card's own dialogs, menus and lists, and
  // an input method have had theirs (on the window, bubbling: whatever stops the key or prevents its
  // default keeps it; the confirm card and the dialog are named, whichever listener runs first).
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (!escFolds(e, { open: expanded, confirming, erasing: erasing !== false, lensCard: lensLive })) return;
      e.preventDefault();
      collapse();
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [expanded, confirming, erasing, lensLive, collapse]);

  // The user's app has the keys: NEO.app's shell passes on a plain Esc it saw go by (the app got it
  // too) — in NEO's window straight to this page (`neo://esc`), in the notch through the core (`esc`,
  // only while this page asked for it, below). Judged on the page as it is when the key comes.
  const awayEsc = useRef<() => void>(() => {});
  awayEsc.current = () => {
    const folds = awayEscFolds({
      open: inNotch ? document.visibilityState === "visible" : expanded,
      focused: document.hasFocus(),
      confirming,
      erasing: erasing !== false,
      lensCard: lensLive,
      pinned,
      holds: chatMustShow || (!!camLive && !csnap.popped) || songHearing || browsingHolds,
      openForMs: performance.now() - shownAt.current,
    });
    if (folds) collapse();
  };
  useEffect(() => {
    if (!hasShell || inNotch) return;
    const un = listen("neo://esc", () => awayEsc.current());
    return () => { un.then((f) => f()).catch(() => {}); };
  }, []);
  useEffect(() => { if (neo.escAt) awayEsc.current(); }, [neo.escAt]);
  // When the chat came on screen: NEO's window unfolding from the orb, however it was asked to (and
  // `expand`, which also brings it forward)…
  useEffect(() => { if (expanded) shownAt.current = performance.now(); }, [expanded]);
  // …in the notch, its panel opening, whichever way it opened…
  useEffect(() => {
    if (!inNotch) return;
    const seen = () => { if (document.visibilityState === "visible") shownAt.current = performance.now(); };
    const un = listen("neo://opened", seen);
    document.addEventListener("visibilitychange", seen);
    return () => { un.then((f) => f()).catch(() => {}); document.removeEventListener("visibilitychange", seen); };
  }, []);
  // …and whether this page wants an Esc pressed elsewhere: while it is on screen without the keys.
  // The core hears each change, and the state again after every (re)connect.
  useEffect(() => {
    if (!inNotch) return;
    let told: boolean | null = null;
    const sync = () => {
      const on = wantsAwayEsc({ visible: document.visibilityState === "visible", focused: document.hasFocus() });
      if (on !== told && neo.send({ cmd: "esc_watch", on })) told = on;
    };
    const soon = () => { window.setTimeout(sync, 0); }; // (hasFocus() follows focus / blur a moment later)
    sync();
    window.addEventListener("focus", soon);
    window.addEventListener("blur", soon);
    document.addEventListener("visibilitychange", sync);
    return () => {
      window.removeEventListener("focus", soon);
      window.removeEventListener("blur", soon);
      document.removeEventListener("visibilitychange", sync);
    };
  }, [neo.helloAt, neo.send]);

  const busy = !["idle", "listening"].includes(shown);
  // No model to ask yet: the chat shows "Give NEO a brain" (the composer stays for simple commands).
  const noBrain = needsBrain(neo.connected, neo.brains);
  const lastReply = [...neo.items].reverse().find((i) => i.kind === "msg" && i.role === "assistant") as Extract<Item, { kind: "msg" }> | undefined;

  // Replies while the page fills the card (focus, compact): a toast above the input — only replies
  // that arrive while the page is filling it, never one from before that the chat already showed.
  const [replyToast, setReplyToast] = useState<{ id: number; text: string } | null>(null);
  const seenReply = useRef<number>(0);
  const pageFills = layout === "focus" || compact;
  const fillsSince = useRef<number>(0); // the newest reply there was when the page took the card
  useEffect(() => { if (pageFills) fillsSince.current = lastReply?.id ?? 0; /* eslint-disable-line react-hooks/exhaustive-deps */ }, [pageFills]);
  useEffect(() => {
    if (!lastReply || !lastReply.final || lastReply.id === seenReply.current) return;
    seenReply.current = lastReply.id;
    if (pageFills && lastReply.id > fillsSince.current) setReplyToast({ id: lastReply.id, text: lastReply.text });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastReply?.id, lastReply?.final]);
  useEffect(() => {
    if (!replyToast) return;
    const t = window.setTimeout(() => setReplyToast(null), 4000);
    return () => clearTimeout(t);
  }, [replyToast]);
  useEffect(() => { if (layout !== "focus" && !compact) setReplyToast(null); }, [layout, compact]);

  // browser_* chips are the card's timeline now: hide them for jobs that have a card.
  const browserJobs = useMemo(() => new Set(neo.items.flatMap((i) => (i.kind === "browser" && i.job ? [i.job] : []))), [neo.items]);
  const latestBrowserSid = useMemo(() => { for (let i = neo.items.length - 1; i >= 0; i--) { const it = neo.items[i]; if (it.kind === "browser") return it.sid; } return null; }, [neo.items]);

  const liveHost = live?.live ? displayHost(live.live.url) : "";
  const orbBadge = camLive ? (
    <span className="orb-badge cam" aria-hidden>{camIcon(12)}<span className="orb-badge-dot" style={{ background: "var(--cam-rec)" }} /></span>
  ) : songHearing ? (
    <span className="orb-badge song" aria-hidden>{noteIcon(11)}</span>
  ) : live?.live ? (
    <span className={"orb-badge" + (live.live.status === "need_user" || live.live.status === "waiting" ? " need" : live.live.controller === "neo" && live.live.status === "working" ? " acting" : "")} aria-hidden>
      {I.globe(12)}{(live.live.status === "need_user" || live.live.status === "waiting") && <span className="orb-badge-dot" />}
    </span>
  ) : null;

  const devLayer = (
    <>
      {MockHud && <Suspense fallback={null}>{createPortal(<MockHud />, document.body)}</Suspense>}
      {SimPopout && simPopout && !inTauri && <Suspense fallback={null}>{createPortal(<SimPopout onClose={bringBack} />, document.body)}</Suspense>}
      {SongHud && <Suspense fallback={null}>{createPortal(<SongHud />, document.body)}</Suspense>}
    </>
  );

  if (!expanded) {
    const openLabel = camLive
      ? "Open NEO — the camera is on"
      : songHearing
      ? "Open NEO — listening for a song"
      : live?.live
      ? live.live.status === "need_user" ? `Open NEO — your turn on ${liveHost}` : `Open NEO — browsing ${liveHost}`
      : neo.connected ? `Open NEO — ${LABEL[shown]}` : "NEO — core offline";
    return (
      <>
        <div className="pill" data-tauri-drag-region role="button" tabIndex={0} aria-label={openLabel} title={neo.connected ? "Click to open · drag to move · ⌥⌘" : "NEO core is offline"} onMouseDown={onOrbDown} onClick={onOrbClick} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); expand(); } }}>
          {/* Tauri only starts a window drag from the element that carries the attribute, so it goes on the canvas too. */}
          <ThinkingOrb state={orb} size={64} theme="dark" speed={ORB_SPEED} {...({ "data-tauri-drag-region": true } as object)} />
          {!neo.connected && <span className="offline-dot" />}
          {neo.connected && orbBadge}
        </div>
        {devLayer}
      </>
    );
  }

  if (lensLive && lcard) {
    // Search Screen's card fills the panel; the chat comes back as it closes.
    return (
      <>
        <div className="card layout-lens snap" style={{ width: cardSize.w, height: cardSize.h }}>
          <LensCard hub={neo.lensCard} snap={lcard} onNatural={setLensNatural} />
        </div>
        {devLayer}
      </>
    );
  }

  const renderItem = (it: Item, idx: number, strip: boolean) => {
    if (confirming && it.kind === "msg" && it === lastReply && idx === neo.items.length - 1) return null;
    if (it.kind === "browser") {
      if (strip) return null;
      const e = bsnap.ended[it.sid];
      if (!e) return null;
      return <EndedRow key={it.id} e={e} landing={landingSid === it.sid} canReopen={it.sid === latestBrowserSid && !live} onReopen={() => void hub.controlSid(it.sid, "reopen").then((r) => { if (!r.ok && r.code === "no_session") hub.pageGone(it.sid); })} />;
    }
    if (it.kind === "camera") {
      if (strip) return null;
      const e = csnap.ended[it.sid];
      return e ? <CameraEndedRow key={it.id} e={e} /> : null;
    }
    if (it.kind === "song") {
      return songSid === it.sid || sshape.ended[it.sid] ? <SongItem key={it.id} hub={song} sid={it.sid} mini={strip} /> : null;
    }
    if (it.kind === "tools") {
      const tools = it.job && browserJobs.has(it.job) ? it.tools.filter((t) => !t.name.startsWith("browser")) : it.tools;
      return tools.length ? <Tools key={it.id} tools={tools} /> : null;
    }
    if (it.role === "user") {
      return (
        <div key={it.id} className={"msg user" + (it.final ? "" : " partial")}>
          {it.attach === "screen" && <span className="attach-chip" title="An area of your screen, from Search Screen">Screen area</span>}
          {it.text}
        </div>
      );
    }
    return (
      <div key={it.id} id={`msg-${it.id}`} className={"msg assistant" + (it.final ? "" : " partial")}>
        <Md text={it.text} />
        {it.meta && !strip && <Meta m={it.meta} />}
      </div>
    );
  };
  const visibleItems = neo.items.filter((i) => (i.kind !== "browser" || bsnap.ended[i.sid]) && (i.kind !== "song" || songSid === i.sid || sshape.ended[i.sid]));

  // The notch's header is one slim line whatever the layout: the small orb, NEO, its state.
  const notchStatus = !neo.connected ? "Reconnecting…" : confirming ? LABEL.confirming : layout === "docked" ? browsingLine(card?.live ?? null) : LABEL[shown];
  const headActions = (
    <div className="head-actions">
      <button className={"icon" + (pinned ? " on" : "")} title={pinned ? "Unpin (auto-hide when idle)" : "Pin open"} aria-label={pinned ? "Unpin" : "Pin open"} aria-pressed={pinned} onClick={() => setPinned((p) => !p)}>{Icon.pin}</button>
      <button className="icon" title="Clear transcript" aria-label="Clear transcript" onClick={neo.clear}>{Icon.broom}</button>
      <button className="icon" title="Collapse (Esc)" aria-label="Collapse" onClick={collapse}>{Icon.collapse}</button>
    </div>
  );
  const header = inNotch ? (
    <header className="head notch">
      <div className="orb xs"><ThinkingOrb state={orb} size={32} theme="dark" speed={ORB_SPEED} /></div>
      <div className="head-text">
        <span className="title">NEO</span>
        <span className={"status" + (neo.connected ? "" : " off")} aria-live="polite">
          {notchStatus}
          {neo.connected && neo.note && <span className="note-inline"> · {neo.note}</span>}
        </span>
      </div>
      {headActions}
    </header>
  ) : layout === "docked" ? (
    <header className="head compact" data-tauri-drag-region>
      <div className="orb sm" data-tauri-drag-region><ThinkingOrb state={orb} size={32} theme="auto" speed={ORB_SPEED} {...({ "data-tauri-drag-region": true } as object)} /></div>
      <div className="head-text" data-tauri-drag-region>
        <div className="title" data-tauri-drag-region>NEO</div>
        <div className={"status" + (neo.connected ? "" : " off")} data-tauri-drag-region aria-live="polite">
          {neo.connected ? (confirming ? LABEL.confirming : browsingLine(card?.live ?? null)) : "Core offline — reconnecting…"}
          {neo.note && <span className="note-inline"> · {neo.note}</span>}
        </div>
      </div>
      <div className="head-actions">
        <button className={"icon" + (pinned ? " on" : "")} title={pinned ? "Unpin (auto-hide when idle)" : "Pin open"} aria-label={pinned ? "Unpin" : "Pin open"} aria-pressed={pinned} onClick={() => setPinned((p) => !p)}>{Icon.pin}</button>
        <button className="icon" title="Clear transcript" aria-label="Clear transcript" onClick={neo.clear}>{Icon.broom}</button>
        <button className="icon" title="Collapse (Esc)" aria-label="Collapse" onClick={collapse}>{Icon.collapse}</button>
      </div>
    </header>
  ) : layout === "chat" ? (
    <header className="head" data-tauri-drag-region>
      <div className="orb" data-tauri-drag-region><ThinkingOrb state={orb} size={64} theme="auto" speed={ORB_SPEED} gravity {...({ "data-tauri-drag-region": true } as object)} /></div>
      <div className="head-text" data-tauri-drag-region>
        <div className="title" data-tauri-drag-region>NEO</div>
        <div className={"status" + (neo.connected ? "" : " off")} data-tauri-drag-region aria-live="polite">
          {neo.connected ? LABEL[shown] : hasShell ? "Core not running — NEO's menu bar → Restart NEO" : "core offline — run  python -m neo2"}
        </div>
        {neo.note && <div className="note" data-tauri-drag-region aria-live="polite">{neo.note}</div>}
      </div>
      <div className="head-actions">
        <button className={"icon" + (pinned ? " on" : "")} title={pinned ? "Unpin (auto-hide when idle)" : "Pin open"} aria-label={pinned ? "Unpin" : "Pin open"} aria-pressed={pinned} onClick={() => setPinned((p) => !p)}>{Icon.pin}</button>
        <button className="icon" title="Clear transcript" aria-label="Clear transcript" onClick={neo.clear}>{Icon.broom}</button>
        <button className="icon" title="Collapse (Esc)" aria-label="Collapse" onClick={collapse}>{Icon.collapse}</button>
      </div>
    </header>
  ) : null;

  const replyToastEl = replyToast ? (
    <button key={replyToast.id} className="reply-toast" onClick={() => {
      setReplyToast(null);
      if (layout === "focus") setBmode("docked");
      window.setTimeout(() => document.getElementById(`msg-${replyToast.id}`)?.scrollIntoView({ block: "nearest" }), MORPH_MS + 60);
    }}>
      <span className="reply-toast-who">NEO</span><span className="reply-toast-text">{replyToast.text}</span>
    </button>
  ) : null;

  const stage = card && (layout === "docked" || layout === "focus") ? (
    <div className={"stage" + (folding ? " folding" : "") + (layout === "focus" ? " focus" : "")}
      onPointerEnter={onStagePointer} onPointerMove={onStagePointer} onPointerLeave={onStagePointer}>
      <BrowserCard
        hub={hub}
        snap={card}
        connected={neo.connected}
        surface={layout === "focus" ? "focus" : "docked"}
        reduced={reduced}
        onExpand={toggleFocus}
        onMinimize={toggleMini}
        onPopout={() => void popOut()}
        focusSignal={viewFocus}
        overlay={replyToastEl}
      />
    </div>
  ) : null;

  const composer = erasing && !neo.ask ? (
    <div className="confirm" role="alertdialog" aria-labelledby="erase-q">
      <div className="confirm-q" id="erase-q">Erase everything NEO remembers about you?</div>
      <div className="confirm-detail">Memories and preferences, past-session summaries, learned requests, the activity log, and logins in NEO's own browser. Your settings and API keys stay. This can't be undone.</div>
      {!neo.connected && <div className="note">NEO's core isn't running, so nothing can be erased right now — menu bar → Restart NEO, then try again.</div>}
      <div className="confirm-btns">
        <button
          className="primary danger-fill"
          disabled={!neo.connected || erasing === "sent"}
          onClick={() => { if (neo.send({ cmd: "forget_all" })) setErasing("sent"); }}
        >
          {erasing === "sent" ? "Erasing…" : "Erase everything"}
        </button>
        <button onClick={() => setErasing(false)} disabled={erasing === "sent"}>Cancel <kbd>esc</kbd></button>
      </div>
    </div>
  ) : neo.ask ? (
    <div className="confirm" role="alertdialog" aria-labelledby="confirm-q" aria-describedby="confirm-detail">
      {inNotch && <div className="confirm-eyebrow" aria-hidden>Needs your OK</div>}
      <div className="confirm-q" id="confirm-q">NEO wants to {neo.ask.summary}.</div>
      {neo.ask.detail && <pre className="confirm-detail" id="confirm-detail">{neo.ask.detail}</pre>}
      <div className={offersTask ? "confirm-btns scoped" : "confirm-btns"}>
        <button className="primary" onClick={() => answer(true)}>{inNotch ? "Allow" : "Yes, do it"} <kbd>⌘⏎</kbd></button>
        {offersTask && (
          <button
            className="task"
            onClick={() => answer(true, "task")}
            title={`Also run the next steps like this${neo.ask.app ? ` in ${neo.ask.app}` : ""} without asking, until this task ends. Anything risky still asks.`}
          >
            <span className="task-label">
              {inNotch ? "Allow for this task" : "Yes for this task"}{neo.ask.app ? <>{" "}<span className="task-app">({neo.ask.app})</span></> : null}
            </span>
            <kbd>⌘⇧⏎</kbd>
          </button>
        )}
        <button onClick={() => answer(false)}>{inNotch ? "Cancel" : "No"} <kbd>esc</kbd></button>
      </div>
    </div>
  ) : (
    <div className="input-row">
      <input
        ref={inputRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        placeholder={!neo.connected && inNotch ? "Reconnecting…" : card ? "Tell NEO what to do next…" : busy ? "Working… type to queue" : "Ask NEO to do anything…"}
        spellCheck={false}
        aria-label="Message NEO"
      />
      {text && !inNotch ? (
        <button className="icon accent" title="Send" aria-label="Send" onClick={() => submit()}>{Icon.send}</button>
      ) : inNotch && busy ? (
        <button className="icon danger" title="Stop" aria-label="Stop" onClick={() => neo.send({ cmd: "stop" })}>{Icon.stop}</button>
      ) : (
        <button
          className={"icon" + (neo.state === "listening" ? " live" : "")}
          title="Hold to talk (or hold Space)"
          aria-label="Hold to talk"
          aria-pressed={neo.state === "listening"}
          onMouseDown={() => neo.send({ cmd: "ptt", on: true })}
          onMouseUp={() => neo.send({ cmd: "ptt", on: false })}
          onMouseLeave={() => neo.send({ cmd: "ptt", on: false })}
          onKeyDown={(e) => { if (e.key === " " && !e.repeat) { e.preventDefault(); neo.send({ cmd: "ptt", on: true }); } }}
          onKeyUp={(e) => { if (e.key === " ") { e.preventDefault(); neo.send({ cmd: "ptt", on: false }); } }}
          onBlur={() => neo.send({ cmd: "ptt", on: false })}
        >
          {Icon.mic}
        </button>
      )}
      {inNotch && <button className={"icon send" + (text ? " accent" : "")} title="Send (Enter)" aria-label="Send" disabled={!text} onClick={() => submit()}>{Icon.send}</button>}
      {busy && !inNotch && <button className="icon danger" title="Stop" aria-label="Stop" onClick={() => neo.send({ cmd: "stop" })}>{Icon.stop}</button>}
    </div>
  );

  // While a question is up in docked mode the confirm card takes the strip's place, so the page keeps its height.
  const showStrip = layout === "docked" && !compact && !neo.ask && !erasing;
  return (
    <>
      <div
        className={`card layout-${layout}` + (compact ? " compact" : "") + (snapSize ? " snap" : "") + (foldFx ? ` fold-${foldFx}` : "")}
        style={{ width: cardSize.w, height: cardSize.h }}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        onTransitionEnd={(e) => { if (e.target === e.currentTarget) setSnapSize(true); }}
      >
        {header}
        {stage}
        {(layout === "chat" || showStrip) && (
          <div className={"log" + (showStrip ? " strip" : "")} ref={logRef} onScroll={onScroll} role="log" aria-live="polite">
            {layout === "chat" && camLive && !csnap.popped && <CameraCard key={camLive.sid} hub={cam} live={camLive} stopping={csnap.stopping} surface="chat" onPopout={inTauri ? () => void camPopOut() : undefined} />}
            {layout === "chat" && camLive && csnap.popped && <CameraPoppedRow hub={cam} live={camLive} stopping={csnap.stopping} onShow={() => void showCameraWindow()} onBringBack={camBringBack} />}
            {layout === "chat" && live && bmode === "mini" && <MiniRow hub={hub} snap={live} onExpand={toggleMini} onStop={() => void hub.control("stop").then((r) => r.ok && hub.toast("Stopped. You're in control.", 2500))} />}
            {layout === "chat" && live && bmode === "popped" && <PoppedRow hub={hub} snap={live} onShow={showWindow} onBringBack={bringBack} />}
            {neo.connected && !showStrip && neo.problems.filter((p) => !(noBrain && isNoModelProblem(p))).map((p) => <div key={p} className="msg system problem" role="alert">{p}</div>)}
            {noBrain && !showStrip && !live && visibleItems.length === 0 && <Onboarding send={neo.send} openKeys={neo.openKeys} setup={neo.setup} />}
            {!noBrain && visibleItems.length === 0 && !showStrip && !live && (
              <div className="empty">
                <div className="empty-hint">Say <b>“Neo”</b>, or try one of these</div>
                <div className="suggest">
                  {SUGGESTIONS.map((s) => <button key={s} onClick={() => submit(s)}>{s}</button>)}
                </div>
              </div>
            )}
            {neo.items.map((it, idx) => renderItem(it, idx, showStrip))}
            {noBrain && !showStrip && !live && visibleItems.length > 0 && <Onboarding tail send={neo.send} openKeys={neo.openKeys} setup={neo.setup} />}
            {keyGuard && !showStrip && (
              <div className="msg system keyguard" role="status">
                {KEY_TYPED}{" "}
                <button className="keyguard-open" onClick={() => { neo.send({ cmd: "open_keys" }); setKeyGuard(false); }}>Open Brains &amp; Keys</button>
              </div>
            )}
            {neo.error && <div className="msg system">{neo.error}</div>}
          </div>
        )}
        {layout !== "chat" && camLive && !csnap.popped && <CameraCard key={camLive.sid} hub={cam} live={camLive} stopping={csnap.stopping} surface="pip" onPopout={inTauri ? () => void camPopOut() : undefined} />}
        {layout !== "chat" && !showStrip && songSid && (
          <div className={"song-pip" + (camLive && !csnap.popped ? " below-cam" : "")}><SongItem hub={song} sid={songSid} mini floating /></div>
        )}
        {inNotch && !neo.connected && <div className="reconnect" role="status"><span className="reconnect-dot" aria-hidden />Reconnecting to NEO…</div>}
        {composer}
        <div className="hide-btn-hint" aria-hidden />
      </div>
      {devLayer}
    </>
  );
}

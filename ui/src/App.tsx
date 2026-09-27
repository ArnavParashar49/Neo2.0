import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ThinkingOrb } from "thinking-orbs";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { openUrl } from "@tauri-apps/plugin-opener";
import { MOCK, useNeo, type Item, type NeoState, type OrbState, type ToolRun, type TurnMeta } from "./ws";
import { useBrowser, type BrowserSnap } from "./browser/hub";
import { BrowserCard } from "./browser/BrowserCard";
import { EndedRow, MiniRow, PoppedRow } from "./browser/BrowserRow";
import { CARD, PILL, browserSizes } from "./browser/sizes";
import { browsingLine, displayHost } from "./browser/text";
import { useReducedMotion } from "./browser/motion";
import { closeBrowserWindow, openBrowserWindow, showBrowserWindow } from "./browser/popout";
import { I } from "./browser/icons";
import { inTauri, placeInitial, resizeAnchored, workArea, type Area, type Size } from "./win";
import "./App.css";
import "./browser/browser.css";

// Dev only (`?mock=browser`): the scripted core's control panel and the simulated pop-out window.
const MockHud = import.meta.env.DEV && MOCK ? lazy(() => import("./browser/mock/Hud")) : null;
const SimPopout = import.meta.env.DEV ? lazy(() => import("./browser/mock/SimWindow")) : null;

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
const MORPH_MS = 280;
type BMode = "docked" | "focus" | "mini" | "popped";

function brainName(m: TurnMeta): string {
  if (m.route === "quick") return "quick action · no model";
  const model = m.model || m.brain;
  const pretty = model
    .replace(/^openai\//, "")
    .replace(/^gemini-(\d+\.\d+)-flash$/, "Gemini $1 Flash")
    .replace(/^gpt-oss-120b$/, "Groq gpt-oss-120b")
    .replace(/^mlx-community\//, "")
    .replace(/^claude-/, "Claude ");
  return pretty || m.brain || "local";
}

function Meta({ m }: { m: TurnMeta }) {
  const secs = (m.ms / 1000).toFixed(m.ms < 10_000 ? 1 : 0) + " s";
  const tools = m.route === "agent" && m.tools ? ` · ${m.tools} tool${m.tools === 1 ? "" : "s"}` : "";
  return <div className="meta">{brainName(m)}{tools} · {secs}</div>;
}

function Tools({ tools }: { tools: ToolRun[] }) {
  return (
    <div className="tools">
      {tools.map((t, i) => (
        <span key={i} className={"chip" + (t.running ? " running" : t.ok === false ? " fail" : " ok")} title={t.summary || JSON.stringify(t.args ?? {})}>
          <span className="dot" />{t.name}
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
          <a href={href} onClick={(e) => { e.preventDefault(); if (href) (inTauri ? openUrl(href) : window.open(href, "_blank")); }}>{children}</a>
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
  const reduced = useReducedMotion();
  const [expanded, setExpanded] = useState(false);
  const [pinned, setPinned] = useState(false);
  // A pending question outranks everything: the card shows exactly what will run.
  const confirming = !!neo.ask;
  const shown = confirming ? "confirming" : neo.state;
  const orb = ORB[shown];
  const [text, setText] = useState("");
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

  // ---- the browser session's place in the overlay -------------------------------------------
  const [bmode, setBmode] = useState<BMode>("docked");
  const [area, setArea] = useState<Area | null>(null);
  const [simPopout, setSimPopout] = useState(false);
  const [viewFocus, setViewFocus] = useState(0);
  // Task end: the card lingers (frozen) while the pointer is on it, then folds away. Derived in
  // render, so there is never a frame between "session over" and "card lingering".
  const [folding, setFolding] = useState(false);
  const [foldedSid, setFoldedSid] = useState<string | null>(null);
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
  const target: Size = !expanded ? PILL : layout === "docked" ? sizes.browse : layout === "focus" ? sizes.focus : CARD;

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
    setExpanded(true);
    // Into the chat input — unless something in the card (a page prompt, a dialog) already has focus.
    setTimeout(() => { const a = document.activeElement; if (!a || a === document.body) inputRef.current?.focus(); }, 60);
  }, []);
  const collapse = useCallback(() => {
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
    if (!init.current) { init.current = placeInitial(PILL).then(() => { winSize.current = PILL; }); return; }
    const my = ++gen.current;
    const run = async () => {
      await init.current;
      if (my !== gen.current) return;
      const from = winSize.current;
      if (!expanded) { await resizeAnchored(PILL); winSize.current = PILL; return; }
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

  // Login / CAPTCHA / 2FA always surface — shown without stealing focus from other apps.
  const needUser = liveStatus === "need_user";
  useEffect(() => {
    if (!needUser) return;
    if (inTauri) void getCurrentWindow().show().catch(() => {});
    setExpanded(true);
    setBmode((m) => (m === "mini" ? "docked" : m));
    setViewFocus((n) => n + 1);
  }, [needUser]);

  // The pop-out window went away without "Bring back" (closed elsewhere): come home.
  const wasPopped = useRef(false);
  useEffect(() => {
    if (livePopped) { wasPopped.current = true; return; }
    if (wasPopped.current && bmode === "popped") { setBmode("docked"); setSimPopout(false); }
    wasPopped.current = false;
  }, [livePopped, bmode]);

  useEffect(() => {
    if (!inTauri) return;
    const un = listen("neo://browser-popin", () => { setBmode("docked"); setExpanded(true); });
    return () => { un.then((f) => f()).catch(() => {}); };
  }, []);

  // Session over: fold the card into its row. A finished task waits while the pointer rests on it.
  const hoverRef = useRef(false);
  hoverRef.current = hover;
  useEffect(() => {
    const prev = lastLive.current?.live;
    if (liveSid || !prev || foldedSid === prev.sid) return;
    const sid = prev.sid;
    const reason = bsnap.ended[sid]?.close.reason;
    const done = () => { setFoldedSid(sid); setFolding(false); setBmode("docked"); };
    if (bmode === "popped" || bmode === "mini" || !expanded) { void closeBrowserWindow(); setSimPopout(false); done(); return; }
    const t0 = performance.now();
    let timer: number | undefined;
    const wait = () => {
      if (reason === "done" && hoverRef.current && performance.now() - t0 < 10_000) { timer = window.setTimeout(wait, 250); return; }
      setFolding(true);
      timer = window.setTimeout(done, reduced ? 120 : 240);
    };
    wait();
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveSid]);

  // ---- modes -------------------------------------------------------------------------------------
  const toggleFocus = useCallback(() => { if (live) { setBmode((m) => (m === "focus" ? "docked" : "focus")); setExpanded(true); } }, [live]);
  const toggleMini = useCallback(() => { if (live) { setBmode((m) => (m === "mini" ? "docked" : "mini")); setExpanded(true); } }, [live]);
  const popOut = useCallback(async () => {
    if (!live?.live) return;
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
      if (!e.metaKey || !e.shiftKey || e.altKey || e.ctrlKey) return;
      const k = e.key.toLowerCase();
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
    const v = () => hub.visibilityChanged();
    document.addEventListener("visibilitychange", v);
    return () => document.removeEventListener("visibilitychange", v);
  }, [hub]);

  // Native summon (hotkey / tray) → expand + arm the voice session.
  useEffect(() => {
    if (!inTauri) return;
    const un = listen("neo://wake", () => { expand(); neo.send({ cmd: "wake" }); });
    return () => { un.then((f) => f()).catch(() => {}); };
  }, [neo.send, expand]);

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
    if (!inTauri) return;
    const un = listen("neo://erase", () => { expand(); setErasing("ask"); });
    return () => { un.then((f) => f()).catch(() => {}); };
  }, [expand]);

  // Erased: close the dialog, and forget the typed history too (↑ must not bring it back).
  useEffect(() => {
    if (!neo.erasedAt) return;
    setErasing(false);
    lastInput.current = "";
    setText("");
  }, [neo.erasedAt]);

  // Dismissed without an answer (collapsed, hidden): the question doesn't linger for next time.
  useEffect(() => { if (!expanded && erasing === "ask") setErasing(false); }, [expanded, erasing]);

  // Auto-collapse after a quiet stretch unless pinned, hovered, being typed into — or browsing
  // on stage, the user driving, or NEO waiting on them.
  const browsingHolds = !!live && (bmode === "docked" || bmode === "focus" || live.live!.controller === "user" || live.live!.status === "need_user");
  useEffect(() => {
    if (!expanded || pinned || hover || neo.state !== "idle" || confirming || erasing || browsingHolds) return;
    const t = window.setTimeout(() => {
      if (document.activeElement === inputRef.current && text) return;
      setExpanded(false);
    }, IDLE_COLLAPSE_MS);
    return () => clearTimeout(t);
  }, [expanded, pinned, hover, neo.state, neo.items, text, confirming, erasing, browsingHolds]);

  // Follow new content only when the user hasn't scrolled up.
  useEffect(() => {
    const el = logRef.current;
    if (el && stickToBottom.current) el.scrollTo({ top: el.scrollHeight, behavior: reduced ? "auto" : "smooth" });
  }, [neo.items, neo.state, layout, bsnap.ended, reduced]);

  const onScroll = () => {
    const el = logRef.current;
    if (el) stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };

  const submit = (t = text) => {
    const v = t.trim();
    if (!v) return;
    lastInput.current = v;
    stickToBottom.current = true;
    neo.send({ cmd: "text", text: v });
    setText("");
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") submit();
    else if (e.key === "Escape") { collapse(); }
    else if (e.key === "ArrowUp" && !text) setText(lastInput.current);
  };

  // Keys while a question is up: ⌘⏎ approves (a stray Enter meant for something else must not),
  // ⎋ declines. The answer names the exact action on the card.
  const answer = useCallback((yes: boolean) => {
    if (neo.ask) neo.send({ cmd: "confirm", yes, action_id: neo.ask.action_id });
  }, [neo.ask, neo.send]);
  useEffect(() => {
    if (!confirming) return;
    const h = (e: KeyboardEvent) => {
      if (e.key === "Enter" && e.metaKey) { e.preventDefault(); answer(true); }
      if (e.key === "Escape") { e.preventDefault(); answer(false); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [confirming, answer]);

  const busy = !["idle", "listening"].includes(shown);
  const lastReply = [...neo.items].reverse().find((i) => i.kind === "msg" && i.role === "assistant") as Extract<Item, { kind: "msg" }> | undefined;

  // Replies while the page fills the card (focus, compact): a toast above the input.
  const [replyToast, setReplyToast] = useState<{ id: number; text: string } | null>(null);
  const seenReply = useRef<number>(0);
  useEffect(() => {
    if (!lastReply || !lastReply.final || lastReply.id === seenReply.current) return;
    seenReply.current = lastReply.id;
    if (layout === "focus" || compact) setReplyToast({ id: lastReply.id, text: lastReply.text });
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
  const orbBadge = live?.live ? (
    <span className={"orb-badge" + (live.live.status === "need_user" || live.live.status === "waiting" ? " need" : live.live.controller === "neo" && live.live.status === "working" ? " acting" : "")} aria-hidden>
      {I.globe(12)}{(live.live.status === "need_user" || live.live.status === "waiting") && <span className="orb-badge-dot" />}
    </span>
  ) : null;

  const devLayer = (
    <>
      {MockHud && <Suspense fallback={null}>{createPortal(<MockHud />, document.body)}</Suspense>}
      {SimPopout && simPopout && !inTauri && <Suspense fallback={null}>{createPortal(<SimPopout onClose={bringBack} />, document.body)}</Suspense>}
    </>
  );

  if (!expanded) {
    const openLabel = live?.live
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

  const renderItem = (it: Item, idx: number, strip: boolean) => {
    if (confirming && it.kind === "msg" && it === lastReply && idx === neo.items.length - 1) return null;
    if (it.kind === "browser") {
      if (strip) return null;
      const e = bsnap.ended[it.sid];
      if (!e) return null;
      return <EndedRow key={it.id} e={e} canReopen={it.sid === latestBrowserSid && !live} onReopen={() => hub.controlSid(it.sid, "reopen")} />;
    }
    if (it.kind === "tools") {
      const tools = it.job && browserJobs.has(it.job) ? it.tools.filter((t) => !t.name.startsWith("browser")) : it.tools;
      return tools.length ? <Tools key={it.id} tools={tools} /> : null;
    }
    if (it.role === "user") return <div key={it.id} className={"msg user" + (it.final ? "" : " partial")}>{it.text}</div>;
    return (
      <div key={it.id} id={`msg-${it.id}`} className={"msg assistant" + (it.final ? "" : " partial")}>
        <Md text={it.text} />
        {it.meta && !strip && <Meta m={it.meta} />}
      </div>
    );
  };
  const visibleItems = neo.items.filter((i) => i.kind !== "browser" || bsnap.ended[i.sid]);

  const header = layout === "docked" ? (
    <header className="head compact" data-tauri-drag-region>
      <div className="orb sm" data-tauri-drag-region><ThinkingOrb state={orb} size={32} theme="dark" speed={ORB_SPEED} {...({ "data-tauri-drag-region": true } as object)} /></div>
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
      <div className="orb" data-tauri-drag-region><ThinkingOrb state={orb} size={64} theme="dark" speed={ORB_SPEED} gravity {...({ "data-tauri-drag-region": true } as object)} /></div>
      <div className="head-text" data-tauri-drag-region>
        <div className="title" data-tauri-drag-region>NEO</div>
        <div className={"status" + (neo.connected ? "" : " off")} data-tauri-drag-region aria-live="polite">
          {neo.connected ? LABEL[shown] : inTauri ? "Core not running — menu bar → Restart NEO" : "core offline — run  python -m neo"}
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

  const stage = card && (layout === "docked" || layout === "focus") ? (
    <div className={"stage" + (folding ? " folding" : "") + (layout === "focus" ? " focus" : "")}>
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
      <div className="confirm-q" id="confirm-q">NEO wants to {neo.ask.summary}.</div>
      {neo.ask.detail && <pre className="confirm-detail" id="confirm-detail">{neo.ask.detail}</pre>}
      <div className="confirm-btns">
        <button className="primary" onClick={() => answer(true)}>Yes, do it <kbd>⌘⏎</kbd></button>
        <button onClick={() => answer(false)}>No <kbd>esc</kbd></button>
      </div>
    </div>
  ) : (
    <div className="input-row">
      <input
        ref={inputRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        placeholder={card ? "Tell NEO what to do next…" : busy ? "Working… type to queue" : "Ask NEO to do anything…"}
        spellCheck={false}
      />
      {text ? (
        <button className="icon accent" title="Send" aria-label="Send" onClick={() => submit()}>{Icon.send}</button>
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
      {busy && <button className="icon danger" title="Stop" aria-label="Stop" onClick={() => neo.send({ cmd: "stop" })}>{Icon.stop}</button>}
    </div>
  );

  // While a question is up in docked mode the confirm card takes the strip's place, so the page keeps its height.
  const showStrip = layout === "docked" && !compact && !neo.ask && !erasing;
  return (
    <>
      <div
        className={`card layout-${layout}` + (compact ? " compact" : "") + (snapSize ? " snap" : "")}
        style={{ width: cardSize.w, height: cardSize.h }}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        onTransitionEnd={(e) => { if (e.target === e.currentTarget) setSnapSize(true); }}
      >
        {header}
        {stage}
        {(layout === "chat" || showStrip) && (
          <div className={"log" + (showStrip ? " strip" : "")} ref={logRef} onScroll={onScroll} role="log" aria-live="polite">
            {layout === "chat" && live && bmode === "mini" && <MiniRow hub={hub} snap={live} onExpand={toggleMini} onStop={() => void hub.control("stop").then((r) => r.ok && hub.toast("Stopped. You're in control.", 2500))} />}
            {layout === "chat" && live && bmode === "popped" && <PoppedRow hub={hub} snap={live} onShow={showWindow} onBringBack={bringBack} />}
            {neo.connected && !showStrip && neo.problems.map((p) => <div key={p} className="msg system problem" role="alert">{p}</div>)}
            {visibleItems.length === 0 && !showStrip && !live && (
              <div className="empty">
                <div className="empty-hint">Say <b>“Neo”</b>, or try one of these</div>
                <div className="suggest">
                  {SUGGESTIONS.map((s) => <button key={s} onClick={() => submit(s)}>{s}</button>)}
                </div>
              </div>
            )}
            {neo.items.map((it, idx) => renderItem(it, idx, showStrip))}
            {neo.error && <div className="msg system">{neo.error}</div>}
          </div>
        )}
        {replyToast && (
          <button key={replyToast.id} className="reply-toast" onClick={() => {
            setReplyToast(null);
            if (layout === "focus") setBmode("docked");
            window.setTimeout(() => document.getElementById(`msg-${replyToast.id}`)?.scrollIntoView({ block: "nearest" }), MORPH_MS + 60);
          }}>
            <span className="reply-toast-who">NEO</span><span className="reply-toast-text">{replyToast.text}</span>
          </button>
        )}
        {composer}
        <div className="hide-btn-hint" aria-hidden />
      </div>
      {devLayer}
    </>
  );
}

// Hide the window fully from the tray / app menu (kept for the Rust side).
export const hideWindow = () => invoke("hide_window").catch(() => {});

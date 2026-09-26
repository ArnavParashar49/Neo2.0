import { useCallback, useEffect, useRef, useState } from "react";
import { ThinkingOrb } from "thinking-orbs";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow, currentMonitor, primaryMonitor, LogicalPosition, LogicalSize } from "@tauri-apps/api/window";
import { openUrl } from "@tauri-apps/plugin-opener";
import { useNeo, type Item, type NeoState, type OrbState, type ToolRun, type TurnMeta } from "./ws";
import "./App.css";

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

const CARD = { w: 400, h: 600 };
const PILL = { w: 84, h: 84 }; // just the floating orb
const IDLE_COLLAPSE_MS = 40_000;
const inTauri = "__TAURI_INTERNALS__" in window;
type Size = { w: number; h: number };

async function workArea() {
  const mon = (await currentMonitor()) ?? (await primaryMonitor());
  if (!mon) return null;
  const s = mon.scaleFactor;
  // Work area excludes the Dock and menu bar so nothing ends up hidden behind them.
  const area = (mon as unknown as { workArea?: { position: { x: number; y: number }; size: { width: number; height: number } } }).workArea;
  const pos = area?.position ?? mon.position, dim = area?.size ?? mon.size;
  return { x: pos.x / s, y: pos.y / s, w: dim.width / s, h: dim.height / s };
}

/** First placement: bottom-right corner of the work area. */
async function placeInitial(size: Size) {
  if (!inTauri) return;
  try {
    const a = await workArea();
    if (!a) return;
    const win = getCurrentWindow();
    await win.setSize(new LogicalSize(size.w, size.h));
    await win.setPosition(new LogicalPosition(a.x + a.w - size.w - 20, a.y + a.h - size.h - 20));
  } catch { /* not inside Tauri */ }
}

/** Resize keeping the window's bottom-right corner where it is (so the card unfolds from the orb,
 *  and the orb lands back where the card was), clamped to the work area. */
async function resizeAnchored(size: Size) {
  if (!inTauri) return;
  try {
    const win = getCurrentWindow();
    const s = await win.scaleFactor();
    const pos = await win.outerPosition();
    const cur = await win.outerSize();
    const right = pos.x / s + cur.width / s, bottom = pos.y / s + cur.height / s;
    let x = right - size.w, y = bottom - size.h;
    const a = await workArea();
    if (a) {
      x = Math.min(Math.max(x, a.x + 8), a.x + a.w - size.w - 8);
      y = Math.min(Math.max(y, a.y + 8), a.y + a.h - size.h - 8);
    }
    await win.setSize(new LogicalSize(size.w, size.h));
    await win.setPosition(new LogicalPosition(x, y));
  } catch { /* not inside Tauri */ }
}

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
  const [expanded, setExpanded] = useState(false);
  const [pinned, setPinned] = useState(false);
  const orb = ORB[neo.state];
  const [text, setText] = useState("");
  const [hover, setHover] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const lastInput = useRef("");
  const init = useRef<Promise<void> | null>(null);
  const press = useRef<{ x: number; y: number } | null>(null);

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
    setTimeout(() => inputRef.current?.focus(), 60);
  }, []);

  useEffect(() => {
    // First mount: park the orb bottom-right. Every later change resizes around the orb's own
    // corner — after the initial placement has finished, so a StrictMode double-run can't race it.
    if (!init.current) { init.current = placeInitial(PILL); return; }
    init.current.then(() => resizeAnchored(expanded ? CARD : PILL));
  }, [expanded]);


  // Native summon (hotkey / tray) → expand + arm the voice session.
  useEffect(() => {
    if (!inTauri) return;
    const un = listen("neo://wake", () => { expand(); neo.send({ cmd: "wake" }); });
    return () => { un.then((f) => f()).catch(() => {}); };
  }, [neo.send, expand]);

  // A confirmation request always surfaces the card.
  useEffect(() => { if (neo.state === "confirming") expand(); }, [neo.state, expand]);

  // Auto-collapse after a quiet stretch unless pinned, hovered, or being typed into.
  useEffect(() => {
    if (!expanded || pinned || hover || neo.state !== "idle") return;
    const t = window.setTimeout(() => {
      if (document.activeElement === inputRef.current && text) return;
      setExpanded(false);
    }, IDLE_COLLAPSE_MS);
    return () => clearTimeout(t);
  }, [expanded, pinned, hover, neo.state, neo.items, text]);

  // Follow new content only when the user hasn't scrolled up.
  useEffect(() => {
    const el = logRef.current;
    if (el && stickToBottom.current) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [neo.items, neo.state]);

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
    else if (e.key === "Escape") { setExpanded(false); }
    else if (e.key === "ArrowUp" && !text) setText(lastInput.current);
  };

  // Global keys while the card is up: ⏎ / ⎋ answer a confirmation.
  useEffect(() => {
    if (neo.state !== "confirming") return;
    const h = (e: KeyboardEvent) => {
      if (e.key === "Enter") { e.preventDefault(); neo.send({ cmd: "confirm", yes: true }); }
      if (e.key === "Escape") { e.preventDefault(); neo.send({ cmd: "confirm", yes: false }); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [neo.state, neo.send]);

  const busy = !["idle", "listening"].includes(neo.state);
  const lastReply = [...neo.items].reverse().find((i) => i.kind === "msg" && i.role === "assistant") as Extract<Item, { kind: "msg" }> | undefined;
  const question = neo.state === "confirming" ? lastReply?.text : undefined;

  if (!expanded) {
    return (
      <div className="pill" data-tauri-drag-region title={neo.connected ? "Click to open · drag to move · ⌥⌘" : "NEO core is offline"} onMouseDown={onOrbDown} onClick={onOrbClick}>
        {/* Tauri only starts a window drag from the element that carries the attribute, so it goes on the canvas too. */}
        <ThinkingOrb state={orb} size={64} theme="dark" speed={ORB_SPEED} {...({ "data-tauri-drag-region": true } as object)} />
        {!neo.connected && <span className="offline-dot" />}
      </div>
    );
  }

  return (
    <div className="card" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
      <header className="head" data-tauri-drag-region>
        <div className="orb" data-tauri-drag-region><ThinkingOrb state={orb} size={64} theme="dark" speed={ORB_SPEED} gravity {...({ "data-tauri-drag-region": true } as object)} /></div>
        <div className="head-text" data-tauri-drag-region>
          <div className="title" data-tauri-drag-region>NEO</div>
          <div className={"status" + (neo.connected ? "" : " off")} data-tauri-drag-region>{neo.connected ? LABEL[neo.state] : "core offline — run  python -m neo"}</div>
          {neo.note && busy && <div className="note" data-tauri-drag-region>{neo.note}</div>}
        </div>
        <div className="head-actions">
          <button className={"icon" + (pinned ? " on" : "")} title={pinned ? "Unpin (auto-hide when idle)" : "Pin open"} onClick={() => setPinned((p) => !p)}>{Icon.pin}</button>
          <button className="icon" title="Clear transcript" onClick={neo.clear}>{Icon.broom}</button>
          <button className="icon" title="Collapse (Esc)" onClick={() => setExpanded(false)}>{Icon.collapse}</button>
        </div>
      </header>

      <div className="log" ref={logRef} onScroll={onScroll}>
        {neo.items.length === 0 && (
          <div className="empty">
            <div className="empty-hint">Say <b>“Hey Neo”</b>, or try one of these</div>
            <div className="suggest">
              {SUGGESTIONS.map((s) => <button key={s} onClick={() => submit(s)}>{s}</button>)}
            </div>
          </div>
        )}
        {neo.items.map((it, idx) =>
          (neo.state === "confirming" && it.kind === "msg" && it === lastReply && idx === neo.items.length - 1) ? null :
          it.kind === "tools" ? (
            <Tools key={it.id} tools={it.tools} />
          ) : it.role === "user" ? (
            <div key={it.id} className={"msg user" + (it.final ? "" : " partial")}>{it.text}</div>
          ) : (
            <div key={it.id} className={"msg assistant" + (it.final ? "" : " partial")}>
              <Md text={it.text} />
              {it.meta && <Meta m={it.meta} />}
            </div>
          ),
        )}
        {neo.error && <div className="msg system">{neo.error}</div>}
      </div>

      {neo.state === "confirming" ? (
        <div className="confirm">
          <div className="confirm-q">{question ?? "Go ahead?"}</div>
          <div className="confirm-btns">
            <button className="primary" onClick={() => neo.send({ cmd: "confirm", yes: true })}>Yes, do it <kbd>⏎</kbd></button>
            <button onClick={() => neo.send({ cmd: "confirm", yes: false })}>No <kbd>esc</kbd></button>
          </div>
        </div>
      ) : (
        <div className="input-row">
          <input
            ref={inputRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKey}
            placeholder={busy ? "Working… type to queue" : "Ask NEO to do anything…"}
            spellCheck={false}
          />
          {text ? (
            <button className="icon accent" title="Send" onClick={() => submit()}>{Icon.send}</button>
          ) : (
            <button
              className={"icon" + (neo.state === "listening" ? " live" : "")}
              title="Hold to talk"
              onMouseDown={() => neo.send({ cmd: "ptt", on: true })}
              onMouseUp={() => neo.send({ cmd: "ptt", on: false })}
              onMouseLeave={() => neo.send({ cmd: "ptt", on: false })}
            >
              {Icon.mic}
            </button>
          )}
          {busy && <button className="icon danger" title="Stop" onClick={() => neo.send({ cmd: "stop" })}>{Icon.stop}</button>}
        </div>
      )}
      <div className="hide-btn-hint" aria-hidden />
    </div>
  );
}

// Hide the window fully from the tray / app menu (kept for the Rust side).
export const hideWindow = () => invoke("hide_window").catch(() => {});

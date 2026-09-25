import { useEffect, useRef, useState } from "react";
import { ThinkingOrb } from "thinking-orbs";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow, currentMonitor, LogicalPosition, LogicalSize } from "@tauri-apps/api/window";
import { useNeo, type OrbState } from "./ws";
import "./App.css";

const LABEL: Record<OrbState, string> = {
  breathing: "Idle",
  listening: "Listening…",
  solving: "Thinking…",
  working: "Working…",
  searching: "Searching…",
  composing: "Speaking…",
  connecting: "Connecting…",
  shaping: "Needs your OK",
};

const CARD = { w: 400, h: 560 };
const PILL = { w: 220, h: 64 };

async function placeBottomRight(size: { w: number; h: number }) {
  try {
    const win = getCurrentWindow();
    const mon = await currentMonitor();
    if (!mon) return;
    const s = mon.scaleFactor;
    // Use the work area (excludes the Dock and menu bar) so the input row is never hidden.
    const area = (mon as unknown as { workArea?: { position: { x: number; y: number }; size: { width: number; height: number } } }).workArea;
    const pos = area?.position ?? mon.position, dim = area?.size ?? mon.size;
    await win.setSize(new LogicalSize(size.w, size.h));
    await win.setPosition(new LogicalPosition(pos.x / s + dim.width / s - size.w - 20, pos.y / s + dim.height / s - size.h - 20));
  } catch { /* not running inside Tauri (vite dev in a browser) */ }
}

export default function App() {
  const neo = useNeo();
  const [expanded, setExpanded] = useState(true);
  const [text, setText] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => { placeBottomRight(expanded ? CARD : PILL); }, [expanded]);

  useEffect(() => {
    if (!("__TAURI_INTERNALS__" in window)) return; // plain browser (vite dev) — no native events
    const un = listen("neo://wake", () => {
      setExpanded(true);
      neo.send({ cmd: "wake" });
      setTimeout(() => inputRef.current?.focus(), 50);
    });
    return () => { un.then((f) => f()).catch(() => {}); };
  }, [neo.send]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [neo.lines, neo.tool]);

  useEffect(() => {
    if (neo.state === "shaping") setExpanded(true);
  }, [neo.state]);

  const submit = () => {
    const t = text.trim();
    if (!t) return;
    neo.send({ cmd: "text", text: t });
    setText("");
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") submit();
    if (e.key === "Escape") { setExpanded(false); invoke("hide_window").catch(() => {}); }
  };

  const busy = !["breathing", "listening"].includes(neo.state);
  const last = [...neo.lines].reverse().find((l) => l.role === "assistant");

  if (!expanded) {
    return (
      <div className="pill" data-tauri-drag-region onClick={() => setExpanded(true)}>
        <ThinkingOrb state={neo.state} size={32} theme="dark" />
        <div className="pill-text">
          <div className="pill-title">NEO</div>
          <div className="pill-sub">{neo.connected ? LABEL[neo.state] : "offline"}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <header className="head" data-tauri-drag-region>
        <ThinkingOrb state={neo.state} size={64} theme="dark" gravity />
        <div className="head-text">
          <div className="title">NEO</div>
          <div className={"status" + (neo.connected ? "" : " off")}>{neo.connected ? LABEL[neo.state] : "core offline — run `python -m neo`"}</div>
          {neo.tool && (
            <div className={"tool" + (neo.tool.running ? " running" : neo.tool.ok === false ? " fail" : "")}>
              {neo.tool.running ? "▶" : neo.tool.ok === false ? "✕" : "✓"} {neo.tool.name}
              {!neo.tool.running && neo.tool.summary ? ` — ${neo.tool.summary.slice(0, 60)}` : ""}
            </div>
          )}
        </div>
        <button className="ghost" title="Collapse (Esc hides)" onClick={() => setExpanded(false)}>–</button>
      </header>

      <div className="log" ref={scrollRef}>
        {neo.lines.length === 0 && (
          <div className="hint">Say <b>“Hey Neo”</b>, press <b>⌘⇧Space</b>, or type below.</div>
        )}
        {neo.lines.map((l, i) => (
          <div key={i} className={"line " + l.role + (l.final ? "" : " partial")}>{l.text}</div>
        ))}
        {neo.error && <div className="line system">{neo.error}</div>}
      </div>

      {neo.state === "shaping" ? (
        <div className="confirm">
          <div className="confirm-q">{last?.text ?? "Go ahead?"}</div>
          <div className="confirm-btns">
            <button className="primary" onClick={() => neo.send({ cmd: "confirm", yes: true })}>Yes, do it</button>
            <button onClick={() => neo.send({ cmd: "confirm", yes: false })}>No</button>
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
          <button
            className={"mic" + (neo.state === "listening" ? " live" : "")}
            title="Push to talk (hold)"
            onMouseDown={() => neo.send({ cmd: "ptt", on: true })}
            onMouseUp={() => neo.send({ cmd: "ptt", on: false })}
            onMouseLeave={() => neo.send({ cmd: "ptt", on: false })}
          >
            ●
          </button>
          {busy && <button className="stop" title="Stop" onClick={() => neo.send({ cmd: "stop" })}>■</button>}
        </div>
      )}
    </div>
  );
}

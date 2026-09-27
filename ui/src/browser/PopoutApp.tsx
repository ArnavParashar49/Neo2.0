// index.html#/browser — the pop-out window: only the browser card, full size. Its red light,
// ⌘W and "Bring back" all return the session to the chat; none of them ends it.
import { useCallback, useEffect, useRef, useState } from "react";
import { ThinkingOrb } from "thinking-orbs";
import { emit } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useNeo } from "../ws";
import { useBrowser } from "./hub";
import { BrowserCard } from "./BrowserCard";
import { useReducedMotion } from "./motion";
import { displayHost } from "./text";
import { inTauri } from "../win";
import { rememberFrame } from "./popout";

export function PopoutApp({ onSimClose }: { onSimClose?: () => void }) {
  const neo = useNeo({ surface: "window" });
  const snap = useBrowser(neo.hub);
  const reduced = useReducedMotion();
  const hadLive = useRef(false);
  const [shown, setShown] = useState(!inTauri);
  const st = snap.live;
  const host = st ? displayHost(st.url) : "";

  const closeSelf = useCallback(() => {
    if (onSimClose) { onSimClose(); return; }
    if (inTauri) void getCurrentWindow().destroy();
  }, [onSimClose]);

  const bringBack = useCallback(async () => {
    if (neo.hub.sid) await neo.hub.control("popin");
    if (inTauri) await emit("neo://browser-popin").catch(() => {});
    closeSelf();
  }, [neo.hub, closeSelf]);

  // Visible after the first paint (no white flash), and titled for Mission Control.
  useEffect(() => {
    if (!inTauri || shown) return;
    const t = window.setTimeout(() => { void getCurrentWindow().show(); setShown(true); }, 400);
    return () => clearTimeout(t);
  }, [shown]);
  useEffect(() => { if (inTauri && host) void getCurrentWindow().setTitle(`NEO Browser — ${host}`).catch(() => {}); }, [host]);

  // Red traffic light: bring the session back rather than ending it.
  useEffect(() => {
    if (!inTauri) return;
    const w = getCurrentWindow();
    const un = w.onCloseRequested(async (e) => { e.preventDefault(); await bringBack(); });
    let t: number | undefined;
    const save = async () => {
      try {
        const s = await w.scaleFactor(), p = await w.outerPosition(), z = await w.innerSize();
        rememberFrame({ x: p.x / s, y: p.y / s, w: z.width / s, h: z.height / s });
      } catch { /* ignore */ }
    };
    const onChange = () => { clearTimeout(t); t = window.setTimeout(save, 400); };
    const a = w.onMoved(onChange), b = w.onResized(onChange);
    return () => { for (const u of [un, a, b]) u.then((f) => f()).catch(() => {}); clearTimeout(t); };
  }, [bringBack]);

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (e.metaKey && !e.altKey && !e.ctrlKey && ((k === "w" && !e.shiftKey) || (k === "o" && e.shiftKey))) {
        e.preventDefault(); e.stopPropagation(); void bringBack();
      }
    };
    window.addEventListener("keydown", h, true);
    return () => window.removeEventListener("keydown", h, true);
  }, [bringBack]);

  // The session ended (or was closed from the chat): this window has nothing left to show.
  useEffect(() => {
    if (st) { hadLive.current = true; return; }
    if (!hadLive.current) return;
    const t = window.setTimeout(closeSelf, 900);
    return () => clearTimeout(t);
  }, [st, closeSelf]);

  useEffect(() => {
    const v = () => neo.hub.visibilityChanged();
    document.addEventListener("visibilitychange", v);
    return () => document.removeEventListener("visibilitychange", v);
  }, [neo.hub]);

  if (!st) {
    return (
      <div className="popout empty" data-tauri-drag-region>
        <ThinkingOrb state={neo.connected ? "breathing" : "connecting"} size={64} theme="auto" />
        <div className="popout-empty-text">{neo.connected ? (hadLive.current ? "The browser session ended." : "Waiting for NEO's browser…") : "Connecting to NEO…"}</div>
      </div>
    );
  }
  return (
    <div className="popout">
      <BrowserCard hub={neo.hub} snap={snap} connected={neo.connected} surface="window" reduced={reduced}
        onPopout={() => void bringBack()} confirmElsewhere={!!neo.ask || st.status === "waiting"} />
    </div>
  );
}

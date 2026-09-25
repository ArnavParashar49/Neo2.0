// Tiny websocket client for the NEO core. Reconnects forever; buffers nothing.
import { useCallback, useEffect, useRef, useState } from "react";

export type OrbState =
  | "breathing" | "listening" | "solving" | "working" | "searching" | "composing" | "connecting" | "shaping";

export interface Line { role: "user" | "assistant" | "system"; text: string; final: boolean; ts: number }
export interface ToolEvent { name: string; ok?: boolean; summary?: string; running: boolean }

export interface NeoEvent { type: string; data: Record<string, any>; ts: number }

const URL = "ws://127.0.0.1:8765";

export function useNeo() {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<OrbState>("breathing");
  const [lines, setLines] = useState<Line[]>([]);
  const [tool, setTool] = useState<ToolEvent | null>(null);
  const [error, setError] = useState<string>("");
  const sock = useRef<WebSocket | null>(null);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;

    const apply = (ev: NeoEvent) => {
      switch (ev.type) {
        case "hello":
          setState(ev.data.state);
          setLines([]);
          for (const r of ev.data.recent ?? []) apply(r);
          break;
        case "state":
          setState(ev.data.state);
          if (ev.data.state !== "working") setTool((t) => (t?.running ? { ...t, running: false } : t));
          break;
        case "transcript": {
          const { role, text, final } = ev.data;
          setLines((prev) => {
            // Replace the most recent still-open line of this role (Live interleaves user/assistant partials).
            let i = prev.length - 1;
            while (i >= 0 && !(prev[i].role === role && !prev[i].final)) i--;
            if (i >= 0) {
              const next = [...prev];
              next[i] = { role, text, final, ts: ev.ts };
              return next;
            }
            return [...prev.slice(-80), { role, text, final, ts: ev.ts }];
          });
          break;
        }
        case "tool_start":
          setTool({ name: ev.data.name, running: true });
          break;
        case "tool_end":
          setTool({ name: ev.data.name, ok: ev.data.ok, summary: ev.data.summary, running: false });
          break;
        case "error":
          setError(String(ev.data.message ?? ""));
          break;
      }
    };

    const connect = () => {
      if (!alive) return;
      const ws = new WebSocket(URL);
      sock.current = ws;
      ws.onopen = () => { if (sock.current === ws) { setConnected(true); setError(""); } };
      ws.onmessage = (m) => { if (sock.current !== ws) return; try { apply(JSON.parse(m.data)); } catch { /* ignore */ } };
      ws.onclose = () => {
        // A stale socket (StrictMode double-mount, or a replaced connection) must not clobber the live one.
        if (sock.current !== ws) return;
        setConnected(false);
        sock.current = null;
        timer = window.setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => { alive = false; if (timer) clearTimeout(timer); sock.current?.close(); };
  }, []);

  const send = useCallback((cmd: Record<string, unknown>) => {
    if (sock.current?.readyState === WebSocket.OPEN) sock.current.send(JSON.stringify(cmd));
  }, []);

  return { connected, state, lines, tool, error, send };
}

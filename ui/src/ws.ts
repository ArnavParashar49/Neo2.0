// Websocket client for the NEO core. Turns the event stream into a transcript timeline:
// user bubbles, tool-activity groups, and assistant replies (streamed partials merge in place).
import { useCallback, useEffect, useRef, useState } from "react";

export type NeoState =
  | "idle" | "listening" | "thinking" | "working" | "searching" | "speaking" | "connecting" | "confirming";
export type OrbState =
  | "breathing" | "listening" | "solving" | "working" | "searching" | "composing" | "connecting" | "shaping" | "weaving";
const MIN_DWELL_MS = 350; // a state must show at least this long before the next one replaces it

export interface TurnMeta { route: string; brain: string; model: string; ms: number; tools: number }
export interface ToolRun { name: string; args?: Record<string, unknown>; ok?: boolean; summary?: string; running: boolean }
export type Item =
  | { kind: "msg"; id: number; role: "user" | "assistant" | "system"; text: string; final: boolean; ts: number; meta?: TurnMeta }
  | { kind: "tools"; id: number; tools: ToolRun[] };

export interface NeoEvent { type: string; data: Record<string, any>; ts: number }

const URL = "ws://127.0.0.1:8765";
let nextId = 1;

export function useNeo() {
  const [connected, setConnected] = useState(false);
  const [state, setStateNow] = useState<NeoState>("idle");
  const lastChange = useRef(0);
  const pending = useRef<number | undefined>(undefined);
  // Rate-limit transitions: rapid back-and-forth (speaking→listening→speaking) collapses into
  // whatever the state is once things settle, instead of strobing the orb.
  const setState = useCallback((next: NeoState) => {
    if (pending.current) { clearTimeout(pending.current); pending.current = undefined; }
    const wait = MIN_DWELL_MS - (Date.now() - lastChange.current);
    const apply = () => { lastChange.current = Date.now(); setStateNow(next); };
    if (wait <= 0) apply(); else pending.current = window.setTimeout(apply, wait);
  }, []);
  const [items, setItems] = useState<Item[]>([]);
  const [error, setError] = useState<string>("");
  const [note, setNote] = useState<string>("");
  const sock = useRef<WebSocket | null>(null);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;

    const apply = (ev: NeoEvent) => {
      switch (ev.type) {
        case "hello":
          setState(ev.data.state);
          setItems([]);
          for (const r of ev.data.recent ?? []) apply(r);
          break;
        case "state":
          setState(ev.data.state);
          if (["breathing", "listening", "shaping"].includes(ev.data.state)) setNote("");
          break;
        case "note":
          setNote(String(ev.data.text ?? ""));
          break;
        case "transcript": {
          const { role, text, final } = ev.data;
          setItems((prev) => {
            // Merge into the most recent still-open message of this role (Live interleaves roles).
            let i = prev.length - 1;
            while (i >= 0) {
              const it = prev[i];
              if (it.kind === "msg" && it.role === role && !it.final) break;
              i--;
            }
            if (i >= 0) {
              const next = [...prev];
              next[i] = { ...(prev[i] as Extract<Item, { kind: "msg" }>), text, final, ts: ev.ts };
              return next;
            }
            return [...prev.slice(-120), { kind: "msg", id: nextId++, role, text, final, ts: ev.ts }];
          });
          break;
        }
        case "tool_start":
          setItems((prev) => {
            const last = prev[prev.length - 1];
            const run: ToolRun = { name: ev.data.name, args: ev.data.args, running: true };
            if (last && last.kind === "tools") {
              const next = [...prev];
              next[next.length - 1] = { ...last, tools: [...last.tools, run] };
              return next;
            }
            return [...prev, { kind: "tools", id: nextId++, tools: [run] }];
          });
          break;
        case "tool_end":
          setItems((prev) => {
            for (let i = prev.length - 1; i >= 0; i--) {
              const it = prev[i];
              if (it.kind !== "tools") continue;
              const j = it.tools.findIndex((t) => t.running && t.name === ev.data.name);
              if (j < 0) continue;
              const tools = [...it.tools];
              tools[j] = { ...tools[j], running: false, ok: ev.data.ok, summary: ev.data.summary };
              const next = [...prev];
              next[i] = { ...it, tools };
              return next;
            }
            return prev;
          });
          break;
        case "turn":
          setItems((prev) => {
            for (let i = prev.length - 1; i >= 0; i--) {
              const it = prev[i];
              if (it.kind === "msg" && it.role === "assistant") {
                const next = [...prev];
                next[i] = { ...it, meta: ev.data as TurnMeta };
                return next;
              }
            }
            return prev;
          });
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
        if (sock.current !== ws) return; // stale socket (StrictMode double-mount) must not clobber the live one
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

  const clear = useCallback(() => setItems([]), []);

  return { connected, state, items, error, note, send, clear };
}

// Websocket client for the NEO core. Turns the event stream into a transcript timeline:
// user bubbles, tool-activity groups, and assistant replies (streamed partials merge in place).
import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { BrowserHub } from "./browser/hub";

export type NeoState =
  | "idle" | "listening" | "thinking" | "working" | "searching" | "speaking" | "connecting" | "confirming";
export type OrbState =
  | "breathing" | "listening" | "solving" | "working" | "searching" | "composing" | "connecting" | "shaping" | "weaving";
const MIN_DWELL_MS = 350; // a state must show at least this long before the next one replaces it

export interface TurnMeta { route: string; brain: string; model: string; ms: number; tools: number }
export interface ToolRun { name: string; args?: Record<string, unknown>; ok?: boolean; summary?: string; running: boolean }
export type Item =
  | { kind: "msg"; id: number; role: "user" | "assistant" | "system"; text: string; final: boolean; ts: number; meta?: TurnMeta; job?: string }
  | { kind: "tools"; id: number; tools: ToolRun[]; job?: string }
  /** Where a browser session began: its card, its live row, or (once ended) its summary row. */
  | { kind: "browser"; id: number; sid: string; job?: string; ts: number };

export interface NeoEvent { type: string; data: Record<string, any>; ts: number }
/** A question NEO is waiting on: exactly what will run if the user says yes. */
export interface ConfirmCard { action_id: string; tool: string; summary: string; detail: string }

const URL = "ws://127.0.0.1:8765";
let nextId = 1;
/** `?mock=browser` (dev builds only): a scripted in-page core instead of the real socket. */
export const MOCK = import.meta.env.DEV && new URLSearchParams(location.search).get("mock") === "browser";

export function useNeo(opts: { surface?: "overlay" | "window" } = {}) {
  const hubRef = useRef<BrowserHub | null>(null);
  if (!hubRef.current) {
    hubRef.current = new BrowserHub(opts.surface ?? "overlay");
    if (import.meta.env.DEV && MOCK) {
      hubRef.current.ignoreHidden = true;
      ((window as unknown as { neoHubs?: BrowserHub[] }).neoHubs ??= []).push(hubRef.current);
    }
  }
  const hub = hubRef.current;
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
  const [ask, setAsk] = useState<ConfirmCard | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [erasedAt, setErasedAt] = useState(0); // bumps when the core confirms an erase
  const sock = useRef<WebSocket | null>(null);
  const noteTimer = useRef<number | undefined>(undefined);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;

    const apply = (ev: NeoEvent) => {
      if (ev.type.startsWith("browser_")) {
        hub.handle(ev.type, ev.data);
        if (ev.type === "browser_open") {
          const { sid, job } = ev.data;
          setItems((prev) => (prev.some((i) => i.kind === "browser" && i.sid === sid) ? prev : [...prev, { kind: "browser", id: nextId++, sid, job, ts: ev.ts }]));
        } else if (ev.type === "browser_close" && ev.data.reason === "erased") {
          setItems((prev) => prev.filter((i) => i.kind !== "browser"));
        }
        return;
      }
      switch (ev.type) {
        case "hello":
          hub.beginHello();
          setState(ev.data.state);
          setItems([]);
          for (const r of ev.data.recent ?? []) apply(r);
          setAsk(ev.data.confirm ?? null);
          setProblems(ev.data.problems ?? []);
          hub.endHello(ev.data.browser);
          {
            // A live session whose browser_open fell out of the replay still needs its anchor.
            const sid = hub.sid;
            if (sid) setItems((prev) => (prev.some((i) => i.kind === "browser" && i.sid === sid) ? prev : [...prev, { kind: "browser", id: nextId++, sid, job: ev.data.browser?.live?.job ?? undefined, ts: ev.ts }]));
          }
          break;
        case "problems":
          setProblems(ev.data.items ?? []);
          break;
        case "erased":
          setItems([]);
          hub.clearEnded();
          setErasedAt(Date.now());
          break;
        case "confirm":
          setAsk(ev.data as ConfirmCard);
          break;
        case "confirm_done":
          setAsk((prev) => (prev && prev.action_id === ev.data.action_id ? null : prev));
          break;
        case "state":
          setState(ev.data.state);
          break;
        case "note":
          // A short-lived line under the title (timeouts, quiet results, fallbacks): shown whatever
          // NEO is doing, gone after a few seconds so it never shows up stale later.
          setNote(String(ev.data.text ?? ""));
          if (noteTimer.current) clearTimeout(noteTimer.current);
          noteTimer.current = window.setTimeout(() => setNote(""), 8000);
          break;
        case "transcript": {
          const { role, text, final, job } = ev.data;
          setItems((prev) => {
            // Replace the most recent still-open line of this role *and job* — several jobs (and
            // the voice session) can be streaming at the same time.
            let i = prev.length - 1;
            while (i >= 0) {
              const it = prev[i];
              if (it.kind === "msg" && it.role === role && !it.final && (it.job ?? "") === (job ?? "")) break;
              i--;
            }
            if (i >= 0) {
              const next = [...prev];
              next[i] = { ...(prev[i] as Extract<Item, { kind: "msg" }>), text, final, ts: ev.ts };
              return next;
            }
            // Trim old lines, but never a browser session's anchor (its row lives there).
            const keep = prev.length > 120 ? prev.filter((it, k) => k >= prev.length - 120 || it.kind === "browser") : prev;
            return [...keep, { kind: "msg", id: nextId++, role, text, final, ts: ev.ts, job }];
          });
          break;
        }
        case "tool_start":
          setItems((prev) => {
            const job = ev.data.job ?? "";
            const run: ToolRun = { name: ev.data.name, args: ev.data.args, running: true };
            // Group with this job's most recent tool row, as long as nothing of its own came after it.
            for (let i = prev.length - 1; i >= 0; i--) {
              const it = prev[i];
              if (it.kind === "msg" && (it.job ?? "") === job && it.role === "assistant" && it.final) break;
              if (it.kind === "tools" && (it.job ?? "") === job) {
                const next = [...prev];
                next[i] = { ...it, tools: [...it.tools, run] };
                return next;
              }
            }
            return [...prev, { kind: "tools", id: nextId++, tools: [run], job }];
          });
          break;
        case "tool_end":
          setItems((prev) => {
            for (let i = prev.length - 1; i >= 0; i--) {
              const it = prev[i];
              if (it.kind !== "tools" || (it.job ?? "") !== (ev.data.job ?? "")) continue;
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
              if (it.kind === "msg" && it.role === "assistant" && (it.job ?? "") === (ev.data.job ?? "")) {
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

    const connect = async () => {
      if (!alive) return;
      // The core only admits clients that know the token NEO.app keeps in ~/.neo/ws-token.
      let ws: WebSocket;
      if (import.meta.env.DEV && MOCK) {
        const { MockSocket } = await import("./browser/mock/socket");
        if (!alive) return;
        ws = new MockSocket() as unknown as WebSocket;
      } else {
        let token = "";
        try { token = await invoke<string>("ws_token"); } catch { /* outside NEO.app: no token, no access */ }
        if (!alive) return;
        ws = new WebSocket(`${URL}/?token=${encodeURIComponent(token)}`);
      }
      ws.binaryType = "arraybuffer"; // live page frames arrive as binary (§10.4)
      sock.current = ws;
      ws.onopen = () => { if (sock.current === ws) { setConnected(true); setError(""); hub.setConnected(true); } };
      ws.onmessage = (m) => {
        if (sock.current !== ws) return;
        if (typeof m.data !== "string") { if (m.data instanceof ArrayBuffer) hub.frame(m.data); return; }
        try { apply(JSON.parse(m.data)); } catch { /* ignore */ }
      };
      ws.onclose = () => {
        if (sock.current !== ws) return; // stale socket (StrictMode double-mount) must not clobber the live one
        setConnected(false);
        hub.setConnected(false);
        setAsk(null); // the next hello says whether it's still waiting
        sock.current = null;
        timer = window.setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => { alive = false; if (timer) clearTimeout(timer); sock.current?.close(); };
  }, []);

  /** Returns whether the command actually went out (the core may be restarting). */
  const send = useCallback((cmd: Record<string, unknown>): boolean => {
    if (sock.current?.readyState !== WebSocket.OPEN) return false;
    sock.current.send(JSON.stringify(cmd));
    return true;
  }, []);
  hub.setSender(send);

  // Clear transcript: rows, posters and thumbnails go too — but a live session keeps its anchor.
  const clear = useCallback(() => {
    const live = hub.sid;
    setItems((prev) => prev.filter((i) => i.kind === "browser" && i.sid === live));
    hub.clearEnded();
  }, [hub]);

  return { connected, state, items, error, note, ask, problems, erasedAt, send, clear, hub };
}

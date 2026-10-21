// Websocket client for the NEO core. Turns the event stream into a transcript timeline:
// user bubbles, tool-activity groups, and assistant replies (streamed partials merge in place).
import { useCallback, useEffect, useRef, useState } from "react";
import { uiHostChanged, wsToken } from "./shell";
import { type UiHost } from "./host";
import { BrowserHub } from "./browser/hub";
import { CameraHub } from "./camera/hub";
import { isCameraFrame } from "./camera/frames";
import { SongHub } from "./song/hub";
import type { BrainsSummary, SetupNeeded } from "./keys/types";

export type NeoState =
  | "idle" | "listening" | "thinking" | "working" | "searching" | "speaking" | "connecting" | "confirming";
export type OrbState =
  | "breathing" | "listening" | "solving" | "working" | "searching" | "composing" | "connecting" | "shaping" | "weaving";
const MIN_DWELL_MS = 350; // a state must show at least this long before the next one replaces it

/** `label`: the brain as the user reads it ("Claude Opus 5.5"), from the core; older cores send none. */
export interface TurnMeta { route: string; brain: string; model: string; label?: string; ms: number; tools: number }
/** `open_keys`'s answer: the core asked NEO.app's shell to open Brains & Keys, or why not. */
export interface OpenKeysResult { ok: boolean; error?: string; at: number }
export interface ToolRun { name: string; args?: Record<string, unknown>; ok?: boolean; summary?: string; running: boolean }
export type Item =
  /** `attach`: what the user's line carried besides their words — "screen": an area of the screen,
   *  from Search Screen's "Continue in NEO" (its pixels are never in the transcript). */
  | { kind: "msg"; id: number; role: "user" | "assistant" | "system"; text: string; final: boolean; ts: number; meta?: TurnMeta; job?: string; attach?: "screen" }
  | { kind: "tools"; id: number; tools: ToolRun[]; job?: string }
  /** Where a browser session began: its card, its live row, or (once ended) its summary row. */
  | { kind: "browser"; id: number; sid: string; job?: string; ts: number }
  /** Where a camera session began: its live card is pinned atop the chat; once off, a row with the answer. */
  | { kind: "camera"; id: number; sid: string; job?: string; ts: number }
  /** Where NEO listened for a song: the listening card, then the song it found (or one calm line). */
  | { kind: "song"; id: number; sid: string; job?: string; ts: number };

export interface NeoEvent { type: string; data: Record<string, any>; ts: number }
/** A question NEO is waiting on: exactly what will run if the user says yes. `scopes`: how else
 *  it may be answered — "task" = a yes for this task, which lets the next calls like it on `app`
 *  run without asking until the task ends (never a risky one). Older cores send neither. */
export interface ConfirmCard { action_id: string; tool: string; summary: string; detail: string; scopes?: string[]; app?: string }
/** How far a yes reaches: just this card, or (when the card offers it) the rest of the task. */
export type ConfirmScope = "task";

// Dev builds only: `VITE_NEO_WS_URL` / `VITE_NEO_WS_TOKEN` point `npm run dev` at a test core (its own port
// and scratch token), so a dev tab never talks to the NEO the user is running. Release builds use NEO.app's.
const DEV_URL = import.meta.env.DEV ? (import.meta.env.VITE_NEO_WS_URL as string | undefined) : undefined;
const DEV_TOKEN = import.meta.env.DEV ? (import.meta.env.VITE_NEO_WS_TOKEN as string | undefined) : undefined;
const URL = DEV_URL || "ws://127.0.0.1:8765";
let nextId = 1;
const MOCK_KIND = import.meta.env.DEV ? new URLSearchParams(location.search).get("mock") : null;
/** `?mock=browser` (dev builds only): a scripted in-page core instead of the real socket. */
export const MOCK = MOCK_KIND === "browser";
/** `?mock=song` (dev builds only): a scripted core that listens for songs (no microphone, no network). */
export const SONG_MOCK = MOCK_KIND === "song";

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
  const camRef = useRef<CameraHub | null>(null);
  if (!camRef.current) {
    camRef.current = new CameraHub(opts.surface ?? "overlay");
    if (import.meta.env.DEV && MOCK) ((window as unknown as { neoCams?: CameraHub[] }).neoCams ??= []).push(camRef.current);
  }
  const cam = camRef.current;
  const songRef = useRef<SongHub | null>(null);
  if (!songRef.current) {
    songRef.current = new SongHub();
    if (import.meta.env.DEV && SONG_MOCK) ((window as unknown as { neoSongs?: SongHub[] }).neoSongs ??= []).push(songRef.current);
  }
  const song = songRef.current;
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
  // NEO's brains as every client may know them (labels and booleans, never a key): the hello's
  // "brains", then each `brains` event. Null: an older core that doesn't say.
  const [brains, setBrains] = useState<BrainsSummary | null>(null);
  const [setup, setSetup] = useState<SetupNeeded | null>(null); // a request needed a model NEO hasn't got
  const [openKeys, setOpenKeys] = useState<OpenKeysResult | null>(null);
  const [erasedAt, setErasedAt] = useState(0); // bumps when the core confirms an erase
  // Where the core says NEO's UI lives: its own window, or the notch (NotchHub's panel). NEO.app's
  // shell follows it — with "notch" it keeps its window hidden and leaves the UI to the host.
  const [uiHost, setUiHost] = useState<UiHost>("window");
  const [helloAt, setHelloAt] = useState(0); // bumps when a (re)connect's replay has landed
  // Search Screen: "Continue in NEO" handed an area to the chat (the chat comes forward), and the
  // overlay is about to take its still (in the notch, the host's panel folds away first).
  const [lensContinueAt, setLensContinueAt] = useState(0);
  const [lensOpeningAt, setLensOpeningAt] = useState(0);
  const sock = useRef<WebSocket | null>(null);
  const noteTimer = useRef<number | undefined>(undefined);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const apply = (ev: NeoEvent) => {
      if (ev.type.startsWith("song_")) {
        // Its place in the transcript: where it opened, or (joined late) where its state first came.
        const opened = song.handle(ev.type, ev.data);
        const sid = ev.type === "song_open" ? ev.data?.sid : opened;
        if (typeof sid === "string" && sid) {
          const job = ev.type === "song_open" ? ev.data?.job : undefined;
          setItems((prev) => (prev.some((i) => i.kind === "song" && i.sid === sid) ? prev : [...prev, { kind: "song", id: nextId++, sid, job, ts: ev.ts }]));
        } else if (ev.type === "song_close" && ev.data?.reason === "erased") {
          setItems((prev) => prev.filter((i) => i.kind !== "song"));
        }
        return;
      }
      if (ev.type.startsWith("camera_")) {
        cam.handle(ev.type, ev.data);
        if (ev.type === "camera_open" && ev.data?.sid) {
          const { sid, job } = ev.data;
          setItems((prev) => (prev.some((i) => i.kind === "camera" && i.sid === sid) ? prev : [...prev, { kind: "camera", id: nextId++, sid, job, ts: ev.ts }]));
        } else if (ev.type === "camera_close" && ev.data?.reason === "erased") {
          setItems((prev) => prev.filter((i) => i.kind !== "camera"));
        }
        return;
      }
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
          cam.beginHello();
          setState(ev.data.state);
          if (ev.data.ui_host === "window" || ev.data.ui_host === "notch") { setUiHost(ev.data.ui_host); void uiHostChanged(ev.data.ui_host); }
          setItems([]);
          for (const r of ev.data.recent ?? []) apply(r);
          setAsk(ev.data.confirm ?? null);
          setProblems(ev.data.problems ?? []);
          setBrains(ev.data.brains && typeof ev.data.brains === "object" ? (ev.data.brains as BrainsSummary) : null);
          hub.endHello(ev.data.browser);
          cam.endHello(ev.data.camera);
          song.endHello();
          setHelloAt(Date.now());
          {
            // A live session whose browser_open fell out of the replay still needs its anchor.
            const sid = hub.sid;
            if (sid) setItems((prev) => (prev.some((i) => i.kind === "browser" && i.sid === sid) ? prev : [...prev, { kind: "browser", id: nextId++, sid, job: ev.data.browser?.live?.job ?? undefined, ts: ev.ts }]));
            const csid = cam.sid; // likewise a live camera session's
            if (csid) setItems((prev) => (prev.some((i) => i.kind === "camera" && i.sid === csid) ? prev : [...prev, { kind: "camera", id: nextId++, sid: csid, ts: ev.ts }]));
            const ssid = song.sid; // and a song card still listening
            if (ssid) setItems((prev) => (prev.some((i) => i.kind === "song" && i.sid === ssid) ? prev : [...prev, { kind: "song", id: nextId++, sid: ssid, ts: ev.ts }]));
          }
          break;
        case "problems":
          setProblems(ev.data.items ?? []);
          break;
        case "brains":
          setBrains(ev.data as BrainsSummary);
          if (ev.data?.ready) setSetup(null);
          break;
        case "setup_needed":
          setSetup({ shell: !!ev.data.shell, opened: !!ev.data.opened, command: String(ev.data.command ?? "") });
          break;
        case "open_keys_result":
          setOpenKeys({ ok: !!ev.data.ok, error: typeof ev.data.error === "string" ? ev.data.error : undefined, at: Date.now() });
          break;
        case "ui_host":
          if (ev.data.host === "window" || ev.data.host === "notch") { setUiHost(ev.data.host); void uiHostChanged(ev.data.host); }
          break;
        case "erased": {
          // A song card still listening keeps its place: its Stop must stay in reach.
          const ssid = song.sid;
          setItems(ssid ? [{ kind: "song", id: nextId++, sid: ssid, ts: ev.ts }] : []);
          hub.clearEnded();
          cam.clearEnded();
          song.clearEnded();
          setErasedAt(Date.now());
          break;
        }
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
          const attach = ev.data.attach === "screen" ? ("screen" as const) : undefined;
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
              next[i] = { ...(prev[i] as Extract<Item, { kind: "msg" }>), text, final, ts: ev.ts, ...(attach ? { attach } : {}) };
              return next;
            }
            // Trim old lines, but never a browser, camera or song session's anchor (its row lives there).
            const keep = prev.length > 120 ? prev.filter((it, k) => k >= prev.length - 120 || it.kind === "browser" || it.kind === "camera" || it.kind === "song") : prev;
            return [...keep, { kind: "msg", id: nextId++, role, text, final, ts: ev.ts, job, ...(attach ? { attach } : {}) }];
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
        case "lens_continue":
          setLensContinueAt(Date.now());
          break;
        case "lens_opening":
          setLensOpeningAt(Date.now());
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
      } else if (import.meta.env.DEV && SONG_MOCK) {
        const { SongMockSocket } = await import("./song/mock/core");
        if (!alive) return;
        ws = new SongMockSocket() as unknown as WebSocket;
      } else {
        let token = "";
        try { token = await wsToken(); } catch { /* outside NEO.app: no token, no access */ }
        if (!token && DEV_TOKEN) token = DEV_TOKEN;
        if (!alive) return;
        ws = new WebSocket(`${URL}/?token=${encodeURIComponent(token)}`);
      }
      ws.binaryType = "arraybuffer"; // live page frames arrive as binary (§10.4)
      sock.current = ws;
      ws.onopen = () => { if (sock.current === ws) { setConnected(true); setError(""); hub.setConnected(true); cam.setConnected(true); song.setConnected(true); } };
      ws.onmessage = (m) => {
        if (sock.current !== ws) return;
        if (typeof m.data !== "string") {
          // One socket, two cards: camera preview frames carry their own NBF1 kind.
          if (m.data instanceof ArrayBuffer) { if (isCameraFrame(m.data)) cam.frame(m.data); else hub.frame(m.data); }
          return;
        }
        try { apply(JSON.parse(m.data)); } catch { /* ignore */ }
      };
      ws.onclose = () => {
        if (sock.current !== ws) return; // stale socket (StrictMode double-mount) must not clobber the live one
        setConnected(false);
        hub.setConnected(false);
        cam.setConnected(false);
        song.setConnected(false);
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
  cam.setSender(send);
  song.setSender(send);

  // Clear transcript: rows, posters and thumbnails go too — but a live session keeps its anchor.
  const clear = useCallback(() => {
    const live = hub.sid, camLive = cam.sid, songLive = song.sid;
    setItems((prev) => prev.filter((i) => (i.kind === "browser" && i.sid === live) || (i.kind === "camera" && i.sid === camLive) || (i.kind === "song" && i.sid === songLive)));
    hub.clearEnded();
    cam.clearEnded();
    song.clearEnded();
  }, [hub, cam, song]);

  return { connected, state, items, error, note, ask, problems, brains, setup, openKeys, erasedAt, helloAt, lensContinueAt, lensOpeningAt, uiHost, send, clear, hub, cam, song };
}

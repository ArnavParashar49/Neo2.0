// The one door to the shell around the page. In NEO's own window that shell is Tauri (its
// commands and events); hosted in the notch it is NotchHub, reached through WKWebView's
// `window.webkit.messageHandlers.neo.postMessage` one way and `window.__neoHostEmit` the other.
// Nothing else in the UI imports `@tauri-apps/*` except the pop-out windows, which only exist
// in NEO's own app (./browser/popout.ts, ./camera/popout.ts); in notch mode no Tauri module is
// ever loaded — they are imported here lazily, and only on the window branch.
import { inNotch } from "./host.ts"; // the extension: node's test runner loads this module as is

/** Whether a shell is there at all (NEO.app, or a host): false in a plain dev tab. */
export const hasShell = inNotch || (typeof window !== "undefined" && "__TAURI_INTERNALS__" in window);

export type Unlisten = () => void;
type Handler = (payload?: unknown) => void;

const handlers = new Map<string, Set<Handler>>();

if (inNotch && typeof window !== "undefined") {
  window.__neoHostEmit = (event, payload) => {
    handlers.get(event)?.forEach((h) => {
      try { h(payload); } catch { /* one listener's error must not stop the rest */ }
    });
  };
}

/** A message to the host (notch mode only). Returns whether a host was there to take it. */
export function post(message: Record<string, unknown>): boolean {
  if (!inNotch) return false;
  const h = window.webkit?.messageHandlers?.neo;
  if (!h) return false;
  try { h.postMessage(message); return true; } catch { return false; }
}

/** The core's websocket token: injected by the host, or handed over by NEO.app. */
export async function wsToken(): Promise<string> {
  if (inNotch) return window.__NEO_WS_TOKEN__ ?? "";
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<string>("ws_token");
}

/** A shell event (`neo://wake`, `neo://erase`, …). */
export async function listen(event: string, cb: () => void): Promise<Unlisten> {
  if (inNotch) {
    const h: Handler = () => cb();
    let set = handlers.get(event);
    if (!set) { set = new Set(); handlers.set(event, set); }
    set.add(h);
    return () => { set.delete(h); };
  }
  const { listen } = await import("@tauri-apps/api/event");
  return listen(event, () => cb());
}

/** Surface the UI without taking focus from the app the user is in. */
export async function showQuiet(): Promise<void> {
  if (inNotch) { post({ type: "show" }); return; }
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("show_quiet");
  } catch {
    try { const { getCurrentWindow } = await import("@tauri-apps/api/window"); await getCurrentWindow().show(); } catch { /* not inside a shell */ }
  }
}

/** Put the UI away (the whole window, or the host's panel). */
export async function hideWindow(): Promise<void> {
  if (inNotch) { post({ type: "collapse" }); return; }
  try { const { invoke } = await import("@tauri-apps/api/core"); await invoke("hide_window"); } catch { /* not inside a shell */ }
}

/** Keys should reach the page: ask the shell for focus. */
export async function focusWindow(): Promise<void> {
  if (inNotch) { post({ type: "focus" }); return; }
  try { const { getCurrentWindow } = await import("@tauri-apps/api/window"); await getCurrentWindow().setFocus(); } catch { /* not inside a shell */ }
}

/** Open a link in the user's default browser (never inside NEO). */
export async function openExternal(href: string): Promise<void> {
  if (inNotch) { post({ type: "open_url", url: href }); return; }
  const { openUrl } = await import("@tauri-apps/plugin-opener");
  await openUrl(href);
}

/** Write the general pasteboard. False when the shell could not (the caller falls back). */
export async function clipboardWrite(text: string): Promise<boolean> {
  if (inNotch) return post({ type: "clipboard", text });
  try { const { invoke } = await import("@tauri-apps/api/core"); await invoke("clipboard_write", { text }); return true; } catch { return false; }
}

/** The page would like this much room (notch mode: the host sizes its panel; a no-op elsewhere). */
export function requestSize(w: number, h: number): void {
  post({ type: "size", w, h });
}

/** The user pinned the panel open (notch mode). */
export function pinHost(on: boolean): void {
  post({ type: "pin", on });
}

/** The core said where the UI should live; NEO.app's shell hides or shows its window accordingly. */
export async function uiHostChanged(host: string): Promise<void> {
  if (inNotch) return;
  try { const { invoke } = await import("@tauri-apps/api/core"); await invoke("ui_host_changed", { host }); } catch { /* an older shell */ }
}

// Where NEO's UI is running: its own Tauri window (the default), or hosted inside another app's
// panel — NotchHub's island, which loads the built page from `neo://app/` in a WKWebView and
// injects `window.__NEO_HOST__ = "notch"` before any script runs (a `?host=notch` query works
// too, for a dev tab). In notch mode there is no window of NEO's own to move, size, show or hide:
// the host owns the frame, and every shell call goes through ./shell.ts instead of Tauri.

export type UiHost = "window" | "notch";

declare global {
  interface Window {
    /** Set by the host before the page's scripts run. */
    __NEO_HOST__?: string;
    /** The core's websocket token, injected by the host (NEO.app hands it over `invoke` instead). */
    __NEO_WS_TOKEN__?: string;
    /** The largest panel the host will give the page, in CSS pixels. */
    __NEO_HOST_AREA__?: { w: number; h: number };
    /** The host calls this (by evaluating JavaScript) to deliver an event such as `neo://wake`;
     *  `neo://show` — the host opened its panel at the user's hand, so the input should take the keys;
     *  or `neo://opened` — the host's panel opened, whichever way (no keys with it). */
    __neoHostEmit?: (event: string, payload?: unknown) => void;
    webkit?: { messageHandlers?: { neo?: { postMessage(message: unknown): void } } };
  }
}

function detect(): UiHost {
  if (typeof window === "undefined") return "window";
  if (window.__NEO_HOST__ === "notch") return "notch";
  try {
    if (new URLSearchParams(window.location.search).get("host") === "notch") return "notch";
  } catch { /* no location (tests) */ }
  return "window";
}

export const HOST: UiHost = detect();
/** Hosted in NotchHub's panel: the page fills it, and the host owns showing and hiding. */
export const inNotch = HOST === "notch";

if (inNotch && typeof document !== "undefined") document.documentElement.dataset.host = "notch";

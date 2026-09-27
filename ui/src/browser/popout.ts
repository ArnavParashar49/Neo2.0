// The pop-out: the same live session in a normal, resizable NEO window (never the user's own
// browser). The overlay opens and closes it; the window renders index.html#/browser.
import { WebviewWindow } from "@tauri-apps/api/webviewWindow";
import { LogicalPosition } from "@tauri-apps/api/window";
import { inTauri, workArea } from "../win";
import { popoutSize } from "./sizes";

export const POPOUT_LABEL = "browser";
const FRAME_KEY = "neo.browserWindow.frame";

type Frame = { x: number; y: number; w: number; h: number };

export function rememberFrame(f: Frame) {
  try { localStorage.setItem(FRAME_KEY, JSON.stringify(f)); } catch { /* storage may be unavailable */ }
}

function recalled(): Frame | null {
  try {
    const f = JSON.parse(localStorage.getItem(FRAME_KEY) ?? "null");
    if (f && [f.x, f.y, f.w, f.h].every((n: unknown) => typeof n === "number" && Number.isFinite(n)) && f.w >= 720 && f.h >= 520) return f;
  } catch { /* ignore */ }
  return null;
}

export async function openBrowserWindow(host: string): Promise<boolean> {
  if (!inTauri) return false;
  try {
    const existing = await WebviewWindow.getByLabel(POPOUT_LABEL);
    if (existing) { await existing.show(); await existing.setFocus(); return true; }
    const a = await workArea();
    const size = popoutSize(a);
    const r = recalled();
    const f: Frame = r ?? { w: size.w, h: size.h, x: a ? a.x + (a.w - size.w) / 2 : 120, y: a ? a.y + (a.h - size.h) / 2 : 80 };
    const w = new WebviewWindow(POPOUT_LABEL, {
      url: "index.html#/browser",
      title: `NEO Browser — ${host}`,
      x: f.x, y: f.y, width: f.w, height: f.h,
      minWidth: 720, minHeight: 520,
      resizable: true,
      decorations: true,
      titleBarStyle: "overlay",
      hiddenTitle: true,
      trafficLightPosition: new LogicalPosition(16, 18),
      transparent: false,
      backgroundColor: "#111114",
      alwaysOnTop: false,
      contentProtected: true,
      visible: false,
      focus: true,
    });
    return await new Promise<boolean>((resolve) => {
      void w.once("tauri://created", () => resolve(true));
      void w.once("tauri://error", () => resolve(false));
    });
  } catch {
    return false;
  }
}

export async function showBrowserWindow() {
  if (!inTauri) return;
  try { const w = await WebviewWindow.getByLabel(POPOUT_LABEL); if (w) { await w.show(); await w.setFocus(); } } catch { /* gone */ }
}

export async function closeBrowserWindow() {
  if (!inTauri) return;
  try { const w = await WebviewWindow.getByLabel(POPOUT_LABEL); if (w) await w.destroy(); } catch { /* gone */ }
}

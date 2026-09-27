// The overlay window's frame. Inside NEO.app this is the real (transparent) Tauri window; in a
// plain browser tab (npm run dev) #root stands in for it, parked bottom-right of a simulated
// desktop, so layouts and the unfold animation can be checked at true size.
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow, currentMonitor, primaryMonitor, LogicalPosition, LogicalSize } from "@tauri-apps/api/window";

export const inTauri = "__TAURI_INTERNALS__" in window;
export type Size = { w: number; h: number };
export type Area = { x: number; y: number; w: number; h: number };

const SIM_MARGIN = 20;
let simSize: Size = { w: 84, h: 84 };

function simApply(size: Size) {
  simSize = size;
  const r = document.getElementById("root");
  if (!r) return;
  r.style.width = `${size.w}px`;
  r.style.height = `${size.h}px`;
}

if (!inTauri) document.documentElement.dataset.sim = "1";

export async function workArea(): Promise<Area | null> {
  if (!inTauri) return { x: 0, y: 0, w: window.innerWidth - 2 * SIM_MARGIN + 16, h: window.innerHeight - 2 * SIM_MARGIN + 16 };
  const mon = (await currentMonitor()) ?? (await primaryMonitor());
  if (!mon) return null;
  const s = mon.scaleFactor;
  // Work area excludes the Dock and menu bar so nothing ends up hidden behind them.
  const area = (mon as unknown as { workArea?: { position: { x: number; y: number }; size: { width: number; height: number } } }).workArea;
  const pos = area?.position ?? mon.position, dim = area?.size ?? mon.size;
  return { x: pos.x / s, y: pos.y / s, w: dim.width / s, h: dim.height / s };
}

/** First placement: bottom-right corner of the work area. */
export async function placeInitial(size: Size) {
  if (!inTauri) { simApply(size); return; }
  try {
    const a = await workArea();
    if (!a) return;
    const win = getCurrentWindow();
    await win.setSize(new LogicalSize(size.w, size.h));
    await win.setPosition(new LogicalPosition(a.x + a.w - size.w - 20, a.y + a.h - size.h - 20));
  } catch { /* not inside Tauri */ }
}

let setFrameCmd: boolean | null = null; // null = unknown until the first try

/** Move + resize in one step when the shell offers `set_frame` (no one-frame jump); otherwise
 *  size then position, as before. */
async function applyFrame(x: number, y: number, w: number, h: number) {
  const win = getCurrentWindow();
  if (setFrameCmd !== false) {
    try { await invoke("set_frame", { x, y, w, h }); setFrameCmd = true; return; } catch { setFrameCmd = false; }
  }
  await win.setSize(new LogicalSize(w, h));
  await win.setPosition(new LogicalPosition(x, y));
}

/** Resize keeping the window's bottom-right corner where it is (so the card unfolds from the orb,
 *  and the orb lands back where the card was), clamped to the work area. Resolves to whether the
 *  anchor held (false = clamped near a screen edge; callers snap instead of animating). */
export async function resizeAnchored(size: Size): Promise<boolean> {
  if (!inTauri) { simApply(size); return true; }
  try {
    const win = getCurrentWindow();
    const s = await win.scaleFactor();
    const pos = await win.outerPosition();
    const cur = await win.outerSize();
    const right = pos.x / s + cur.width / s, bottom = pos.y / s + cur.height / s;
    let x = right - size.w, y = bottom - size.h;
    const ax = x, ay = y;
    const a = await workArea();
    if (a) {
      x = Math.min(Math.max(x, a.x + 8), a.x + a.w - size.w - 8);
      y = Math.min(Math.max(y, a.y + 8), a.y + a.h - size.h - 8);
    }
    await applyFrame(x, y, size.w, size.h);
    return Math.abs(x - ax) < 1 && Math.abs(y - ay) < 1;
  } catch { return true; /* not inside Tauri */ }
}

export function currentSimSize(): Size { return simSize; }

// The camera's pop-out: the same live preview in a small NEO window (index.html#/camera). The
// overlay opens and closes it; closing it brings the card back to the chat and never leaves the
// camera on without a visible preview.
import { WebviewWindow } from "@tauri-apps/api/webviewWindow";
import { LogicalPosition } from "@tauri-apps/api/window";
import { inTauri, workArea } from "../win";

export const CAMERA_POPOUT_LABEL = "camera";
export const CAMERA_POPIN_EVENT = "neo://camera-popin";

export async function openCameraWindow(): Promise<boolean> {
  if (!inTauri) return false;
  try {
    const existing = await WebviewWindow.getByLabel(CAMERA_POPOUT_LABEL);
    if (existing) { await existing.show(); await existing.setFocus(); return true; }
    const a = await workArea();
    const w = 560, h = 520;
    const win = new WebviewWindow(CAMERA_POPOUT_LABEL, {
      url: "index.html#/camera",
      title: "NEO Camera",
      x: a ? a.x + (a.w - w) / 2 : 160, y: a ? a.y + (a.h - h) / 2 : 120, width: w, height: h,
      minWidth: 360, minHeight: 340,
      resizable: true,
      decorations: true,
      titleBarStyle: "overlay",
      hiddenTitle: true,
      trafficLightPosition: new LogicalPosition(16, 18),
      transparent: false,
      backgroundColor: "#111114",
      alwaysOnTop: true, // the preview is the camera's visible sign: keep it in sight
      contentProtected: true, // never in screenshots or screen recordings
      visible: false,
      focus: true,
    });
    return await new Promise<boolean>((resolve) => {
      void win.once("tauri://created", () => resolve(true));
      void win.once("tauri://error", () => resolve(false));
    });
  } catch {
    return false;
  }
}

export async function showCameraWindow() {
  if (!inTauri) return;
  try { const w = await WebviewWindow.getByLabel(CAMERA_POPOUT_LABEL); if (w) { await w.show(); await w.setFocus(); } } catch { /* gone */ }
}

export async function closeCameraWindow() {
  if (!inTauri) return;
  try { const w = await WebviewWindow.getByLabel(CAMERA_POPOUT_LABEL); if (w) await w.destroy(); } catch { /* gone */ }
}

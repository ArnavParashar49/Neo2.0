// index.html#/camera — the camera's pop-out window: only the camera card, full size. Its red
// light and ⌘W bring the card back to the chat (the camera stays on until the answer, or Stop);
// when the camera turns off, the window closes itself.
import { useCallback, useEffect, useRef, useState } from "react";
import { emit } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useNeo } from "../ws";
import { inTauri } from "../win";
import { useCamera } from "./hub";
import { CameraCard, camIcon } from "./CameraCard";
import { CAMERA_POPIN_EVENT } from "./popout";

export function CameraPopoutApp() {
  const neo = useNeo({ surface: "window" });
  const snap = useCamera(neo.cam);
  const live = snap.live;
  const hadLive = useRef(false);
  const [shown, setShown] = useState(!inTauri);

  const closeSelf = useCallback(() => { if (inTauri) void getCurrentWindow().destroy(); }, []);
  const bringBack = useCallback(async () => {
    if (inTauri) await emit(CAMERA_POPIN_EVENT).catch(() => {});
    closeSelf();
  }, [closeSelf]);

  useEffect(() => {
    if (!inTauri || shown) return;
    const t = window.setTimeout(() => { void getCurrentWindow().show(); setShown(true); }, 300);
    return () => clearTimeout(t);
  }, [shown]);

  useEffect(() => {
    if (!inTauri) return;
    const un = getCurrentWindow().onCloseRequested(async (e) => { e.preventDefault(); await bringBack(); });
    return () => { un.then((f) => f()).catch(() => {}); };
  }, [bringBack]);

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.metaKey && !e.altKey && !e.ctrlKey && !e.shiftKey && e.key.toLowerCase() === "w") { e.preventDefault(); void bringBack(); }
    };
    window.addEventListener("keydown", h, true);
    return () => window.removeEventListener("keydown", h, true);
  }, [bringBack]);

  // The camera is off: nothing left to show here.
  useEffect(() => {
    if (live) { hadLive.current = true; return; }
    if (!hadLive.current) return;
    const t = window.setTimeout(closeSelf, 600);
    return () => clearTimeout(t);
  }, [live, closeSelf]);

  useEffect(() => {
    const v = () => neo.cam.visibilityChanged();
    document.addEventListener("visibilitychange", v);
    return () => document.removeEventListener("visibilitychange", v);
  }, [neo.cam]);

  if (!live) {
    return (
      <div className="cam-popout empty" data-tauri-drag-region>
        {camIcon(28)}
        <div>{neo.connected ? (hadLive.current ? "The camera is off." : "Waiting for NEO's camera…") : "Connecting to NEO…"}</div>
      </div>
    );
  }
  return (
    <div className="cam-popout">
      <CameraCard hub={neo.cam} live={live} stopping={snap.stopping} surface="window" onPopout={() => void bringBack()} />
    </div>
  );
}

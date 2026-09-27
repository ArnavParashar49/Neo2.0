// Dev-only stand-in for the pop-out window when running in a plain browser tab: a draggable,
// resizable macOS-style window on the simulated desktop, running the real pop-out app (with
// its own connection to the mock core, like a second Tauri window would have).
import { useEffect, useRef, useState } from "react";
import { PopoutApp } from "../PopoutApp";
import { popoutSize } from "../sizes";
import "./mock.css";

export default function SimWindow({ onClose }: { onClose: () => void }) {
  const init = () => {
    const s = popoutSize({ x: 0, y: 0, w: window.innerWidth, h: window.innerHeight });
    const w = Math.min(s.w, window.innerWidth - 60), h = Math.min(s.h, window.innerHeight - 60);
    const x = window.innerWidth - w - 380 >= 24 ? Math.max(24, Math.min(362, window.innerWidth - w - 24)) : 24;
    return { x, y: Math.max(16, Math.round((window.innerHeight - h) / 2) - 10), w, h };
  };
  const [f, setF] = useState(init);
  const drag = useRef<{ dx: number; dy: number } | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const move = (e: PointerEvent) => { const d = drag.current; if (d) setF((p) => ({ ...p, x: Math.max(-p.w + 120, e.clientX - d.dx), y: Math.max(0, e.clientY - d.dy) })); };
    const up = () => { drag.current = null; };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
  }, []);

  const onDown = (e: React.PointerEvent) => {
    const t = e.target as HTMLElement;
    if (!t.closest("[data-tauri-drag-region]") || t.closest("button, input, a")) return;
    drag.current = { dx: e.clientX - f.x, dy: e.clientY - f.y };
  };

  return (
    <div ref={ref} className="sim-window" style={{ left: f.x, top: f.y, width: f.w, height: f.h }} onPointerDown={onDown}>
      <div className="sim-traffic" aria-hidden={false}>
        <button className="r" onClick={onClose} aria-label="Close window (brings the browser back to the chat)" title="Close (brings it back)" />
        <span className="y" /><span className="g" />
      </div>
      <PopoutApp onSimClose={onClose} />
    </div>
  );
}

// The camera card in NEO's chat: the live preview (mirrored, like a selfie view), what NEO sees
// (boxes and labels), where it is (starting → live → looking → answered), the answer, and a Stop
// that turns the camera off at once. Ended sessions leave a row with the answer — never an image.
import { useEffect, useId, useRef, useState } from "react";
import type { Frame } from "../browser/frames";
import { I } from "../browser/icons";
import type { CameraHub } from "./hub";
import { boxStyle } from "./frames";
import type { CamEnded, CamLive, CamStatus } from "./types";
import "./camera.css";

/** Selfie view: moving the object left moves it left on screen. Boxes are flipped to match. */
export const MIRROR = true;

const STEPS: { key: Exclude<CamStatus, "off">; label: string }[] = [
  { key: "starting", label: "Starting" },
  { key: "live", label: "Live" },
  { key: "analysing", label: "Looking" },
  { key: "answered", label: "Answered" },
];

export const camIcon = (s = 14) => (
  <svg viewBox="0 0 24 24" width={s} height={s} aria-hidden focusable="false" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 8.5A2.5 2.5 0 0 1 5.5 6h1.8l1.5-2h6.4l1.5 2h1.8A2.5 2.5 0 0 1 21 8.5v9a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z" />
    <circle cx="12" cy="12.5" r="3.5" />
  </svg>
);

function Steps({ status, compact }: { status: CamStatus; compact?: boolean }) {
  const at = STEPS.findIndex((s) => s.key === status);
  if (compact) {
    const cur = STEPS[Math.max(0, at)];
    return <span className={"cam-chip active" + (cur.key === "analysing" ? " busy" : "")}>{cur.label}</span>;
  }
  return (
    <div className="cam-steps" role="status" aria-label={`Camera: ${STEPS[Math.max(0, at)].label}`}>
      {STEPS.map((s, i) => (
        <span key={s.key} className={"cam-chip" + (i < at ? " done" : i === at ? " active" : "") + (i === at && s.key === "analysing" ? " busy" : "")}>
          {s.label}
        </span>
      ))}
    </div>
  );
}

/** The mirrored preview canvas with NEO's boxes over it. Paints straight from the frame pipe. */
function Preview({ hub, live, stopping }: { hub: CameraHub; live: CamLive; stopping: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const got = useRef(false);
  const [hasFrame, setHasFrame] = useState(false);
  const [aspect, setAspect] = useState(4 / 3);

  useEffect(() => {
    const paint = (f: Frame) => {
      const c = canvasRef.current;
      if (!c) return;
      const { width, height } = f.bmp;
      if (!width || !height) return;
      if (c.width !== width || c.height !== height) { c.width = width; c.height = height; setAspect(width / height); }
      c.getContext("2d")?.drawImage(f.bmp, 0, 0);
      if (!got.current) { got.current = true; setHasFrame(true); }
    };
    if (hub.frames.latest) paint(hub.frames.latest);
    return hub.frames.onPaint(paint);
  }, [hub]);

  // Off: blank the canvas at once — the last picture never lingers on screen.
  useEffect(() => {
    if (!stopping) return;
    const c = canvasRef.current;
    c?.getContext("2d")?.clearRect(0, 0, c.width, c.height);
  }, [stopping]);

  return (
    <div className={"cam-preview" + (hasFrame && !stopping ? " has-frame" : "")} style={{ aspectRatio: String(Math.min(2, Math.max(0.5, aspect))) }}>
      <canvas ref={canvasRef} className={"cam-canvas" + (MIRROR ? " mirrored" : "")} aria-label="Live camera preview" role="img" />
      {(!hasFrame || stopping) && (
        <div className="cam-placeholder">
          {camIcon(22)}
          <span>{stopping ? "Turning the camera off…" : "Turning the camera on…"}</span>
        </div>
      )}
      {hasFrame && !stopping && (
        <div className="cam-boxes" aria-hidden>
          {live.boxes.map((b, i) => (
            <div key={b.id ?? `i${i}`} className={"cam-box" + (b.y < 0.08 ? " label-in" : "")} style={boxStyle(b, MIRROR)}>
              {b.label && <span className="cam-box-label">{b.label}{b.score > 0 ? ` ${Math.round(b.score * 100)}%` : ""}</span>}
            </div>
          ))}
        </div>
      )}
      {live.status === "analysing" && !stopping && <div className="cam-scan" aria-hidden />}
    </div>
  );
}

export interface CameraCardProps {
  hub: CameraHub;
  live: CamLive;
  stopping: boolean;
  /** chat: in the transcript · pip: a small corner card while the browser fills the overlay · window: the pop-out. */
  surface: "chat" | "pip" | "window";
  onPopout?: () => void;
}

export function CameraCard({ hub, live, stopping, surface, onPopout }: CameraCardProps) {
  const key = useId();
  // A mounted card is a visible preview: the core may stream only while one is on screen.
  useEffect(() => {
    hub.setView(key, true);
    return () => hub.setView(key, null);
  }, [hub, key]);

  const pip = surface === "pip";
  const stop = () => { hub.stop(); };
  return (
    <section className={`cam-card ${surface}` + (live.status === "analysing" ? " analysing" : "")} aria-label="Camera">
      <header className="cam-head" {...(surface === "window" ? { "data-tauri-drag-region": true } : {})}>
        <span className="cam-rec" title="The camera is on"><span className="cam-rec-dot" />{pip ? "On" : "Camera on"}</span>
        {!pip && live.reason && <span className="cam-reason" title={live.reason}>{live.reason}</span>}
        {pip && <Steps status={live.status} compact />}
        <span className="cam-head-actions">
          {onPopout && (
            <button className="br-icon cam-icon-btn" onClick={onPopout} title={surface === "window" ? "Bring back to NEO's chat (⌘W)" : "Open in its own window"} aria-label={surface === "window" ? "Bring back" : "Pop out"}>
              {surface === "window" ? I.popin(14) : I.popout(14)}
            </button>
          )}
          <button className="br-pill bad sm" onClick={stop} disabled={stopping} aria-disabled={stopping} title="Turn the camera off now">
            {I.stop(9)}<span>{stopping ? "Stopping…" : "Stop"}</span>
          </button>
        </span>
      </header>
      <Preview hub={hub} live={live} stopping={stopping} />
      {!pip && (
        <div className="cam-foot">
          <Steps status={live.status} />
          {live.labels.length > 0 && (
            <div className="cam-labels" aria-label="What NEO sees">
              {live.labels.map((l) => <span key={l} className="cam-label">{l}</span>)}
            </div>
          )}
          {live.answer ? (
            <div className="cam-answer" aria-live="polite">{live.answer}</div>
          ) : live.status === "analysing" ? (
            <div className="cam-answer pending">Looking…</div>
          ) : null}
          {live.status === "answered" && !stopping && <div className="cam-note">The camera turns off by itself in a moment — ask a follow-up to keep looking.</div>}
        </div>
      )}
    </section>
  );
}

/** The camera is showing in its own window: a sticky row keeps it visible (and stoppable) here. */
export function CameraPoppedRow({ hub, live, stopping, onShow, onBringBack }: { hub: CameraHub; live: CamLive; stopping: boolean; onShow: () => void; onBringBack: () => void }) {
  return (
    <div className="cam-row sticky">
      <button className="cam-row-main" onClick={onShow} title="Show the camera window">
        <span className="cam-row-icon">{camIcon(16)}<span className="cam-rec-dot on-icon" /></span>
        <span className="cam-row-text">
          <span className="cam-row-l1">Camera on · in its own window</span>
          <span className="cam-row-l2">{live.answer || live.reason || "NEO is looking"}</span>
        </span>
      </button>
      <span className="cam-row-actions">
        <button className="br-pill ghost sm" onClick={onBringBack} title="Bring back to NEO's chat">{I.popin(13)}<span>Bring back</span></button>
        <button className="br-pill bad sm" onClick={() => hub.stop()} disabled={stopping} title="Turn the camera off now">{I.stop(9)}<span>{stopping ? "Stopping…" : "Stop"}</span></button>
      </span>
    </div>
  );
}

const ENDED_WHY: Record<string, string> = {
  stopped: "Stopped", gone: "Turned off", error: "Couldn't finish", replaced: "Turned off", closed: "Turned off", timeout: "Turned off",
};

/** Where a camera session was: the answer and how long the camera was on. No picture is kept. */
export function CameraEndedRow({ e }: { e: CamEnded }) {
  const secs = e.ms >= 1000 ? `${(e.ms / 1000).toFixed(e.ms < 10_000 ? 1 : 0)} s` : "";
  return (
    <div className="cam-row ended" data-cam-sid={e.sid}>
      <span className="cam-row-icon">{camIcon(16)}</span>
      <span className="cam-row-text">
        <span className="cam-row-l1">Camera · off{secs && <span className="cam-row-ms"> · on for {secs}</span>}</span>
        <span className="cam-row-l2" title={e.answer}>{e.answer || ENDED_WHY[e.reason] || "Turned off"}</span>
      </span>
    </div>
  );
}

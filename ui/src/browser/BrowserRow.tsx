// A browser session as a row in the transcript: live (minimized, 2 fps thumbnail), popped out
// (it lives in its own window), or ended (poster, summary, the steps it took).
import { useEffect, useId, useRef, useState } from "react";
import type { BrowserHub, BrowserSnap, EndedSession } from "./hub";
import type { Frame } from "./frames";
import { Timeline } from "./Timeline";
import { I } from "./icons";
import { captionFor, displayHost, fmtDuration } from "./text";

/** A small live thumbnail drawn straight from the frame pipe (cover-fit, ≤ 2 fps). */
export function LiveThumb({ hub, w = 88, h = 55 }: { hub: BrowserHub; w?: number; h?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const key = useId();
  // Frames flow only to views that say they are showing (the core sends this one at 2 fps).
  useEffect(() => {
    hub.setView(key, { surface: "mini", w, h, visible: true });
    return () => hub.setView(key, null);
  }, [hub, key, w, h]);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    let last = 0;
    const draw = (f: Frame, force = false) => {
      const now = performance.now();
      if (!force && now - last < 450) return;
      last = now;
      const ctx = c.getContext("2d");
      if (!ctx) return;
      const bw = f.bmp.width, bh = f.bmp.height;
      const s = Math.max(c.width / bw, c.height / bh);
      const sw = c.width / s, sh = c.height / s;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(f.bmp, 0, 0, sw, sh, 0, 0, c.width, c.height); // top-left anchored: the top of the page reads best
    };
    if (hub.frames.latest) draw(hub.frames.latest, true);
    return hub.frames.onPaint((f) => draw(f));
  }, [hub, w, h]);
  return <canvas ref={ref} className="br-row-canvas" style={{ width: w, height: h }} aria-hidden />;
}

function Fav({ src }: { src?: string | null }) {
  return src ? <img className="br-row-fav" src={src} alt="" draggable={false} /> : <span className="br-row-fav glyph" aria-hidden>{I.globe(10)}</span>;
}

function rowCaption(snap: BrowserSnap): string {
  const st = snap.live!;
  switch (st.status) {
    case "starting": return "Starting NEO's browser…";
    case "thinking": return "Thinking…";
    case "waiting": return "Waiting for your OK";
    case "need_user": return st.need_user?.title ?? "Your turn";
    case "user": return "You're in control";
    case "error": return "Something went wrong";
  }
  if (snap.action) { const c = captionFor(snap.action); return c.text + (c.ctx ? " " + c.ctx : ""); }
  return st.caption ?? "Browsing";
}

/** Live and minimized: sticky at the top of the chat log. */
export function MiniRow({ hub, snap, onExpand, onStop }: { hub: BrowserHub; snap: BrowserSnap; onExpand: () => void; onStop: () => void }) {
  const st = snap.live!;
  const acting = st.controller === "neo" && (st.status === "working" || st.status === "thinking");
  const need = st.status === "need_user" || st.status === "waiting";
  return (
    <div className={"br-row live sticky" + (acting ? " acting" : "") + (need ? " need" : "")}>
      <button className="br-row-main" onClick={onExpand} aria-label={`NEO's browser on ${displayHost(st.url)}: ${rowCaption(snap)}. Expand`}>
        <span className="br-row-thumb"><LiveThumb hub={hub} /><Fav src={st.favicon} /></span>
        <span className="br-row-text">
          <span className="br-row-l1"><span className="br-row-host">{displayHost(st.url)}</span><span className="br-live"><span className="br-live-dot" />Live</span></span>
          <span className="br-row-l2" dir="auto">{rowCaption(snap)}</span>
        </span>
      </button>
      <span className="br-row-actions">
        <button className="br-icon" aria-label="Expand" title="Expand (⌘⇧M)" onClick={onExpand}>{I.expand()}</button>
        {st.job && <button className="br-icon bad" aria-label="Stop NEO" title="Stop NEO (⌘.)" onClick={onStop}>{I.stop()}</button>}
      </span>
    </div>
  );
}

/** Popped out: the chat keeps a small live row. */
export function PoppedRow({ hub, snap, onShow, onBringBack }: { hub: BrowserHub; snap: BrowserSnap; onShow: () => void; onBringBack: () => void }) {
  const st = snap.live!;
  return (
    <div className="br-row live sticky popped">
      <button className="br-row-main" onClick={onShow} aria-label={`NEO's browser on ${displayHost(st.url)} is in its own window. Show window`}>
        <span className="br-row-thumb"><LiveThumb hub={hub} /><Fav src={st.favicon} /></span>
        <span className="br-row-text">
          <span className="br-row-l1"><span className="br-row-host">{displayHost(st.url)}</span></span>
          <span className="br-row-l2">In its own window</span>
        </span>
      </button>
      <span className="br-row-actions text stack">
        <button className="br-pill sm ghost" onClick={onShow}>Show window</button>
        <button className="br-pill sm" onClick={onBringBack} title="Bring back (⌘⇧O)">Bring back</button>
      </span>
    </div>
  );
}

/** Ended: where browsing began in the transcript. */
export function EndedRow({ e, canReopen, onReopen }: { e: EndedSession; canReopen: boolean; onReopen: () => void }) {
  const [open, setOpen] = useState(false);
  const c = e.close;
  const host = displayHost(c.url);
  const n = c.steps || e.steps.length;
  const line2 = [`${n} ${n === 1 ? "step" : "steps"}`, fmtDuration(c.duration_ms), c.summary].filter(Boolean).join(" · ");
  return (
    <div className="br-ended">
      <div className={"br-row ended" + (open ? " open" : "")}>
        <button className="br-row-main" aria-expanded={open} aria-controls={`ended-${c.sid}`} onClick={() => setOpen((o) => !o)}
          aria-label={`Browsed ${host}. ${line2}. ${open ? "Hide" : "Show"} steps`}>
          <span className="br-row-thumb">
            {c.poster ? <img className="br-row-poster" src={c.poster} alt="" draggable={false} /> : <span className="br-row-poster empty">{I.globe(18)}</span>}
            <Fav src={c.favicon} />
          </span>
          <span className="br-row-text">
            <span className="br-row-l1"><span className="br-row-host">Browsed {host}</span><span className={"br-chev" + (open ? " open" : "")} aria-hidden>{I.chevronDown(12)}</span></span>
            <span className="br-row-l2" title={line2}>{line2}</span>
          </span>
        </button>
        {canReopen && c.alive && (
          <span className="br-row-actions text">
            <button className="br-pill sm" onClick={onReopen}>Reopen</button>
          </span>
        )}
      </div>
      {open && <Timeline id={`ended-${c.sid}`} steps={e.steps} startedAt={e.open?.started_at ?? 0} variant="inline" />}
    </div>
  );
}

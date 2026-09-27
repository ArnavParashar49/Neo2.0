// What NEO did, step by step, with small thumbnails: a drawer over the bottom of the live view,
// or an inline list under an ended row.
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { BrowserStep } from "./types";
import { I, stepIcon } from "./icons";
import { fmtDuration, toSec } from "./text";

const STATUS_WORD: Record<BrowserStep["status"], string> = {
  running: "in progress", ok: "done", fail: "failed", waiting: "waiting for your OK", declined: "declined", skipped: "skipped",
};

const PREVIEW_H = 125;

export function stepTime(s: BrowserStep, startedAt: number) {
  return fmtDuration((toSec(s.ts) - toSec(startedAt)) * 1000);
}

export function Timeline({ steps, startedAt, variant, onClose, id }: {
  steps: BrowserStep[]; startedAt: number; variant: "drawer" | "inline"; onClose?: () => void; id?: string;
}) {
  const listRef = useRef<HTMLOListElement>(null);
  const stick = useRef(true);
  const [preview, setPreview] = useState<{ src: string; top: number; left: number } | null>(null);
  const last = steps[steps.length - 1];
  const total = last ? stepTime(last, startedAt) : "0:00";

  useLayoutEffect(() => {
    const el = listRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [steps.length, last?.status]);

  useEffect(() => {
    if (variant !== "drawer" || !onClose) return;
    const h = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || !listRef.current?.closest(".br-drawer")?.contains(document.activeElement)) return;
      e.preventDefault(); e.stopPropagation(); onClose();
    };
    window.addEventListener("keydown", h, true);
    return () => window.removeEventListener("keydown", h, true);
  }, [variant, onClose]);

  // The larger preview never covers the list it belongs to: in the drawer it floats just above
  // the drawer (over the page), lined up with the thumbnails; under an ended row it sits to the
  // right of the hovered row's thumbnail.
  const show = (s: BrowserStep, el: HTMLElement) => {
    if (!s.thumb) { setPreview(null); return; }
    const host = listRef.current?.parentElement;
    if (!host) return;
    const hb = host.getBoundingClientRect(), rb = el.getBoundingClientRect();
    const tb = el.querySelector(".br-step-thumb")?.getBoundingClientRect() ?? rb;
    if (variant === "drawer") {
      setPreview({ src: s.thumb, top: -(PREVIEW_H + 8), left: Math.max(6, tb.left - hb.left) });
    } else {
      const top = rb.top - hb.top + rb.height / 2 - PREVIEW_H / 2;
      setPreview({ src: s.thumb, top: Math.max(-PREVIEW_H / 2, top), left: tb.right - hb.left + 10 });
    }
  };

  const list = (
    <ol
      ref={listRef}
      className="br-steps"
      aria-label="Steps NEO took"
      onScroll={(e) => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48; }}
      onMouseLeave={() => setPreview(null)}
    >
      {steps.length === 0 && <li className="br-steps-empty">NEO's steps will show up here.</li>}
      {steps.map((s, i) => (
        <li key={s.step_id}>
          <button
            className={`br-step st-${s.status}` + (s.by === "user" ? " by-user" : "")}
            aria-label={`Step ${i + 1}: ${s.label}, ${stepTime(s, startedAt)}, ${STATUS_WORD[s.status]}`}
            title={s.status === "fail" && s.reason ? s.reason : undefined}
            onMouseEnter={(e) => show(s, e.currentTarget)}
            onFocus={(e) => show(s, e.currentTarget)}
            onBlur={() => setPreview(null)}
          >
            <span className="br-step-thumb">{s.thumb ? <img src={s.thumb} alt="" draggable={false} /> : s.by === "user" ? <span className="br-step-thumb-icon">{I.hand(14)}</span> : <span className="br-shimmer" />}</span>
            <span className="br-step-kind" aria-hidden>{s.by === "user" ? I.hand() : stepIcon(s.kind)}</span>
            <span className="br-step-label" dir="auto">{s.label}</span>
            {s.status === "running" && <span className="br-spin xs" aria-hidden />}
            {s.status === "fail" && <span className="br-step-mark bad" aria-hidden>{I.warn(13)}</span>}
            {s.status === "waiting" && <span className="br-step-mark warn" aria-hidden>{I.wait(13)}</span>}
            {s.status === "declined" && <span className="br-step-tag">Declined</span>}
            <span className="br-step-time">{stepTime(s, startedAt)}</span>
          </button>
        </li>
      ))}
    </ol>
  );

  return (
    <div className={variant === "drawer" ? "br-drawer" : "br-inline-steps"} id={id} data-noinput onPointerDown={(e) => e.stopPropagation()}>
      {variant === "drawer" && (
        <div className="br-drawer-head">
          <span className="br-drawer-title">Steps</span>
          <span className="br-drawer-count">{steps.length} · {total}</span>
          <span className="br-grow" />
          <button className="br-icon sm" aria-label="Close steps" title="Close steps (Esc)" onClick={onClose}>{I.close(14)}</button>
        </div>
      )}
      {list}
      {preview && <img className="br-step-preview" src={preview.src} alt="" style={{ top: preview.top, left: preview.left }} />}
    </div>
  );
}

// Browser card icons: 24-unit strokes like the rest of the overlay (§4).
import type { ReactNode } from "react";

function S({ children, size = 15, fill = false }: { children: ReactNode; size?: number; fill?: boolean }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} aria-hidden focusable="false"
      fill={fill ? "currentColor" : "none"} stroke={fill ? "none" : "currentColor"} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      {children}
    </svg>
  );
}

export const I = {
  back: (s?: number) => <S size={s}><path d="M15 5l-7 7 7 7" /></S>,
  forward: (s?: number) => <S size={s}><path d="M9 5l7 7-7 7" /></S>,
  reload: (s?: number) => <S size={s}><path d="M20 12a8 8 0 1 1-2.34-5.66M20 4v5h-5" /></S>,
  close: (s?: number) => <S size={s}><path d="M7 7l10 10M17 7L7 17" /></S>,
  lock: (s = 12) => <S size={s}><rect x="5" y="11" width="14" height="9" rx="2" /><path d="M8 11V8a4 4 0 0 1 8 0v3" /></S>,
  popout: (s?: number) => <S size={s}><path d="M14 4h6v6M20 4l-8 8M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4" /></S>,
  popin: (s?: number) => <S size={s}><path d="M10 20H4v-6M4 20l8-8M6 10V6a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-4" /></S>,
  expand: (s?: number) => <S size={s}><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" /></S>,
  restore: (s?: number) => <S size={s}><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" /></S>,
  minimize: (s?: number) => <S size={s}><path d="M6 12h12" /></S>,
  hand: (s = 14) => <S size={s}><path d="M8 13V5.5a1.5 1.5 0 0 1 3 0V12M11 11V4.5a1.5 1.5 0 0 1 3 0V12M14 11.5V6a1.5 1.5 0 0 1 3 0v8a6 6 0 0 1-6 6h-.5a6 6 0 0 1-4.9-2.5L3.3 13.6a1.5 1.5 0 0 1 2.4-1.8L8 14" /></S>,
  sparkle: (s = 14) => <S size={s}><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" /></S>,
  steps: (s = 14) => <S size={s}><path d="M9 6h11M9 12h11M9 18h11M4.5 6h.01M4.5 12h.01M4.5 18h.01" /></S>,
  stop: (s = 10) => <S size={s} fill><rect x="6" y="6" width="12" height="12" rx="2" /></S>,
  click: (s = 14) => <S size={s}><path d="M5 3l14 7-6 2-2 6z" /></S>,
  type: (s = 14) => <S size={s}><rect x="3" y="6" width="18" height="12" rx="2" /><path d="M7 10h.01M11 10h.01M15 10h.01M7 14h10" /></S>,
  read: (s = 14) => <S size={s}><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></S>,
  scroll: (s = 14) => <S size={s}><path d="M12 5v14M8 15l4 4 4-4" /></S>,
  globe: (s = 14) => <S size={s}><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" /></S>,
  wait: (s = 14) => <S size={s}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></S>,
  key: (s = 14) => <S size={s}><circle cx="8" cy="15" r="4" /><path d="M11 12l9-9M17 6l2 2M15 8l2 2" /></S>,
  shield: (s = 14) => <S size={s}><path d="M12 3l8 3v6c0 4.5-3.4 8.3-8 9-4.6-.7-8-4.5-8-9V6z" /></S>,
  message: (s = 14) => <S size={s}><path d="M4 5h16v11H9l-5 4z" /><path d="M8 10h8" /></S>,
  warn: (s = 14) => <S size={s}><path d="M12 4l9 16H3zM12 10v4M12 17h.01" /></S>,
  chevron: (s = 12) => <S size={s}><path d="M6 15l6-6 6 6" /></S>,
  chevronDown: (s = 12) => <S size={s}><path d="M6 9l6 6 6-6" /></S>,
  check: (s = 14) => <S size={s}><path d="M5 12.5l4.5 4.5L19 7.5" /></S>,
  copy: (s = 14) => <S size={s}><rect x="8" y="8" width="12" height="12" rx="2" /><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" /></S>,
  keyboard: (s = 14) => <S size={s}><rect x="3" y="6" width="18" height="12" rx="2" /><path d="M7 10h.01M11 10h.01M15 10h.01M7 14h10" /></S>,
};

export type StepIconKind = "click" | "type" | "key" | "select" | "scroll" | "read" | "navigate" | "wait" | "move" | "user" | "confirm" | "dialog" | "need_user" | "error";

export function stepIcon(kind: StepIconKind) {
  switch (kind) {
    case "click": case "move": return I.click();
    case "type": case "key": return I.type();
    case "select": return I.steps();
    case "scroll": return I.scroll();
    case "read": return I.read();
    case "navigate": return I.globe();
    case "wait": return I.wait();
    case "user": return I.hand();
    case "confirm": return I.shield();
    case "dialog": return I.message();
    case "need_user": return I.key();
    case "error": return I.warn();
  }
  return I.click();
}

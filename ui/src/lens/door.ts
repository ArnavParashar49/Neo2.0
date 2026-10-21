// Search Screen's door to NEO.app's shell (ui/src-tauri/src/lens.rs): the phase events of its own
// window, and six commands that answer only that window. Tauri's modules are loaded lazily and
// only inside NEO.app — never in the notch (NotchHub hosts the chat page, not this one), never in
// a plain browser tab.

import { inNotch } from "../host.ts";

export interface PhaseEvent {
  /** The shell's session number: a newer one replaces everything of an older one. */
  g: number;
  phase: string;
  to_arm_ms?: number;
  why?: string;
  via?: string;
  reduce_motion?: boolean;
  reduce_transparency?: boolean;
  increase_contrast?: boolean;
  voiceover?: boolean;
  safe_top?: number;
  /** The overlay's display (`id`: its CGDirectDisplayID — the notch's island may be on another). */
  display?: { id?: number; x: number; y: number; w: number; h: number; scale: number } | null;
  /** Who draws the glow: the shell ("native", Core Animation) or this page ("web": the shell couldn't). */
  glow?: "native" | "web";
  /** Where the pointer was on the display (display-local points): Return picks the window under it. */
  pointer?: { x: number; y: number } | null;
}

/** lens_hello: the phase as it is now, and the ticket once the hold has armed (or it opened). */
export interface ShellHello extends PhaseEvent {
  ticket?: string | null;
  /** The core heard the ticket (false: no core of this app's is listening yet). */
  core?: boolean;
  /** Screen Recording is allowed. */
  sc?: boolean;
  /** How long the phase has been on (ms): a late page catches up. */
  since_ms?: number;
  hold_ms?: number;
}

export interface Door {
  hello(): Promise<ShellHello | null>;
  painted(): Promise<void>;
  close(why: string): Promise<void>;
  peek(hidden: boolean): Promise<void>;
  allowSc(): Promise<void>;
  /** An area is selected (or no longer is): the shell's glow narrows and dims, or comes back. */
  glow(selected: boolean): Promise<void>;
  onPhase(cb: (ev: PhaseEvent) => void): Promise<() => void>;
}

export interface DoorDeps {
  invoke(cmd: string, args?: Record<string, unknown>): Promise<unknown>;
  /** Listen to an event sent to this window. */
  listen(event: string, cb: (payload: unknown) => void): Promise<() => void>;
}

export const PHASE_EVENT = "neo://lens-phase";
/** Exactly these: lens.rs answers them for this window alone. */
export const COMMANDS = ["lens_hello", "lens_painted", "lens_close", "lens_peek", "lens_allow_sc", "lens_glow"] as const;

const quiet = async (p: Promise<unknown>) => { try { await p; } catch { /* the shell said no, or is gone */ } };

export function makeDoor(deps: DoorDeps): Door {
  return {
    async hello() {
      try {
        const v = await deps.invoke("lens_hello");
        return v && typeof v === "object" ? (v as ShellHello) : null;
      } catch {
        return null; // not NEO.app's own Search Screen window: nothing to show
      }
    },
    painted: () => quiet(deps.invoke("lens_painted")),
    close: (why) => quiet(deps.invoke("lens_close", { why: String(why).slice(0, 16) })),
    peek: (hidden) => quiet(deps.invoke("lens_peek", { hidden })),
    allowSc: () => quiet(deps.invoke("lens_allow_sc")),
    glow: (selected) => quiet(deps.invoke("lens_glow", { selected: !!selected })),
    onPhase: (cb) => deps.listen(PHASE_EVENT, (p) => {
      if (p && typeof p === "object" && typeof (p as PhaseEvent).g === "number" && typeof (p as PhaseEvent).phase === "string") cb(p as PhaseEvent);
    }),
  };
}

/** The door inside NEO.app's own window, or null anywhere else. */
export async function tauriDoor(): Promise<Door | null> {
  if (inNotch || typeof window === "undefined" || !("__TAURI_INTERNALS__" in window)) return null;
  const [{ invoke }, { getCurrentWebviewWindow }] = await Promise.all([import("@tauri-apps/api/core"), import("@tauri-apps/api/webviewWindow")]);
  const me = getCurrentWebviewWindow();
  return makeDoor({
    invoke: (cmd, args) => invoke(cmd, args),
    listen: (event, cb) => me.listen(event, (e) => cb(e.payload)), // this window's events only
  });
}

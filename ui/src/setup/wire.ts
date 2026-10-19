// Setup's line to NEO.app's shell: Tauri commands (each answers only the Setup window the shell
// created: setupwin.rs `may_use`) and the `setup_progress` event, which carries the whole state.
// `?mock=<view>` (dev builds only, `npm run dev` then index.html?mock=welcome#/setup) swaps in a
// scripted shell: no Tauri, no core, no network — for working on the page.
import type { Snapshot, View } from "./types.ts";
import { mockSnapshot, MOCK_VIEWS, playMock } from "./mock.ts";

export type Unlisten = () => void;

export interface SetupWire {
  state(): Promise<Snapshot>;
  /** Every snapshot the shell sends. */
  on(cb: (s: Snapshot) => void): Unlisten;
  call(cmd: string, args?: Record<string, unknown>): Promise<void>;
}

export function tauriWire(): SetupWire {
  let unlisten: Promise<Unlisten> | null = null;
  return {
    async state() {
      const { invoke } = await import("@tauri-apps/api/core");
      return invoke<Snapshot>("setup_state");
    },
    on(cb) {
      unlisten = import("@tauri-apps/api/event").then(({ listen }) => listen<Snapshot>("setup_progress", (e) => cb(e.payload)));
      return () => { void unlisten?.then((f) => f()); };
    },
    async call(cmd, args) {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke(cmd, args ?? {});
    },
  };
}

/** The scripted shell for `?mock=` (only in dev builds: main.tsx never reaches it otherwise). */
export function mockWire(view: string): SetupWire {
  const start: View = (MOCK_VIEWS as readonly string[]).includes(view) ? (view as View) : "welcome";
  let snap = mockSnapshot(start);
  const subs = new Set<(s: Snapshot) => void>();
  const set = (next: Snapshot) => { snap = next; subs.forEach((f) => f(snap)); };
  const plays: (() => void)[] = [];
  return {
    async state() { return snap; },
    on(cb) { subs.add(cb); return () => { subs.delete(cb); }; },
    async call(cmd, args) {
      if (cmd === "setup_cancel") plays.splice(0).forEach((stop) => stop()); // (what's playing stops, as Cancel would)
      plays.push(playMock(cmd, args ?? {}, () => snap, set));
    },
  };
}

/** Which mock the address asks for (dev builds only). */
export function mockKind(search: string, dev: boolean): string | null {
  if (!dev) return null;
  return new URLSearchParams(search).get("mock");
}

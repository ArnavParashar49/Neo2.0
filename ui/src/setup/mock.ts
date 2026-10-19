// `?mock=<view>`: a scripted shell for working on Setup's page in a browser (dev builds only).
// Each view starts from a believable state, and the buttons play out what the shell would do —
// a download with progress, a failure for `?mock=offline`, the checklist's models arriving.
import type { Fetch, Snapshot, Stage, View } from "./types.ts";

export const MOCK_VIEWS = [
  "move", "unsupported", "no_engine", "welcome", "new_engine", "repair", "engine", "checklist", "updates", "uninstall", "offline", "disk_full",
] as const;

const fetchOf = (state: Fetch["state"], done = 0, total = 0): Fetch => ({
  state, items: [], done, total, rate: 0, eta_s: null, code: null, error: null,
});

export function mockSnapshot(view: View | string): Snapshot {
  const base: Snapshot = {
    view: "welcome",
    product: "NEO",
    version: "2.2.0",
    sandbox: false,
    place: "ok",
    macos: "15.6",
    plan: {
      app_version: "2.2.0", neo2_version: "2.2.0", host: "github.com", asset: "neo-engine-0123456789abcdef-macos14-arm64.tar.xz",
      engine_bytes: 225_000_000, downloaded: 0, engine_disk: 1_475_000_000, models_bytes: 1_700_000_000,
      need_bytes: 4_400_000_000, free_bytes: 182_000_000_000, from_engine: null,
    },
    manifest_problem: null,
    engine: { stage: "idle", done: 0, total: 0, rate: 0, eta_s: null, error: null },
    engine_running: false,
    engine_from_file: false,
    engine_ok: false,
    essential: fetchOf("idle"),
    optional: {},
    here: {},
    perms: { accessibility: false, screen_recording: false, microphone: "not_determined", camera: "not_determined" },
    core: "not set up — open NEO Setup",
    core_ready: false,
    login_allowed: true,
    login_on: true,
    brain_opened: false,
    done: false,
    update: { auto: true, checking: false, latest: null, offer: null, last_check: 0, error: null, page: null },
    uninstall: { items: [], sizing: false, running: false, outcome: null, bundle_id: "com.arnav.neo" },
    setup_log: "~/.neo/logs/setup.log",
  };
  switch (view) {
    case "move": return { ...base, view: "move", place: "volume" };
    case "unsupported": return { ...base, view: "unsupported", macos: "13.6" };
    case "no_engine": return { ...base, view: "no_engine", plan: null };
    case "new_engine": return { ...base, view: "new_engine", plan: { ...base.plan!, from_engine: "fedcba9876543210", app_version: "2.3.0" } };
    case "repair": return { ...base, view: "repair", engine_ok: true, core: "running" };
    case "engine": return { ...base, view: "engine", engine_running: true, engine: { stage: "downloading", done: 84_000_000, total: 225_000_000, rate: 6_200_000, eta_s: 23, error: null } };
    case "offline":
      return { ...base, view: "engine", engine: { stage: "failed", done: 12_000_000, total: 225_000_000, rate: 0, eta_s: null, error: { code: "offline", message: "Can't reach github.com — check your connection, then Retry." } } };
    case "disk_full":
      return { ...base, view: "engine", engine: { stage: "failed", done: 0, total: 0, rate: 0, eta_s: null, error: { code: "disk_full", message: "NEO needs 4.4 GB free on this Mac to set up, and 2.1 GB is free. Make some room, then Retry." } } };
    case "checklist":
      return {
        ...base, view: "checklist", engine_ok: true, core: "running", core_ready: true, brain_opened: true,
        perms: { accessibility: true, screen_recording: false, microphone: "not_determined", camera: "denied" },
        essential: { ...fetchOf("running", 610_000_000, 1_620_000_000), eta_s: 190, rate: 5_300_000 },
        here: { local_voice: false, browser: false, vision: false },
      };
    case "updates":
      return { ...base, view: "updates", engine_ok: true, update: { ...base.update, latest: "2.2.1", offer: "2.2.1", last_check: Math.floor(Date.now() / 1000) - 3600, page: "https://github.com/ArnavParashar49/Neo2.0/releases/tag/v2.2.1" } };
    case "uninstall":
      return {
        ...base, view: "uninstall", engine_ok: true,
        uninstall: {
          ...base.uninstall,
          items: [
            { id: "app", label: "NEO.app and its login item", note: "The app, its login item, caches and web data.", shown: ["/Applications/NEO.app"], bytes: 6_100_000, default: true, locked: true },
            { id: "engine", label: "NEO's engine", note: "Python and NEO's packages (Application Support/NEO). NotchHub's NEO page goes with it.", shown: ["~/Library/Application Support/NEO"], bytes: 1_520_000_000, default: true, locked: false },
            { id: "models", label: "NEO's models in the Hugging Face cache", note: "Only the models NEO downloaded; the rest of that cache stays.", shown: ["~/.cache/huggingface/hub/models--convaiinnovations--laya"], bytes: 1_480_000_000, default: true, locked: false },
            { id: "model_files", label: "NEO's other models", note: "The wake word, camera boxes and memory search (~/.neo/models).", shown: ["~/.neo/models"], bytes: 160_000_000, default: true, locked: false },
            { id: "chromium", label: "Chromium for NEO's browser", note: "Playwright's browsers — other tools on this Mac may use them too.", shown: ["~/Library/Caches/ms-playwright"], bytes: 557_000_000, default: false, locked: false },
            { id: "data", label: "My NEO data", note: "Keys, memory, settings and logs (~/.neo). Leave it to keep them for later.", shown: ["~/.neo"], bytes: 42_000_000, default: false, locked: false },
          ],
        },
      };
    default: return base;
  }
}

const STEPS: { stage: Stage; ms: number; total?: number }[] = [
  { stage: "preparing", ms: 400 },
  { stage: "downloading", ms: 5000, total: 225_000_000 },
  { stage: "verifying", ms: 900, total: 225_000_000 },
  { stage: "unpacking", ms: 2200, total: 31_412 },
  { stage: "optimizing", ms: 1800 },
  { stage: "installing", ms: 700 },
  { stage: "testing", ms: 1400 },
  { stage: "finishing", ms: 300 },
];

/** What a button does in the mock: returns how to stop what it started. */
export function playMock(cmd: string, args: Record<string, unknown>, get: () => Snapshot, set: (s: Snapshot) => void): () => void {
  const timers: ReturnType<typeof setTimeout>[] = [];
  const later = (ms: number, f: () => void) => { timers.push(setTimeout(f, ms)); };
  const stop = () => timers.forEach(clearTimeout);
  const s = get();
  switch (cmd) {
    case "setup_start": case "setup_retry": case "setup_use_file": {
      const failOffline = new URLSearchParams(globalThis.location?.search ?? "").get("fail") === "offline";
      set({ ...s, view: "engine", engine_running: true, engine_from_file: cmd === "setup_use_file", engine: { stage: "preparing", done: 0, total: 0, rate: 0, eta_s: null, error: null } });
      let t = 0;
      for (const step of STEPS) {
        const ticks = step.total ? 10 : 1;
        for (let i = 1; i <= ticks; i++) {
          later((t += step.ms / ticks), () => {
            const total = step.total ?? 0;
            const done = total ? Math.floor((total * i) / ticks) : 0;
            if (failOffline && step.stage === "downloading" && i === 4) {
              set({ ...get(), engine_running: false, engine: { stage: "failed", done, total, rate: 0, eta_s: null, error: { code: "offline", message: "Can't reach github.com — check your connection, then Retry." } } });
              stop();
              return;
            }
            const rate = step.stage === "downloading" ? 45_000_000 : 0;
            set({ ...get(), engine: { stage: step.stage, done, total, rate, eta_s: rate ? Math.ceil((total - done) / rate) : null, error: null } });
          });
        }
      }
      later(t + 300, () => {
        set({ ...get(), view: "checklist", engine_running: false, engine_ok: true, core: "running", engine: { stage: "done", done: 1, total: 1, rate: 0, eta_s: null, error: null } });
        playModels(get, set, later);
        later(2500, () => set({ ...get(), core_ready: true, brain_opened: true }));
      });
      break;
    }
    case "setup_cancel":
      set({ ...s, engine_running: false, engine: { ...s.engine, stage: "cancelled", error: { code: "cancelled", message: "Setup was cancelled. Retry picks up where it stopped." } } });
      break;
    case "setup_permission": {
      const what = String(args.what ?? "");
      later(600, () => {
        const p = { ...get().perms };
        if (what === "accessibility") p.accessibility = true;
        if (what === "screen_recording") p.screen_recording = true;
        if (what === "microphone") p.microphone = "granted";
        if (what === "camera") p.camera = "granted";
        set({ ...get(), perms: p });
      });
      break;
    }
    case "setup_open":
      if (args.what === "keys") set({ ...s, brain_opened: true });
      break;
    case "setup_login":
      set({ ...s, login_on: Boolean(args.on) });
      break;
    case "setup_optional": {
      const g = String(args.group);
      if (!args.on) { set({ ...s, optional: { ...s.optional, [g]: { ...fetchOf("cancelled") } } }); break; }
      const total = g === "local_voice" ? 2_898_977_287 : g === "browser" ? 160_000_000 : 42_000_000;
      for (let i = 0; i <= 20; i++) {
        later(i * 250, () => set({ ...get(), optional: { ...get().optional, [g]: { ...fetchOf(i === 20 ? "done" : "running", Math.floor((total * i) / 20), total), eta_s: (20 - i) * 0.25 } } }));
      }
      later(21 * 250, () => set({ ...get(), here: { ...get().here, [g]: true } }));
      break;
    }
    case "setup_models":
      playModels(get, set, later);
      break;
    case "setup_update":
      if (args.action === "check") {
        set({ ...s, update: { ...s.update, checking: true } });
        later(900, () => set({ ...get(), update: { ...get().update, checking: false, last_check: Math.floor(Date.now() / 1000) } }));
      } else if (args.action === "skip") {
        set({ ...s, update: { ...s.update, offer: null } });
      } else if (args.action === "auto") {
        set({ ...s, update: { ...s.update, auto: Boolean(args.on) } });
      }
      break;
    case "setup_uninstall":
      if (args.action === "run") {
        set({ ...s, uninstall: { ...s.uninstall, running: true } });
        later(1500, () => set({ ...get(), uninstall: { ...get().uninstall, running: false, outcome: { moved: ["/Applications/NEO.app", "~/Library/Application Support/NEO"], failed: [], permissions_reset: true } } }));
      }
      break;
    case "setup_done": case "setup_quit":
      set({ ...s, done: true });
      break;
  }
  return stop;
}

function playModels(get: () => Snapshot, set: (s: Snapshot) => void, later: (ms: number, f: () => void) => void) {
  const total = 1_620_000_000;
  for (let i = 0; i <= 30; i++) {
    later(i * 300, () => set({ ...get(), essential: { ...fetchOf(i === 30 ? "done" : "running", Math.floor((total * i) / 30), total), rate: 40_000_000, eta_s: (30 - i) * 0.3 } }));
  }
}

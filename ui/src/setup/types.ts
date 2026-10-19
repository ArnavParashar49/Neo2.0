// What NEO.app's shell says of Setup (ui/src-tauri/src/setupwin.rs: `Snapshot`), as one object each
// time something changes (the `setup_progress` event) and on request (`setup_state`).

export type View =
  | "move" | "unsupported" | "no_engine" | "welcome" | "new_engine" | "repair" | "engine" | "checklist" | "updates" | "uninstall";

/** setup.rs `Stage`: where an engine install is. */
export type Stage =
  | "idle" | "preparing" | "downloading" | "verifying" | "unpacking" | "optimizing" | "installing" | "testing" | "finishing"
  | "done" | "failed" | "cancelled";

export interface Failure { code: string; message: string }

export interface EngineProgress {
  stage: Stage;
  done: number;
  total: number;
  rate: number;
  eta_s: number | null;
  error: Failure | null;
}

export interface Plan {
  app_version: string;
  neo2_version: string;
  host: string;
  asset: string;
  engine_bytes: number;
  downloaded: number;
  engine_disk: number;
  models_bytes: number;
  need_bytes: number;
  free_bytes: number | null;
  from_engine: string | null;
}

export interface FetchItem {
  id: string;
  label: string;
  source: string;
  size: number;
  state: "waiting" | "running" | "here" | "done" | "skipped" | "failed" | string;
  done: number;
  total: number;
  stage: string | null;
  message: string | null;
}

/** One `neo2 fetch` run, as its JSON lines said. */
export interface Fetch {
  state: "idle" | "running" | "done" | "failed" | "cancelled" | string;
  items: FetchItem[];
  done: number;
  total: number;
  rate: number;
  eta_s: number | null;
  code: string | null;
  error: string | null;
}

/** granted | denied | restricted | not_determined | unknown */
export type MediaStatus = string;

export interface UninstallItem {
  id: string;
  label: string;
  note: string;
  shown: string[];
  bytes: number;
  default: boolean;
  locked: boolean;
}

export interface Outcome { moved: string[]; failed: string[]; permissions_reset: boolean | null }

export interface Snapshot {
  view: View;
  product: string;
  version: string;
  sandbox: boolean;
  place: "ok" | "translocated" | "volume";
  macos: string;
  plan: Plan | null;
  manifest_problem: string | null;
  engine: EngineProgress;
  engine_running: boolean;
  engine_from_file: boolean;
  engine_ok: boolean;
  essential: Fetch;
  optional: Record<string, Fetch>;
  here: Record<string, boolean>;
  perms: { accessibility: boolean; screen_recording: boolean; microphone: MediaStatus; camera: MediaStatus };
  core: string;
  core_ready: boolean;
  login_allowed: boolean;
  login_on: boolean;
  brain_opened: boolean;
  done: boolean;
  update: {
    auto: boolean; checking: boolean; latest: string | null; offer: string | null; last_check: number; error: string | null; page: string | null;
  };
  uninstall: { items: UninstallItem[]; sizing: boolean; running: boolean; outcome: Outcome | null; bundle_id: string };
  setup_log: string;
}

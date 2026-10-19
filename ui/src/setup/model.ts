// The words and numbers NEO Setup shows (index.html#/setup): everything here is a plain function of
// what the shell said (./types.ts), so the tests can hold each screen to its words.
import { etaText, sizeText } from "../keys/localvoice.ts";
import type { EngineProgress, Fetch, MediaStatus, Plan, Snapshot, Stage, UninstallItem, View } from "./types.ts";

export { etaText, sizeText };

/** What the optional downloads are, as neo2/core/fetch.py sizes them. */
export const OPTIONAL = [
  { group: "local_voice", title: "On-device voice", detail: "Talk to NEO without a key: Parakeet listens and Kokoro speaks, on this Mac.", size: 2_898_977_287 },
  { group: "browser", title: "Web browsing", detail: "Chromium, for the tasks NEO does in a browser of its own.", size: 362_000_000 },
  { group: "vision", title: "Camera object boxes", detail: "D-FINE draws a box round what NEO sees through the camera.", size: 42_000_000 },
] as const;

export function rateText(bps: number): string {
  if (!Number.isFinite(bps) || bps <= 0) return "";
  return bps >= 1e6 ? `${(bps / 1e6).toFixed(1)} MB/s` : `${Math.max(1, Math.round(bps / 1e3))} KB/s`;
}

const num = (n: number) => Math.round(n).toLocaleString("en-US");

/** Each stage's share of the whole install's bar (the download is most of the wait). */
const SPAN: Readonly<Record<Stage, [number, number]>> = {
  idle: [0, 0], preparing: [0, 1], downloading: [1, 70], verifying: [70, 74], unpacking: [74, 88], optimizing: [88, 95],
  installing: [95, 97], testing: [97, 99], finishing: [99, 100], done: [100, 100], failed: [0, 0], cancelled: [0, 0],
};

/** 0–100 for the bar, or null for a stage that can't be measured (it shows as busy). */
export function enginePercent(p: EngineProgress): number | null {
  const [a, b] = SPAN[p.stage] ?? [0, 0];
  if (p.stage === "done") return 100;
  if (p.stage === "failed" || p.stage === "cancelled" || p.stage === "idle") return null;
  if (p.total > 0) return Math.floor(a + ((b - a) * Math.min(p.done, p.total)) / p.total);
  return ["optimizing", "installing", "testing"].includes(p.stage) ? null : a;
}

/** The line under the bar: what's happening, how far, how long. */
export function stageText(p: EngineProgress, plan: Plan | null): string {
  const of = (unit: (n: number) => string) => (p.total > 0 ? `${unit(p.done) || "0 MB"} of ${unit(p.total)}` : "");
  const extra = [p.rate > 0 ? rateText(p.rate) : "", etaText(p.eta_s)].filter(Boolean).join(" · ");
  switch (p.stage) {
    case "idle": return "";
    case "preparing": return "Getting ready…";
    case "downloading": return `Downloading NEO's engine — ${of(sizeText) || "starting"}${extra ? ` · ${extra}` : ""}`;
    case "verifying": return `Checking the download… ${p.total > 0 ? `${Math.floor((100 * p.done) / p.total)}%` : ""}`.trim();
    case "unpacking": return p.total > 0 ? `Unpacking — ${num(Math.min(p.done, p.total))} of ${num(p.total)} files` : "Unpacking…";
    case "optimizing": return "Optimizing for this Mac… (about a minute)";
    case "installing": return `Installing NEO${plan ? ` ${plan.neo2_version}` : ""}…`;
    case "testing": return "Checking that NEO's engine starts…";
    case "finishing": return "Finishing…";
    case "done": return "NEO's engine is installed.";
    case "cancelled": return p.error?.message ?? "Setup was cancelled. Retry picks up where it stopped.";
    case "failed": return p.error?.message ?? "Setup stopped.";
  }
  return "";
}

export type FixButton = "retry" | "use_file" | "logs" | "releases";

/** What a failure offers, by its code (setup.rs `Failure`). */
export function fixesFor(code: string | undefined): FixButton[] {
  switch (code) {
    case "offline": return ["retry", "use_file"];
    case "not_found": return ["releases", "use_file"];
    case "checksum": return ["retry", "use_file"];
    case "disk_full": return ["retry"];
    case "busy": return [];
    case "cancelled": return ["retry", "use_file"];
    case "smoke": case "unpack": return ["retry", "logs"];
    default: return ["retry", "use_file", "logs"];
  }
}

export interface PlanRow { what: string; from: string; size: string; disk: string }

/** The Welcome page's table: what will be downloaded, from where, how big. */
export function planRows(plan: Plan): PlanRow[] {
  const left = Math.max(0, plan.engine_bytes - plan.downloaded);
  return [
    {
      what: "NEO's engine: Python and every package NEO runs",
      from: plan.host,
      size: plan.downloaded > 0 && left > 0 ? `${sizeText(left)} left of ${sizeText(plan.engine_bytes)}` : sizeText(plan.engine_bytes),
      disk: sizeText(plan.engine_disk),
    },
    {
      what: "Models: NEO's reflex, the wake word and memory search",
      from: "Hugging Face, alphacephei.com",
      size: sizeText(plan.models_bytes),
      disk: sizeText(plan.models_bytes),
    },
  ];
}

/** Free space as Finder rounds it: "182 GB", "4.4 GB". */
function freeText(n: number): string {
  return n >= 10e9 ? `${Math.floor(n / 1e9)} GB` : sizeText(n);
}

/** "NEO needs about 4.4 GB of free space; this Mac has 182 GB." — and whether there's room. */
export function roomLine(plan: Plan): { ok: boolean; text: string } {
  const need = sizeText(plan.need_bytes);
  if (plan.free_bytes == null) return { ok: true, text: `NEO needs about ${need} of free space.` };
  const ok = plan.free_bytes >= plan.need_bytes;
  return {
    ok,
    text: ok
      ? `NEO needs about ${need} of free space; this Mac has ${freeText(plan.free_bytes)}.`
      : `NEO needs about ${need} of free space, and this Mac has ${freeText(plan.free_bytes)}. Make some room first.`,
  };
}

export function mediaText(status: MediaStatus): string {
  switch (status) {
    case "granted": return "Allowed";
    case "denied": return "Turned off — switch it on in System Settings";
    case "restricted": return "Blocked by this Mac's settings";
    case "not_determined": return "Not asked yet";
    default: return "";
  }
}

export type RowState = "ok" | "todo" | "wait" | "note";
export interface Row { id: string; title: string; detail: string; state: RowState; action?: { label: string; cmd: string; arg?: string } }

/** The checklist's rows, each with what it needs and its button. */
export function checklistRows(s: Snapshot): Row[] {
  const p = s.perms;
  const media = (id: "microphone" | "camera", title: string, why: string): Row => {
    const st = p[id];
    if (st === "granted") return { id, title, detail: why, state: "ok" };
    if (!s.core_ready) return { id, title, detail: `${why} (NEO asks once its engine has started.)`, state: "wait" };
    if (st === "denied" || st === "restricted") {
      return { id, title, detail: `${why} ${mediaText(st)}.`, state: "todo", action: { label: "Open Settings…", cmd: "setup_open", arg: `${id}_settings` } };
    }
    return { id, title, detail: why, state: "todo", action: { label: "Allow…", cmd: "setup_permission", arg: id } };
  };
  return [
    {
      id: "brain",
      title: "A brain",
      detail: "Paste a free Gemini key, or pick another provider, in Brains & Keys.",
      state: s.brain_opened ? "ok" : "todo",
      action: { label: s.brain_opened ? "Brains & Keys…" : "Open Brains & Keys…", cmd: "setup_open", arg: "keys" },
    },
    p.accessibility
      ? { id: "accessibility", title: "Accessibility", detail: "So NEO can use your apps for you.", state: "ok" }
      : { id: "accessibility", title: "Accessibility", detail: "So NEO can use your apps for you. Turn NEO on in the list.", state: "todo", action: { label: "Allow…", cmd: "setup_permission", arg: "accessibility" } },
    p.screen_recording
      ? { id: "screen_recording", title: "Screen Recording", detail: "So NEO can see what's on screen when you ask.", state: "ok" }
      : {
        id: "screen_recording", title: "Screen Recording",
        detail: "So NEO can see what's on screen when you ask. Once you turn it on, macOS asks to Quit & Reopen NEO — that's expected.",
        state: "todo", action: { label: "Allow…", cmd: "setup_permission", arg: "screen_recording" },
      },
    media("microphone", "Microphone", "So you can talk to NEO."),
    media("camera", "Camera (optional)", "Only when you ask NEO to look at something."),
    { id: "later", title: "Calendars, Reminders and Contacts", detail: "macOS asks the first time NEO uses them.", state: "note" },
  ];
}

/** The models' line: how far `neo2 fetch --essential` is. */
export function fetchLine(f: Fetch, what = "NEO's models"): string {
  switch (f.state) {
    case "running": {
      if (f.total <= 0) return `Getting ready to download ${what}…`;
      const pct = Math.floor((100 * Math.min(f.done, f.total)) / f.total);
      const eta = etaText(f.eta_s);
      return `Downloading ${what} — ${sizeText(f.done) || "0 MB"} of ${sizeText(f.total)} (${pct}%)${eta ? ` · ${eta}` : ""}`;
    }
    case "done": return `${what[0].toUpperCase()}${what.slice(1)} are on this Mac.`;
    case "cancelled": return "Stopped.";
    case "failed": return f.error ?? "The download didn't finish.";
    default: return "";
  }
}

export function fetchPercent(f: Fetch): number | null {
  if (f.state === "done") return 100;
  if (f.state !== "running" || f.total <= 0) return null;
  return Math.floor((100 * Math.min(f.done, f.total)) / f.total);
}

export interface OptionalRow { group: string; title: string; detail: string; size: string; here: boolean; fetch: Fetch | null; label: string }

export function optionalRows(s: Snapshot): OptionalRow[] {
  return OPTIONAL.map((o) => {
    const f = s.optional[o.group] ?? null;
    const here = s.here[o.group] === true || f?.state === "done";
    const running = f?.state === "running";
    return {
      group: o.group, title: o.title, detail: o.detail, size: sizeText(o.size), here, fetch: f,
      label: here ? "On this Mac" : running ? "Cancel" : f?.state === "failed" ? "Try again" : `Download (${sizeText(o.size)})`,
    };
  });
}

/** Each page's heading. */
export function title(view: View, product: string): string {
  switch (view) {
    case "move": return `Move ${product} to Applications`;
    case "unsupported": return `${product} needs a newer Mac`;
    case "no_engine": return `This ${product} has no engine to install`;
    case "welcome": return `Set up ${product}`;
    case "new_engine": return `${product} needs a newer engine`;
    case "repair": return `Repair ${product}'s engine`;
    case "engine": return `Setting up ${product}`;
    case "checklist": return "Setup checklist";
    case "updates": return "Updates";
    case "uninstall": return `Uninstall ${product}`;
  }
  return product;
}

/** What Uninstall will move, in all. */
export function uninstallTotal(items: UninstallItem[], chosen: ReadonlySet<string>): number {
  return items.filter((i) => i.locked || chosen.has(i.id)).reduce((n, i) => n + (i.bytes || 0), 0);
}

export function defaultChosen(items: UninstallItem[]): Set<string> {
  return new Set(items.filter((i) => i.default || i.locked).map((i) => i.id));
}

/** The Updates page's line. */
export function updateLine(u: Snapshot["update"], version: string, product: string): string {
  if (u.checking) return "Checking…";
  if (u.offer) return `${product} ${u.offer} is out (this is ${version}).`;
  if (u.error) return u.error;
  if (u.last_check > 0) return `${product} ${version} is the newest version.`;
  return `This is ${product} ${version}.`;
}

/** When the last check was, in words ("today", "3 days ago"). */
export function agoText(at: number, now = Date.now() / 1000): string {
  if (!at) return "never";
  const days = Math.floor((now - at) / 86400);
  if (days <= 0) return "today";
  return days === 1 ? "yesterday" : `${days} days ago`;
}

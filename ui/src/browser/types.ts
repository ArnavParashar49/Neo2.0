// NEO in-chat browser: wire protocol v1 (see the design spec, §10). Every coordinate is page
// viewport CSS px (relative to the visible viewport's top-left, not the document).

export type Sid = string;
export interface Rect { x: number; y: number; w: number; h: number }
export type Controller = "neo" | "user";
export type Status = "starting" | "working" | "thinking" | "waiting" | "need_user" | "user" | "error" | "ended";
export type ActionKind = "move" | "click" | "type" | "key" | "select" | "scroll" | "read" | "navigate" | "wait";
export interface Viewport { w: number; h: number; dpr: number }
export type Secure = "secure" | "insecure" | "local" | "internal" | "error";

/** A session card starts (or reopens). */
export interface BrowserOpen {
  sid: Sid; job: string; reason: "task" | "reopen" | "user";
  url: string; title: string; viewport: Viewport;
  controller: Controller; status: Status; started_at: number;
}

export interface NeedUser {
  reason: "login" | "captcha" | "2fa" | "confirm" | "other";
  site: string;
  title: string; sub: string;
  detail?: string;
  action_id?: string;
  rect?: Rect;
}

/** Full snapshot of the chrome state (≤ 10/s). */
export interface BrowserState {
  sid: Sid; seq: number;
  url: string; title: string;
  favicon?: string | null;
  secure: Secure;
  loading: boolean; progress: number | null;
  can_back: boolean; can_forward: boolean;
  controller: Controller; status: Status;
  viewport: Viewport;
  tabs: { count: number; active: number; list?: { title: string; url: string; favicon?: string | null }[] };
  popped: boolean;
  job: string | null;
  need_user: NeedUser | null;
  error: { code: string; message: string } | null;
  caption: string | null;
}

export interface BrowserAction {
  sid: Sid; aid: number; phase: "intent" | "done" | "fail";
  kind: ActionKind;
  x?: number; y?: number;
  rect?: Rect;
  label?: string;
  role?: string;
  field?: string;
  text?: string;
  secret?: boolean;
  key?: string;
  button?: "left" | "right" | "middle"; clicks?: number;
  dir?: "up" | "down" | "left" | "right"; amount?: number;
  url?: string;
  ms?: number;
  gated?: boolean;
  caption?: string;
  reason?: string;
}

export type StepKind = ActionKind | "user" | "confirm" | "dialog" | "need_user" | "error";
export interface BrowserStep {
  sid: Sid; step_id: number; ts: number;
  kind: StepKind;
  label: string;
  status: "running" | "ok" | "fail" | "waiting" | "declined" | "skipped";
  by: "neo" | "user";
  url?: string;
  thumb?: string | null;
  duration_ms?: number;
  reason?: string;
}

export interface BrowserClose {
  sid: Sid; reason: "done" | "stopped" | "closed" | "error" | "erased" | "replaced";
  alive: boolean;
  url: string; title: string; favicon?: string | null;
  poster?: string | null;
  steps: number; duration_ms: number;
  summary: string;
}

export interface BrowserHint {
  sid: Sid;
  cursor?: string;
  focus?: { rect: Rect; kind: "text" | "password" | "textarea" | "contenteditable" | "select" | "other" } | null;
}

export interface BrowserSelect {
  sid: Sid; select_id: string; rect: Rect; multiple: boolean;
  options: { value: string; label: string; selected: boolean; disabled: boolean; group?: string }[];
}

export interface BrowserDialog {
  sid: Sid; dialog_id: string; kind: "alert" | "confirm" | "prompt" | "beforeunload";
  message: string;
  default?: string; origin: string;
}

export type ErrCode = "bad_sid" | "not_user" | "no_session" | "busy" | "invalid" | "rate" | "blocked_url" | "unsupported" | "timeout";
export interface BrowserResult { req: string; ok: boolean; code?: ErrCode; message?: string }
export interface BrowserClipboard { sid: Sid; req: string; text: string }

/** `hello.data.browser`: everything a (re)connecting client needs to rebuild the card. */
export interface BrowserHelloSnapshot {
  v: number;
  live: BrowserState | null;
  open?: BrowserOpen | null;
  steps: BrowserStep[];
  action: BrowserAction | null;
  dialog: BrowserDialog | null;
  select: BrowserSelect | null;
}

export type Surface = "docked" | "focus" | "window" | "mini" | "hidden";
export type ControlAction =
  | "take" | "release" | "stop" | "back" | "forward" | "reload" | "hard_reload" | "stop_loading"
  | "navigate" | "tab" | "select" | "dialog" | "copy" | "cut" | "popout" | "popin" | "expand" | "collapse"
  | "close" | "reopen" | "restart";

/** Decoded binary frame header (NBF1). */
export interface FrameMeta {
  sid: Sid; fid: number; kind: 1 | 2; ts: number;
  vw: number; vh: number; iw: number; ih: number; dpr: number;
  sx: number; sy: number; flags: number; mime: string;
}

// What the core says about NEO's brains (neo2/core/brains.py), as the page reads it. Every client
// hears the summary; only the Brains & Keys window hears brains_detail and its commands' results.
// No key value ever comes back: at most its last 4 characters, and those only to the window.

/** One role as every client sees it: a brain reference and its label, and whether it can answer. */
export interface RoleSummary { ref: string; provider: string; label: string; ok: boolean }

/** The `brains` event and the hello's "brains". */
export interface BrainsSummary {
  ready: boolean;
  brain: RoleSummary;
  quick: RoleSummary;
  eyes: RoleSummary | null;
  voice: "live" | "local" | "off" | string;
  providers: Record<string, boolean>;
  restart_needed: boolean;
}

/** `setup_needed`: a request needed a model NEO hasn't got (neo2/core/onboarding.py). */
export interface SetupNeeded { shell: boolean; opened: boolean; command: string }

export type KeySource = "env" | "saved" | "checkout_env" | "dotenv" | "none";
export interface KeyEntry { set: boolean; source: KeySource | string; last4: string | null; extra?: number; saved?: string | null; also?: string[] }

export interface CatalogTemplate {
  id: string; label: string; base_url: string; env: string | null; key_url: string; key_hint: string;
  models: Record<string, string>; tools: boolean; vision: boolean; short_tool_ids: boolean; needs_key: boolean;
}
export interface CatalogEntry {
  id: string; label: string; transport: string; cost: string; key_url: string; key_hint: string;
  roles: string[]; env: string | null; templates?: CatalogTemplate[];
}

export interface ShownRef { ref: string | null; valid: boolean; typed?: string }
export interface RoleDetail extends ShownRef { source: string }

export interface CustomEndpoint {
  id: string; label: string; template: string; base_url: string; model: string; tools: boolean; vision: boolean;
  context: number; reasoning?: boolean; token_param?: string; short_tool_ids?: boolean; env?: string | null;
  on_device?: boolean; reachable?: boolean | null; saved?: boolean;
}

/** brains_detail: for the Brains & Keys window alone. */
export interface BrainsDetail {
  id?: string;
  catalog: CatalogEntry[];
  keys: Record<string, KeyEntry>;
  roles: Record<string, RoleDetail> & { backup_brains?: { refs: ShownRef[] | null; source: string } };
  custom: CustomEndpoint[];
  voice: { value: string; source: string; live_possible: boolean };
  cloud_eyes: { value: string; source: string };
  daily_budgets: { value: Record<string, number>; source: string; prefill: Record<string, number> };
  store: { path: string; ok: boolean; problem?: string | null; old_copies?: number };
  summary?: BrainsSummary;
}

export interface ModelEntry { id: string; label: string; sees: boolean; tools: boolean; context: number | null; tier?: string; price?: unknown; note?: string }
export interface ModelsResult {
  id?: string; provider: string; ok: boolean; models: ModelEntry[]; defaults: Record<string, string>;
  error?: string | null; message?: string;
}

export type KeyStatus = "ok" | "bad_key" | "no_access" | "no_credit" | "rate_limited" | "not_found" | "unreachable" | "timeout" | "error";
export interface KeyTestResult {
  id?: string; provider: string; ok: boolean; status: KeyStatus | string; http: number | null; message: string;
  models: number | null; credit: Record<string, unknown> | null;
}

/** Any command's answer: what every result shares, then its own fields. */
export interface Result { id?: string; ok: boolean; error?: string; [k: string]: unknown }

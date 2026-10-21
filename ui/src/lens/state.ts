// Search Screen's overlay as one pure reducer: the shell's phases, the core's messages, the user's
// clicks, drags and keys go in; the next state and what to do about it (send this, close, peek…)
// come out. LensApp.tsx runs the effects; the tests drive it with fakes. The pointer merely moving
// never comes here (LensApp writes a corner being dragged and smart pick's outline directly).

import { atLeast, box, iou, nudge, toRect, type Box, type Pt } from "./geometry.ts";
import { DOUBLE_MS, pick, regionAt, tabStops, windowAt } from "./pick.ts";
import { FLY_RESET_MS, FLY_RESET_REDUCED_MS, flyPlan, type FlyPlan } from "./fly.ts";
import {
  answerProblem, emptyResult, MAX_THREAD, NOTE_MS, privacyLine as resultPrivacyLine, replyProblem, resultAction, resultMsg,
  type Result, type ResultAction, type ResultOp, type Turn,
} from "./result.ts";
import { send, type CoreMsg, type How, type LensHello, type Regions } from "./wire.ts";
import type { PhaseEvent, ShellHello } from "./door.ts";

export { answerProblem, MAX_THREAD, NOTE_MS, replyProblem, type Turn };

export type Phase = "idle" | "warm" | "preview" | "armed" | "open" | "cancel" | "closing";
/** A card in place of the selection: no Screen Recording, no core yet, the core went, no line to it. */
export type Card = null | "permission" | "starting" | "restarted" | "unreachable";

export interface Flags { reduceMotion: boolean; reduceTransparency: boolean; increaseContrast: boolean; voiceover: boolean }

/** An area on the overlay: its results (./result.ts, shared with the card in the notch) and where
 *  it is and how far along. */
export interface Sel extends Result {
  box: Box;
  how: How;
  /** The box the core was last asked about (an adjustment re-asks only when it moved enough). */
  searched: Box;
  /** Sent to the core (it waits for the line and for the still). */
  sent: boolean;
  /** No still: the core captures the area live, with the overlay out of the way. */
  live: boolean;
  captured: "none" | "pending" | "ok" | "failed";
  /** Handed to the notch (NotchHub shows the results instead of the sheet): `due` — not asked yet (a
   *  pick waits for a double-click, live mode for the capture); `waiting` — asked; `ok` — it flies;
   *  `no` — the sheet shows here as before. */
  handoff: Handoff;
  /** Hand off without waiting for a double-click (it was one, or it isn't a pick). */
  quick: boolean;
}

export type Handoff = "none" | "due" | "waiting" | "ok" | "no";
/** An area on its way to the notch: no sheet, no adjusting, no nudging. */
export const handingOff = (sel: Sel | null): boolean => !!sel && (sel.handoff === "due" || sel.handoff === "waiting");

export interface LensState {
  g: number;
  phase: Phase;
  /** When the phase began (the page's clock, ms), and for a preview how long until it arms. */
  phaseAt: number;
  toArmMs: number;
  via: string;
  why: string;
  flags: Flags;
  safeTop: number;
  W: number;
  H: number;
  /** Who draws the glow: NEO.app's shell (Core Animation) or, when it can't, this page. */
  glow: "native" | "web";
  /** The overlay's display (CGDirectDisplayID), when the shell says. */
  displayId: number | null;
  /** The area flying into the notch (the overlay is closing). */
  fly: FlyPlan | null;
  /** The notch said no this session (its camera or Allow / Cancel hold the island): the areas after
   *  it — an adjustment, a new drag — stay here with the sheet rather than ask again. */
  notchRefused: boolean;
  ticket: string | null;
  /** The core heard the ticket (null: not known yet). */
  core: boolean | null;
  /** Screen Recording, as the shell sees it. */
  sc: boolean;
  socket: "idle" | "connecting" | "open" | "failed";
  hello: LensHello | null;
  frozen: "pending" | "ok" | "no";
  frozenWhy: string;
  still: string | null;
  regions: Regions | null;
  sel: Sel | null;
  nextSel: number;
  nextQid: number;
  card: Card;
  prefill: string | null;
  /** Tab's candidate (an index into tabStops), -1 for none. */
  candidate: number;
  pendingPick: { at: Pt; double: boolean } | null;
  /** Bumped by each arrow key / adjustment: only the last one's timer re-asks. */
  moveV: number;
  /** The language the user last translated into (kept by the page across sessions). */
  translateTo: string | null;
  coreState: string;
}

export type KeyInfo = {
  key: string;
  meta?: boolean;
  shift?: boolean;
  alt?: boolean;
  ctrl?: boolean;
  /** An input method is composing (the IME gets Esc and Return). */
  composing?: boolean;
  /** The sheet's question box has the keys. */
  inComposer?: boolean;
  /** Text is selected on the page (⌘C copies that, as usual). */
  domSelection?: boolean;
  pointer?: Pt;
};

export type Action =
  | { t: "phase"; ev: PhaseEvent; now: number }
  | { t: "shell"; hello: ShellHello; now: number }
  | { t: "size"; W: number; H: number }
  | { t: "socket"; status: "connecting" | "open" | "failed" }
  | { t: "msg"; msg: CoreMsg }
  | { t: "still"; g: number; url: string }
  | { t: "reset"; g: number }
  | { t: "select"; box: Box; how: How }
  | { t: "click"; at: Pt; double: boolean }
  | { t: "pick_now"; g: number }
  | { t: "adjusted"; box: Box }
  | { t: "move_due"; v: number }
  | { t: "handoff_due"; n: number }
  | { t: "handoff_late"; n: number }
  | { t: "key"; key: KeyInfo }
  | { t: "chip"; kind: string; id?: string; to?: string }
  | { t: "translate_to"; to: string }
  | { t: "ask"; q: string }
  | { t: "ask_again" }
  | { t: "continue"; text: string }
  | { t: "listen" }
  | { t: "note_clear"; n: number; id: string }
  | { t: "close"; why: string }
  | { t: "allow_sc" };

export type Effect =
  | { t: "send"; cmd: Record<string, unknown> }
  /** Live mode: the overlay steps aside (peek), then this goes; lens_captured brings it back. */
  | { t: "send_live"; cmd: Record<string, unknown> }
  | { t: "peek"; hidden: boolean }
  | { t: "close"; why: string }
  | { t: "allow_sc" }
  | { t: "connect"; ticket: string }
  | { t: "disconnect" }
  | { t: "shell_hello" }
  | { t: "later"; ms: number; action: Action }
  | { t: "revoke"; url: string }
  | { t: "focus_composer"; text: string }
  /** The shell's glow: an area is selected now (narrow and dim), or not. */
  | { t: "glow"; selected: boolean }
  /** The area's crop, cut from the still now, ready to fly the moment the notch says yes. */
  | { t: "fly_prep"; box: Box };

/** An adjusted area is asked about again only when it changed this much. */
export const REASK_IOU = 0.92;
export const ADJUST_WAIT_MS = 250;
export const NUDGE_WAIT_MS = 400;
/** A click waits this long at most for smart pick's regions (they come as soon as the still is in). */
export const PICK_WAIT_MS = 3000;
export const FADE_MS = 220;
/** A pick hands off once a double-click can no longer make it a paragraph. */
export const HANDOFF_PICK_MS = DOUBLE_MS + 20;
/** No answer to `lens_handoff` this long (the core waits at most ~4.3 s): the sheet, here. */
export const HANDOFF_GIVE_UP_MS = 5000;
export const CARD_CLOSE_MS = 3000;

const NO_FLAGS: Flags = { reduceMotion: false, reduceTransparency: false, increaseContrast: false, voiceover: false };

export function initial(W = 0, H = 0, translateTo: string | null = null): LensState {
  return {
    g: 0, phase: "idle", phaseAt: 0, toArmMs: 0, via: "hold", why: "", flags: NO_FLAGS, safeTop: 0, W, H, glow: "web", displayId: null, fly: null, notchRefused: false,
    ticket: null, core: null, sc: true, socket: "idle", hello: null, frozen: "pending", frozenWhy: "", still: null,
    regions: null, sel: null, nextSel: 1, nextQid: 1, card: null, prefill: null, candidate: -1, pendingPick: null,
    moveV: 0, translateTo, coreState: "",
  };
}

/** A session's own things gone (its still, its line, its areas); the page's own kept. */
function fresh(s: LensState, g: number): LensState {
  return { ...initial(s.W, s.H, s.translateTo), g, flags: s.flags, safeTop: s.safeTop, glow: s.glow, displayId: s.displayId };
}

function teardown(s: LensState): Effect[] {
  const fx: Effect[] = [];
  if (s.socket !== "idle") fx.push({ t: "disconnect" });
  if (s.still) fx.push({ t: "revoke", url: s.still });
  return fx;
}

const PHASES: Record<string, Phase> = {
  hidden: "idle", warm: "warm", preview: "preview", armed: "armed", open: "open", cancel: "cancel", closing: "closing",
};

function flagsOf(ev: PhaseEvent, prev: Flags): Flags {
  return {
    reduceMotion: typeof ev.reduce_motion === "boolean" ? ev.reduce_motion : prev.reduceMotion,
    reduceTransparency: typeof ev.reduce_transparency === "boolean" ? ev.reduce_transparency : prev.reduceTransparency,
    increaseContrast: typeof ev.increase_contrast === "boolean" ? ev.increase_contrast : prev.increaseContrast,
    voiceover: typeof ev.voiceover === "boolean" ? ev.voiceover : prev.voiceover,
  };
}

function onPhase(s: LensState, ev: PhaseEvent, now: number, sinceMs = 0): [LensState, Effect[]] {
  if (!Number.isFinite(ev.g) || ev.g < s.g) return [s, []]; // an older session's
  const fx: Effect[] = [];
  let st = s;
  if (ev.g > s.g) {
    fx.push(...teardown(s));
    st = fresh(s, ev.g);
  }
  if (ev.phase === "gone") return [{ ...st, card: "restarted" }, fx];
  const phase = PHASES[ev.phase];
  if (!phase) return [st, fx];
  const changed = phase !== st.phase;
  st = {
    ...st,
    phase,
    phaseAt: changed ? now - Math.max(0, sinceMs) : st.phaseAt,
    toArmMs: typeof ev.to_arm_ms === "number" && phase === "preview" ? ev.to_arm_ms : st.toArmMs,
    via: typeof ev.via === "string" && ev.via ? ev.via : st.via,
    why: typeof ev.why === "string" ? ev.why : st.why,
    flags: flagsOf(ev, st.flags),
    safeTop: typeof ev.safe_top === "number" && Number.isFinite(ev.safe_top) ? ev.safe_top : st.safeTop,
    glow: ev.glow === "native" || ev.glow === "web" ? ev.glow : st.glow,
    displayId: typeof ev.display?.id === "number" && Number.isInteger(ev.display.id) && ev.display.id >= 0 ? ev.display.id : st.displayId,
  };
  if ((phase === "armed" || phase === "open") && !st.ticket) fx.push({ t: "shell_hello" });
  if (changed && (phase === "cancel" || phase === "closing")) {
    if (st.socket !== "idle") fx.push({ t: "disconnect" });
    st = { ...st, socket: "idle" };
    // Closing into the notch: the page stays for the flight (the shell orders it out at 440 ms).
    const notch = phase === "closing" && st.why === "notch";
    const ms = notch ? (st.flags.reduceMotion ? FLY_RESET_REDUCED_MS : FLY_RESET_MS) : FADE_MS;
    fx.push({ t: "later", ms, action: { t: "reset", g: st.g } });
  }
  return [st, fx];
}

/** The pending area goes out once there is a line and the still is known (taken, or not: live).
 *  With NotchHub able to show the results (the core's hello says `notch`), it is handed off too:
 *  at once, or — a pick — once a double-click can't make it a paragraph, or — live — once it is
 *  captured. */
function flush(s: LensState): [LensState, Effect[]] {
  const sel = s.sel;
  if (!sel || sel.sent || s.socket !== "open" || s.frozen === "pending" || s.card) return [s, []];
  const live = s.frozen === "no";
  const cmd = send.select(sel.n, toRect(sel.box), sel.how, live);
  const fx: Effect[] = [live ? { t: "send_live", cmd } : { t: "send", cmd }];
  let next: Sel = { ...sel, sent: true, live, captured: live ? "pending" : "none" };
  if (s.hello?.notch && !s.notchRefused) {
    next = { ...next, handoff: "due" };
    if (live) fx.push({ t: "later", ms: HANDOFF_GIVE_UP_MS, action: { t: "handoff_late", n: sel.n } });
    else if (!sel.quick) fx.push({ t: "later", ms: HANDOFF_PICK_MS, action: { t: "handoff_due", n: sel.n } });
    else return askHandoff({ ...s, sel: next }, fx);
  }
  return [{ ...s, sel: next }, fx];
}

/** Ask the core to show the area in the notch (the crop is cut now, so it can fly the moment it may). */
function askHandoff(s: LensState, fx: Effect[] = []): [LensState, Effect[]] {
  const sel = s.sel;
  if (!sel || sel.handoff !== "due") return [s, fx];
  const out: Effect[] = [...fx, { t: "send", cmd: send.handoff(sel.n, s.translateTo) }];
  if (!sel.live) out.push({ t: "fly_prep", box: sel.box });
  out.push({ t: "later", ms: HANDOFF_GIVE_UP_MS, action: { t: "handoff_late", n: sel.n } });
  return [{ ...s, sel: { ...sel, handoff: "waiting" } }, out];
}

/** The notch won't (or didn't say): the sheet, here — and in live mode the overlay comes back. */
function noHandoff(s: LensState, n: number): [LensState, Effect[]] {
  const sel = s.sel;
  if (!sel || sel.n !== n || !handingOff(sel)) return [s, []];
  return [{ ...s, sel: { ...sel, handoff: "no" }, notchRefused: true }, sel.live ? [{ t: "peek", hidden: false }] : []];
}

function newSel(s: LensState, b: Box, how: How, double = false): [LensState, Effect[]] {
  if (s.phase !== "open" || s.card || !s.W || !s.H) return [s, []];
  const area = atLeast(b, s.W, s.H);
  const first = s.nextSel === 1;
  const sel: Sel = {
    ...emptyResult(s.nextSel, first ? s.prefill : null),
    box: area, how, searched: area, sent: false, live: false, captured: "none", handoff: "none", quick: how !== "pick" || double,
  };
  return flush({ ...s, sel, nextSel: s.nextSel + 1, candidate: -1, pendingPick: null });
}

function withSel(s: LensState, n: number, f: (sel: Sel) => Sel): LensState {
  return s.sel && s.sel.n === n ? { ...s, sel: f(s.sel) } : s;
}

/** The overlay's command for one of the results' ops (the card in the notch sends its own). */
function opCmd(n: number, o: ResultOp): Record<string, unknown> {
  switch (o.op) {
    case "ask": return send.ask(n, o.qid, o.q);
    case "ask_again": return send.ask(n, 0, "");
    case "translate": return send.translate(n, o.to);
    case "act": return send.act(n, o.id);
    case "continue": return send.cont(n, o.text);
    case "listen": return send.listen(n);
  }
}

/** A click or key on the sheet's results (./result.ts), as the overlay's commands. */
function onResult(s: LensState, a: ResultAction): [LensState, Effect[]] {
  const sel = s.sel;
  if (!sel) return a.t === "translate_to" ? [{ ...s, translateTo: a.to }, []] : [s, []];
  const out = resultAction(sel, a, s.nextQid);
  const fx: Effect[] = out.ops.map((o) => ({ t: "send", cmd: opCmd(sel.n, o) }));
  return [{ ...s, sel: out.r, nextQid: out.nextQid, translateTo: out.translateTo ?? s.translateTo }, fx];
}

function onMsg(s: LensState, m: CoreMsg): [LensState, Effect[]] {
  switch (m.t) {
    case "lens_hello": {
      const h = m.data;
      let st: LensState = { ...s, hello: h, frozen: h.frozen === "ok" ? "ok" : h.frozen === "no" && s.frozen !== "ok" ? "no" : s.frozen, prefill: h.prefill ?? s.prefill };
      if (!h.perms.screen && st.phase === "open") st = { ...st, card: "permission" };
      return flush(st);
    }
    case "lens_frozen": {
      const why = m.data.why ?? "";
      let st: LensState = { ...s, frozen: m.data.ok ? "ok" : "no", frozenWhy: m.data.ok ? "" : why };
      if (!m.data.ok && why === "no_permission") st = { ...st, card: "permission" };
      return flush(st);
    }
    case "lens_regions": {
      const st = { ...s, regions: m.data };
      const pp = st.pendingPick;
      // The first part (lines, paragraphs) picks a line or paragraph now; anything else waits for
      // the things (an object, the window), up to PICK_WAIT_MS.
      return pp && pickable(m.data, pp.at, st, pp.double) ? doPick(st) : [st, []];
    }
    case "lens_captured": {
      if (!s.sel || s.sel.n !== m.data.sel) return [s, []];
      const st = withSel(s, m.data.sel, (x) => ({ ...x, captured: m.data.ok ? "ok" : "failed" }));
      // Into the notch: the overlay stays out of the way (no sheet will show here).
      if (st.sel?.handoff === "due") return m.data.ok ? askHandoff(st) : [{ ...st, sel: { ...st.sel, handoff: "no" } }, [{ t: "peek", hidden: false }]];
      return [st, [{ t: "peek", hidden: false }]];
    }
    case "lens_handoff": {
      const h = m.data, sel = s.sel;
      if (!sel || sel.n !== h.sel || sel.handoff !== "waiting" || s.phase !== "open") return [s, []]; // an older area's, or closing anyway
      if (!h.ok) return noHandoff(s, sel.n);
      const same = h.display === null || s.displayId === null || h.display === s.displayId;
      const fly = sel.live ? null : flyPlan(sel.box, s.W, s.H, s.safeTop, same, s.flags.reduceMotion);
      return [{ ...s, sel: { ...sel, handoff: "ok" }, fly }, [{ t: "close", why: "notch" }]];
    }
    case "lens_local":
    case "lens_answer":
    case "lens_links":
    case "lens_reply":
    case "lens_translation":
      return [withSel(s, m.data.sel, (x) => resultMsg(x, m).r), []];
    case "lens_done": {
      const d = m.data;
      if (!s.sel || s.sel.n !== d.sel) return [s, []];
      const { r, noteClear } = resultMsg(s.sel, m);
      return [{ ...s, sel: r }, noteClear ? [{ t: "later", ms: noteClear.ms, action: { t: "note_clear", n: d.sel, id: noteClear.id } }] : []];
    }
    case "state":
      return [{ ...s, coreState: m.data.state }, []];
    case "lens_listening":
      return [{ ...s, coreState: m.data.on ? "listening" : "" }, []];
    case "lens_refused":
      return [s, []]; // (stale: an older area's; the newer one's answers come)
  }
  return [s, []];
}

/** Whether a click at `at` can be picked now: all the regions are in, or the first part already
 *  has the line (or paragraph) under it. */
function pickable(r: Regions | null, at: Pt, s: LensState, double: boolean): boolean {
  if (!r) return false;
  if (r.done !== false) return true;
  const hit = regionAt(r, at, s.W, s.H, double);
  return !!hit && (hit.source === "line" || hit.source === "block");
}

function doPick(s: LensState): [LensState, Effect[]] {
  const pp = s.pendingPick;
  if (!pp) return [s, []];
  const p = pick(s.regions, pp.at, s.W, s.H, pp.double);
  return newSel({ ...s, pendingPick: null }, p.box, "pick", pp.double);
}

/** Whether the keys are the sheet's or the page's: a printable key with an area goes to the question box. */
const printable = (k: KeyInfo) => k.key.length === 1 && !k.meta && !k.ctrl && k.key !== " ";

function onKey(s: LensState, k: KeyInfo): [LensState, Effect[]] {
  if (k.composing) return [s, []];
  if (k.key === "Escape") return s.phase === "idle" ? [s, []] : [s, [{ t: "close", why: "esc" }]];
  if (s.phase !== "open" || s.card) return [s, []];
  if (k.key === "Enter" && k.meta) return s.sel ? [s, [{ t: "send", cmd: send.cont(s.sel.n, "") }]] : [s, []];
  if (k.inComposer) return [s, []];
  const { W, H } = s;
  const stops = () => tabStops(s.regions, W, H);
  if (k.key === "Tab") {
    if (s.sel) return [s, []]; // with an area, Tab walks the sheet's controls
    const all = stops();
    if (!all.length) return [s, []];
    const n = all.length;
    const i = s.candidate < 0 ? (k.shift ? n - 1 : 0) : (s.candidate + (k.shift ? -1 : 1) + n) % n;
    return [{ ...s, candidate: i }, []];
  }
  if (k.key === "Enter") {
    const all = stops();
    if (s.candidate >= 0 && s.candidate < all.length) return newSel(s, all[s.candidate], "key");
    if (s.sel) return [s, []];
    const at = k.pointer ?? { x: W / 2, y: H / 2 };
    return newSel(s, windowAt(s.regions, at) ?? box(0, 0, W, H), "key"); // one key, no pointer path (WCAG 2.5.7)
  }
  if (k.meta && k.key.toLowerCase() === "a") return newSel(s, box(0, 0, W, H), "all");
  if (k.meta && k.key.toLowerCase() === "c") {
    if (k.domSelection || !s.sel) return [s, []];
    const copy = s.sel.local?.actions.find((a) => a.kind === "copy_text");
    if (!copy) return [s, []];
    return [withSel(s, s.sel.n, (x) => ({ ...x, notes: { ...x.notes, [copy.id]: "…" } })), [{ t: "send", cmd: send.act(s.sel.n, copy.id) }]];
  }
  const arrows: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
  if (handingOff(s.sel) || s.sel?.handoff === "ok") return [s, []]; // on its way to the notch
  if (arrows[k.key] && s.sel) {
    const step = k.shift ? 10 : 1;
    const [dx, dy] = arrows[k.key];
    const b = nudge(s.sel.box, dx * step, dy * step, !!k.alt, W, H);
    const v = s.moveV + 1;
    return [{ ...withSel(s, s.sel.n, (x) => ({ ...x, box: b })), moveV: v }, [{ t: "later", ms: NUDGE_WAIT_MS, action: { t: "move_due", v } }]];
  }
  if (printable(k) && s.sel) return [s, [{ t: "focus_composer", text: k.key }]];
  return [s, []];
}

/** The reducer: one step, and the shell's glow told whenever an area appears or goes while open. */
export function reduce(s: LensState, a: Action): [LensState, Effect[]] {
  const [next, fx] = step(s, a);
  if (next.glow === "native" && next.phase === "open" && !!s.sel !== !!next.sel) return [next, [...fx, { t: "glow", selected: !!next.sel }]];
  return [next, fx];
}

/** Whether this page draws the glow itself (the shell said it couldn't, or an older shell said nothing). */
export const showsWebGlow = (h: { glow?: string } | null | undefined): boolean => h?.glow !== "native";

function step(s: LensState, a: Action): [LensState, Effect[]] {
  switch (a.t) {
    case "phase":
      return onPhase(s, a.ev, a.now);
    case "shell": {
      const h = a.hello;
      if (!Number.isFinite(h.g) || h.g < s.g) return [s, []];
      let [st, fx] = onPhase(s, h, a.now, h.since_ms ?? 0);
      if (h.g !== st.g) return [st, fx];
      st = { ...st, sc: h.sc !== false, core: typeof h.core === "boolean" ? h.core : st.core };
      const ticket = typeof h.ticket === "string" && /^[0-9a-f]{64}$/.test(h.ticket) ? h.ticket : null;
      fx = fx.filter((e) => !(e.t === "shell_hello" && ticket));
      if (ticket) st = { ...st, ticket };
      const live = st.phase === "armed" || st.phase === "open";
      if (st.ticket && live && st.socket === "idle") {
        fx.push({ t: "connect", ticket: st.ticket });
        st = { ...st, socket: "connecting" };
      }
      if (st.phase === "open" && !st.card) {
        if (st.core === false) {
          st = { ...st, card: "starting" };
          fx.push({ t: "later", ms: CARD_CLOSE_MS, action: { t: "close", why: "cancel" } });
        } else if (!st.sc) {
          st = { ...st, card: "permission" };
        }
      }
      return [st, fx];
    }
    case "size":
      return [{ ...s, W: a.W, H: a.H }, []];
    case "socket": {
      if (a.status === "open") {
        const st: LensState = { ...s, socket: "open" };
        const [f, fx] = flush(st);
        return [f, [{ t: "send", cmd: send.hello(s.flags.voiceover) }, ...fx]];
      }
      if (a.status === "failed") {
        const st: LensState = { ...s, socket: "failed" };
        if (s.phase === "open" && !s.card) {
          return [{ ...st, card: "unreachable" }, [{ t: "later", ms: CARD_CLOSE_MS, action: { t: "close", why: "gone" } }]];
        }
        return [st, []];
      }
      return [{ ...s, socket: s.socket === "idle" ? "idle" : "connecting" }, []];
    }
    case "msg":
      return onMsg(s, a.msg);
    case "still":
      if (a.g !== s.g) return [s, [{ t: "revoke", url: a.url }]];
      return [{ ...s, still: a.url }, s.still && s.still !== a.url ? [{ t: "revoke", url: s.still }] : []];
    case "reset":
      if (a.g !== s.g || (s.phase !== "cancel" && s.phase !== "closing")) return [s, []];
      return [{ ...fresh(s, s.g), phase: "idle" }, teardown({ ...s, socket: "idle" })];
    case "select":
      return newSel(s, a.box, a.how);
    case "click": {
      if (s.phase !== "open" || s.card) return [s, []];
      if (!pickable(s.regions, a.at, s, a.double) && s.frozen !== "no") {
        // Smart pick waits for what the core found in the still (a pulsing dot shows), at most PICK_WAIT_MS.
        return [{ ...s, pendingPick: { at: a.at, double: a.double } }, [{ t: "later", ms: PICK_WAIT_MS, action: { t: "pick_now", g: s.g } }]];
      }
      return newSel(s, pick(s.regions, a.at, s.W, s.H, a.double).box, "pick", a.double);
    }
    case "pick_now":
      return a.g === s.g ? doPick(s) : [s, []];
    case "adjusted": {
      if (!s.sel || handingOff(s.sel) || s.sel.handoff === "ok") return [s, []];
      const v = s.moveV + 1;
      const st = { ...withSel(s, s.sel.n, (x) => ({ ...x, box: atLeast(a.box, s.W, s.H) })), moveV: v };
      return [st, [{ t: "later", ms: ADJUST_WAIT_MS, action: { t: "move_due", v } }]];
    }
    case "handoff_due":
      return s.sel && s.sel.n === a.n && !s.sel.live ? askHandoff(s) : [s, []];
    case "handoff_late":
      return noHandoff(s, a.n);
    case "move_due": {
      if (a.v !== s.moveV || !s.sel || handingOff(s.sel)) return [s, []];
      if (iou(s.sel.box, s.sel.searched) >= REASK_IOU) return [s, []];
      return newSel(s, s.sel.box, "adjust");
    }
    case "key":
      return onKey(s, a.key);
    case "chip":
    case "translate_to":
    case "ask":
    case "ask_again":
    case "continue":
    case "listen":
      return onResult(s, a);
    case "note_clear":
      return [withSel(s, a.n, (x) => resultAction(x, { t: "note_clear", id: a.id }, s.nextQid).r), []];
    case "close":
      return [s, [{ t: "close", why: a.why }]];
    case "allow_sc":
      return [s, [{ t: "allow_sc" }]];
  }
  return [s, []];
}

/** What the hint pill says (null: hidden — there is an area, or a card). */
export function hint(s: LensState): string | null {
  if (s.phase !== "open" || s.sel || s.card) return null;
  if (s.prefill) return `Circle what you mean — “${s.prefill}”`;
  return "Circle or drag over anything · Click to pick · Esc to close";
}

/** The privacy line under the sheet (./result.ts): every place the area, its text or a search
 *  phrase from it went — "Stayed on this Mac" only when nothing left it. */
export function privacyLine(s: LensState): string {
  return resultPrivacyLine(s.sel, s.hello);
}

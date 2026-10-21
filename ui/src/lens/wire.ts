// Search Screen's messages with the core, as the core sends them (neo2/core/lens/live.py; the
// protocol is written down in research-2026-10-21-lens/core-protocol.md). Everything that arrives is
// checked and cut to size here, so the overlay never shows more than the protocol allows; everything
// that goes out is built here, with the protocol's own limits.

export type Rect = [number, number, number, number]; // x, y, w, h in display-local points
export type How = "drag" | "lasso" | "pick" | "key" | "adjust" | "all";

export interface Brain { label: string; local: boolean }
/** Where an area (or what it holds) went: a brain the picture or its text was sent to, or the
 *  search phrase sent to NEO's web search — every one asked, not only the one that answered. */
export interface Sent { label: string; local: boolean; what: "picture" | "text" | "query" }
export interface SearchProvider { word: string; label: string; image: boolean; text: boolean }
export interface Lang { code: string; name: string }

export interface LensHello {
  v: number;
  open: boolean;
  frozen: "pending" | "ok" | "no";
  display: { w: number; h: number; scale: number } | null;
  via: string;
  perms: { screen: boolean };
  eyes: Brain | null;
  eyesWhy: "" | "no_brain" | "local_only";
  ask: boolean;
  search: SearchProvider | null;
  langs: Lang[];
  prefill: string | null;
  /** A hint from the core: NotchHub hosts NEO and could show the results (a hand-off may work now). */
  notch: boolean;
}

export interface Regions {
  lines: { box: Rect; block: number }[];
  blocks: Rect[];
  objects: Rect[];
  instances: Rect[];
  windows: Rect[];
  texts: string[];
  /** False for the first part (lines, paragraphs, windows): the things are still coming. */
  done?: boolean;
}

export type ActionKind =
  | "copy_text" | "open" | "mail" | "call" | "wifi" | "code" | "copy_image" | "image_search" | "text_search" | "brains";
export const ACTION_KINDS: readonly ActionKind[] = [
  "copy_text", "open", "mail", "call", "wifi", "code", "copy_image", "image_search", "text_search", "brains",
];
export interface Action { id: string; kind: ActionKind; label: string; detail?: string; warn?: "userinfo" | "punycode" }

export interface LocalData {
  sel: number;
  text: string;
  lines: number;
  lang: { code: string; name: string; conf: number } | null;
  translateTo: string;
  dark: boolean;
  actions: Action[];
  thumb?: string;
}

export type AnswerError =
  | "no_brain" | "local_only" | "offline" | "timeout" | "failed" | "dark" | "refused" | "off" | "key" | "busy";
export interface AnswerData {
  sel: number;
  state: "thinking" | "done" | "failed";
  kind?: string;
  title?: string;
  answer?: string;
  query?: string;
  brain?: Brain;
  error?: AnswerError;
  actions: Action[];
  sentTo?: Sent[];
}
export interface LinkRow { id: string; title: string; host: string }
export interface ReplyData { sel: number; qid: number; state: "thinking" | "done" | "failed"; text?: string; brain?: Brain; error?: string; sentTo?: Sent[] }
export interface TranslationData { sel: number; to: string; text?: string; error?: string; sentTo?: Sent[] }
export interface DoneData { sel: number; id: string; ok: boolean; note: string; close: boolean }

export type CoreMsg =
  | { t: "lens_hello"; data: LensHello }
  | { t: "lens_frozen"; data: { ok: boolean; ms?: number; why?: string } }
  | { t: "lens_regions"; data: Regions }
  | { t: "lens_captured"; data: { sel: number; ok: boolean; error?: string } }
  | { t: "lens_local"; data: LocalData }
  | { t: "lens_answer"; data: AnswerData }
  | { t: "lens_links"; data: { sel: number; rows: LinkRow[]; sentTo?: Sent[] } }
  | { t: "lens_reply"; data: ReplyData }
  | { t: "lens_translation"; data: TranslationData }
  | { t: "lens_done"; data: DoneData }
  | { t: "lens_refused"; data: { cmd: string; error: string } }
  | { t: "lens_listening"; data: { on: boolean } }
  | { t: "lens_handoff"; data: HandoffData }
  | { t: "state"; data: { state: string } };

/** The core's answer to `lens_handoff`: NotchHub shows the area's results (`display`: its island's
 *  display, when known), or it won't (`why`), and the sheet shows here as before. */
export interface HandoffData { sel: number; ok: boolean; display: number | null; why: string }
export const HANDOFF_WHYS = ["no_notch", "unshown", "timeout", "stale"] as const;

// Lengths the page never goes past, whatever arrives.
export const MAX = { label: 90, detail: 120, title: 60, answer: 1200, text: 8000, reply: 2000, host: 120, q: 500, cont: 2000, lang: 16 } as const;
const MAX_BOXES = 400;
const ACTION_ID = /^a\d{1,3}$/;

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj => (v && typeof v === "object" && !Array.isArray(v) ? (v as Obj) : {});
const str = (v: unknown, max: number): string => (typeof v === "string" ? v.slice(0, max) : "");
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const int = (v: unknown): number | null => (typeof v === "number" && Number.isInteger(v) ? v : null);
const bool = (v: unknown): boolean => v === true;

function rect(v: unknown): Rect | null {
  if (!Array.isArray(v) || v.length < 4) return null;
  const [x, y, w, h] = v.map(num);
  if (x === null || y === null || w === null || h === null || w <= 0 || h <= 0) return null;
  return [x, y, w, h];
}

function rects(v: unknown): Rect[] {
  if (!Array.isArray(v)) return [];
  const out: Rect[] = [];
  for (const r of v.slice(0, MAX_BOXES)) {
    const b = rect(r);
    if (b) out.push(b);
  }
  return out;
}

function brain(v: unknown): Brain | null {
  const o = obj(v);
  const label = str(o.label, 60).trim();
  return label ? { label, local: bool(o.local) } : null;
}

const WHATS: readonly Sent["what"][] = ["picture", "text", "query"];

/** `sent_to`, checked (undefined when the message carried none). */
export function sentTo(v: unknown): Sent[] | undefined {
  if (!Array.isArray(v)) return undefined;
  const out: Sent[] = [];
  for (const x of v.slice(0, 12)) {
    const b = brain(x);
    const what = obj(x).what as Sent["what"];
    if (b && WHATS.includes(what)) out.push({ ...b, what });
  }
  return out;
}

export function action(v: unknown): Action | null {
  const o = obj(v);
  const id = str(o.id, 8);
  const kind = str(o.kind, 20) as ActionKind;
  const label = str(o.label, MAX.label).trim();
  if (!ACTION_ID.test(id) || !ACTION_KINDS.includes(kind) || !label) return null;
  const a: Action = { id, kind, label };
  const detail = str(o.detail, MAX.detail).trim();
  if (detail) a.detail = detail;
  if (o.warn === "userinfo" || o.warn === "punycode") a.warn = o.warn;
  return a;
}

function actions(v: unknown): Action[] {
  if (!Array.isArray(v)) return [];
  return v.slice(0, 40).map(action).filter((a): a is Action => a !== null);
}

/** What the results need of a hello (the overlay's lens_hello, a card's snapshot): the brain an
 *  area goes to first, Ask, the search provider, the Translate menu. */
export function resultHello(v: unknown): Pick<LensHello, "eyes" | "eyesWhy" | "ask" | "search" | "langs"> {
  const o = obj(v);
  const s = obj(o.search);
  const label = str(s.label, 40).trim();
  const why = o.eyes_why === "no_brain" || o.eyes_why === "local_only" ? o.eyes_why : "";
  const langs: Lang[] = [];
  if (Array.isArray(o.langs)) {
    for (const l of o.langs.slice(0, 40)) {
      const lo = obj(l);
      const code = str(lo.code, MAX.lang).trim(), name = str(lo.name, 40).trim();
      if (code && name) langs.push({ code, name });
    }
  }
  return {
    eyes: brain(o.eyes),
    eyesWhy: why,
    ask: o.ask !== false,
    search: label ? { word: str(s.word, 40), label, image: bool(s.image), text: bool(s.text) } : null,
    langs,
  };
}

export function hello(v: unknown): LensHello {
  const o = obj(v);
  const d = obj(o.display);
  const w = num(d.w), h = num(d.h), scale = num(d.scale);
  const prefill = str(o.prefill, MAX.q).trim();
  return {
    v: int(o.v) ?? 1,
    open: bool(o.open),
    frozen: o.frozen === "ok" || o.frozen === "pending" ? o.frozen : "no",
    display: w && h && scale ? { w, h, scale } : null,
    via: str(o.via, 10),
    perms: { screen: obj(o.perms).screen !== false },
    ...resultHello(o),
    prefill: prefill || null,
    notch: o.notch === true,
  };
}

export function regions(v: unknown): Regions {
  const o = obj(v);
  const lines: Regions["lines"] = [];
  if (Array.isArray(o.lines)) {
    for (const l of o.lines.slice(0, MAX_BOXES)) {
      const b = rect(l);
      if (!b) continue;
      const block = Array.isArray(l) ? int(l[4]) : null;
      lines.push({ box: b, block: block ?? -1 });
    }
  }
  const texts = Array.isArray(o.texts) ? o.texts.slice(0, MAX_BOXES).map((t) => str(t, 120)) : [];
  return { lines, blocks: rects(o.blocks), objects: rects(o.objects), instances: rects(o.instances), windows: rects(o.windows), texts, done: o.done !== false };
}

const ANSWER_ERRORS: readonly AnswerError[] = ["no_brain", "local_only", "offline", "timeout", "failed", "dark", "refused", "off", "key", "busy"];
const sent = (v: unknown) => { const s = sentTo(v); return s ? { sentTo: s } : {}; };

/** One text message from the core, checked; null for anything that isn't Search Screen's. */
export function parseMsg(text: string): CoreMsg | null {
  let raw: unknown;
  try { raw = JSON.parse(text); } catch { return null; }
  const m = obj(raw);
  return parseLens(typeof m.type === "string" ? m.type : "", m.data);
}

/** One of Search Screen's messages (its type and data, already JSON), checked; null for anything
 *  else. The card in the notch reads the same ones (./card.ts), with its `cid` beside. */
export function parseLens(type: string, data: unknown): CoreMsg | null {
  const d = obj(data);
  const sel = int(d.sel);
  switch (type) {
    case "hello": // the websocket's own hello carries the same lens_hello
      return { t: "lens_hello", data: hello(d.lens) };
    case "lens_hello":
      return { t: "lens_hello", data: hello(d) };
    case "lens_frozen":
      return { t: "lens_frozen", data: { ok: bool(d.ok), ms: num(d.ms) ?? undefined, why: str(d.why, 20) || undefined } };
    case "lens_regions":
      return { t: "lens_regions", data: regions(d) };
    case "state":
      return typeof d.state === "string" ? { t: "state", data: { state: d.state.slice(0, 20) } } : null;
    case "lens_refused":
      return { t: "lens_refused", data: { cmd: str(d.cmd, 32), error: str(d.error, 20) } };
    case "lens_listening":
      return { t: "lens_listening", data: { on: bool(d.on) } };
  }
  if (sel === null) return null;
  switch (type) {
    case "lens_captured":
      return { t: "lens_captured", data: { sel, ok: bool(d.ok), error: str(d.error, 20) || undefined } };
    case "lens_local": {
      const lang = obj(d.lang);
      const code = str(lang.code, MAX.lang), name = str(lang.name, 40);
      const thumb = typeof d.thumb === "string" && d.thumb.startsWith("data:image/jpeg;base64,") && d.thumb.length < 200_000 ? d.thumb : undefined;
      return {
        t: "lens_local",
        data: {
          sel,
          text: str(d.text, MAX.text),
          lines: Math.max(0, int(d.lines) ?? 0),
          lang: code && name ? { code, name, conf: num(lang.conf) ?? 0 } : null,
          translateTo: str(d.translate_to, MAX.lang),
          dark: bool(d.dark),
          actions: actions(d.actions),
          ...(thumb ? { thumb } : {}),
        },
      };
    }
    case "lens_answer": {
      const state = d.state === "done" || d.state === "failed" ? d.state : "thinking";
      const error = ANSWER_ERRORS.includes(d.error as AnswerError) ? (d.error as AnswerError) : state === "failed" ? "failed" : undefined;
      const b = brain(d.brain);
      return {
        t: "lens_answer",
        data: {
          sel, state,
          kind: str(d.kind, 12) || undefined,
          title: str(d.title, MAX.title).trim() || undefined,
          answer: str(d.answer, MAX.answer).trim() || undefined,
          query: str(d.query, 120).trim() || undefined,
          ...(b ? { brain: b } : {}),
          ...(error ? { error } : {}),
          actions: actions(d.actions),
          ...sent(d.sent_to),
        },
      };
    }
    case "lens_links": {
      const rows: LinkRow[] = [];
      if (Array.isArray(d.rows)) {
        for (const r of d.rows.slice(0, 8)) {
          const ro = obj(r);
          const id = str(ro.id, 8), title = str(ro.title, MAX.label).trim(), host = str(ro.host, MAX.host).trim();
          if (ACTION_ID.test(id) && title && host) rows.push({ id, title, host });
        }
      }
      return { t: "lens_links", data: { sel, rows, ...sent(d.sent_to) } };
    }
    case "lens_reply": {
      const qid = int(d.qid);
      if (qid === null) return null;
      const state = d.state === "done" || d.state === "failed" ? d.state : "thinking";
      const b = brain(d.brain);
      return {
        t: "lens_reply",
        data: { sel, qid, state, text: str(d.text, MAX.reply).trim() || undefined, ...(b ? { brain: b } : {}), error: str(d.error, 20) || undefined, ...sent(d.sent_to) },
      };
    }
    case "lens_translation":
      return { t: "lens_translation", data: { sel, to: str(d.to, MAX.lang), text: typeof d.text === "string" ? d.text.slice(0, MAX.text) : undefined, error: str(d.error, 20) || undefined, ...sent(d.sent_to) } };
    case "lens_done":
      return { t: "lens_done", data: { sel, id: str(d.id, 8), ok: bool(d.ok), note: str(d.note, 20), close: bool(d.close) } };
    case "lens_handoff": {
      const disp = int(d.display);
      const why = (HANDOFF_WHYS as readonly string[]).includes(d.why as string) ? (d.why as string) : "";
      return { t: "lens_handoff", data: { sel, ok: bool(d.ok), display: disp !== null && disp >= 0 && disp <= 0xffffffff ? disp : null, why } };
    }
  }
  return null;
}

// ---- the card in the notch ------------------------------------------------------------------------

/** A card's id: 32 lowercase hex, the core's own. */
export const CID = /^[0-9a-f]{32}$/;
export const CARD_WHYS = ["esc", "x", "away", "fold", "unshown", "timeout", "action", "continue", "chat", "stop", "erased", "replaced", "idle", "gone", "done"] as const;

/** Everything a card shows when it opens (or when the page asks again): lens_card's `card`. */
export interface CardData {
  sel: number;
  thumb: string | null;
  question: string | null;
  translateTo: string;
  hello: Pick<LensHello, "eyes" | "eyesWhy" | "ask" | "search" | "langs">;
  local: LocalData | null;
  answer: AnswerData | null;
  links: LinkRow[];
  thread: (ReplyData & { q: string })[];
  translation: TranslationData | null;
  sentTo: Sent[];
}
export type CardMsg = { cid: string; open: true; card: CardData | null } | { cid: string; open: false; why: string };

const thumbOf = (v: unknown): string | null =>
  typeof v === "string" && v.startsWith("data:image/jpeg;base64,") && v.length < 200_000 ? v : null;

/** `lens_card`, checked: a card opening (with what it shows), or closing (and why); null for a
 *  malformed one. */
export function parseCard(data: unknown): CardMsg | null {
  const d = obj(data);
  const cid = typeof d.cid === "string" && CID.test(d.cid) ? d.cid : null;
  if (!cid) return null;
  if (d.open !== true) {
    const why = (CARD_WHYS as readonly string[]).includes(d.why as string) ? (d.why as string) : "done";
    return { cid, open: false, why };
  }
  if (!d.card || typeof d.card !== "object") return { cid, open: true, card: null };
  const c = obj(d.card);
  const sel = int(c.sel);
  if (sel === null) return null;
  const as = <T extends CoreMsg["t"]>(t: T, v: unknown) => {
    if (!v || typeof v !== "object") return null;
    const m = parseLens(t, { ...obj(v), sel });
    return m && m.t === t ? (m.data as Extract<CoreMsg, { t: T }>["data"]) : null;
  };
  const thread: (ReplyData & { q: string })[] = [];
  if (Array.isArray(c.thread)) {
    for (const t of c.thread.slice(-12)) {
      const r = as("lens_reply", t);
      if (r) thread.push({ ...r, q: str(obj(t).q, MAX.q) });
    }
  }
  return {
    cid, open: true,
    card: {
      sel,
      thumb: thumbOf(c.thumb),
      question: str(c.question, MAX.q).trim() || null,
      translateTo: str(c.translate_to, MAX.lang),
      hello: resultHello(c.hello),
      local: as("lens_local", c.local),
      answer: as("lens_answer", c.answer),
      links: as("lens_links", c.links)?.rows ?? [],
      thread,
      translation: as("lens_translation", c.translation),
      sentTo: sentTo(c.sent_to) ?? [],
    },
  };
}


// ---- the still: one binary NBF1 frame of kind 4 ------------------------------------------------

export const KIND_LENS = 4;
const HEADER_LEN = 64;
const MAGIC = 0x4e424631; // "NBF1", big-endian
const MIME: Record<number, string> = { 1: "image/jpeg", 2: "image/webp", 3: "image/png" };

export interface Still { vw: number; vh: number; iw: number; ih: number; dpr: number; mime: string; image: Uint8Array }

/** Search Screen's still, or null for anything else (the browser's frames, kinds 1–2; the camera's, 3). */
export function parseStill(buf: ArrayBuffer): Still | null {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength <= HEADER_LEN) return null;
  const dv = new DataView(buf);
  if (dv.getUint32(0, false) !== MAGIC || dv.getUint8(4) !== 1 || dv.getUint8(5) !== KIND_LENS) return null;
  const hl = dv.getUint16(6, true);
  if (hl < HEADER_LEN || hl >= buf.byteLength) return null;
  const vw = dv.getUint16(36, true), vh = dv.getUint16(38, true), iw = dv.getUint16(40, true), ih = dv.getUint16(42, true);
  if (!vw || !vh || !iw || !ih) return null;
  return { vw, vh, iw, ih, dpr: dv.getUint16(44, true) / 100, mime: MIME[dv.getUint8(56)] ?? "image/jpeg", image: new Uint8Array(buf, hl) };
}

// ---- what the page sends -------------------------------------------------------------------------

const fin = (v: number) => (Number.isFinite(v) ? Math.round(v * 10) / 10 : 0);
const cut = (s: string, max: number) => s.replace(/\s+/g, " ").trim().slice(0, max);

export const send = {
  hello: (texts: boolean) => ({ cmd: "lens_hello", v: 1, ...(texts ? { texts: true } : {}) }),
  select: (sel: number, r: Rect, how: How, live = false) => ({
    cmd: "lens_select", sel, rect: r.map(fin) as Rect, how, ...(live ? { live: true } : {}),
  }),
  ask: (sel: number, qid: number, q: string) => ({ cmd: "lens_ask", sel, qid, q: cut(q, MAX.q) }),
  translate: (sel: number, to: string) => ({ cmd: "lens_translate", sel, to: to.slice(0, MAX.lang) }),
  act: (sel: number, id: string) => ({ cmd: "lens_act", sel, id }),
  cont: (sel: number, text: string) => ({ cmd: "lens_continue", sel, text: cut(text, MAX.cont) }),
  listen: (sel: number) => ({ cmd: "lens_listen", sel }),
  /** Show this area's results in the notch instead of the sheet (`to`: the language Translate uses). */
  handoff: (sel: number, to: string | null) => ({ cmd: "lens_handoff", sel, to: /^[A-Za-z0-9_-]{1,16}$/.test(to ?? "") ? (to as string) : "" }),
  /** The overlay gave up waiting for the answer to `handoff`: the area stays here (the core closes
   *  any card it made of it, and sends its results here again). */
  handoffCancel: (sel: number) => ({ cmd: "lens_handoff", sel, cancel: true }),
};

/** What NEO's page in the notch sends about a card: each names its cid. */
export const cardSend = {
  hello: () => ({ cmd: "lens_card_hello" }),
  ask: (cid: string, qid: number, q: string) => ({ cmd: "lens_card_ask", cid, qid, q: cut(q, MAX.q) }),
  translate: (cid: string, to: string) => ({ cmd: "lens_card_translate", cid, to: to.slice(0, MAX.lang) }),
  act: (cid: string, id: string) => ({ cmd: "lens_card_act", cid, id }),
  cont: (cid: string, text: string) => ({ cmd: "lens_card_continue", cid, text: cut(text, MAX.cont) }),
  listen: (cid: string) => ({ cmd: "lens_card_listen", cid }),
  close: (cid: string, why: "esc" | "x" | "chat") => ({ cmd: "lens_card_close", cid, why }),
};

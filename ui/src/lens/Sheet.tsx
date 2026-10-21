// The results sheet beside the selection: what it is (the Eyes brain's short answer, "Looking…"
// until then), what can be done with it (chips), shopping or place links, the text read on this
// Mac (and its translation), follow-up questions and their answers, a question box with the mic,
// "Continue in NEO", and where the area went (the privacy line).
//
// It is placed by `transform` (and slides there by one, on the compositor), never by left/top; its
// thumbnail is the small picture the core makes of the area (never a copy cut from the full still).

import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import { chipsFor, langName, type Chip } from "./chips";
import { placeSheet, type Side } from "./place";
import { answerProblem, privacyLine, replyProblem, type Action, type LensState, type Sel } from "./state";
import { clipboardWrite } from "../shell";

const KIND: Record<string, string> = {
  product: "Product", place: "Place", plant: "Plant", animal: "Animal", food: "Food", art: "Art",
  text: "Text", math: "Math", code: "Code", error: "Error", chart: "Chart",
};

const Icon = {
  close: <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M6 6l12 12M18 6L6 18" /></svg>,
  mic: <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></svg>,
  send: <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19V5M5 12l7-7 7 7" /></svg>,
  down: <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 9l6 6 6-6" /></svg>,
  warn: <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l9 16H3zM12 10v4M12 17h.01" /></svg>,
};

interface Props {
  s: LensState;
  sel: Sel;
  dispatch: (a: Action) => void;
  composer: RefObject<HTMLInputElement | null>;
  q: string;
  setQ: (q: string) => void;
}

function Thumb({ thumb }: { thumb?: string }) {
  return <div className="lx-thumb" aria-hidden="true">{thumb && <img src={thumb} alt="" />}</div>;
}

function Chips({ s, sel, dispatch }: { s: LensState; sel: Sel; dispatch: (a: Action) => void }) {
  const [more, setMore] = useState(false);
  const [langs, setLangs] = useState(false);
  const { shown, more: rest } = chipsFor(sel.local, sel.answer, s.hello, s.translateTo);
  const list = more ? [...shown, ...rest] : shown;
  if (!list.length) return null;
  const click = (c: Chip) => (c.kind === "translate" ? dispatch({ t: "chip", kind: "translate", to: c.to }) : dispatch({ t: "chip", kind: c.kind, id: c.id }));
  const options = s.hello?.langs ?? [];
  return (
    <div className="lx-chips" role="toolbar" aria-label="Actions">
      {list.map((c) => {
        const note = c.id ? sel.notes[c.id] : undefined;
        const cls = "lx-chip" + (c.warn ? " warn" : "") + (c.kind === "image_search" ? " primary" : "") + (note && note !== "…" ? " done" : "");
        const chip = (
          <button key={c.key} className={cls} onClick={() => click(c)} title={c.detail ? `${c.label} — ${c.detail}` : c.label} aria-busy={note === "…"}>
            {c.warn && Icon.warn}
            <span className="label">{note && note !== "…" ? note : c.label}</span>
            {c.detail && !note && <span className="detail">{c.detail}</span>}
          </button>
        );
        if (c.kind !== "translate" || !options.length) return chip;
        return (
          <span key={c.key} className="lx-menu">
            {chip}
            <button className="lx-chip" aria-label="Translate into…" aria-haspopup="listbox" aria-expanded={langs} onClick={() => setLangs((v) => !v)}>{Icon.down}</button>
            {langs && (
              <div className="lx-menu-list" role="listbox" aria-label="Translate into">
                {options.map((l) => (
                  <button key={l.code} role="option" aria-selected={l.code === c.to} onClick={() => { setLangs(false); dispatch({ t: "translate_to", to: l.code }); }}>{l.name}</button>
                ))}
              </div>
            )}
          </span>
        );
      })}
      {!more && rest.length > 0 && <button className="lx-chip" onClick={() => setMore(true)}>More</button>}
    </div>
  );
}

function AnswerSlot({ s, sel, dispatch }: { s: LensState; sel: Sel; dispatch: (a: Action) => void }) {
  const a = sel.answer;
  if (sel.captured === "failed") {
    return (
      <div className="lx-answer problem" role="status">
        Couldn't capture the screen.
        <div className="lx-row-actions"><button className="lx-btn" onClick={() => dispatch({ t: "select", box: sel.box, how: sel.how })}>Try again</button></div>
      </div>
    );
  }
  if (!a || a.state === "thinking") {
    return <div className="lx-skel" aria-label="Looking" role="status"><i /><i /><i /></div>;
  }
  if (a.state === "done") return <div className="lx-answer" aria-live="polite">{a.answer ?? a.title ?? ""}</div>;
  const problem = answerProblem(a, s.hello?.eyesWhy ?? "");
  const again = a.error === "timeout" || a.error === "failed" || a.error === "offline" || a.error === "busy";
  // NEO_LENS_ASK off: Ask. Came out black: Ask anyway (a dark page, a black terminal NEO couldn't read).
  const label = a.error === "off" ? "Ask NEO" : a.error === "dark" ? "Ask anyway" : again ? "Try again" : null;
  return (
    <div className="lx-answer problem" role="status" aria-live="polite">
      {problem}
      {label && (
        <div className="lx-row-actions">
          <button className={"lx-btn" + (a.error === "off" ? " primary" : "")} onClick={() => dispatch({ t: "ask_again" })}>{label}</button>
        </div>
      )}
    </div>
  );
}

function TextSection({ sel }: { sel: Sel }) {
  const local = sel.local;
  const tr = sel.translation;
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"orig" | "tr">("orig");
  useEffect(() => { if (tr?.text) { setOpen(true); setTab("tr"); } }, [tr?.text]);
  if (!local?.text && !tr) return null;
  const lines = local?.lines ?? 0;
  const shown = tab === "tr" && tr?.text ? tr.text : local?.text ?? "";
  return (
    <div className="lx-text">
      <div className="lx-text-head">
        <button className="lx-fold" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          Text{lines ? ` · ${lines} line${lines === 1 ? "" : "s"}` : ""} {open ? "▴" : "▾"}
        </button>
        {tr && (
          <span className="lx-seg" role="group" aria-label="Show">
            <button aria-pressed={tab === "orig"} onClick={() => { setTab("orig"); setOpen(true); }}>Original</button>
            <button aria-pressed={tab === "tr"} onClick={() => { setTab("tr"); setOpen(true); }}>{tr.pending ? "Translating…" : langName(tr.to)}</button>
          </span>
        )}
        {open && shown && <button className="lx-fold" onClick={() => void clipboardWrite(shown)}>Copy</button>}
      </div>
      {open && <pre>{tab === "tr" && tr?.error ? "Couldn't translate this." : shown}</pre>}
    </div>
  );
}

function Thread({ sel }: { sel: Sel }) {
  if (!sel.thread.length) return null;
  return (
    <div className="lx-thread" aria-live="polite">
      {sel.thread.map((t) => (
        <div key={t.qid} style={{ display: "contents" }}>
          {t.q && <div className="lx-q">{t.q}</div>}
          <div className={"lx-a" + (t.state === "thinking" ? " thinking" : "")}>
            {t.state === "thinking" ? "Thinking…" : t.state === "failed" ? replyProblem(t.error) : t.text}
          </div>
        </div>
      ))}
    </div>
  );
}

export function Sheet({ s, sel, dispatch, composer, q, setQ }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [h, setH] = useState(260);
  const side = useRef<Side | undefined>(undefined);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setH(el.scrollHeight));
    ro.observe(el);
    const inner = el.firstElementChild;
    if (inner) ro.observe(inner);
    return () => ro.disconnect();
  }, []);
  // A new question or answer scrolls into view.
  const turns = sel.thread.length + sel.thread.filter((t) => t.state !== "thinking").length;
  useEffect(() => {
    const sc = ref.current?.firstElementChild;
    if (sc && turns) sc.scrollTo({ top: sc.scrollHeight, behavior: "smooth" });
  }, [turns]);
  const place = placeSheet(sel.box, h, s.W, s.H, s.safeTop, side.current);
  useEffect(() => { side.current = place.side; }, [place.side]);

  const a = sel.answer;
  const thinking = !a || a.state === "thinking";
  const title = sel.captured === "failed" ? "This area" : a?.state === "done" ? a.title ?? "Here's what NEO found" : thinking ? "Looking…" : sel.local?.text ? "Text on screen" : "This area";
  const brain = a?.brain?.label ?? (thinking && s.hello?.eyes && s.hello.ask ? s.hello.eyes.label : "");
  const sub = [a?.kind ? KIND[a.kind] : "", brain].filter(Boolean).join(" · ");
  const listening = s.coreState === "listening";
  const ask = () => { if (q.trim()) { dispatch({ t: "ask", q }); setQ(""); } };

  return (
    <div className="lx-sheet-at" style={{ transform: `translate3d(${place.x}px, ${place.y}px, 0)` }}>
    <div
      ref={ref}
      className="lx-sheet"
      role="dialog"
      aria-label="Search Screen results"
      style={{ width: place.w, maxHeight: place.maxH }}
      onPointerDown={(e) => e.stopPropagation()}
    >
      <div className="lx-sheet-scroll">
        <div className="lx-head">
          <Thumb thumb={sel.local?.thumb} />
          <div className="lx-titles">
            <div className={"lx-title" + (thinking && sel.captured !== "failed" ? " lx-shimmer" : "")}>{title}</div>
            {sub && <div className="lx-sub">{sub}</div>}
          </div>
          <button className="lx-x" aria-label="Close Search Screen" onClick={() => dispatch({ t: "close", why: "cancel" })}>{Icon.close}</button>
        </div>
        <Chips s={s} sel={sel} dispatch={dispatch} />
        {sel.question && <div className="lx-q" style={{ alignSelf: "flex-start" }}>{sel.question}</div>}
        <AnswerSlot s={s} sel={sel} dispatch={dispatch} />
        {sel.links.length > 0 && (
          <div className="lx-links" role="list">
            {sel.links.map((r) => (
              <button key={r.id} className="lx-link" role="listitem" onClick={() => dispatch({ t: "chip", kind: "open", id: r.id })} title={`${r.title} — ${r.host}`}>
                <span className="lx-avatar" aria-hidden="true">{(r.host.replace(/^www\./, "")[0] ?? "?").toUpperCase()}</span>
                <div><div className="t">{r.title}</div><div className="h">{r.host}</div></div>
              </button>
            ))}
          </div>
        )}
        <TextSection sel={sel} />
        <Thread sel={sel} />
      </div>
      <div className="lx-compose">
        <input
          ref={composer}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={listening ? "Listening…" : "Ask about this…"}
          aria-label="Ask about this area"
          maxLength={500}
          spellCheck={false}
          onKeyDown={(e) => {
            if (e.key !== "Enter" || e.nativeEvent.isComposing) return;
            e.preventDefault();
            e.stopPropagation();
            if (e.metaKey) dispatch({ t: "continue", text: q });
            else ask();
          }}
        />
        <button className={"lx-round" + (listening ? " live" : "")} aria-label="Ask by voice" aria-pressed={listening} onClick={() => dispatch({ t: "listen" })}>{Icon.mic}</button>
        <button className="lx-round" aria-label="Ask" disabled={!q.trim()} onClick={ask}>{Icon.send}</button>
      </div>
      <div className="lx-foot">
        <button className="lx-continue" onClick={() => dispatch({ t: "continue", text: q })}>Continue in NEO →</button>
        <div className="lx-privacy" title={privacyLine(s)}>{privacyLine(s)}</div>
      </div>
    </div>
    </div>
  );
}

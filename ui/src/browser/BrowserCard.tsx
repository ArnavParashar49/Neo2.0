// NEO's browser, live in the chat: a dark glass frame around the real page. A thin top bar
// (navigation, address, window buttons), the live view, and a footer that always says who is
// driving. A soft glow breathes around the frame while NEO is in control; the frame goes calm
// and neutral when the user is.
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { ThinkingOrb } from "thinking-orbs";
import type { BrowserHub, BrowserSnap } from "./hub";
import type { BrowserState, NeedUser } from "./types";
import { BrowserView } from "./BrowserView";
import { Timeline } from "./Timeline";
import { I } from "./icons";
import { hostOf, isLocalHost, q, splitHost, displayHost } from "./text";
import { reservedShortcut } from "./input";

export type CardSurface = "docked" | "focus" | "window";

export interface CardProps {
  hub: BrowserHub;
  snap: BrowserSnap;
  connected: boolean;
  surface: CardSurface;
  reduced: boolean;
  /** Overlay: docked ⇄ focus. */
  onExpand?: () => void;
  onMinimize?: () => void;
  /** Overlay: pop out. Window: bring back. */
  onPopout?: () => void;
  /** A pending approval lives in NEO's chat (the pop-out window has no composer). */
  confirmElsewhere?: boolean;
  /** Bumped by the host to put keyboard focus on the page (need_user). */
  focusSignal?: number;
  /** Floats over the live view (the overlay's reply toast in focus / compact layouts). */
  overlay?: ReactNode;
}

type Ring = "acting" | "thinking" | "starting" | "waiting" | "user" | "need" | "error" | "offline" | "ended";

/** The job is over but the page is still NEO's: nothing is happening, and the card folds soon. */
export function isFinished(st: BrowserState, connected: boolean): boolean {
  return connected && st.controller === "neo" && !st.job && (st.status === "thinking" || st.status === "working");
}

function ringOf(st: BrowserState, connected: boolean): Ring {
  if (st.status === "ended" || isFinished(st, connected)) return "ended";
  if (!connected) return "offline";
  if (st.status === "error") return "error";
  if (st.status === "need_user") return "need";
  if (st.controller === "user") return "user";
  if (st.status === "waiting") return "waiting";
  if (st.status === "starting") return "starting";
  if (st.status === "thinking") return "thinking";
  return "acting";
}

const NEED_VERB: Record<NeedUser["reason"], string> = {
  login: "log in", captcha: "solve the check", "2fa": "enter the code", confirm: "check this", other: "finish the step",
};

function IconBtn({ label, title, onClick, children, disabled, pressed, className = "" }: {
  label: string; title?: string; onClick?: () => void; children: ReactNode; disabled?: boolean; pressed?: boolean; className?: string;
}) {
  return (
    <button
      className={"br-icon " + className}
      aria-label={label}
      title={title ?? label}
      aria-disabled={disabled || undefined}
      aria-pressed={pressed}
      onClick={() => { if (!disabled) onClick?.(); }}
    >
      {children}
    </button>
  );
}

/** A pill button whose width glides between labels (Take control ⇄ Hand back to NEO). */
function MorphPill({ variant, className, onClick, title, disabled, children, reduced }: {
  variant: string; className: string; onClick: () => void; title?: string; disabled?: boolean; children: ReactNode; reduced: boolean;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const prevW = useRef<number | null>(null);
  useLayoutEffect(() => {
    const b = ref.current;
    if (!b) return;
    const w = b.offsetWidth;
    if (prevW.current != null && Math.abs(prevW.current - w) > 1 && !reduced) {
      b.animate([{ width: `${prevW.current}px` }, { width: `${w}px` }], { duration: 200, easing: "cubic-bezier(0.2, 0.8, 0.2, 1)" });
    }
    prevW.current = w;
  }, [variant, reduced]);
  return (
    <button ref={ref} className={"br-pill " + className} onClick={() => { if (!disabled) onClick(); }} title={title} aria-disabled={disabled || undefined}>
      <span key={variant} className="br-pill-in">{children}</span>
    </button>
  );
}

// ---- address capsule ------------------------------------------------------------------------
function Address({ st, userDriving, editSignal, onNavigate, disabled }: {
  st: BrowserState; userDriving: boolean; editSignal: number; onNavigate: (url: string) => void; disabled: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [pop, setPop] = useState(false);
  const [copied, setCopied] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const host = hostOf(st.url);
  const { sub: fullSub, reg } = splitHost(host);
  // A long subdomain gives way from its left, so the part touching the real domain stays.
  const sub = fullSub.length > 18 ? "…" + fullSub.slice(-17) : fullSub;
  const title = (st.title || "").slice(0, 200);

  useEffect(() => { if (editSignal && userDriving) setEditing(true); }, [editSignal, userDriving]);
  useEffect(() => { if (!userDriving) setEditing(false); }, [userDriving]);
  useEffect(() => { if (editing) { const i = inputRef.current; if (i) { i.focus(); i.select(); } } }, [editing]);
  useEffect(() => {
    if (!pop) return;
    const h = (e: PointerEvent) => { if (!boxRef.current?.contains(e.target as Node)) setPop(false); };
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") { e.stopPropagation(); setPop(false); } };
    window.addEventListener("pointerdown", h, true);
    window.addEventListener("keydown", k, true);
    return () => { window.removeEventListener("pointerdown", h, true); window.removeEventListener("keydown", k, true); };
  }, [pop]);

  const security = st.secure === "secure" ? <span className="br-sec lock" title="Secure connection" aria-label="Secure">{I.lock()}</span>
    : st.secure === "insecure" ? <span className="br-chip-sm warn">Not secure</span>
    : st.secure === "local" || isLocalHost(host) ? <span className="br-chip-sm">Local</span>
    : st.secure === "error" ? <span className="br-sec bad" aria-label="Couldn't load">{I.warn(12)}</span>
    : null;

  const fav = st.loading ? <span className="br-spin fav" aria-hidden />
    : st.favicon ? <img className="br-fav" src={st.favicon} alt="" draggable={false} />
    : <span className="br-fav glyph" aria-hidden>{I.globe(14)}</span>;

  if (editing) {
    return (
      <div className="br-addr editing" ref={boxRef}>
        <span className="br-fav glyph" aria-hidden>{I.globe(14)}</span>
        <input
          ref={inputRef}
          className="br-addr-input"
          defaultValue={st.url}
          aria-label="Address"
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); onNavigate(e.currentTarget.value); setEditing(false); }
            else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); setEditing(false); }
          }}
          onBlur={() => setEditing(false)}
        />
      </div>
    );
  }

  return (
    <div className="br-addr-wrap" ref={boxRef}>
      <button
        className="br-addr"
        aria-label={`${host}${title ? " — " + title : ""}. ${userDriving ? "Edit address (⌘L)" : "Show full address"}`}
        title={userDriving ? "Edit address (⌘L)" : st.url}
        aria-disabled={disabled || undefined}
        aria-expanded={userDriving ? undefined : pop}
        onClick={() => { if (disabled) return; if (userDriving) setEditing(true); else setPop((p) => !p); }}
      >
        {fav}
        {security}
        <span className="br-host" dir="ltr" title={host}>{sub && <span className="br-host-sub">{sub}</span>}<span className="br-host-reg">{reg || st.url}</span></span>
        {title && <><span className="br-addr-sep" aria-hidden>—</span><span className="br-addr-title" dir="auto">{title}</span></>}
        {!title && st.status === "starting" && <span className="br-shimmer title" aria-hidden />}
      </button>
      {pop && (
        <div className="br-pop url" role="dialog" aria-label="Page address">
          <div className="br-pop-url">{st.url}</div>
          <button
            className="br-pill ghost sm"
            onClick={() => {
              void navigator.clipboard?.writeText(st.url).then(() => { setCopied(true); window.setTimeout(() => setCopied(false), 1400); }).catch(() => {});
            }}
          >
            {copied ? I.check(13) : I.copy(13)}<span>{copied ? "Copied" : "Copy link"}</span>
          </button>
        </div>
      )}
    </div>
  );
}

function Progress({ loading, progress, reduced }: { loading: boolean; progress: number | null; reduced: boolean }) {
  const [phase, setPhase] = useState<"idle" | "run" | "done">(loading ? "run" : "idle");
  useEffect(() => {
    if (loading) { setPhase("run"); return; }
    setPhase((p) => (p === "run" ? "done" : p));
    const t = window.setTimeout(() => setPhase("idle"), 420);
    return () => clearTimeout(t);
  }, [loading]);
  if (phase === "idle") return null;
  const det = phase === "run" && progress != null && progress > 0;
  return (
    <div className={"br-progress " + phase + (det ? " det" : "") + (reduced ? " reduced" : "")} aria-hidden>
      <div className="br-progress-bar" style={det ? { width: `${Math.round(progress! * 100)}%` } : undefined} />
    </div>
  );
}

function Tabs({ st, onPick }: { st: BrowserState; onPick: (i: number) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const h = (e: PointerEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    window.addEventListener("pointerdown", h, true);
    return () => window.removeEventListener("pointerdown", h, true);
  }, [open]);
  if (st.tabs.count <= 1) return null;
  return (
    <div className="br-tabs" ref={ref}>
      <button className="br-tabs-chip" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>{st.tabs.count} tabs</button>
      {open && (
        <div className="br-pop menu" role="menu">
          {(st.tabs.list ?? Array.from({ length: st.tabs.count }, (_, i) => ({ title: `Tab ${i + 1}`, url: "", favicon: null }))).map((t, i) => (
            <button key={i} role="menuitemradio" aria-checked={i === st.tabs.active} className={"br-menu-row" + (i === st.tabs.active ? " on" : "")} onClick={() => { setOpen(false); onPick(i); }}>
              {t.favicon ? <img className="br-fav" src={t.favicon} alt="" /> : <span className="br-fav glyph">{I.globe(14)}</span>}
              <span className="br-menu-label" dir="auto">{t.title || displayHost(t.url) || `Tab ${i + 1}`}</span>
              {i === st.tabs.active && I.check(13)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---- the card ----------------------------------------------------------------------------------
export function BrowserCard({ hub, snap, connected, surface, reduced, onExpand, onMinimize, onPopout, confirmElsewhere, focusSignal = 0, overlay }: CardProps) {
  const st = snap.live!;
  const ring = ringOf(st, connected);
  const finished = isFinished(st, connected);
  const userDriving = connected && st.controller === "user" && (st.status === "user" || st.status === "need_user");
  const neoDriving = !finished && connected && st.controller === "neo" && (st.status === "working" || st.status === "thinking" || st.status === "waiting");
  const job = st.job;
  const nu = st.status === "need_user" ? st.need_user : null;
  const [stepsOpen, setStepsOpen] = useState(false);
  const [closeAsk, setCloseAsk] = useState(false);
  const [editSignal, setEditSignal] = useState(0);
  const [viewFocus, setViewFocus] = useState(0);
  const suppressToast = useRef(false);
  const shellRef = useRef<HTMLElement>(null);
  const host = displayHost(st.url);
  const off = !connected;

  // Need-user prompts take the keyboard to the page so the user can type straight away.
  useEffect(() => { if (nu) setViewFocus((n) => n + 1); }, [nu?.reason, nu?.title]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (focusSignal) setViewFocus((n) => n + 1); }, [focusSignal]);

  // Who's driving is never in doubt: say so when it changes.
  const prev = useRef({ controller: st.controller, status: st.status, sid: st.sid });
  useEffect(() => {
    const p = prev.current;
    prev.current = { controller: st.controller, status: st.status, sid: st.sid };
    if (p.sid !== st.sid) return;
    if (st.status === "need_user" && p.status !== "need_user" && nu) hub.toast(`Your turn: ${NEED_VERB[nu.reason]}, then hand back`, 3000);
    else if (p.controller === "neo" && st.controller === "user" && st.status === "user") {
      if (!suppressToast.current) hub.toast("You're in control. NEO is paused.", 2500);
    } else if (p.controller === "user" && st.controller === "neo") hub.toast("NEO is back in control", 2000);
    suppressToast.current = false;
  }, [st.controller, st.status, st.sid, hub, nu]);

  useEffect(() => { if (!neoDriving) setCloseAsk(false); }, [neoDriving]);

  const take = useCallback(() => {
    if (off) return;
    if (st.status === "waiting" && snap.action?.gated) {
      suppressToast.current = true;
      hub.toast(`You took over. NEO won't click ${q(snap.action.label)}.`, 3000);
    }
    void hub.control("take");
  }, [hub, off, st.status, snap.action]);
  const release = useCallback(() => { if (!off) void hub.control(job || nu ? "release" : "close"); }, [hub, off, job, nu]);
  const stop = useCallback(() => {
    if (off || !job) return;
    suppressToast.current = true;
    void hub.control("stop").then((r) => hub.toast(r.ok ? "Stopped. You're in control." : "Couldn't stop NEO", 2500));
  }, [hub, off, job]);
  const close = useCallback((force = false) => {
    if (off && !force) return;
    if (neoDriving && job && !force) { setCloseAsk(true); return; }
    setCloseAsk(false);
    void hub.control("close", force ? { force: true } : {});
  }, [hub, off, neoDriving, job]);
  const nav = useCallback((action: "back" | "forward" | "reload" | "hard_reload" | "stop_loading") => {
    if (off) return;
    if (neoDriving) suppressToast.current = false;
    void hub.control(action);
  }, [hub, off, neoDriving]);
  const navigate = useCallback((raw: string) => {
    let v = raw.trim();
    if (!v) return;
    if (!/^[a-z][a-z0-9+.-]*:/i.test(v)) v = (/^(localhost|\d{1,3}(\.\d{1,3}){3})(:\d+)?(\/|$)/.test(v) ? "http://" : "https://") + v;
    let u: URL;
    try { u = new URL(v); } catch { hub.toast("That doesn't look like a web address", 2200); return; }
    if (u.protocol !== "http:" && u.protocol !== "https:") { hub.toast("Only web addresses (http or https) can be opened", 2600); return; }
    if (u.href.length > 2048) { hub.toast("That address is too long", 2200); return; }
    void hub.control("navigate", { url: u.href }).then((r) => { if (!r.ok && r.code === "blocked_url") hub.toast("NEO's browser won't open that address", 2600); });
  }, [hub]);
  const toggleControl = useCallback(() => { if (userDriving) release(); else if (neoDriving || finished || st.status === "starting") take(); }, [userDriving, neoDriving, finished, st.status, release, take]);

  // Shortcuts (capture phase, never forwarded to the page).
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const r = reservedShortcut(e);
      if (!r) return;
      switch (r) {
        case "control": toggleControl(); break;
        case "stop": if (!job) return; stop(); break;
        case "address": if (!userDriving) return; setEditSignal((n) => n + 1); break;
        case "back": case "forward": case "reload": case "hard_reload": nav(r); break;
        case "focus_next": case "focus_prev": {
          const a = document.activeElement as HTMLElement | null;
          if (!a?.classList.contains("br-sink")) return;
          const all = [...document.querySelectorAll<HTMLElement>("button, [href], input, textarea, select, [tabindex]")]
            .filter((el) => el.tabIndex >= 0 && !el.hasAttribute("disabled") && el.offsetParent !== null && !el.classList.contains("br-sink") && !el.classList.contains("br-view"));
          const pos = a.compareDocumentPosition.bind(a);
          const next = r === "focus_next"
            ? all.find((el) => pos(el) & Node.DOCUMENT_POSITION_FOLLOWING)
            : [...all].reverse().find((el) => pos(el) & Node.DOCUMENT_POSITION_PRECEDING);
          next?.focus();
          break;
        }
        default: return; // expand / minimize / pop out / close window: the host handles them
      }
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener("keydown", h, true);
    return () => window.removeEventListener("keydown", h, true);
  }, [toggleControl, stop, nav, job, userDriving]);

  const pending = snap.pending;
  const taking = !!pending.take;
  // The Done button nudges when the prompt changes under it ("Signed in? Hand back to NEO.").
  const nuTitle = useRef<string | null>(null);
  const [nudge, setNudge] = useState(0);
  useEffect(() => {
    const t = nu?.title ?? null;
    if (t && nuTitle.current && t !== nuTitle.current) setNudge((n) => n + 1);
    nuTitle.current = t;
  }, [nu?.title]);
  const releasing = !!pending.release;
  const isWindow = surface === "window";
  const drag = surface !== "docked" ? { "data-tauri-drag-region": true } : {};
  const navTitle = (base: string, key: string) => neoDriving ? `${base} — takes control (${key})` : `${base} (${key})`;

  const statusEl = (() => {
    if (ring === "ended") return <><span className="br-status-glyph ok">{I.check(14)}</span><span className="br-fg">Finished</span></>;
    if (off) return <><span className="br-dot off" />Offline</>;
    switch (ring) {
      case "starting": return <><span className="br-dot dim" />Starting…</>;
      case "acting": return <>{isWindow ? <ThinkingOrb state="working" size={20} theme="auto" /> : <span className="br-dot live" />}NEO is browsing</>;
      case "thinking": return <>{isWindow ? <ThinkingOrb state="solving" size={20} theme="auto" /> : <span className="br-dot live slow" />}NEO is thinking</>;
      case "waiting": return <><span className="br-status-glyph warn">{I.wait(14)}</span>{confirmElsewhere ? "Waiting for your OK in NEO's chat" : "Waiting for your OK below"}</>;
      case "user": return <><span className="br-status-glyph">{I.hand(14)}</span><span className="br-fg">You're in control</span></>;
      case "error": return <><span className="br-status-glyph bad">{I.warn(14)}</span>{st.error?.code === "crashed" ? "This page crashed" : st.error?.code === "browser_closed" ? "Browser closed" : `Couldn't reach ${host}`}</>;
      default: return null;
    }
  })();

  const needGlyph = nu ? (nu.reason === "login" ? I.key(14) : nu.reason === "captcha" ? I.shield(14) : nu.reason === "2fa" ? I.message(14) : I.warn(14)) : null;
  const needSub = nu ? (nu.reason === "other" && nu.detail ? `NEO: “${nu.detail.slice(0, 120)}”` : nu.sub) : "";

  return (
    <section ref={shellRef} className={`br-shell ring-${ring} surf-${surface}` + (taking && neoDriving ? " taking" : "")} role="region" aria-label={`NEO's browser — ${host}`}>
      <div className="br-glow" aria-hidden><div className="br-halo" /><div className="br-ring a" /><div className="br-ring b" /></div>
      <div className="br-warn" aria-hidden><div className="br-halo warn" /><div className="br-ring solid" /></div>
      <div className="br-card">
        <div className="br-top" {...drag}>
          {isWindow && <div className="br-traffic-space" data-tauri-drag-region />}
          <div className="br-nav">
            <IconBtn label="Back" title={navTitle("Back", "⌘[")} disabled={!st.can_back || off} onClick={() => nav("back")}>{I.back()}</IconBtn>
            <IconBtn label="Forward" title={navTitle("Forward", "⌘]")} disabled={!st.can_forward || off} onClick={() => nav("forward")}>{I.forward()}</IconBtn>
            {st.loading
              ? <IconBtn label="Stop loading" disabled={off} onClick={() => nav("stop_loading")}>{I.close()}</IconBtn>
              : <IconBtn label="Reload" title={navTitle("Reload", "⌘R")} disabled={off || st.status === "starting"} onClick={() => nav("reload")}>{I.reload()}</IconBtn>}
          </div>
          <Address st={st} userDriving={userDriving} editSignal={editSignal} onNavigate={navigate} disabled={off} />
          <Tabs st={st} onPick={(i) => void hub.control("tab", { index: i })} />
          <div className="br-grow" {...drag} />
          <div className="br-top-actions">
            {isWindow ? (
              <>
                <button className="br-pill ghost sm" onClick={onPopout} title="Bring back to NEO's chat (⌘W)">{I.popin(14)}<span>Bring back</span></button>
                <IconBtn label="Close browser" onClick={() => close()}>{I.close()}</IconBtn>
              </>
            ) : (
              <>
                <IconBtn label="Pop out" title="Pop out (⌘⇧O)" disabled={off} onClick={onPopout}>{I.popout()}</IconBtn>
                <IconBtn label="Minimize to chat" title="Minimize to chat (⌘⇧M)" onClick={onMinimize}>{I.minimize()}</IconBtn>
                <IconBtn label={surface === "focus" ? "Exit expanded view" : "Expand"} title={surface === "focus" ? "Exit expanded view (⌘⇧E)" : "Expand (⌘⇧E)"} pressed={surface === "focus"} onClick={onExpand}>
                  {surface === "focus" ? I.restore() : I.expand()}
                </IconBtn>
                <IconBtn label="Close browser" onClick={() => close()}>{I.close()}</IconBtn>
              </>
            )}
          </div>
          <Progress loading={st.loading && !off} progress={st.progress} reduced={reduced} />
        </div>

        <div className="br-viewwrap">
          <BrowserView hub={hub} snap={snap} surface={surface} connected={connected} reduced={reduced} onTakeControl={take} focusSignal={viewFocus} drawerOpen={stepsOpen} taking={taking} finished={finished} />
          {overlay}
          {stepsOpen && <Timeline id={`steps-${st.sid}`} steps={snap.steps} startedAt={snap.open?.started_at ?? 0} variant="drawer" onClose={() => setStepsOpen(false)} />}
          {snap.dialog && <DialogSheet hub={hub} snap={snap} />}
          {snap.select && <SelectPopover hub={hub} snap={snap} />}
        </div>

        {nu ? (
          <div className="br-need" role="alert">
            <div className="br-need-badge" aria-hidden>{needGlyph}</div>
            <div className="br-need-text" title={`${nu.title} ${needSub}`.trim()}>
              <span className="br-need-title">{nu.title}</span>
              {needSub && <span className="br-need-sub"><span aria-hidden> · </span>{needSub}</span>}
            </div>
            <button key={nudge} className={"br-pill solid" + (nudge ? " nudge" : "")} onClick={release} aria-disabled={off || releasing || undefined} title="Hand back to NEO (⌘⇧U)">
              {releasing ? <span className="br-spin sm dark" /> : null}Done — hand back
            </button>
            {job && <button className="br-pill bad" onClick={stop} title="Stop NEO (⌘.)">{I.stop()}<span>Stop</span></button>}
          </div>
        ) : closeAsk ? (
          <div className="br-foot ask" role="alertdialog" aria-label="NEO is still working. Stop and close?">
            <span className="br-foot-status"><span className="br-status-glyph warn">{I.warn(14)}</span><span className="br-fg">NEO is still working. Stop and close?</span></span>
            <span className="br-grow" />
            <button className="br-pill bad" autoFocus onClick={() => close(true)} onKeyDown={(e) => { if (e.key === "Escape") { e.stopPropagation(); setCloseAsk(false); } }}>Stop &amp; close</button>
            <button className="br-pill" onClick={() => setCloseAsk(false)}>Cancel</button>
          </div>
        ) : (
          <div className="br-foot" {...(isWindow ? drag : {})}>
            <span className="br-foot-status" aria-live="polite">{statusEl}</span>
            <StepsToggle steps={snap.steps} open={stepsOpen} onToggle={() => setStepsOpen((o) => !o)} controls={`steps-${st.sid}`} />
            <span className="br-grow" {...(isWindow ? drag : {})} />
            {st.status !== "error" && st.status !== "starting" && st.status !== "ended" && (
              userDriving ? (
                <MorphPill variant="release" className="accent" reduced={reduced} disabled={off || releasing} onClick={release} title={job ? "Hand back to NEO (⌘⇧U)" : "Done (⌘⇧U)"}>
                  {releasing ? <span className="br-spin sm" /> : job ? I.sparkle() : I.check(14)}<span>{job ? "Hand back to NEO" : "Done"}</span>
                </MorphPill>
              ) : (
                <MorphPill variant="take" className="" reduced={reduced} disabled={off || taking || !(neoDriving || finished)} onClick={take} title="Take control (⌘⇧U)">
                  {taking ? <span className="br-spin sm" /> : I.hand()}<span>{taking ? "Taking control…" : "Take control"}</span>
                </MorphPill>
              )
            )}
            {job && st.status !== "ended" && <button className="br-pill bad" onClick={stop} aria-disabled={off || undefined} title="Stop NEO (⌘.)">{I.stop()}<span>Stop</span></button>}
          </div>
        )}
      </div>
    </section>
  );
}

function StepsToggle({ steps, open, onToggle, controls }: { steps: { step_id: number }[]; open: boolean; onToggle: () => void; controls: string }) {
  // The glyph and the count: page thumbnails at footer size read as smudges (they live in the drawer).
  return (
    <button className="br-steps-toggle" aria-expanded={open} aria-controls={controls} title="Steps" onClick={onToggle}>
      <span className="br-steps-glyph" aria-hidden>{I.steps(14)}</span>
      <span className="br-steps-count">{steps.length} {steps.length === 1 ? "step" : "steps"}</span>
      <span className={"br-chev" + (open ? " open" : "")} aria-hidden>{I.chevron(12)}</span>
    </button>
  );
}

// ---- JS dialogs and <select> -------------------------------------------------------------------
function DialogSheet({ hub, snap }: { hub: BrowserHub; snap: BrowserSnap }) {
  const d = snap.dialog!;
  const [text, setText] = useState(d.default ?? "");
  const ref = useRef<HTMLDivElement>(null);
  const answer = (accept: boolean) => {
    void hub.control("dialog", { dialog_id: d.dialog_id, accept, ...(d.kind === "prompt" && accept ? { text: text.slice(0, 2000) } : {}) });
    hub.dismiss("dialog", d.dialog_id);
  };
  useEffect(() => {
    const el = ref.current;
    el?.querySelector<HTMLElement>(d.kind === "prompt" ? "input" : ".br-pill.solid")?.focus();
  }, [d.dialog_id, d.kind]);
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); answer(false); }
    else if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); answer(true); }
    else if (e.key === "Tab") {
      const f = [...(ref.current?.querySelectorAll<HTMLElement>("input, button") ?? [])];
      const i = f.indexOf(document.activeElement as HTMLElement);
      const n = f[(i + (e.shiftKey ? -1 : 1) + f.length) % f.length];
      if (n) { e.preventDefault(); n.focus(); }
    }
  };
  const origin = d.origin || displayHost(snap.live!.url);
  return (
    <div className="br-sheet-scrim" data-noinput onPointerDown={(e) => e.stopPropagation()}>
      <div className="br-sheet" role="alertdialog" aria-modal="true" aria-labelledby="br-dlg-origin" aria-describedby="br-dlg-msg" ref={ref} onKeyDown={onKey}>
        <div className="br-sheet-origin" id="br-dlg-origin">{origin} says</div>
        <div className="br-sheet-msg" id="br-dlg-msg" dir="auto">{d.message.slice(0, 2000) || (d.kind === "beforeunload" ? "Leave this page? Changes you made may not be saved." : "")}</div>
        {d.kind === "prompt" && <input className="br-sheet-input" value={text} onChange={(e) => setText(e.target.value)} aria-label="Answer" spellCheck={false} />}
        <div className="br-sheet-btns">
          {d.kind !== "alert" && <button className="br-pill" onClick={() => answer(false)}>{d.kind === "beforeunload" ? "Stay" : "Cancel"}</button>}
          <button className="br-pill solid" onClick={() => answer(true)}>{d.kind === "beforeunload" ? "Leave" : "OK"}</button>
        </div>
      </div>
    </div>
  );
}

function SelectPopover({ hub, snap }: { hub: BrowserHub; snap: BrowserSnap }) {
  const s = snap.select!;
  const opts = s.options.slice(0, 500);
  const [active, setActive] = useState(() => Math.max(0, opts.findIndex((o) => o.selected)));
  const [pos, setPos] = useState<{ left: number; top: number; minW: number; maxH: number } | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const typed = useRef({ s: "", t: 0 });

  useLayoutEffect(() => {
    // Anchor to the <select> in view px: the canvas is placed at the frame rect.
    const wrap = ref.current?.parentElement;
    const canvas = wrap?.querySelector("canvas");
    if (!wrap || !canvas || !snap.live) return;
    const wb = wrap.getBoundingClientRect(), cb = canvas.getBoundingClientRect();
    const sc = cb.width / (snap.live.viewport.w || 1);
    const x = cb.left - wb.left + s.rect.x * sc;
    const below = cb.top - wb.top + (s.rect.y + s.rect.h) * sc + 4;
    const above = cb.top - wb.top + s.rect.y * sc - 4;
    const room = wb.height - below - 8;
    const maxH = Math.min(280, Math.max(room, above - 8));
    const top = room >= Math.min(160, opts.length * 30 + 12) ? below : Math.max(8, above - Math.min(280, opts.length * 30 + 12));
    setPos({ left: Math.max(8, Math.min(x, wb.width - 180)), top, minW: Math.max(140, s.rect.w * sc), maxH });
  }, [s, snap.live, opts.length]);

  useEffect(() => { ref.current?.focus(); }, [s.select_id]);
  useEffect(() => { ref.current?.querySelector(`[data-i="${active}"]`)?.scrollIntoView({ block: "nearest" }); }, [active]);

  const pick = (i: number) => {
    const o = opts[i];
    if (!o || o.disabled) return;
    void hub.control("select", { select_id: s.select_id, value: o.value });
    hub.dismiss("select", s.select_id);
  };
  const cancel = () => hub.dismiss("select", s.select_id);
  const move = (d: number) => {
    let i = active;
    for (let k = 0; k < opts.length; k++) { i = (i + d + opts.length) % opts.length; if (!opts[i].disabled) break; }
    setActive(i);
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(active); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); cancel(); }
    else if (e.key.length === 1) {
      const now = performance.now();
      typed.current = { s: (now - typed.current.t < 700 ? typed.current.s : "") + e.key.toLowerCase(), t: now };
      const i = opts.findIndex((o) => !o.disabled && o.label.toLowerCase().startsWith(typed.current.s));
      if (i >= 0) setActive(i);
    }
  };
  useEffect(() => {
    const h = (e: PointerEvent) => { if (!ref.current?.contains(e.target as Node)) cancel(); };
    window.addEventListener("pointerdown", h, true);
    return () => window.removeEventListener("pointerdown", h, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.select_id]);

  let lastGroup: string | undefined;
  return (
    <div
      ref={ref}
      className="br-pop select"
      role="listbox"
      aria-label="Choose an option"
      aria-activedescendant={`br-opt-${active}`}
      tabIndex={-1}
      data-noinput
      onKeyDown={onKey}
      onPointerDown={(e) => e.stopPropagation()}
      style={pos ? { left: pos.left, top: pos.top, minWidth: pos.minW, maxHeight: pos.maxH } : { visibility: "hidden" }}
    >
      {opts.map((o, i) => {
        const head = o.group && o.group !== lastGroup ? <div className="br-opt-group" role="presentation">{o.group}</div> : null;
        lastGroup = o.group;
        return (
          <div key={i}>
            {head}
            <div
              id={`br-opt-${i}`}
              data-i={i}
              role="option"
              aria-selected={o.selected}
              aria-disabled={o.disabled || undefined}
              className={"br-opt" + (i === active ? " active" : "") + (o.disabled ? " disabled" : "")}
              onPointerEnter={() => !o.disabled && setActive(i)}
              onClick={() => pick(i)}
            >
              <span className="br-opt-check">{o.selected ? I.check(13) : null}</span>
              <span dir="auto">{o.label}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

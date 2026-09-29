// The song card in NEO's chat, Shazam-style: while NEO listens, a ring that pulses with how loud
// it is, what it listens to and for how long, and a Stop; then a short shimmer while it matches;
// then the song — cover (a data URL the core fetched: nothing here loads a remote image), title,
// artist, album · year · genre, and buttons that ask the core to open the song's links. No match,
// an error or a stop leave one calm line. Older songs shrink to a row as newer ones arrive.
import { useEffect, useState, type CSSProperties, type ReactNode } from "react";
import { I } from "../browser/icons";
import { useSongSession, type SongHub } from "./hub";
import {
  ERROR_TEXT, NOT_FOUND_TEXT, STOPPED_TEXT, announce, clockAt, eqBars, extraLine, fmtSecs, hearing, shownLevel, sourceLabel, viewOf,
  type SongSession, type SongView,
} from "./state";
import type { SongLinkKind } from "./types";
import "./song.css";

export const noteIcon = (s = 14) => (
  <svg viewBox="0 0 24 24" width={s} height={s} aria-hidden focusable="false" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 18V6.2l10-2.2v11.8" />
    <circle cx="6.5" cy="18" r="2.5" />
    <circle cx="16.5" cy="15.8" r="2.5" />
  </svg>
);
const playIcon = (s = 11) => (
  <svg viewBox="0 0 24 24" width={s} height={s} aria-hidden focusable="false" fill="currentColor"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.4-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" /></svg>
);

const LINKS: { which: SongLinkKind; label: string; icon: () => ReactNode }[] = [
  { which: "apple_music", label: "Apple Music", icon: () => playIcon(10) },
  { which: "shazam", label: "Shazam", icon: () => I.popout(12) },
];

/** Ticks while NEO listens, so the seconds run on between the core's updates. */
function useClock(on: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!on) return;
    setNow(Date.now());
    const t = window.setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, [on]);
  return now;
}

const lvl = (level: number) => ({ "--lvl": shownLevel(level).toFixed(3) }) as CSSProperties;

/** The listening ring: two ripples whose strength and a core whose size follow the loudness. */
function Orb({ level, small }: { level: number; small?: boolean }) {
  return (
    <span className={"song-orb" + (small ? " sm" : "")} style={lvl(level)} aria-hidden>
      {!small && <><span className="song-ripple" /><span className="song-ripple r2" /></>}
      <span className="song-orb-core">{noteIcon(small ? 12 : 20)}</span>
    </span>
  );
}

function Meter({ level, seed }: { level: number; seed: number }) {
  return (
    <span className="song-eq" aria-hidden>
      {eqBars(level, seed).map((h, i) => <i key={i} style={{ transform: `scaleY(${h.toFixed(3)})` }} />)}
    </span>
  );
}

function StopButton({ hub, x, small }: { hub: SongHub; x: SongSession; small?: boolean }) {
  const label = x.stopping ? "Stopping…" : "Stop";
  return small ? (
    <button className="br-icon song-stop-sm" onClick={() => hub.stop()} disabled={x.stopping} aria-disabled={x.stopping} aria-label="Stop listening" title="Stop listening now">
      {I.stop(9)}
    </button>
  ) : (
    <button className="br-pill bad sm song-stop" onClick={() => hub.stop()} disabled={x.stopping} aria-disabled={x.stopping} aria-label={x.stopping ? "Stopping" : "Stop listening"} title="Stop listening now">
      {I.stop(9)}<span>{label}</span>
    </button>
  );
}

function LinkButton({ hub, sid, which, label, icon }: { hub: SongHub; sid: string; which: SongLinkKind; label: string; icon: ReactNode }) {
  const [opening, setOpening] = useState(false);
  useEffect(() => {
    if (!opening) return;
    const t = window.setTimeout(() => setOpening(false), 1600);
    return () => clearTimeout(t);
  }, [opening]);
  return (
    <button
      className="br-pill sm song-link"
      onClick={() => { if (!opening && hub.openLink(sid, which)) setOpening(true); }}
      aria-disabled={opening}
      aria-label={`Open in ${label}`}
      title={`Open in ${label}`}
    >
      {icon}<span>{opening ? "Opening…" : label}</span>
    </button>
  );
}

/** The one line a card without a song leaves: no match, an error, or a stop. */
function noteLine(x: SongSession, view: SongView): { text: string; dim: string } {
  if (view === "error") return { text: x.message ?? ERROR_TEXT, dim: "" };
  if (view === "stopped") return { text: STOPPED_TEXT, dim: x.elapsedS >= 1 ? ` · after ${fmtSecs(x.elapsedS)}` : "" };
  return { text: x.message ?? NOT_FOUND_TEXT, dim: x.elapsedS >= 1 ? ` · listened ${fmtSecs(x.elapsedS)}` : "" };
}

function Listening({ hub, x, connected, view }: { hub: SongHub; x: SongSession; connected: boolean; view: SongView }) {
  const now = useClock(connected);
  const secs = clockAt(x, now);
  const pct = x.listenS > 0 ? Math.min(1, secs / x.listenS) : 0;
  return (
    <section className={`song-card live ${view}`} aria-label="Song recognition" data-song-sid={x.sid}>
      <div className="song-listen">
        <Orb level={x.level} />
        <div className="song-listen-text" aria-hidden>
          <div className="song-listen-title">
            <span className={"song-state" + (view === "matching" ? " shimmer" : "")}>{view === "matching" ? "Matching…" : "Listening…"}</span>
            <Meter level={x.level} seed={x.at} />
          </div>
          <div className="song-listen-sub">{connected ? `${sourceLabel(x.source)} · ${fmtSecs(secs)}` : "Reconnecting…"}</div>
          {x.message && <div className="song-listen-note" title={x.message}>{x.message}</div>}
        </div>
        <span className="song-sr">{announce(x)}</span>
        <StopButton hub={hub} x={x} />
      </div>
      <div className="song-progress" aria-hidden><span style={{ transform: `scaleX(${pct.toFixed(4)})` }} /></div>
      {view === "matching" && <span className="song-sweep" aria-hidden />}
    </section>
  );
}

/** The newest song: its links are the ones the core still opens. */
function Found({ hub, x }: { hub: SongHub; x: SongSession }) {
  const r = x.result!;
  const extra = extraLine(r);
  const links = LINKS.filter((l) => r.links[l.which]);
  return (
    <section className="song-card found" aria-label={announce(x)} data-song-sid={x.sid}>
      {r.cover && <img className="song-backdrop" src={r.cover} alt="" aria-hidden draggable={false} />}
      <div className="song-found">
        <div className="song-cover">
          {r.cover ? <img src={r.cover} alt={`Cover art for ${r.album ?? r.title}`} draggable={false} /> : noteIcon(24)}
        </div>
        <div className="song-meta">
          <div className="song-title" title={r.title}>{r.title}</div>
          {r.artist && <div className="song-artist" title={r.artist}>{r.artist}</div>}
          {extra && <div className="song-extra" title={extra}>{extra}</div>}
        </div>
      </div>
      <div className="song-foot">
        <span className="song-when">{x.elapsedS > 0 ? `Recognised in ${fmtSecs(x.elapsedS, true)}` : "Recognised"}</span>
        {links.map((l) => <LinkButton key={l.which} hub={hub} sid={x.sid} which={l.which} label={l.label} icon={l.icon()} />)}
      </div>
    </section>
  );
}

/** A song from before: a row with its cover, title and artist (its links belonged to its own turn). */
function PastSong({ x }: { x: SongSession }) {
  const r = x.result!;
  const second = [r.artist, r.album, r.year].filter(Boolean).join(" · ");
  return (
    <div className="song-row past" data-song-sid={x.sid}>
      <span className="song-row-icon cover">{r.cover ? <img src={r.cover} alt="" draggable={false} /> : noteIcon(15)}</span>
      <span className="song-row-text">
        <span className="song-row-l1 strong" title={r.title}>{r.title}</span>
        {second && <span className="song-row-l2" title={second}>{second}</span>}
      </span>
    </div>
  );
}

function Note({ x, view }: { x: SongSession; view: SongView }) {
  const { text, dim } = noteLine(x, view);
  return (
    <div className={`song-row line ${view}`} data-song-sid={x.sid}>
      <span className="song-row-icon">{view === "error" ? I.warn(15) : noteIcon(15)}</span>
      <span className="song-row-text">
        <span className="song-row-l1" title={text}>{text}<span className="song-row-dim">{dim}</span></span>
      </span>
    </div>
  );
}

export interface SongCardProps {
  hub: SongHub;
  x: SongSession;
  /** The newest session (a found song keeps its full card and links; older ones shrink to a row). */
  latest: boolean;
  connected: boolean;
}

/** A song session in the transcript, whichever face it has now. While NEO listens it stays in
 *  sight (sticky to the log's edge) with its Stop. */
export function SongCard({ hub, x, latest, connected }: SongCardProps) {
  const view = viewOf(x);
  if (view === "listening" || view === "matching") return <Listening hub={hub} x={x} connected={connected} view={view} />;
  if (view === "found") return latest ? <Found hub={hub} x={x} /> : <PastSong x={x} />;
  return <Note x={x} view={view} />;
}

/** One compact line (the chat strip under a docked browser, or a small card over the page when
 *  there is no chat to show it in): listening with a Stop, or what was found. */
export function SongMini({ hub, x, connected, floating }: SongCardProps & { floating?: boolean }) {
  const view = viewOf(x);
  const live = hearing(x);
  const now = useClock(live && connected);
  const r = x.result;
  let icon: ReactNode, text: string, dim = "";
  if (live) {
    icon = <Orb level={x.level} small />;
    text = view === "matching" ? "Matching…" : `Listening ${sourceLabel(x.source)}`;
    dim = connected ? ` · ${fmtSecs(clockAt(x, now))}` : " · reconnecting…";
  } else if (view === "found" && r) {
    icon = r.cover ? <img src={r.cover} alt="" draggable={false} /> : noteIcon(13);
    text = r.title;
    dim = r.artist ? ` — ${r.artist}` : "";
  } else {
    icon = view === "error" ? I.warn(13) : noteIcon(13);
    ({ text, dim } = noteLine(x, view));
  }
  return (
    <div className={`song-mini ${view}` + (floating ? " floating" : "")} data-song-sid={x.sid} {...(floating ? { role: "status" } : {})}>
      <span className={"song-mini-icon" + (view === "found" && r?.cover ? " cover" : "")}>{icon}</span>
      <span className="song-mini-text" aria-hidden={live}>
        <span className={view === "matching" ? "song-state shimmer" : ""}>{text}</span><span className="song-row-dim">{dim}</span>
      </span>
      {live && <span className="song-sr">{announce(x)}</span>}
      {live && <StopButton hub={hub} x={x} small />}
    </div>
  );
}

/** A song card by sid, following its own session (level updates re-render it alone): the full card
 *  in the chat, or one line (`mini`) in the docked browser's strip or floating over the page. */
export function SongItem({ hub, sid, mini, floating }: { hub: SongHub; sid: string; mini?: boolean; floating?: boolean }) {
  const { x, latest, connected } = useSongSession(hub, sid);
  if (!x) return null;
  return mini
    ? <SongMini hub={hub} x={x} latest={latest} connected={connected} floating={floating} />
    : <SongCard hub={hub} x={x} latest={latest} connected={connected} />;
}

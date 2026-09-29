// Dev-only control panel for the song mock core (?mock=song): pick a run, loop them, change speed,
// flip the theme, drop the connection (the card asks song_hello again) or restart the core (a live
// card gets no answer and ends as "gone").
import { useEffect, useState } from "react";
import { songMock, type SongRun } from "./core";
import "../../browser/mock/mock.css";

const RUNS: [SongRun | "loop", string][] = [
  ["found", "Found"], ["not_found", "No match"], ["error", "Error"],
  ["stopped", "Stopped"], ["listening", "Listening"], ["matching", "Matching"],
];

export default function Hud() {
  const core = songMock();
  const [open, setOpen] = useState(true);
  const [speed, setSpeed] = useState(core.speed);
  const [light, setLight] = useState(document.documentElement.dataset.theme === "light");
  const [cur, setCur] = useState(core.current);
  useEffect(() => {
    const fn = () => setCur(core.current);
    core.listeners.add(fn);
    return () => { core.listeners.delete(fn); };
  }, [core]);
  const go = (k: SongRun | "loop") => {
    const u = new URL(location.href);
    if (k === "loop") u.searchParams.delete("state"); else u.searchParams.set("state", k);
    history.replaceState(null, "", u);
    core.pick(k);
  };
  return (
    <div className={"mock-hud" + (open ? "" : " closed")} role="complementary" aria-label="Song mock controls">
      <div className="mock-hud-head">
        <span className="mock-dot" /><b>NEO song mock</b><span className="mock-sub">scripted · no mic · no network</span>
        <button className="mock-x" onClick={() => setOpen((o) => !o)} aria-label={open ? "Hide mock controls" : "Show mock controls"}>{open ? "–" : "+"}</button>
      </div>
      {open && (
        <>
          <button className={"mock-run" + (cur === "loop" ? " on" : "")} onClick={() => go("loop")}>▶ Loop: found → no match → found → error</button>
          <div className="mock-grid">
            {RUNS.map(([id, label]) => <button key={id} className={cur === id ? "on" : ""} onClick={() => go(id)}>{label}</button>)}
          </div>
          <div className="mock-row">
            <span>Speed</span>
            {[0.5, 1, 2].map((x) => <button key={x} className={speed === x ? "on" : ""} onClick={() => { setSpeed(x); core.setSpeed(x); }}>{x}×</button>)}
            <span className="mock-gap" />
            <span>Theme</span>
            <button className={!light ? "on" : ""} onClick={() => { setLight(false); document.documentElement.dataset.theme = "dark"; }}>Dark</button>
            <button className={light ? "on" : ""} onClick={() => { setLight(true); document.documentElement.dataset.theme = "light"; }}>Light</button>
          </div>
          <div className="mock-row">
            <button onClick={() => core.kill(1500)}>Reconnect</button>
            <button onClick={() => core.kill(1500, true)}>Restart core</button>
          </div>
          <p className="mock-tip">Ask “what song is this?” in the chat, or press Stop on the card. Links are never opened: the core notes which one NEO would open.</p>
        </>
      )}
    </div>
  );
}

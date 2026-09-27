// Dev-only control panel for the mock core (?mock=browser): run the scripted task, jump to any
// state in the spec, change speed, flip the theme, kill the core to see "Reconnecting".
import { useState } from "react";
import { mockCore } from "./server";
import "./mock.css";

const STATES: [string, string][] = [
  ["starting", "Starting"], ["loading", "Loading"], ["acting", "NEO acting"], ["thinking", "Thinking"],
  ["waiting", "Needs OK"], ["user", "You drive"], ["login", "Login"], ["captcha", "CAPTCHA"],
  ["2fa", "2FA code"], ["other", "Other step"], ["dialog", "JS dialog"], ["select", "<select>"],
  ["error", "Crashed"], ["offline", "Core offline"], ["ended", "Ended row"],
];

export default function Hud() {
  const core = mockCore();
  const [open, setOpen] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [light, setLight] = useState(document.documentElement.dataset.theme === "light");
  const [cur, setCur] = useState(() => new URLSearchParams(location.search).get("state") ?? "");
  const go = (s: string) => {
    setCur(s);
    const u = new URL(location.href);
    if (s) u.searchParams.set("state", s); else u.searchParams.delete("state");
    history.replaceState(null, "", u);
    if (s) void core.jump(s); else void core.shop();
  };
  return (
    <div className={"mock-hud" + (open ? "" : " closed")} role="complementary" aria-label="Mock controls">
      <div className="mock-hud-head">
        <span className="mock-dot" /><b>NEO mock core</b><span className="mock-sub">scripted · no network</span>
        <button className="mock-x" onClick={() => setOpen((o) => !o)} aria-label={open ? "Hide mock controls" : "Show mock controls"}>{open ? "–" : "+"}</button>
      </div>
      {open && (
        <>
          <button className={"mock-run" + (cur === "" ? " on" : "")} onClick={() => go("")}>▶ Run the shopping task</button>
          <div className="mock-grid">
            {STATES.map(([id, label]) => <button key={id} className={cur === id ? "on" : ""} onClick={() => go(id)}>{label}</button>)}
          </div>
          <div className="mock-row">
            <span>Speed</span>
            {[0.5, 1, 2].map((x) => <button key={x} className={speed === x ? "on" : ""} onClick={() => { setSpeed(x); core.setSpeed(x); }}>{x}×</button>)}
            <span className="mock-gap" />
            <span>Theme</span>
            <button className={!light ? "on" : ""} onClick={() => { setLight(false); document.documentElement.dataset.theme = "dark"; }}>Dark</button>
            <button className={light ? "on" : ""} onClick={() => { setLight(true); document.documentElement.dataset.theme = "light"; }}>Light</button>
          </div>
          <p className="mock-tip">Take control, then click, scroll and type in the page — the store responds (try signing in). <kbd>⌘⇧U</kbd> take/hand back · <kbd>⌘⇧E</kbd> expand · <kbd>⌘⇧M</kbd> minimize · <kbd>⌘⇧O</kbd> pop out · <kbd>⌘.</kbd> stop</p>
        </>
      )}
    </div>
  );
}

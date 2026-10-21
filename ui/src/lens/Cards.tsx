// What shows in place of a selection when Search Screen can't work yet (design §13): no Screen
// Recording, no core of NEO.app's yet, the core restarted, or no line to it.

import type { Action, Card as CardKind } from "./state";

export function Card({ kind, dispatch }: { kind: Exclude<CardKind, null>; dispatch: (a: Action) => void }) {
  const close = (why: string) => dispatch({ t: "close", why });
  if (kind === "permission") {
    return (
      <div className="lx-card" role="alertdialog" aria-labelledby="lx-card-t" aria-describedby="lx-card-d">
        <h1 id="lx-card-t">NEO can't see your screen yet</h1>
        <p id="lx-card-d">Search Screen needs Screen Recording. Only the area you choose is ever used, and nothing is saved.</p>
        <div className="lx-row-actions">
          <button className="lx-btn primary" autoFocus onClick={() => dispatch({ t: "allow_sc" })}>Open Privacy Settings</button>
          <button className="lx-btn" onClick={() => close("cancel")}>Not now</button>
        </div>
        <div className="foot">Just allowed it? Quit and reopen NEO.</div>
      </div>
    );
  }
  const words: Record<Exclude<CardKind, null | "permission">, [string, string]> = {
    starting: ["NEO is starting", "Try Search Screen again in a moment."],
    restarted: ["NEO restarted", "Close Search Screen and try again."],
    unreachable: ["Search Screen couldn't reach NEO", "Try again in a moment."],
  };
  const [title, body] = words[kind];
  return (
    <div className="lx-card" role="alertdialog" aria-labelledby="lx-card-t" aria-describedby="lx-card-d">
      <h1 id="lx-card-t">{title}</h1>
      <p id="lx-card-d">{body}</p>
      <div className="lx-row-actions">
        <button className="lx-btn" autoFocus onClick={() => close(kind === "starting" ? "cancel" : "gone")}>Close</button>
      </div>
    </div>
  );
}

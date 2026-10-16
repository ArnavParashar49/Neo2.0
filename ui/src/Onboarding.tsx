// The chat's "Give NEO a brain" card (see ./onboard.ts): in NEO's window and in the notch alike.
import { useEffect, useRef, useState } from "react";
import { hasShell, openExternal } from "./shell";
import { inNotch } from "./host";
import { ALSO, BODY, COMMAND, GEMINI_KEY_URL, GET_KEY, SET_UP, TITLE, afterOpenKeys, afterSetupNeeded, cardNote, type CardState } from "./onboard";
import type { OpenKeysResult } from "./ws";
import type { SetupNeeded } from "./keys/types";

const OPENING_GIVE_UP_MS = 6000; // an older core never answers open_keys: the button comes back

/** `tail`: under the chat's last lines (a chat already under way), rather than as the empty chat. */
export function Onboarding({ send, openKeys, setup, tail = false }: {
  tail?: boolean;
  send: (cmd: Record<string, unknown>) => boolean;
  openKeys: OpenKeysResult | null;
  setup: SetupNeeded | null;
}) {
  const [state, setState] = useState<CardState>(() => afterSetupNeeded(setup));
  const asked = useRef(0);

  // The core's answer to this card's own ask (not one from before it).
  useEffect(() => {
    if (openKeys && asked.current && openKeys.at >= asked.current) setState(afterOpenKeys(openKeys));
  }, [openKeys]);
  // A request that needed a model: the core opened the window itself, or there's no NEO.app.
  useEffect(() => {
    const s = afterSetupNeeded(setup);
    if (s !== "idle") setState(s);
  }, [setup]);
  useEffect(() => {
    if (state !== "opening" && state !== "busy" && state !== "opened") return;
    const t = window.setTimeout(() => setState((s) => (s === "terminal" ? s : "idle")), state === "opening" ? OPENING_GIVE_UP_MS : 8000);
    return () => clearTimeout(t);
  }, [state]);

  const setUp = () => {
    asked.current = Date.now();
    if (send({ cmd: "open_keys" })) setState("opening");
  };
  const getKey = () => {
    if (hasShell) void openExternal(GEMINI_KEY_URL).catch(() => {});
    else window.open(GEMINI_KEY_URL, "_blank", "noopener");
  };
  const note = cardNote(state);

  return (
    <section className={"onboard" + (tail ? " tail" : "")} aria-labelledby="onboard-title">
      <div className="onboard-title" id="onboard-title">{TITLE}</div>
      <p className="onboard-body">{BODY}</p>
      <div className="onboard-btns">
        <button className="primary" onClick={setUp} disabled={state === "opening"}>{state === "opening" ? "Opening…" : SET_UP}</button>
        <button onClick={getKey}>{GET_KEY} <span aria-hidden>↗</span></button>
      </div>
      {note && (
        <div className={"onboard-note" + (state === "terminal" ? " cli" : "")} role="status">
          {state === "terminal" ? <>In a terminal, run: <code>{COMMAND}</code></> : note}
        </div>
      )}
      {!inNotch || state !== "terminal" ? <p className="onboard-also">{ALSO}</p> : null}
    </section>
  );
}

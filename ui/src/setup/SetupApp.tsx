// index.html#/setup — NEO Setup: installs NEO's engine on a new Mac, then is the checklist; also the
// menu's Repair, Check for updates and Uninstall pages. Only NEO.app's shell opens this window
// (ui/src-tauri/src/setupwin.rs); everything it shows is the shell's own snapshot (./types.ts), and
// every button is one of the shell's commands (./wire.ts). `?mock=<view>` (dev builds) plays it all
// without a shell.
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  agoText, checklistRows, defaultChosen, enginePercent, fetchLine, fetchPercent, fixesFor, optionalRows, planRows, roomLine, sizeText,
  stageText, title, uninstallTotal, updateLine, type FixButton, type Row,
} from "./model";
import { mockKind, mockWire, tauriWire, type SetupWire } from "./wire";
import type { Snapshot } from "./types";
import "./setup.css";

const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

function errorText(e: unknown): string {
  const s = typeof e === "string" ? e : e instanceof Error ? e.message : String(e);
  switch (s) {
    case "busy": return "Setup is already running.";
    case "no_manifest": return "This build of NEO has no engine to install.";
    case "not_ready": return "NEO's engine is still starting — try again in a moment.";
    case "forbidden": return "This window can't do that.";
    default: return s;
  }
}

function Bar({ pct }: { pct: number | null }) {
  return (
    <div className={"sx-bar" + (pct == null ? " busy" : "")} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct ?? undefined}>
      <div style={pct == null ? undefined : { width: `${pct}%` }} />
    </div>
  );
}

function Btn({ children, onClick, primary, disabled, danger }: { children: ReactNode; onClick: () => void; primary?: boolean; disabled?: boolean; danger?: boolean }) {
  return <button type="button" className={"sx-btn" + (primary ? " primary" : "") + (danger ? " danger" : "")} disabled={disabled} onClick={onClick}>{children}</button>;
}

type Call = (cmd: string, args?: Record<string, unknown>) => void;

export function SetupApp() {
  const wire: SetupWire = useMemo(() => {
    const kind = mockKind(location.search, import.meta.env.DEV);
    return kind ? mockWire(kind) : tauriWire();
  }, []);
  const [s, setS] = useState<Snapshot | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    const off = wire.on((next) => setS(next));
    wire.state().then(setS).catch((e) => setProblem(errorText(e)));
    return off;
  }, [wire]);

  // Shown once painted (the shell shows it anyway after a moment); ⌘W closes it.
  useEffect(() => {
    if (!inTauri) return;
    const t = window.setTimeout(async () => {
      try { const { getCurrentWindow } = await import("@tauri-apps/api/window"); const w = getCurrentWindow(); await w.show(); await w.setFocus(); } catch { /* the shell shows it */ }
    }, 60);
    const onKey = async (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "w" && e.metaKey && !e.shiftKey && !e.altKey && !e.ctrlKey) {
        e.preventDefault();
        try { const { getCurrentWindow } = await import("@tauri-apps/api/window"); await getCurrentWindow().close(); } catch { /* not in the app */ }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => { clearTimeout(t); window.removeEventListener("keydown", onKey); };
  }, []);

  const call: Call = useCallback((cmd, args) => {
    setProblem(null);
    wire.call(cmd, args).catch((e) => setProblem(errorText(e)));
  }, [wire]);

  // The checklist looks again whenever it comes up (permissions, what's on this Mac).
  const view = s?.view;
  useEffect(() => { if (view === "checklist") call("setup_check"); }, [view, call]);

  if (!s) {
    return <div className="sx"><header className="sx-head"><span className="sx-title">NEO Setup</span></header><main className="sx-main"><p className="sx-dim">{problem ?? "Loading…"}</p></main></div>;
  }
  return (
    <div className="sx">
      <header className="sx-head" data-tauri-drag-region>
        <span className="sx-title" data-tauri-drag-region>{title(s.view, s.product)}</span>
        {s.sandbox && <span className="sx-badge">Sandbox</span>}
      </header>
      <main className="sx-main">
        {problem && <div className="sx-alert" role="alert">{problem}</div>}
        <Body s={s} call={call} />
      </main>
    </div>
  );
}

function Body({ s, call }: { s: Snapshot; call: Call }) {
  switch (s.view) {
    case "move": return <Move s={s} call={call} />;
    case "unsupported": return <Unsupported s={s} call={call} />;
    case "no_engine": return <NoEngine s={s} call={call} />;
    case "welcome": case "new_engine": case "repair": return <Start s={s} call={call} />;
    case "engine": return <Engine s={s} call={call} />;
    case "checklist": return <Checklist s={s} call={call} />;
    case "updates": return <Updates s={s} call={call} />;
    case "uninstall": return <Uninstall s={s} call={call} />;
  }
  return null;
}

function Move({ s, call }: { s: Snapshot; call: Call }) {
  return (
    <section className="sx-section">
      <p className="sx-lead">
        {s.product} is running from {s.place === "volume" ? "its disk image" : "a temporary copy macOS made of it"}. Drag {s.product} into your
        Applications folder, then open it from there.
      </p>
      <p className="sx-dim">{s.product} keeps its engine, and opens at login, from the copy in Applications — so it doesn't set anything up from here.</p>
      <div className="sx-btns">
        <Btn primary onClick={() => call("setup_open", { what: "applications" })}>Show Applications</Btn>
        <Btn onClick={() => call("setup_quit")}>Quit</Btn>
      </div>
    </section>
  );
}

function Unsupported({ s, call }: { s: Snapshot; call: Call }) {
  return (
    <section className="sx-section">
      <p className="sx-lead">{s.product} needs macOS 14 Sonoma or newer, on a Mac with Apple silicon.{s.macos ? ` This Mac runs macOS ${s.macos}.` : ""}</p>
      <div className="sx-btns"><Btn onClick={() => call("setup_quit")}>Quit</Btn></div>
    </section>
  );
}

function NoEngine({ s, call }: { s: Snapshot; call: Call }) {
  return (
    <section className="sx-section">
      <p className="sx-lead">This build of {s.product} doesn't carry an engine to install — it's a developer build.</p>
      <p className="sx-dim">Build {s.product} from its source with <code>scripts/install-app.sh</code>, or download {s.product} from its releases page.</p>
      {s.manifest_problem && <p className="sx-dim">The app's engine manifest: {s.manifest_problem}</p>}
      <div className="sx-btns">
        <Btn onClick={() => call("setup_open", { what: "releases" })}>Releases page</Btn>
        <Btn onClick={() => call("setup_open", { what: "logs" })}>Open Logs</Btn>
      </div>
    </section>
  );
}

/** Welcome, a newer engine for an update, or Repair: what will happen, then the button. */
function Start({ s, call }: { s: Snapshot; call: Call }) {
  const plan = s.plan;
  if (!plan) return <NoEngine s={s} call={call} />;
  const room = roomLine(plan);
  const engine = sizeText(plan.engine_bytes);
  const lead =
    s.view === "new_engine"
      ? `${s.product} ${plan.app_version} needs a newer engine (${engine}). It's installed beside the old one, then ${s.product} switches over and removes the old one.`
      : s.view === "repair"
        ? `Repair downloads ${s.product}'s engine again (${engine}) and reinstalls it. ${s.product} stops for a minute while it switches over; your settings, keys and memory stay as they are.`
        : `${s.product} runs its assistant on this Mac. Setting it up downloads:`;
  const button = s.view === "new_engine" ? "Update the engine" : s.view === "repair" ? "Repair" : `Set up ${s.product}`;
  return (
    <section className="sx-section">
      <p className="sx-lead">{lead}</p>
      {s.view === "welcome" && (
        <table className="sx-table">
          <thead><tr><th>What</th><th>From</th><th>Download</th><th>On disk</th></tr></thead>
          <tbody>
            {planRows(plan).map((r) => <tr key={r.what}><td>{r.what}</td><td>{r.from}</td><td className="n">{r.size}</td><td className="n">{r.disk}</td></tr>)}
          </tbody>
        </table>
      )}
      <p className={"sx-dim" + (room.ok ? "" : " sx-warn")}>{room.text}</p>
      {s.view === "welcome" && (
        <p className="sx-dim">
          About 1–3 minutes for the engine, then a few more for the models while you set up the rest. Nothing about you is sent: these come
          straight from the sites above. Optional extras — the on-device voice, web browsing, camera boxes — are offered afterwards.
        </p>
      )}
      <div className="sx-btns">
        <Btn primary disabled={!room.ok || s.engine_running} onClick={() => call("setup_start")}>{button}</Btn>
        <button type="button" className="sx-link" onClick={() => call("setup_use_file")}>Use a downloaded file…</button>
      </div>
      <p className="sx-foot">
        A network that blocks the download? Get <code>{plan.asset}</code> from {s.product}'s release page on another connection, then choose “Use a
        downloaded file…”.
      </p>
    </section>
  );
}

const FIX_LABEL: Readonly<Record<FixButton, string>> = { retry: "Retry", use_file: "Use a downloaded file…", logs: "Open the setup log", releases: "Releases page" };

function Engine({ s, call }: { s: Snapshot; call: Call }) {
  const p = s.engine;
  const failed = p.stage === "failed" || p.stage === "cancelled";
  const fix = (f: FixButton) => {
    if (f === "retry") call("setup_retry");
    else if (f === "use_file") call("setup_use_file");
    else if (f === "logs") call("setup_open", { what: "logs" });
    else call("setup_open", { what: "releases" });
  };
  return (
    <section className="sx-section">
      {!failed && <Bar pct={enginePercent(p)} />}
      <p className={failed ? "sx-alert" : "sx-status"} role={failed ? "alert" : "status"}>{stageText(p, s.plan)}</p>
      {s.engine_from_file && !failed && <p className="sx-dim">From the file you chose.</p>}
      {!failed && p.stage !== "done" && (
        <p className="sx-dim">
          {s.product} downloads its engine{s.plan ? ` from ${s.plan.host}` : ""}, checks it against the checksum built into the app, and installs it
          in Application Support. Nothing else on this Mac changes.
        </p>
      )}
      <div className="sx-btns">
        {s.engine_running && <Btn onClick={() => call("setup_cancel")}>Cancel</Btn>}
        {p.stage === "done" && <Btn primary onClick={() => call("setup_done")}>Close</Btn>}
        {failed && fixesFor(p.error?.code).map((f, i) => <Btn key={f} primary={i === 0} onClick={() => fix(f)}>{FIX_LABEL[f]}</Btn>)}
      </div>
      {failed && <p className="sx-foot">The setup log is in {s.setup_log}.</p>}
    </section>
  );
}

function Dot({ state }: { state: Row["state"] }) {
  return <span className={`sx-dot ${state}`} aria-hidden="true" />;
}

function Checklist({ s, call }: { s: Snapshot; call: Call }) {
  const models = s.essential;
  const modelsPct = fetchPercent(models);
  const modelsHere = s.here.essential === true || models.state === "done";
  return (
    <>
      <section className="sx-section">
        <h2>{s.product}'s models</h2>
        {modelsHere ? (
          <p className="sx-ok">NEO's models are on this Mac.</p>
        ) : models.state === "idle" ? (
          <p className="sx-dim">{s.engine_ok ? "Getting ready to download NEO's models…" : "Waiting for NEO's engine."}</p>
        ) : (
          <>
            {models.state === "running" && <Bar pct={modelsPct} />}
            <p className={models.state === "failed" ? "sx-alert" : "sx-status"}>{fetchLine(models)}</p>
            {(models.state === "failed" || models.state === "cancelled") && <div className="sx-btns"><Btn onClick={() => call("setup_models")}>Try again</Btn></div>}
          </>
        )}
        <p className="sx-dim">You can carry on below while they download: {s.product} uses them as they arrive.</p>
      </section>
      <section className="sx-section">
        <h2>Let {s.product} work</h2>
        <ul className="sx-rows">
          {checklistRows(s).map((r) => (
            <li key={r.id} className="sx-row">
              <Dot state={r.state} />
              <div className="sx-row-t"><strong>{r.title}</strong><span className="sx-dim">{r.detail}</span></div>
              {r.action && r.state !== "ok" && <Btn onClick={() => call(r.action!.cmd, { what: r.action!.arg })}>{r.action.label}</Btn>}
              {r.action && r.state === "ok" && r.id === "brain" && <Btn onClick={() => call("setup_open", { what: "keys" })}>{r.action.label}</Btn>}
            </li>
          ))}
        </ul>
      </section>
      <section className="sx-section">
        <h2>Extras (optional)</h2>
        <ul className="sx-rows">
          {optionalRows(s).map((o) => {
            const running = o.fetch?.state === "running";
            const pct = o.fetch ? fetchPercent(o.fetch) : null;
            return (
              <li key={o.group} className="sx-row">
                <Dot state={o.here ? "ok" : running ? "wait" : "note"} />
                <div className="sx-row-t">
                  <strong>{o.title}</strong>
                  <span className="sx-dim">{o.detail}</span>
                  {running && <Bar pct={pct} />}
                  {o.fetch && (running || o.fetch.state === "failed") && <span className={o.fetch.state === "failed" ? "sx-dim sx-bad" : "sx-dim"}>{fetchLine(o.fetch, o.title.toLowerCase())}</span>}
                </div>
                {o.here ? <span className="sx-ok sx-small">On this Mac</span> : <Btn disabled={!s.engine_ok} onClick={() => call("setup_optional", { group: o.group, on: !running })}>{o.label}</Btn>}
              </li>
            );
          })}
        </ul>
      </section>
      <section className="sx-section">
        <label className={"sx-check" + (s.login_allowed ? "" : " off")}>
          <input type="checkbox" checked={s.login_allowed && s.login_on} disabled={!s.login_allowed} onChange={(e) => call("setup_login", { on: e.target.checked })} />
          Open {s.product} at login
        </label>
        {!s.login_allowed && <p className="sx-dim">Move {s.product} into Applications to open it at login.</p>}
      </section>
      <div className="sx-btns sx-end">
        <span className="sx-dim">Say “Neo”, or press ⌥⌘ (⌘⇧Space works too), when you're ready. This list stays in the menu: Setup checklist…</span>
        <Btn primary onClick={() => call("setup_done")}>Done</Btn>
      </div>
    </>
  );
}

function Updates({ s, call }: { s: Snapshot; call: Call }) {
  const u = s.update;
  return (
    <section className="sx-section">
      <p className="sx-lead">{updateLine(u, s.version, s.product)}</p>
      <div className="sx-btns">
        {u.offer && <Btn primary onClick={() => call("setup_update", { action: "open" })}>Download {s.product} {u.offer}…</Btn>}
        {u.offer && <Btn onClick={() => call("setup_update", { action: "skip" })}>Skip this version</Btn>}
        <Btn disabled={u.checking} onClick={() => call("setup_update", { action: "check" })}>Check now</Btn>
      </div>
      <label className="sx-check">
        <input type="checkbox" checked={u.auto} onChange={(e) => call("setup_update", { action: "auto", on: e.target.checked })} />
        Check for updates once a day
      </label>
      <p className="sx-dim">Last checked: {agoText(u.last_check)}. The check fetches one small file from GitHub and sends nothing about you.</p>
      <p className="sx-foot">
        To update, download the new disk image and drag {s.product} into Applications, replacing this one. Your settings, keys and memory stay; the
        engine updates itself on the next start if the new version needs it.
      </p>
    </section>
  );
}

function Uninstall({ s, call }: { s: Snapshot; call: Call }) {
  const items = s.uninstall.items;
  const [chosen, setChosen] = useState<Set<string> | null>(null);
  const [reset, setReset] = useState(true);
  const [confirm, setConfirm] = useState(false);
  const picked = chosen ?? defaultChosen(items);
  const out = s.uninstall.outcome;
  if (out) {
    return (
      <section className="sx-section">
        <p className="sx-lead">{s.product} is in the Trash. Empty the Trash to free the space.</p>
        {out.failed.length > 0 && <p className="sx-alert">Not moved: {out.failed.join("; ")}</p>}
        {out.permissions_reset === false && <p className="sx-dim">Its permissions couldn't be reset: remove {s.product} under System Settings › Privacy & Security.</p>}
        <div className="sx-btns"><Btn primary onClick={() => call("setup_quit")}>Quit {s.product}</Btn></div>
      </section>
    );
  }
  const total = uninstallTotal(items, picked);
  const toggle = (id: string, on: boolean) => {
    const next = new Set(picked);
    if (on) next.add(id); else next.delete(id);
    setChosen(next);
  };
  return (
    <section className="sx-section">
      <p className="sx-lead">Everything goes to the Trash, so you can still put it back.</p>
      <ul className="sx-rows">
        {items.map((i) => (
          <li key={i.id} className="sx-row">
            <input type="checkbox" checked={i.locked || picked.has(i.id)} disabled={i.locked || s.uninstall.running} onChange={(e) => toggle(i.id, e.target.checked)} aria-label={i.label} />
            <div className="sx-row-t"><strong>{i.label}</strong><span className="sx-dim">{i.note}</span><span className="sx-path">{i.shown.join(", ")}</span></div>
            <span className="sx-size">{s.uninstall.sizing && !i.bytes ? "…" : sizeText(i.bytes)}</span>
          </li>
        ))}
        <li className="sx-row">
          <input type="checkbox" checked={reset} disabled={s.uninstall.running} onChange={(e) => setReset(e.target.checked)} aria-label="Reset permissions" />
          <div className="sx-row-t"><strong>Reset {s.product}'s permissions</strong><span className="sx-dim">Accessibility, Screen Recording, Microphone and the rest — for {s.uninstall.bundle_id} only.</span></div>
        </li>
      </ul>
      <p className="sx-dim">Your notes in Documents › NEO Notes stay where they are.</p>
      {!confirm ? (
        <div className="sx-btns"><Btn danger disabled={s.uninstall.running || items.length === 0} onClick={() => setConfirm(true)}>Move to Trash…</Btn></div>
      ) : (
        <div className="sx-confirm" role="alertdialog" aria-label="Move to Trash?">
          <span>Move {sizeText(total) || "them"} to the Trash? {s.product} stops now and quits afterwards.</span>
          <div className="sx-btns">
            <Btn onClick={() => setConfirm(false)} disabled={s.uninstall.running}>Cancel</Btn>
            <Btn danger disabled={s.uninstall.running} onClick={() => call("setup_uninstall", { action: "run", chosen: [...picked], reset })}>
              {s.uninstall.running ? "Moving…" : "Move to Trash"}
            </Btn>
          </div>
        </div>
      )}
    </section>
  );
}

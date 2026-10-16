// index.html#/keys — Brains & Keys: pick NEO's brains, paste API keys, add an OpenAI-compatible
// endpoint, the voice and daily caps. Only NEO.app's shell opens this window (ui/src-tauri/src/keys.rs);
// it talks to the core as its one `keys` client (./wire.ts). Keys flow one way: a pasted key goes out
// in the one command that carries it and is dropped from the page's state as it goes; what comes
// back is whether a key is set, where from, and its last 4 characters.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { openExternal } from "../shell";
import { KeysWire, tauriDeps, type WireStatus } from "./wire";
import { KEYED, QUICK, TEMPLATE_KEY_URLS, costBadge, keyPage, providerOf, show } from "./catalog";
import { cleanKey, hint } from "./keyshape";
import {
  FIELD, badges, budgetRows, budgetsAfter, cliLines, configured, endpointBody, endpointProblem, errorText,
  formFrom, formFromEndpoint, formFromTemplate, headline, keyMustBeRetyped, modelOptions, oldCopiesText, onThisMac,
  providerChoices, providersInUse,
  quickSetup, quickSetupWith, refFor, rolesChanges, rolesErrorText, shadowNote, sourceText, templates, testText,
  voiceChoice, type EndpointForm, type Role, type RoleForm,
} from "./model";
import type { BrainsDetail, BrainsSummary, CustomEndpoint, ModelEntry, ModelsResult, Result } from "./types";
import "./keys.css";

type Ask = (cmd: string, body?: Record<string, unknown>) => Promise<Result>;
type Msg = { tone: "ok" | "bad" | "info"; text: string } | null;
const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const SECRET_INPUT = { autoComplete: "off", spellCheck: false, autoCapitalize: "off", autoCorrect: "off", "data-1p-ignore": true, "data-lpignore": "true" } as const;

function Line({ msg }: { msg: Msg }) {
  if (!msg) return null;
  return <div className={`kx-msg ${msg.tone}`} role={msg.tone === "bad" ? "alert" : "status"}>{msg.text}</div>;
}

function Badges({ items }: { items: string[] }) {
  return <>{items.map((b) => <span key={b} className={"kx-badge" + (b === "Paid" ? " paid" : b === "Free" ? " free" : "")}>{b}</span>)}</>;
}

function KeyLink({ url, label = "Get a key" }: { url: string | null; label?: string }) {
  const page = keyPage(url);
  if (!page) return null;
  return <button type="button" className="kx-link" onClick={() => void openExternal(page).catch(() => {})}>{label} ↗</button>;
}

export function KeysApp() {
  const wireRef = useRef<KeysWire | null>(null);
  const [status, setStatus] = useState<WireStatus>("connecting");
  const [summary, setSummary] = useState<BrainsSummary | null>(null);
  const [detail, setDetail] = useState<BrainsDetail | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [models, setModels] = useState<Record<string, ModelsResult>>({});
  // The first run's quick path while NEO has no model; once shown it stays until the user leaves it.
  const [mode, setMode] = useState<"auto" | "quick" | "all">("auto");
  const [restarting, setRestarting] = useState<Msg>(null);
  const asked = useRef(new Set<string>());
  const again = useRef<number | undefined>(undefined);

  const ask: Ask = useCallback((cmd, body) => wireRef.current?.request(cmd, body) ?? Promise.resolve({ ok: false, error: "offline" }), []);

  const refresh = useCallback(async () => {
    const r = await ask("brains_get");
    if (Array.isArray((r as unknown as BrainsDetail).catalog)) {
      const d = r as unknown as BrainsDetail;
      setDetail(d);
      if (d.summary) setSummary(d.summary);
    }
  }, [ask]);
  const refreshSoon = useCallback(() => {
    if (again.current) clearTimeout(again.current);
    again.current = window.setTimeout(() => void refresh(), 250);
  }, [refresh]);

  const loadModels = useCallback(async (pid: string, fresh = false): Promise<ModelsResult | null> => {
    if (!fresh && asked.current.has(pid)) return null;
    asked.current.add(pid);
    const r = await ask("models_list", { provider: pid, refresh: fresh });
    if (!Array.isArray((r as unknown as ModelsResult).models)) { asked.current.delete(pid); return null; }
    const m = r as unknown as ModelsResult;
    setModels((prev) => ({ ...prev, [pid]: m }));
    return m;
  }, [ask]);

  useEffect(() => {
    const w = new KeysWire(tauriDeps());
    wireRef.current = w;
    const offs = [
      w.on("status", setStatus),
      w.on("hello", (h) => {
        if (h.brains) setSummary(h.brains);
        setProblems(Array.isArray(h.problems) ? h.problems : []);
        asked.current.clear(); // a core that restarted: its lists afresh
        setRestarting(null);
        void refresh();
      }),
      w.on("brains", (b) => { setSummary(b); refreshSoon(); }),
      w.on("problems", setProblems),
    ];
    w.start();
    return () => { offs.forEach((f) => f()); w.stop(); if (wireRef.current === w) wireRef.current = null; };
  }, [refresh, refreshSoon]);

  // Each provider the roles use, or that has a key: its models, for the menus.
  useEffect(() => { for (const pid of providersInUse(detail)) void loadModels(pid); }, [detail, loadModels]);

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

  const restart = async () => {
    setRestarting({ tone: "info", text: "Restarting NEO…" });
    const r = await ask("brains_restart");
    setRestarting(r.ok ? { tone: "info", text: "NEO is restarting. This window reconnects by itself." } : { tone: "bad", text: errorText(String(r.error ?? "")) });
  };

  useEffect(() => { if (mode === "auto" && detail && summary) setMode(summary.ready ? "all" : "quick"); }, [mode, detail, summary]);
  const ready = summary?.ready;
  const quick = mode === "quick" && !!detail;
  const live = status === "open";

  return (
    <div className="kx">
      <header className="kx-head" data-tauri-drag-region>
        <div className="kx-title" data-tauri-drag-region>Brains &amp; Keys</div>
        <div className={"kx-sub" + (ready ? " ok" : "")} data-tauri-drag-region aria-live="polite">{live ? headline(detail, ready, summary?.brain?.label) : statusLine(status)}</div>
      </header>
      <main className="kx-main">
        <StatusBanner status={status} />
        {live && summary?.restart_needed && (
          <div className="kx-banner warn" role="status">
            <div><b>Restart NEO to switch the voice.</b> The voice reads its key when NEO starts.</div>
            <button className="kx-btn primary" onClick={() => void restart()}>Restart NEO</button>
          </div>
        )}
        <Line msg={restarting} />
        {live && detail && !detail.store.ok && (
          <div className="kx-banner bad" role="alert">
            {detail.store.problem === "unavailable"
              ? "NEO can't keep keys here right now (its key file is out of reach)."
              : `NEO couldn't read ${detail.store.path}: it counts as empty, and is set aside (as .bad-…) the next time you save a key.`}
          </div>
        )}
        {live && detail && (detail.store.old_copies ?? 0) > 0 && (
          <div className="kx-banner warn" role="status">
            {oldCopiesText(detail.store.old_copies ?? 0, detail.store.path)}
          </div>
        )}
        {live && problems.filter((p) => !(quick && /^NEO has no AI model yet\b/.test(p))).slice(0, 4).map((p) => <div key={p} className="kx-problem">{p}</div>)}
        {live && !detail && <div className="kx-loading">Reading NEO's brains…</div>}
        {live && detail && quick && <QuickStart detail={detail} ask={ask} refresh={refresh} loadModels={loadModels} onSkip={() => setMode("all")} />}
        {live && detail && !quick && (
          <>
            <Roles detail={detail} summary={summary} models={models} ask={ask} loadModels={loadModels} refresh={refresh} />
            <Providers detail={detail} ask={ask} refresh={refresh} loadModels={loadModels} />
            <Endpoints detail={detail} ask={ask} refresh={refresh} loadModels={loadModels} />
            {detail.voice.live_possible && <Voice detail={detail} ask={ask} refresh={refresh} />}
            <Spending detail={detail} ask={ask} refresh={refresh} />
          </>
        )}
        <footer className="kx-foot">
          <p>Keys are kept in ~/.neo/api_keys.json, readable only by your Mac account, and sent only to the provider you picked.</p>
          {inTauri && (
            <button className="kx-link" onClick={() => { void import("@tauri-apps/api/core").then(({ invoke }) => invoke("edit_env_file")).catch(() => {}); }}>
              Open ~/.neo/.env (advanced)
            </button>
          )}
        </footer>
      </main>
    </div>
  );
}

function statusLine(s: WireStatus): string {
  switch (s) {
    case "no_link": return "NEO's core runs outside NEO.app";
    case "no_core": return "NEO's core isn't running";
    case "forbidden": return "Open Brains & Keys from NEO's menu";
    default: return "Connecting to NEO…";
  }
}

function StatusBanner({ status }: { status: WireStatus }) {
  if (status === "open" || status === "stopped") return null;
  if (status === "connecting") return <div className="kx-loading" role="status">Waiting for NEO's core. This window connects by itself as soon as it answers.</div>;
  if (status === "no_link") {
    return (
      <div className="kx-banner" role="status">
        <div>NEO's core was started outside NEO.app (from a terminal), so this window can't reach it. Set keys and brains there instead:</div>
        <pre className="kx-cli">{cliLines().join("\n")}</pre>
        <div className="kx-dim">Providers: gemini, anthropic, openai, groq, cerebras, openrouter. Or quit that core and restart NEO from its menu.</div>
      </div>
    );
  }
  if (status === "no_core") return <div className="kx-banner bad" role="alert">NEO's core isn't running. Choose Restart NEO from NEO's menu (in the notch: NotchHub's Settings → NEO).</div>;
  return <div className="kx-banner bad" role="alert">This window can't reach NEO's keys. Open Brains &amp; Keys from NEO's menu.</div>;
}

// ---- first run ---------------------------------------------------------------------------------

function QuickStart({ detail, ask, refresh, loadModels, onSkip }: { detail: BrainsDetail; ask: Ask; refresh: () => Promise<void>; loadModels: (pid: string, fresh?: boolean) => Promise<ModelsResult | null>; onSkip: () => void }) {
  const [pick, setPick] = useState("gemini");
  const [done, setDone] = useState<string | null>(null);
  return (
    <section className="kx-section kx-quick" aria-labelledby="kx-quick-h">
      <h2 id="kx-quick-h">Give NEO a brain</h2>
      <p className="kx-lead">Pick a provider and paste its key. NEO checks it and sets it up for tasks, quick replies and looking at the screen.</p>
      <div className="kx-cards" role="radiogroup" aria-label="Provider">
        {QUICK.map((c) => (
          <button key={c.id} role="radio" aria-checked={pick === c.id} className={"kx-card" + (pick === c.id ? " on" : "")} onClick={() => { setPick(c.id); setDone(null); }}>
            <span className="kx-card-t">{c.title}</span>
            <span className="kx-card-l">{c.line}</span>
          </button>
        ))}
      </div>
      {done ? (
        <div className="kx-done" role="status">
          <div>{done}</div>
          <button className="kx-btn primary" onClick={onSkip}>See all settings</button>
        </div>
      ) : pick === "this_mac" ? (
        <ThisMac detail={detail} ask={ask} refresh={refresh} onDone={setDone} />
      ) : (
        <QuickKey key={pick} pid={pick} ask={ask} refresh={refresh} loadModels={loadModels} onDone={setDone} />
      )}
      <button className="kx-link kx-skip" onClick={onSkip}>Show all settings</button>
    </section>
  );
}

function QuickKey({ pid, ask, refresh, loadModels, onDone }: { pid: string; ask: Ask; refresh: () => Promise<void>; loadModels: (pid: string, fresh?: boolean) => Promise<ModelsResult | null>; onDone: (text: string) => void }) {
  const p = show(pid);
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);
  const [choose, setChoose] = useState<ModelEntry[] | null>(null); // OpenRouter: the user picks a model
  const [chosen, setChosen] = useState("");
  const shape = hint(pid, value);

  const finish = async (plan: Record<string, unknown>) => {
    const r = await ask("brains_set", plan);
    if (!r.ok) { setMsg({ tone: "bad", text: rolesErrorText(String(r.error ?? ""), r.field as string | undefined, r.message as string | undefined) }); return; }
    await refresh();
    const warn = Array.isArray(r.warnings) && r.warnings.length ? ` ${(r.warnings as string[]).join(". ")}.` : "";
    onDone(`${p?.short ?? pid} is NEO's brain now.${warn}`);
  };

  const connect = async () => {
    const c = cleanKey(value);
    if (!c.ok) { setMsg({ tone: "bad", text: c.problem }); return; }
    setValue(""); // dropped from the page as it goes
    setBusy(true);
    setMsg({ tone: "info", text: "Checking the key…" });
    try {
      const r = await ask("key_save", { provider: pid, key: c.key, test: true });
      if (!r.ok) { setMsg({ tone: "bad", text: errorText(String(r.error ?? ""), r.message as string | undefined) }); return; }
      setMsg({ tone: "info", text: "Key saved. Setting up…" });
      const m = await loadModels(pid, true);
      const plan = quickSetup(pid, m?.defaults);
      if (plan) { await finish(plan); return; }
      const fits = (m?.models ?? []).filter((x) => x.tools);
      setChoose(fits.length ? fits : m?.models ?? []);
      setChosen((fits[0] ?? m?.models?.[0])?.id ?? "");
      setMsg({ tone: "info", text: "Key saved. Pick the model NEO should use." });
    } finally {
      setBusy(false);
    }
  };

  if (choose) {
    const model = choose.find((x) => x.id === chosen) ?? null;
    return (
      <div className="kx-panel">
        <label className="kx-field">
          <span>Model</span>
          <select value={chosen} onChange={(e) => setChosen(e.target.value)}>
            {choose.map((x) => <option key={x.id} value={x.id}>{x.label}{x.sees ? "" : " · text only"}{x.tools ? "" : " · no tools"}</option>)}
          </select>
        </label>
        {!choose.length && <div className="kx-dim">The list didn't come. Type a model id under all settings.</div>}
        <div className="kx-row-btns">
          <button className="kx-btn primary" disabled={!model || busy} onClick={() => { if (model) { setBusy(true); void finish(quickSetupWith(pid, model)).finally(() => setBusy(false)); } }}>Use this model</button>
        </div>
        <Line msg={msg} />
      </div>
    );
  }
  return (
    <form className="kx-panel" onSubmit={(e) => { e.preventDefault(); void connect(); }}>
      <label className="kx-field">
        <span>{p?.label ?? pid} API key</span>
        <input type="password" value={value} onChange={(e) => setValue(e.target.value)} placeholder={p?.keyHint ? `${p.keyHint}…` : "Paste the key"} aria-describedby="kx-quick-hint" {...SECRET_INPUT} />
      </label>
      <div className="kx-hintrow" id="kx-quick-hint">
        <KeyLink url={p?.keyUrl ?? null} label={pid === "gemini" ? "Get a free key" : "Get a key"} />
        {shape && <span className="kx-shape">{shape}</span>}
      </div>
      <div className="kx-row-btns">
        <button type="submit" className="kx-btn primary" disabled={busy || !value.trim()}>{busy ? "Checking…" : "Connect"}</button>
      </div>
      <Line msg={msg} />
    </form>
  );
}

const MAC_SPOTS = [
  { template: "ollama", label: "Ollama", base: "http://localhost:11434/v1" },
  { template: "lmstudio", label: "LM Studio", base: "http://localhost:1234/v1" },
] as const;

function ThisMac({ detail, ask, refresh, onDone }: { detail: BrainsDetail; ask: Ask; refresh: () => Promise<void>; onDone: (text: string) => void }) {
  const [found, setFound] = useState<Record<string, ModelEntry[] | false | null>>({});
  const [neo, setNeo] = useState<boolean | null>(null);
  const [chosen, setChosen] = useState<Record<string, string>>({});
  const [msg, setMsg] = useState<Msg>(null);
  const [busy, setBusy] = useState(false);
  const known = useRef(detail);
  known.current = detail;
  const base = (t: string, fallback: string) => templates(known.current).find((x) => x.id === t)?.base_url || fallback;

  // What answers here: NEO's own model, Ollama, LM Studio (each asked on this Mac only).
  const look = useCallback(async () => {
    setFound({}); setNeo(null);
    const local = await ask("key_test", { provider: "local" });
    setNeo(!!local.ok);
    for (const s of MAC_SPOTS) {
      const t = templates(known.current).find((x) => x.id === s.template);
      const form = { ...formFromTemplate(t), base_url: t?.base_url || s.base, label: s.label };
      const r = await ask("models_list", { provider: "custom", endpoint: endpointBody(form, { forProbe: true }), refresh: true });
      const list = r.ok && Array.isArray(r.models) ? (r.models as ModelEntry[]) : false;
      setFound((f) => ({ ...f, [s.template]: list }));
      if (list && list[0]) setChosen((c) => ({ ...c, [s.template]: c[s.template] ?? list[0].id }));
    }
  }, [ask]);
  useEffect(() => { void look(); }, [look]);

  const use = async (template: string, label: string, url: string) => {
    const model = chosen[template];
    if (!model) return;
    setBusy(true);
    try {
      const form: EndpointForm = { ...formFromTemplate(templates(detail).find((x) => x.id === template)), label, base_url: url, model };
      const r = await ask("custom_save", { endpoint: endpointBody(form) });
      if (!r.ok || typeof r.provider !== "string") { setMsg({ tone: "bad", text: errorText(String(r.error ?? "")) }); return; }
      const s = await ask("brains_set", quickSetupWith(r.provider, { id: model, sees: form.vision, tools: form.tools }));
      if (!s.ok) { setMsg({ tone: "bad", text: rolesErrorText(String(s.error ?? ""), s.field as string | undefined, s.message as string | undefined) }); return; }
      await refresh();
      onDone(`${label} on this Mac is NEO's brain now (${model}).`);
    } finally { setBusy(false); }
  };
  const useNeo = async () => {
    setBusy(true);
    try {
      const s = await ask("brains_set", { brain: "local", fast_brain: "local", light_brain: "" });
      if (!s.ok) { setMsg({ tone: "bad", text: rolesErrorText(String(s.error ?? ""), s.field as string | undefined, s.message as string | undefined) }); return; }
      await refresh();
      onDone("NEO's own model on this Mac is its brain now.");
    } finally { setBusy(false); }
  };

  return (
    <div className="kx-panel">
      <p className="kx-dim">A model on this Mac is free and works offline; it can't see the screen, so NEO works through what apps say about themselves.</p>
      {MAC_SPOTS.map((s) => {
        const f = found[s.template];
        return (
          <div key={s.template} className="kx-mac">
            <div className="kx-mac-h"><b>{s.label}</b> <span className="kx-dim">{base(s.template, s.base).replace(/^http:\/\//, "")}</span></div>
            {f === undefined || f === null ? <span className="kx-dim">Looking…</span> : f === false ? <span className="kx-dim">Not running.</span> : f.length === 0 ? <span className="kx-dim">Running, with no model yet.</span> : (
              <div className="kx-mac-use">
                <select aria-label={`${s.label} model`} value={chosen[s.template] ?? ""} onChange={(e) => setChosen((c) => ({ ...c, [s.template]: e.target.value }))}>
                  {f.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
                </select>
                <button className="kx-btn primary" disabled={busy} onClick={() => void use(s.template, s.label, base(s.template, s.base))}>Use it</button>
              </div>
            )}
          </div>
        );
      })}
      <div className="kx-mac">
        <div className="kx-mac-h"><b>NEO's own model (Gemma)</b></div>
        {neo === null ? <span className="kx-dim">Looking…</span> : neo ? (
          <button className="kx-btn primary" disabled={busy} onClick={() => void useNeo()}>Use it</button>
        ) : (
          <span className="kx-dim">In a terminal, run <code>python -m neo2 --local</code> and keep it running.</span>
        )}
      </div>
      <div className="kx-row-btns"><button className="kx-btn" onClick={() => void look()} disabled={busy}>Look again</button></div>
      <Line msg={msg} />
    </div>
  );
}

// ---- Your brain -----------------------------------------------------------------------------------

function Roles({ detail, summary, models, ask, loadModels, refresh }: {
  detail: BrainsDetail; summary: BrainsSummary | null; models: Record<string, ModelsResult>; ask: Ask;
  loadModels: (pid: string, fresh?: boolean) => Promise<ModelsResult | null>; refresh: () => Promise<void>;
}) {
  const [edits, setEdits] = useState<RoleForm | null>(null);
  const [msg, setMsg] = useState<Msg>(null);
  const [busy, setBusy] = useState(false);
  const form = edits ?? formFrom(detail);
  const changes = useMemo(() => rolesChanges(form, detail), [form, detail]);
  const dirty = Object.keys(changes).length > 0;
  // (functional: a provider pick waits for its model list, and must not undo a change made meanwhile)
  const set = (patch: Partial<RoleForm>) => { setEdits((prev) => ({ ...(prev ?? formFrom(detail)), ...patch })); setMsg(null); };

  const save = async () => {
    setBusy(true);
    try {
      const r = await ask("brains_set", changes);
      if (!r.ok) { setMsg({ tone: "bad", text: rolesErrorText(String(r.error ?? ""), r.field as string | undefined, r.message as string | undefined) }); return; }
      setEdits(null);
      const notes: string[] = [];
      if (Array.isArray(r.warnings)) notes.push(...(r.warnings as string[]));
      if (Array.isArray(r.shadowed) && r.shadowed.length) notes.push(`${(r.shadowed as string[]).map((f) => `NEO_${f.toUpperCase()}`).join(", ")} in the environment still wins`);
      setMsg({ tone: notes.length ? "info" : "ok", text: notes.length ? `Saved. ${notes.join(". ")}.` : "Saved." });
      await refresh();
    } finally { setBusy(false); }
  };

  const brainChoices = providerChoices("brain", detail);
  const backups = form.backup_brains;
  const backupOptions = brainChoices.map((c) => ({ id: c.id, ref: models[c.id]?.defaults?.brain ?? (c.id === "local" ? "local" : refFor(c.id, null)), label: c.label }))
    .filter((o) => o.ref !== form.brain && (o.id !== "openrouter" || !!models.openrouter?.defaults?.brain));

  return (
    <section className="kx-section" aria-labelledby="kx-roles-h">
      <h2 id="kx-roles-h">Your brain</h2>
      {(["brain", "quick", "eyes"] as Role[]).map((role) => (
        <RoleRow key={role} role={role} form={form} detail={detail} summary={summary} models={models} loadModels={loadModels} onPick={(ref) => set({ [FIELD[role]]: ref } as Partial<RoleForm>)} />
      ))}
      <div className="kx-subrow">
        <div className="kx-subrow-h">Backups</div>
        <div className="kx-chips" role="group" aria-label="Backups">
          <button className={"kx-chip" + (backups === null ? " on" : "")} aria-pressed={backups === null} onClick={() => set({ backup_brains: null })}>Automatic</button>
          {backupOptions.map((o) => {
            const on = !!backups?.includes(o.ref);
            return (
              <button key={o.ref} className={"kx-chip" + (on ? " on" : "")} aria-pressed={on}
                disabled={!on && (backups?.length ?? 0) >= 4}
                onClick={() => set({ backup_brains: on ? (backups ?? []).filter((x) => x !== o.ref) : [...(backups ?? []), o.ref] })}>
                {o.label}
              </button>
            );
          })}
        </div>
        <div className="kx-dim">{backups === null ? "When the brain can't answer, NEO tries Quick replies' brain, then the model on this Mac." : backups.length ? `Tried in this order: ${backups.join(", ")}.` : "No backups: when the brain can't answer, the task stops."}</div>
      </div>
      <div className="kx-subrow">
        <div className="kx-subrow-h">Screenshots</div>
        <div className="kx-radios" role="radiogroup" aria-label="Screenshots">
          <label><input type="radio" name="kx-eyes" checked={form.cloud_eyes === "off"} onChange={() => set({ cloud_eyes: "off" })} /> Only the brains I picked</label>
          <label><input type="radio" name="kx-eyes" checked={form.cloud_eyes !== "off"} onChange={() => set({ cloud_eyes: detail.cloud_eyes.value === "on" ? "on" : "auto" })} /> Other brains with a key may help</label>
        </div>
        {detail.cloud_eyes.source === "env" && <div className="kx-dim">NEO_CLOUD_EYES in the environment wins over this.</div>}
      </div>
      <div className="kx-row-btns">
        <button className="kx-btn primary" disabled={!dirty || busy} onClick={() => void save()}>{busy ? "Saving…" : "Save"}</button>
        {dirty && <button className="kx-btn" onClick={() => { setEdits(null); setMsg(null); }} disabled={busy}>Undo changes</button>}
      </div>
      <Line msg={msg} />
    </section>
  );
}

function RoleRow({ role, form, detail, summary, models, loadModels, onPick }: {
  role: Role; form: RoleForm; detail: BrainsDetail; summary: BrainsSummary | null; models: Record<string, ModelsResult>;
  loadModels: (pid: string, fresh?: boolean) => Promise<ModelsResult | null>; onPick: (ref: string) => void;
}) {
  const field = FIELD[role];
  const value = (form as unknown as Record<string, string>)[field] ?? "";
  const provider = providerOf(value);
  const choices = providerChoices(role, detail);
  const known = choices.some((c) => c.id === provider);
  const options = provider ? modelOptions(provider, role, models[provider], value, detail) : [];
  const picked = options.find((o) => o.ref === value);
  const rd = detail.roles[field];
  const live = role === "brain" ? summary?.brain : role === "quick" ? summary?.quick : summary?.eyes;
  const [loading, setLoading] = useState(false);

  const pickProvider = async (pid: string) => {
    if (!pid) { onPick(""); return; }
    setLoading(true);
    try {
      const list = models[pid] ?? (await loadModels(pid, true)) ?? undefined;
      const want = list?.defaults?.[role] ?? (role === "eyes" ? list?.defaults?.brain : undefined);
      const opts = modelOptions(pid, role, list, null, detail);
      // No list at all (the core away): the provider's own default model, where it has one.
      const ref = (want && opts.some((o) => o.ref === want) ? want : opts[0]?.ref) ?? (pid === "openrouter" ? "" : refFor(pid, null));
      if (ref) onPick(ref);
    } finally { setLoading(false); }
  };

  const title = role === "brain" ? "Brain" : role === "quick" ? "Quick replies" : "Eyes";
  const what = role === "brain" ? "Tasks on your Mac, with tools" : role === "quick" ? "Chat, summaries, memory, dictation" : "Steps that look at the screen or a photo";
  return (
    <div className="kx-role">
      <div className="kx-role-h">
        <div className="kx-role-t">{title}</div>
        <div className="kx-dim">{what}</div>
      </div>
      <div className="kx-role-pick">
        <select aria-label={`${title}: provider`} value={provider} onChange={(e) => void pickProvider(e.target.value)} disabled={loading}>
          {role === "eyes" && <option value="">Same as Brain</option>}
          {!known && provider && <option value={provider}>{show(provider)?.short ?? provider} (not set up)</option>}
          {!provider && role !== "eyes" && <option value="">Pick a provider</option>}
          {choices.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
        </select>
        {provider && (
          <select aria-label={`${title}: model`} value={value} onChange={(e) => onPick(e.target.value)} disabled={loading}>
            {options.map((o) => <option key={o.ref} value={o.ref}>{o.label}</option>)}
          </select>
        )}
      </div>
      <div className="kx-role-meta">
        {provider && <Badges items={badges(provider, picked, detail)} />}
        {live && value === (detail.roles[field]?.ref ?? "") && live.ref && !live.ok && <span className="kx-badge bad">Can't answer now</span>}
        {rd && !rd.valid && <span className="kx-warn">{`NEO_${field.toUpperCase()}`} isn't a brain NEO knows{rd.typed ? ` (${rd.typed})` : ""}.</span>}
        {rd?.source === "env" && <span className="kx-dim">NEO_{field.toUpperCase()} in the environment wins over this.</span>}
        {role === "quick" && picked && badges(provider, picked, detail).includes("Paid") && <span className="kx-dim">Paid: every summary and memory note costs a little. A cheap model suits this.</span>}
      </div>
    </div>
  );
}

// ---- Providers ------------------------------------------------------------------------------------

function Providers({ detail, ask, refresh, loadModels }: { detail: BrainsDetail; ask: Ask; refresh: () => Promise<void>; loadModels: (pid: string, fresh?: boolean) => Promise<ModelsResult | null> }) {
  return (
    <section className="kx-section" aria-labelledby="kx-prov-h">
      <h2 id="kx-prov-h">Providers</h2>
      {KEYED.map((pid) => <ProviderRow key={pid} pid={pid} detail={detail} ask={ask} refresh={refresh} loadModels={loadModels} />)}
    </section>
  );
}

function ProviderRow({ pid, detail, ask, refresh, loadModels }: { pid: string; detail: BrainsDetail; ask: Ask; refresh: () => Promise<void>; loadModels: (pid: string, fresh?: boolean) => Promise<ModelsResult | null> }) {
  const p = show(pid)!;
  const entry = detail.keys[pid];
  const env = detail.catalog.find((c) => c.id === pid)?.env ?? null;
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState("");
  const [extra, setExtra] = useState("");
  const [busy, setBusy] = useState<null | "test" | "save" | "remove">(null);
  const [sure, setSure] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);
  const shape = hint(pid, value);
  const shadow = shadowNote(entry, env);
  const badge = costBadge(p.cost);
  const close = () => { setOpen(false); setValue(""); setExtra(""); };

  const test = async () => {
    setBusy("test"); setMsg({ tone: "info", text: "Testing…" });
    try {
      let body: Record<string, unknown> = { provider: pid };
      if (value.trim()) {
        const c = cleanKey(value);
        if (!c.ok) { setMsg({ tone: "bad", text: c.problem }); return; }
        body = { provider: pid, key: c.key };
      }
      const r = await ask("key_test", body);
      setMsg({ tone: r.ok ? "ok" : "bad", text: testText(r) });
    } finally { setBusy(null); }
  };

  const save = async () => {
    const c = cleanKey(value);
    if (!c.ok) { setMsg({ tone: "bad", text: c.problem }); return; }
    const more = pid === "gemini" ? extra.split(",").map((x) => x.trim()).filter(Boolean) : [];
    for (const m of more) { const e = cleanKey(m); if (!e.ok) { setMsg({ tone: "bad", text: `Extra key: ${e.problem}` }); return; } }
    setValue(""); setExtra(""); // dropped from the page as they go
    setBusy("save"); setMsg({ tone: "info", text: "Checking and saving…" });
    try {
      const r = await ask("key_save", { provider: pid, key: c.key, ...(more.length ? { extra: more } : {}), test: true });
      if (!r.ok) { setMsg({ tone: "bad", text: errorText(String(r.error ?? ""), r.message as string | undefined) }); return; }
      const k = r.key as { last4?: string | null } | undefined;
      const parts = [`Saved ••••${k?.last4 ?? ""}.`];
      if (r.shadowed_by === "env") parts.push(`The ${env ?? "environment"} variable still wins over it.`);
      if (typeof r.daily_budget === "number") parts.push(`NEO spends at most $${r.daily_budget.toFixed(2)} a day on it (Spending, below).`);
      setMsg({ tone: "ok", text: parts.join(" ") });
      setOpen(false);
      await refresh();
      void loadModels(pid, true);
    } finally { setBusy(null); }
  };

  const remove = async () => {
    setBusy("remove"); setSure(false);
    try {
      const r = await ask("key_remove", { provider: pid });
      if (!r.ok) { setMsg({ tone: "bad", text: errorText(String(r.error ?? "")) }); return; }
      const still = r.still_set === "env" ? ` A key is still set by ${env ?? "the environment"}.` : r.still_set ? " A key is still set in a .env file." : "";
      setMsg({ tone: "ok", text: (r.removed === false ? "Nothing was saved in NEO." : "Removed from NEO.") + still });
      await refresh();
    } finally { setBusy(null); }
  };

  return (
    <div className="kx-prov">
      <div className="kx-prov-h">
        <div className="kx-prov-name">
          <span className="kx-prov-t">{p.label}</span>
          <span className={`kx-badge ${badge.tone}`}>{badge.text}</span>
        </div>
        <div className="kx-prov-btns">
          {entry?.set && <button className="kx-btn sm" onClick={() => void test()} disabled={!!busy}>{busy === "test" ? "Testing…" : "Test"}</button>}
          <button className="kx-btn sm" onClick={() => (open ? close() : setOpen(true))} disabled={!!busy}>{open ? "Cancel" : entry?.set ? "Replace" : "Add key"}</button>
          {entry?.source === "saved" && !sure && <button className="kx-btn sm danger" onClick={() => setSure(true)} disabled={!!busy}>Remove</button>}
          {sure && <><button className="kx-btn sm danger-fill" onClick={() => void remove()}>Remove key</button><button className="kx-btn sm" onClick={() => setSure(false)}>Keep</button></>}
        </div>
      </div>
      <div className="kx-prov-status">
        <span className={entry?.set ? "kx-set" : "kx-dim"}>{sourceText(entry)}</span>
        {entry?.set && (entry.also?.length ?? 0) > 0 && <span className="kx-dim"> · also in {entry.also!.map((a) => (a === "saved" ? "NEO" : a === "env" ? "the environment" : a === "dotenv" ? "~/.neo/.env" : "the checkout's .env")).join(", ")}</span>}
      </div>
      <div className="kx-dim kx-blurb">{p.blurb}</div>
      {shadow && <div className="kx-dim">{shadow}</div>}
      {open && (
        <form className="kx-panel" onSubmit={(e) => { e.preventDefault(); void save(); }}>
          <label className="kx-field">
            <span>{entry?.set ? "New key" : "Key"}</span>
            <input type="password" autoFocus value={value} onChange={(e) => setValue(e.target.value)} placeholder={`${p.keyHint}…`} {...SECRET_INPUT} />
          </label>
          <div className="kx-hintrow">
            <KeyLink url={p.keyUrl} />
            {shape && <span className="kx-shape">{shape}</span>}
          </div>
          {pid === "gemini" && (
            <details className="kx-adv">
              <summary>Extra keys (advanced)</summary>
              <label className="kx-field">
                <span>More Gemini keys, separated by commas: NEO moves to the next when one runs out of free quota</span>
                <input type="password" value={extra} onChange={(e) => setExtra(e.target.value)} placeholder="AIza…, AIza…" {...SECRET_INPUT} />
              </label>
            </details>
          )}
          <div className="kx-row-btns">
            <button type="button" className="kx-btn" onClick={() => void test()} disabled={!!busy || !value.trim()}>Test</button>
            <button type="submit" className="kx-btn primary" disabled={!!busy || !value.trim()}>{busy === "save" ? "Saving…" : "Save"}</button>
          </div>
        </form>
      )}
      <Line msg={msg} />
    </div>
  );
}

// ---- Other endpoint (OpenAI-compatible) ---------------------------------------------------------

function Endpoints({ detail, ask, refresh, loadModels }: { detail: BrainsDetail; ask: Ask; refresh: () => Promise<void>; loadModels: (pid: string, fresh?: boolean) => Promise<ModelsResult | null> }) {
  const [editing, setEditing] = useState<EndpointForm | null>(null);
  return (
    <section className="kx-section" aria-labelledby="kx-ep-h">
      <h2 id="kx-ep-h">Other endpoint (OpenAI-compatible)</h2>
      <p className="kx-lead">xAI, DeepSeek, Mistral, Together, Ollama, LM Studio, or any server that speaks OpenAI's API.</p>
      {detail.custom.map((cp) => <EndpointRow key={cp.id} cp={cp} detail={detail} ask={ask} refresh={refresh} onEdit={() => setEditing(formFromEndpoint(cp))} />)}
      {editing ? (
        <EndpointEditor key={editing.id ?? "new"} start={editing} detail={detail} ask={ask} refresh={refresh} loadModels={loadModels} onClose={() => setEditing(null)} />
      ) : (
        <div className="kx-row-btns"><button className="kx-btn" onClick={() => setEditing(formFromTemplate(templates(detail)[0]))} disabled={detail.custom.length >= 10}>Add an endpoint</button></div>
      )}
    </section>
  );
}

function EndpointRow({ cp, detail, ask, refresh, onEdit }: { cp: CustomEndpoint; detail: BrainsDetail; ask: Ask; refresh: () => Promise<void>; onEdit: () => void }) {
  const [msg, setMsg] = useState<Msg>(null);
  const [busy, setBusy] = useState(false);
  const [sure, setSure] = useState(false);
  const entry = detail.keys[cp.id];
  let host = cp.base_url;
  try { host = new URL(cp.base_url).host; } catch { /* as typed */ }
  const state = cp.on_device ? (cp.reachable ? "Answering on this Mac" : "On this Mac, not answering now") : sourceText(entry);
  return (
    <div className="kx-prov">
      <div className="kx-prov-h">
        <div className="kx-prov-name"><span className="kx-prov-t">{cp.label}</span><span className={"kx-badge " + (cp.on_device ? "mac" : "paid")}>{cp.on_device ? "On this Mac" : "Paid"}</span></div>
        <div className="kx-prov-btns">
          <button className="kx-btn sm" disabled={busy} onClick={async () => { setBusy(true); setMsg({ tone: "info", text: "Testing…" }); const r = await ask("key_test", { provider: cp.id }); setMsg({ tone: r.ok ? "ok" : "bad", text: testText(r) }); setBusy(false); }}>Test</button>
          {cp.saved !== false && <button className="kx-btn sm" onClick={onEdit} disabled={busy}>Edit</button>}
          {cp.saved !== false && !sure && <button className="kx-btn sm danger" onClick={() => setSure(true)} disabled={busy}>Remove</button>}
          {sure && <>
            <button className="kx-btn sm danger-fill" onClick={async () => { setSure(false); setBusy(true); const r = await ask("custom_remove", { provider: cp.id }); setBusy(false); if (!r.ok) setMsg({ tone: "bad", text: errorText(String(r.error ?? "")) }); else await refresh(); }}>Remove endpoint</button>
            <button className="kx-btn sm" onClick={() => setSure(false)}>Keep</button>
          </>}
        </div>
      </div>
      <div className="kx-prov-status"><span className="kx-set">{host}</span><span className="kx-dim"> · {cp.model} · {cp.tools ? "tools" : "no tools"} · {cp.vision ? "sees" : "text only"} · {state}</span></div>
      {cp.saved === false && <div className="kx-dim">From NEO_CUSTOM_PROVIDERS: change it there.</div>}
      <Line msg={msg} />
    </div>
  );
}

function EndpointEditor({ start, detail, ask, refresh, loadModels, onClose }: { start: EndpointForm; detail: BrainsDetail; ask: Ask; refresh: () => Promise<void>; loadModels: (pid: string, fresh?: boolean) => Promise<ModelsResult | null>; onClose: () => void }) {
  const [f, setF] = useState<EndpointForm>(start);
  const [key, setKey] = useState("");
  const [list, setList] = useState<ModelEntry[]>([]);
  const [busy, setBusy] = useState<null | "test" | "tools" | "save">(null);
  const [msg, setMsg] = useState<Msg>(null);
  const t = templates(detail).find((x) => x.id === f.template);
  const local = onThisMac(f.base_url);
  const problem = endpointProblem(f);
  const retype = keyMustBeRetyped(f, detail) && !key.trim(); // (its saved key never follows a new host)
  const keyHintText = t?.key_hint ? hint(`custom:${t.id}`, key) : null;
  const edit = (patch: Partial<EndpointForm>) => { setF((x) => ({ ...x, ...patch })); setMsg(null); };
  const keyBody = (): { key?: string } | null => {
    if (!key.trim()) return {};
    const c = cleanKey(key);
    if (!c.ok) { setMsg({ tone: "bad", text: c.problem }); return null; }
    return { key: c.key };
  };

  const test = async () => {
    const k = keyBody();
    if (!k) return;
    setBusy("test"); setMsg({ tone: "info", text: "Testing…" });
    try {
      // An endpoint being edited is tested as the form has it, with its saved key unless one is typed.
      const endpoint = endpointBody(f, { forProbe: true });
      const provider = f.id ?? "custom";
      const r = await ask("key_test", { provider, endpoint, ...k });
      setMsg({ tone: r.ok ? "ok" : "bad", text: testText(r) });
      if (r.ok) {
        const m = await ask("models_list", { provider, endpoint, refresh: true, ...k });
        if (Array.isArray(m.models)) setList(m.models as ModelEntry[]);
      }
    } finally { setBusy(null); }
  };
  const checkTools = async () => {
    const k = keyBody();
    if (!k || problem) { if (problem) setMsg({ tone: "bad", text: problem }); return; }
    setBusy("tools"); setMsg({ tone: "info", text: "Asking the model to call a tool…" });
    try {
      const r = await ask("key_test", { provider: f.id ?? "custom", check: "tools", endpoint: endpointBody(f), ...k });
      setMsg({ tone: r.ok ? "ok" : "bad", text: testText(r) });
    } finally { setBusy(null); }
  };
  const save = async () => {
    if (problem) { setMsg({ tone: "bad", text: problem }); return; }
    const k = keyBody();
    if (!k) return;
    setKey(""); // dropped from the page as it goes
    setBusy("save"); setMsg({ tone: "info", text: "Saving…" });
    try {
      const r = await ask("custom_save", { endpoint: endpointBody(f), ...k });
      if (!r.ok) { setMsg({ tone: "bad", text: errorText(String(r.error ?? ""), r.message as string | undefined) }); return; }
      await refresh();
      if (typeof r.provider === "string") void loadModels(r.provider, true);
      onClose();
    } finally { setBusy(null); }
  };

  return (
    <form className="kx-panel kx-ep" onSubmit={(e) => { e.preventDefault(); void save(); }}>
      {!f.id && (
        <label className="kx-field">
          <span>Start from</span>
          <select value={f.template} onChange={(e) => { setF(formFromTemplate(templates(detail).find((x) => x.id === e.target.value))); setList([]); setMsg(null); }}>
            {templates(detail).map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
      )}
      <div className="kx-grid2">
        <label className="kx-field"><span>Name</span><input value={f.label} maxLength={40} onChange={(e) => edit({ label: e.target.value })} spellCheck={false} /></label>
        <label className="kx-field"><span>Base URL</span><input value={f.base_url} onChange={(e) => edit({ base_url: e.target.value })} placeholder="https://…/v1" spellCheck={false} autoCapitalize="off" autoCorrect="off" /></label>
      </div>
      <label className="kx-field">
        <span>{retype ? "Key — the base URL moved to another host: paste the key again" : local ? "Key (usually none on this Mac)" : f.id && detail.keys[f.id]?.set ? `Key — ${sourceText(detail.keys[f.id])}; paste to replace` : "Key"}</span>
        <input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder={t?.key_hint ? `${t.key_hint}…` : local ? "" : "Paste the key"} {...SECRET_INPUT} />
      </label>
      <div className="kx-hintrow">
        <KeyLink url={TEMPLATE_KEY_URLS[f.template] ?? null} />
        {keyHintText && <span className="kx-shape">{keyHintText}</span>}
        {t?.env && <span className="kx-dim">{t.env} in the environment wins over a key saved here (only at {t.base_url.replace(/^https:\/\//, "").split("/")[0]}).</span>}
      </div>
      <div className="kx-grid2">
        <label className="kx-field">
          <span>Model</span>
          <input value={f.model} onChange={(e) => edit({ model: e.target.value })} list="kx-ep-models" placeholder="Test to list the models, or type an id" spellCheck={false} autoCapitalize="off" autoCorrect="off" />
          <datalist id="kx-ep-models">{list.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}</datalist>
        </label>
        <label className="kx-field"><span>Context (tokens)</span><input type="number" min={4096} max={2000000} step={1024} value={f.context} onChange={(e) => edit({ context: Math.round(Number(e.target.value) || 0) })} /></label>
      </div>
      <div className="kx-toggles">
        <label><input type="checkbox" checked={f.tools} onChange={(e) => edit({ tools: e.target.checked })} /> Calls tools (needed for the Brain)</label>
        <label><input type="checkbox" checked={f.vision} onChange={(e) => edit({ vision: e.target.checked })} /> Sees pictures</label>
      </div>
      <div className="kx-row-btns">
        <button type="button" className="kx-btn" onClick={() => void test()} disabled={!!busy}>{busy === "test" ? "Testing…" : "Test"}</button>
        <button type="button" className="kx-btn" onClick={() => void checkTools()} disabled={!!busy} title="Sends one tiny request with a dummy tool">{busy === "tools" ? "Checking…" : "Check tools · uses a few tokens"}</button>
        <span className="kx-spacer" />
        <button type="button" className="kx-btn" onClick={onClose} disabled={busy === "save"}>Cancel</button>
        <button type="submit" className="kx-btn primary" disabled={!!busy || !!problem || retype}>{busy === "save" ? "Saving…" : "Save"}</button>
      </div>
      {problem && !msg && <div className="kx-dim">{problem}</div>}
      <Line msg={msg} />
    </form>
  );
}

// ---- Voice -------------------------------------------------------------------------------------

function Voice({ detail, ask, refresh }: { detail: BrainsDetail; ask: Ask; refresh: () => Promise<void> }) {
  const [msg, setMsg] = useState<Msg>(null);
  const [busy, setBusy] = useState(false);
  const now = voiceChoice(detail.voice.value);
  const pick = async (v: "auto" | "local") => {
    if ((v === "auto" && now === "live") || (v === "local" && now === "local")) return;
    setBusy(true);
    try {
      const r = await ask("brains_set", { voice: v });
      if (!r.ok) { setMsg({ tone: "bad", text: rolesErrorText(String(r.error ?? ""), "voice") }); return; }
      setMsg({ tone: "ok", text: r.restart_needed ? "Saved. Restart NEO to switch the voice." : "Saved." });
      await refresh();
    } finally { setBusy(false); }
  };
  return (
    <section className="kx-section" aria-labelledby="kx-voice-h">
      <h2 id="kx-voice-h">Voice</h2>
      <div className="kx-radios col" role="radiogroup" aria-label="Voice">
        <label><input type="radio" name="kx-voice" checked={now === "live"} disabled={busy} onChange={() => void pick("auto")} /> <b>Gemini Live</b> — natural voice; answers quick questions itself</label>
        <label><input type="radio" name="kx-voice" checked={now === "local"} disabled={busy} onChange={() => void pick("local")} /> <b>On this Mac</b> — your brain answers everything; works offline</label>
      </div>
      {now === "off" && <div className="kx-dim">The voice is off (NEO_VOICE=off). Picking one turns it on.</div>}
      {detail.voice.source === "env" && <div className="kx-dim">NEO_VOICE in the environment wins over this.</div>}
      <Line msg={msg} />
    </section>
  );
}

// ---- Spending ----------------------------------------------------------------------------------

function Spending({ detail, ask, refresh }: { detail: BrainsDetail; ask: Ask; refresh: () => Promise<void> }) {
  const rows = budgetRows(detail);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [msg, setMsg] = useState<Msg>(null);
  const [busy, setBusy] = useState(false);
  const shown = (pid: string, value: number | null, suggested: number | null) =>
    edits[pid] ?? (value !== null ? value.toFixed(2) : suggested !== null ? suggested.toFixed(2) : "");
  const pending: Record<string, string> = { ...edits };
  for (const r of rows) if (r.value === null && r.suggested !== null && pending[r.pid] === undefined) pending[r.pid] = r.suggested.toFixed(2);
  const dirty = Object.entries(pending).some(([pid, v]) => {
    const was = detail.daily_budgets.value[pid];
    const x = v.trim() === "" ? null : Number(v.trim().replace(/^\$/, ""));
    return (x ?? null) !== (was ?? null);
  });
  const save = async () => {
    const after = budgetsAfter(detail, pending);
    if ("bad" in after) { setMsg({ tone: "bad", text: `${show(after.bad)?.short ?? after.bad}: a cap is dollars a day, 0 to 1000 (empty: no cap).` }); return; }
    setBusy(true);
    try {
      const r = await ask("brains_set", { daily_budgets: after.budgets });
      if (!r.ok) { setMsg({ tone: "bad", text: rolesErrorText(String(r.error ?? ""), "daily_budgets") }); return; }
      setEdits({});
      setMsg({ tone: "ok", text: "Saved." });
      await refresh();
    } finally { setBusy(false); }
  };
  const cerebras = configured("cerebras", detail);
  if (!rows.length && !cerebras) return null;
  return (
    <section className="kx-section" aria-labelledby="kx-spend-h">
      <h2 id="kx-spend-h">Spending</h2>
      <p className="kx-lead">The most NEO spends a day on each paid provider. At 80% it says so; then it skips that provider until midnight.</p>
      {rows.map((r) => (
        <label key={r.pid} className="kx-spend">
          <span className="kx-spend-t">{r.label}</span>
          <span className="kx-money">$<input inputMode="decimal" value={shown(r.pid, r.value, r.suggested)} placeholder="no cap" onChange={(e) => { setEdits((x) => ({ ...x, [r.pid]: e.target.value })); setMsg(null); }} aria-label={`${r.label}: dollars a day`} /> a day</span>
          {r.value === null && r.suggested !== null && edits[r.pid] === undefined && <span className="kx-dim">suggested</span>}
        </label>
      ))}
      {rows.some((r) => r.pid === "openrouter") && (
        <div className="kx-dim">OpenRouter: each model is counted at the price OpenRouter lists for it; one NEO hasn't seen listed, at the highest price NEO knows.</div>
      )}
      {rows.some((r) => r.pid.startsWith("custom:")) && (
        <div className="kx-dim">NEO has no price list for other endpoints: it counts their use at the highest price it knows ($10 in, $50 out per million tokens), so a cap there is reached early. It's only suggested until you save it.</div>
      )}
      {cerebras && <div className="kx-dim">Cerebras keeps its own daily budget (NEO_CEREBRAS_DAILY_BUDGET, $0.50 unless set).</div>}
      {detail.daily_budgets.source === "env" && <div className="kx-dim">NEO_DAILY_BUDGETS in the environment wins over this.</div>}
      {rows.length > 0 && <div className="kx-row-btns"><button className="kx-btn primary" disabled={!dirty || busy} onClick={() => void save()}>{busy ? "Saving…" : "Save caps"}</button></div>}
      <Line msg={msg} />
    </section>
  );
}

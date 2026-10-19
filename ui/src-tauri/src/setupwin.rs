//! NEO Setup: the window (index.html#/setup, ui/src/setup/SetupApp.tsx) that installs NEO's engine
//! on a new Mac, then becomes the checklist; also the menu's Repair NEO's engine…, Check for
//! updates… and Uninstall NEO… pages.
//!
//! Like Brains & Keys (keys.rs) the shell alone creates the window and its commands answer only
//! that window (`may_use`: label `setup`, created here, NEO's own page at #/setup). It talks to the
//! shell, never to the core's websocket: Setup runs before there is a core. What the page shows
//! comes as one `setup_progress` event carrying the whole state (`Snapshot`), sent when it changes.
//!
//! The views, in order on a new Mac:
//!
//! - `move`: NEO runs from its disk image or a translocated copy — "drag NEO to Applications, then
//!   open it from there". Nothing else happens at that launch: no core, no login item, no prompts.
//! - `unsupported`: older than macOS 14 (LSMinimumSystemVersion stops that first).
//! - `welcome`: what will be downloaded, from where, how big, and the room it needs; Set up NEO.
//! - `engine`: the install (setup.rs), with Cancel, Retry and "Use a downloaded file…".
//! - `checklist`: once the engine works the core starts, Brains & Keys opens, and
//!   `neo2 fetch --essential --json` brings the models (its JSON lines are the progress). Then a row
//!   each: a brain, Accessibility, Screen Recording, the microphone and camera (the core asks: the
//!   prompt is credited to NEO.app), the optional downloads (on-device voice, web browsing, camera
//!   boxes: `neo2 fetch --<group> --json`) and Open at login, ticked, written only from Applications.
//!   Done (or closing it) is "setup done": permissions and the login item no longer wait for it.
//! - `new_engine`: an update that needs another engine (case f); `repair`: Repair NEO's engine…;
//!   `no_engine`: a build with nothing to install (a developer build: scripts/install-app.sh).
//! - `updates` (update.rs) and `uninstall` (uninstall.rs).

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::webview::{Color, NewWindowResponse};
use tauri::{AppHandle, Emitter, LogicalPosition, Manager, Runtime, TitleBarStyle, Url, WebviewUrl, WebviewWindow, WebviewWindowBuilder, WindowEvent};

use crate::setup::{self, Decision, Place};
use crate::{core, keys, perms, uninstall, update};

pub const LABEL: &str = "setup";
const ROUTE: &str = "/setup";
const PAGE: &str = "index.html#/setup";
const SHOW_ANYWAY: Duration = Duration::from_millis(1500);
const EMIT_EVERY: Duration = Duration::from_millis(150);
/// The optional downloads, by `neo2 fetch` group.
pub const OPTIONAL: [&str; 3] = ["local_voice", "browser", "vision"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum View {
    Move,
    Unsupported,
    NoEngine,
    Welcome,
    NewEngine,
    Repair,
    Engine,
    Checklist,
    Updates,
    Uninstall,
}

// ---- one `neo2 fetch` run, as its JSON lines say -------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FetchItem {
    pub id: String,
    pub label: String,
    pub source: String,
    pub size: u64,
    /// waiting | running | here | done | skipped | failed
    pub state: String,
    pub done: u64,
    pub total: u64,
    pub stage: Option<String>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Fetch {
    /// idle | running | done | failed | cancelled
    pub state: String,
    pub items: Vec<FetchItem>,
    pub done: u64,
    pub total: u64,
    pub rate: f64,
    pub eta_s: Option<f64>,
    pub code: Option<String>,
    pub error: Option<String>,
}

impl Default for Fetch {
    fn default() -> Self {
        Fetch { state: "idle".into(), items: Vec::new(), done: 0, total: 0, rate: 0.0, eta_s: None, code: None, error: None }
    }
}

/// One event of `neo2 fetch --json` (neo2/core/fetch.py) into the run's state.
pub fn apply(f: &mut Fetch, ev: &serde_json::Value) {
    let s = |k: &str| ev.get(k).and_then(|v| v.as_str()).map(String::from);
    let n = |k: &str| ev.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
    let id = s("id").unwrap_or_default();
    match ev.get("event").and_then(|v| v.as_str()).unwrap_or("") {
        "plan" => {
            f.items = ev
                .get("items")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .map(|it| {
                            let g = |k: &str| it.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let size = it.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
                            let here = it.get("here").and_then(|v| v.as_bool()).unwrap_or(false);
                            let skip = it.get("skip").and_then(|v| v.as_str()).map(String::from);
                            let state = if skip.is_some() { "skipped" } else if here { "here" } else { "waiting" };
                            FetchItem {
                                id: g("id"),
                                label: g("label"),
                                source: g("source"),
                                size,
                                state: state.into(),
                                done: if here { size } else { 0 },
                                total: size,
                                stage: None,
                                message: skip,
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            f.total = n("size");
            f.done = 0;
        }
        "start" => {
            if let Some(it) = f.items.iter_mut().find(|i| i.id == id) {
                it.state = "running".into();
            }
        }
        "progress" => {
            if let Some(it) = f.items.iter_mut().find(|i| i.id == id) {
                it.done = n("done");
                it.total = n("total").max(it.done);
            }
            f.done = n("all_done");
            f.total = n("all_total").max(f.done);
            f.rate = ev.get("rate").and_then(|v| v.as_f64()).unwrap_or(0.0);
            f.eta_s = ev.get("eta_s").and_then(|v| v.as_f64());
        }
        "stage" => {
            if let Some(it) = f.items.iter_mut().find(|i| i.id == id) {
                it.stage = s("stage");
            }
        }
        "item" => {
            let state = s("state").unwrap_or_default();
            if let Some(it) = f.items.iter_mut().find(|i| i.id == id) {
                if state == "done" || state == "here" {
                    it.done = it.total;
                }
                it.state = state.clone();
                it.message = s("message").or(it.message.take());
            }
            if state == "failed" && f.error.is_none() {
                f.code = s("code");
                f.error = s("message");
            }
        }
        "done" => {
            let cancelled = ev.get("cancelled").and_then(|v| v.as_bool()).unwrap_or(false);
            let ok = ev.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
            f.state = if cancelled { "cancelled" } else if ok { "done" } else { "failed" }.into();
            if f.error.is_none() {
                f.error = s("message");
                f.code = s("code");
            }
            if ok {
                f.done = f.total;
                f.eta_s = Some(0.0);
            }
        }
        _ => {}
    }
}

/// Whether a `--status` plan says every item of `group` is here (or skipped by a setting).
pub fn group_here(plan: &serde_json::Value, group: &str) -> Option<bool> {
    let items = plan.get("items")?.as_array()?;
    let mine: Vec<&serde_json::Value> = items.iter().filter(|i| i.get("group").and_then(|g| g.as_str()) == Some(group)).collect();
    if mine.is_empty() {
        return None;
    }
    Some(mine.iter().all(|i| i.get("here").and_then(|h| h.as_bool()).unwrap_or(false) || i.get("skip").is_some_and(|s| !s.is_null())))
}

// ---- the window's state ---------------------------------------------------------------------------

#[derive(Default)]
pub struct Setup(Mutex<State>);

struct State {
    open: Option<u64>,
    made: u64,
    view: View,
    engine: setup::Progress,
    task: Option<setup::Task>,
    from_file: bool,
    essential: Fetch,
    optional: BTreeMap<String, Fetch>,
    pids: BTreeMap<String, u32>,
    here: BTreeMap<String, bool>,
    microphone: String,
    camera: String,
    login_on: bool,
    brain_opened: bool,
    checking_update: bool,
    update_error: Option<String>,
    uninstall_items: Vec<uninstall::Item>,
    sizing: bool,
    uninstalling: bool,
    outcome: Option<uninstall::Outcome>,
    last_emit: Option<Instant>,
}

impl Default for State {
    fn default() -> Self {
        State {
            open: None,
            made: 0,
            view: View::Welcome,
            engine: setup::Progress::default(),
            task: None,
            from_file: false,
            essential: Fetch::default(),
            optional: BTreeMap::new(),
            pids: BTreeMap::new(),
            here: BTreeMap::new(),
            microphone: "unknown".into(),
            camera: "unknown".into(),
            login_on: true,
            brain_opened: false,
            checking_update: false,
            update_error: None,
            uninstall_items: Vec::new(),
            sizing: false,
            uninstalling: false,
            outcome: None,
            last_emit: None,
        }
    }
}

#[derive(Serialize)]
struct Plan {
    app_version: String,
    neo2_version: String,
    host: String,
    asset: String,
    engine_bytes: u64,
    downloaded: u64,
    engine_disk: u64,
    models_bytes: u64,
    need_bytes: u64,
    free_bytes: Option<u64>,
    from_engine: Option<String>,
}

#[derive(Serialize)]
struct Perms {
    accessibility: bool,
    screen_recording: bool,
    microphone: String,
    camera: String,
}

#[derive(Serialize)]
struct UpdateInfo {
    auto: bool,
    checking: bool,
    latest: Option<String>,
    offer: Option<String>,
    last_check: u64,
    error: Option<String>,
    page: Option<String>,
}

#[derive(Serialize)]
struct UninstallInfo {
    items: Vec<uninstall::Item>,
    sizing: bool,
    running: bool,
    outcome: Option<uninstall::Outcome>,
    bundle_id: &'static str,
}

/// Everything the page shows.
#[derive(Serialize)]
struct Snapshot {
    view: View,
    product: &'static str,
    version: String,
    sandbox: bool,
    place: Place,
    macos: String,
    plan: Option<Plan>,
    manifest_problem: Option<String>,
    engine: setup::Progress,
    engine_running: bool,
    engine_from_file: bool,
    /// An engine runs (or can): the checklist has a core behind it.
    engine_ok: bool,
    essential: Fetch,
    optional: BTreeMap<String, Fetch>,
    here: BTreeMap<String, bool>,
    perms: Perms,
    core: String,
    core_ready: bool,
    login_allowed: bool,
    login_on: bool,
    brain_opened: bool,
    done: bool,
    update: UpdateInfo,
    uninstall: UninstallInfo,
    setup_log: String,
}

fn product() -> &'static str {
    if core::SANDBOX {
        "NEO Sandbox"
    } else {
        "NEO"
    }
}

fn setup_done_marker() -> PathBuf {
    core::home().join(".neo").join("setup-done")
}

/// Setup has been finished on this Mac (or isn't needed: a developer engine).
pub fn done() -> bool {
    setup_done_marker().exists()
}

fn plan(decision: &Decision) -> Option<Plan> {
    let m = setup::manifest()?;
    let paths = setup::Paths::here();
    let downloaded = fs::metadata(paths.part(&m.engine.id)).map(|x| x.len()).unwrap_or(0).min(m.engine.bytes);
    Some(Plan {
        app_version: m.app_version.clone(),
        neo2_version: m.neo2_version.clone(),
        host: m.host(),
        asset: m.asset_name(),
        engine_bytes: m.engine.bytes,
        downloaded,
        engine_disk: m.engine.unpacked_bytes,
        models_bytes: m.models_bytes(),
        need_bytes: setup::bytes_needed(m, downloaded, false),
        free_bytes: setup::free_bytes(&paths.root),
        from_engine: match decision {
            Decision::NewEngine { from } => Some(from.clone()),
            _ => None,
        },
    })
}

fn snapshot(app: &AppHandle) -> Snapshot {
    let decision = core::decide_now();
    let core_state = app.state::<core::Core>();
    let core_status = core_state.status.lock().map(|s| s.clone()).unwrap_or_default();
    let remembered = update::load();
    let version = app.package_info().version.to_string();
    let s = app.state::<Setup>();
    let st = s.0.lock().unwrap();
    let offer = update::offer(&remembered, &version);
    Snapshot {
        view: st.view,
        product: product(),
        version: version.clone(),
        sandbox: core::SANDBOX,
        place: setup::placement_now(),
        macos: setup::macos_version().map(|(a, b)| format!("{a}.{b}")).unwrap_or_default(),
        plan: plan(&decision),
        manifest_problem: setup::bundle().and_then(|b| b.problem.clone()),
        engine: st.engine.clone(),
        engine_running: st.task.is_some(),
        engine_from_file: st.from_file,
        engine_ok: core::launch_for(&decision).is_some(),
        essential: st.essential.clone(),
        optional: st.optional.clone(),
        here: st.here.clone(),
        perms: Perms {
            accessibility: perms::accessibility(),
            screen_recording: perms::screen_recording(),
            microphone: st.microphone.clone(),
            camera: st.camera.clone(),
        },
        core: core_status,
        core_ready: core_state.listening(),
        login_allowed: core::may_open_at_login(),
        login_on: st.login_on,
        brain_opened: st.brain_opened,
        done: done(),
        update: UpdateInfo {
            auto: remembered.auto,
            checking: st.checking_update,
            latest: remembered.latest.clone(),
            page: offer.as_deref().map(update::page),
            offer,
            last_check: remembered.last_check,
            error: st.update_error.clone(),
        },
        uninstall: UninstallInfo {
            items: st.uninstall_items.clone(),
            sizing: st.sizing,
            running: st.uninstalling,
            outcome: st.outcome.clone(),
            bundle_id: core::BUNDLE_ID,
        },
        setup_log: setup::log_path().display().to_string().replace(&core::home().display().to_string(), "~"),
    }
}

/// Tell the page (now, or within EMIT_EVERY when `soon`: a progress tick).
fn notify(app: &AppHandle, soon: bool) {
    {
        let s = app.state::<Setup>();
        let mut st = s.0.lock().unwrap();
        if st.open.is_none() {
            return;
        }
        if soon && st.last_emit.is_some_and(|t| t.elapsed() < EMIT_EVERY) {
            return;
        }
        st.last_emit = Some(Instant::now());
    }
    let snap = snapshot(app);
    let _ = app.emit_to(LABEL, "setup_progress", &snap);
}

fn with<R>(app: &AppHandle, f: impl FnOnce(&mut State) -> R) -> R {
    let s = app.state::<Setup>();
    let mut st = s.0.lock().unwrap();
    f(&mut st)
}

// ---- the window -------------------------------------------------------------------------------------

fn ours(app: &AppHandle) -> bool {
    with(app, |st| st.open.is_some())
}

/// Whether a window may use Setup's commands: labelled `setup`, created here, NEO's own page at
/// #/setup.
pub fn may_use(label: &str, ours: bool, url: Option<&Url>, dev: bool) -> bool {
    let Some(url) = url else { return false };
    label == LABEL && ours && keys::app_origin(url, dev) && url.fragment() == Some(ROUTE)
}

fn allowed(app: &AppHandle, window: &WebviewWindow) -> Result<(), String> {
    let url = window.url().ok();
    if may_use(window.label(), ours(app), url.as_ref(), cfg!(debug_assertions)) {
        Ok(())
    } else {
        core::log("a window asked for one of Setup's commands and was refused");
        Err("forbidden".into())
    }
}

fn front(w: &WebviewWindow) {
    let _ = w.unminimize();
    let _ = w.show();
    let _ = w.set_focus();
}

/// Open the window at `view` (or bring it up as it is). On the main thread.
pub fn open(app: &AppHandle, view: View) {
    with(app, |st| {
        // An install that is running (or just failed) keeps its page until the user leaves it.
        let busy = st.view == View::Engine && (st.task.is_some() || st.engine.error.is_some());
        if !busy || matches!(view, View::Updates | View::Uninstall) {
            st.view = view;
        }
    });
    if let Some(w) = app.get_webview_window(LABEL) {
        if ours(app) {
            front(&w);
            notify(app, false);
            return;
        }
        core::log("a Setup window the shell didn't create was closed");
        let _ = w.destroy();
        return;
    }
    let id = with(app, |st| {
        st.made += 1;
        st.open = Some(st.made);
        st.made
    });
    let dev = cfg!(debug_assertions);
    let built = WebviewWindowBuilder::new(app, LABEL, WebviewUrl::App(PAGE.into()))
        .title(if core::SANDBOX { "NEO Sandbox Setup" } else { "NEO Setup" })
        .inner_size(600.0, 660.0)
        .min_inner_size(500.0, 540.0)
        .resizable(true)
        .center()
        .decorations(true)
        .title_bar_style(TitleBarStyle::Overlay)
        .hidden_title(true)
        .traffic_light_position(LogicalPosition::new(16.0, 20.0))
        .transparent(false)
        .background_color(Color(0x11, 0x11, 0x14, 0xff))
        .always_on_top(false)
        .visible(false)
        .focused(true)
        .on_navigation(move |url| keys::may_navigate(url, dev))
        .on_new_window(|_, _| NewWindowResponse::Deny)
        .build();
    let w = match built {
        Ok(w) => w,
        Err(e) => {
            with(app, |st| {
                if st.open == Some(id) {
                    st.open = None;
                }
            });
            core::log(&format!("Setup didn't open: {e}"));
            return;
        }
    };
    let handle = app.clone();
    w.on_window_event(move |e| {
        if let WindowEvent::Destroyed = e {
            closed(&handle, id);
        }
    });
    let handle = app.clone();
    thread::spawn(move || {
        thread::sleep(SHOW_ANYWAY);
        let h = handle.clone();
        let _ = handle.run_on_main_thread(move || {
            if let Some(w) = h.get_webview_window(LABEL) {
                if ours(&h) && !w.is_visible().unwrap_or(true) {
                    front(&w);
                }
            }
        });
    });
}

fn closed(app: &AppHandle, id: u64) {
    let view = with(app, |st| {
        if st.open != Some(id) {
            return None;
        }
        st.open = None;
        Some(st.view)
    });
    match view {
        // "Drag NEO to Applications": this copy has nothing else to do.
        Some(View::Move) | Some(View::Unsupported) => app.exit(0),
        // The checklist closed with an engine that works: setup is done (the menu reopens it).
        Some(View::Checklist) if core::launch_for(&core::decide_now()).is_some() => finish(app),
        // The uninstalled app has nothing left to run.
        Some(View::Uninstall) if with(app, |st| st.outcome.is_some()) => app.exit(0),
        _ => {}
    }
}

/// Every page load: a `setup` window this shell didn't create is closed.
pub fn page_loaded<R: Runtime>(webview: &tauri::Webview<R>) {
    if webview.label() != LABEL {
        return;
    }
    let app = webview.app_handle();
    let mine = app.state::<Setup>().0.lock().map(|s| s.open.is_some()).unwrap_or(false);
    if !mine {
        core::log("a Setup window the shell didn't create was closed");
        let _ = webview.window().destroy();
    }
}

// ---- at launch, and what the supervisor says ------------------------------------------------------

/// What this launch has to show before anything else: the Move page (a disk image or translocated
/// copy), or an unsupported macOS. None: carry on.
pub fn blocked_at_launch() -> Option<View> {
    if setup::placement_now() != Place::Ok {
        return Some(View::Move);
    }
    if setup::macos_version().is_some_and(|v| v < (14, 0)) {
        return Some(View::Unsupported);
    }
    None
}

/// Before the core starts: what an earlier setup left, NotchHub's copy of the page, and what this
/// launch will run.
pub fn prepare_launch() -> Decision {
    let paths = setup::Paths::here();
    setup::cleanup(&paths, setup::manifest().map(|m| m.engine.id.as_str()));
    if let Some(b) = setup::bundle() {
        if let Err(e) = setup::sync_ui(&b.ui(), &paths) {
            setup::log(&format!("NEO's page for NotchHub couldn't be copied: {e}"));
        }
    }
    core::decide_now()
}

/// The page for a launch that can't run a core yet.
pub fn view_for(decision: &Decision) -> View {
    match decision {
        Decision::NewEngine { .. } => View::NewEngine,
        Decision::NoEngine => View::NoEngine,
        _ => View::Welcome,
    }
}

/// The supervisor found no engine to run (from its thread).
pub fn needs(app: &AppHandle, decision: Decision) {
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || {
        if with(&handle, |st| st.task.is_some()) {
            return; // an install is running: its own page says what happens
        }
        let view = if decision == Decision::Install && setup::current_release(&setup::Paths::here()).is_some() { View::Repair } else { view_for(&decision) };
        open(&handle, view);
    });
}

/// A core said hello: the old engine an update replaced can go now, and the checklist asks what
/// macOS says of the microphone and the camera.
pub fn core_said_hello(app: &AppHandle) {
    let paths = setup::Paths::here();
    if let Some(id) = setup::current_release(&paths) {
        setup::sweep_engines(&paths, Some(&id));
    }
    if ours(app) {
        let core = app.state::<core::Core>();
        for what in core::PERMISSION_WHATS {
            core.permission(what, false);
        }
    }
    notify(app, false);
}

/// What the core says macOS answered (`FromCore::Permission`).
pub fn permission_said(app: &AppHandle, what: &str, status: &str) {
    with(app, |st| match what {
        "microphone" => st.microphone = status.to_string(),
        "camera" => st.camera = status.to_string(),
        _ => {}
    });
    notify(app, false);
}

/// Something the page shows changed outside it (the status loop: permissions, the core).
pub fn tick(app: &AppHandle) {
    notify(app, true);
}

/// Setup is done: permissions and the login item stop waiting for it.
pub fn finish(app: &AppHandle) {
    if done() {
        return;
    }
    let version = app.package_info().version.to_string();
    let _ = fs::create_dir_all(setup_done_marker().parent().unwrap());
    let _ = fs::write(setup_done_marker(), format!("{version}\n"));
    perms::mark_asked(&version); // the checklist offered them: no prompts at the next launch
    let on = with(app, |st| st.login_on);
    if core::may_open_at_login() {
        let _ = core::set_launch_at_login(on);
        core::remember_login_choice();
    }
    crate::login_item_changed(app);
    setup::log(&format!("setup done (open at login: {})", if on && core::may_open_at_login() { "on" } else { "off" }));
}

// ---- the engine --------------------------------------------------------------------------------------

fn start_install(app: &AppHandle, source: setup::Source) -> Result<(), String> {
    let (Some(bundle), Some(m)) = (setup::bundle(), setup::manifest()) else {
        return Err("no_manifest".into());
    };
    let Some(wheel) = bundle.wheel() else { return Err("no_manifest".into()) };
    let task = setup::Task::default();
    let from_file = matches!(source, setup::Source::File(_));
    let began = with(app, |st| {
        if st.task.is_some() {
            return false;
        }
        st.task = Some(task.clone());
        st.from_file = from_file;
        st.view = View::Engine;
        st.engine = setup::Progress { stage: setup::Stage::Preparing, ..Default::default() };
        true
    });
    if !began {
        return Err("busy".into());
    }
    notify(app, false);
    let handle = app.clone();
    thread::spawn(move || {
        let paths = setup::Paths::here();
        let core = handle.state::<core::Core>().inner().clone();
        let report = |p: &setup::Progress| {
            let stage_changed = with(&handle, |st| {
                let changed = st.engine.stage != p.stage;
                st.engine = p.clone();
                changed
            });
            notify(&handle, !stage_changed);
        };
        let res = setup::install(&paths, m, &wheel, &source, &setup::Opts::here(), &task, &report, &|| core.hold());
        core.release(); // whatever happened: the core runs whatever engine there is now
        let ok = res.is_ok();
        with(&handle, |st| {
            st.task = None;
            match &res {
                Ok(()) => {
                    st.engine = setup::Progress { stage: setup::Stage::Done, done: 1, total: 1, ..Default::default() };
                    // A first setup goes on to the checklist; an update's or a repair's says it's done.
                    if !done() {
                        st.view = View::Checklist;
                    }
                }
                Err(f) => {
                    let stage = if f.is_cancel() { setup::Stage::Cancelled } else { setup::Stage::Failed };
                    st.engine = setup::Progress { stage, error: Some(f.clone()), ..st.engine.clone() };
                }
            }
        });
        match &res {
            Ok(()) => setup::log("the engine is in: the core starts"),
            Err(f) => setup::log(&format!("setup stopped: {} ({})", f.message, f.code)),
        }
        notify(&handle, false);
        if ok {
            after_engine(&handle);
        }
    });
    Ok(())
}

/// The engine works: NEO's orb, Brains & Keys on a first setup (it waits for the core itself), the
/// models, and what's here.
fn after_engine(app: &AppHandle) {
    let first = !done();
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || {
        if !crate::in_notch(&handle) {
            if let Some(w) = handle.get_webview_window("main") {
                let _ = w.show(); // hidden while there was no core behind it
            }
        }
        if first {
            keys::open(&handle);
            with(&handle, |st| st.brain_opened = true);
        }
    });
    run_fetch(app, "essential");
    refresh_here(app);
}

/// The checklist, up to date: what macOS says of each permission, what's here, and the essential
/// models (fetch says "here" at once for what is — a launch after Quit & Reopen picks them up again).
pub fn open_checklist(app: &AppHandle) {
    open(app, View::Checklist);
    if core::launch_for(&core::decide_now()).is_some() {
        run_fetch(app, "essential");
        refresh_here(app);
    }
    let core = app.state::<core::Core>();
    for w in core::PERMISSION_WHATS {
        core.permission(w, false);
    }
}

// ---- `neo2 fetch` --------------------------------------------------------------------------------

fn fetch_command(args: &[&str]) -> Option<Command> {
    let launch = core::launch_for(&core::decide_now())?;
    let mut cmd = Command::new(&launch.python);
    cmd.args(launch.module()).arg("fetch").args(args).current_dir(&launch.cwd);
    core::core_env(&mut cmd);
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    Some(cmd)
}

fn flag(group: &str) -> Option<&'static str> {
    match group {
        "essential" => Some("--essential"),
        "local_voice" => Some("--local-voice"),
        "browser" => Some("--browser"),
        "vision" => Some("--vision"),
        _ => None,
    }
}

fn fetch_state<'a>(st: &'a mut State, group: &str) -> &'a mut Fetch {
    if group == "essential" {
        &mut st.essential
    } else {
        st.optional.entry(group.to_string()).or_default()
    }
}

/// `neo2 fetch --<group> --json`, its lines into the page, one run per group at a time.
fn run_fetch(app: &AppHandle, group: &'static str) {
    let Some(flag) = flag(group) else { return };
    let started = with(app, |st| {
        if st.pids.contains_key(group) {
            return false;
        }
        *fetch_state(st, group) = Fetch { state: "running".into(), ..Fetch::default() };
        true
    });
    if !started {
        return;
    }
    let Some(mut cmd) = fetch_command(&[flag, "--json"]) else {
        with(app, |st| {
            let f = fetch_state(st, group);
            f.state = "failed".into();
            f.error = Some("NEO's engine isn't set up yet.".into());
        });
        notify(app, false);
        return;
    };
    let handle = app.clone();
    thread::spawn(move || {
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                with(&handle, |st| {
                    let f = fetch_state(st, group);
                    f.state = "failed".into();
                    f.error = Some(format!("The download didn't start: {e}"));
                });
                notify(&handle, false);
                return;
            }
        };
        with(&handle, |st| st.pids.insert(group.to_string(), child.id()));
        setup::log(&format!("neo2 fetch {flag}: started"));
        if let Some(err) = child.stderr.take() {
            thread::spawn(move || {
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    setup::log(&format!("fetch {flag}: {line}"));
                }
            });
        }
        if let Some(out) = child.stdout.take() {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                let Ok(ev) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
                let big = ev.get("event").and_then(|e| e.as_str()) != Some("progress");
                with(&handle, |st| apply(fetch_state(st, group), &ev));
                notify(&handle, !big);
            }
        }
        let code = child.wait().ok().and_then(|s| s.code());
        with(&handle, |st| {
            st.pids.remove(group);
            let f = fetch_state(st, group);
            if f.state == "running" {
                // It ended without a word (killed): say so.
                f.state = if code == Some(0) { "done" } else if code == Some(130) || code.is_none() { "cancelled" } else { "failed" }.into();
                if f.state == "failed" && f.error.is_none() {
                    f.error = Some(format!("The download stopped (exit {}). The setup log has the details.", code.unwrap_or(-1)));
                }
            }
        });
        setup::log(&format!("neo2 fetch {flag}: exit {code:?}"));
        notify(&handle, false);
        refresh_here(&handle);
    });
}

fn cancel_fetch(app: &AppHandle, group: &str) {
    if let Some(pid) = with(app, |st| st.pids.get(group).copied()) {
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }
    }
}

/// Every download Setup started stops (the app is quitting, or uninstalling).
pub fn stop_all(app: &AppHandle) {
    let pids: Vec<u32> = with(app, |st| st.pids.values().copied().collect());
    for pid in pids {
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }
    }
}

/// What's on this Mac already, by group (`neo2 fetch --status --json`, all of them).
fn refresh_here(app: &AppHandle) {
    let Some(mut cmd) = fetch_command(&["--status", "--json", "--essential", "--local-voice", "--browser", "--vision"]) else { return };
    let handle = app.clone();
    thread::spawn(move || {
        let Ok(out) = cmd.output() else { return };
        let text = String::from_utf8_lossy(&out.stdout);
        let Some(plan) = text.lines().filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()).find(|v| v.get("event").and_then(|e| e.as_str()) == Some("plan")) else {
            return;
        };
        with(&handle, |st| {
            for g in ["essential", "local_voice", "browser", "vision"] {
                if let Some(h) = group_here(&plan, g) {
                    st.here.insert(g.to_string(), h);
                }
            }
        });
        notify(&handle, false);
    });
}

// ---- the commands -----------------------------------------------------------------------------------

#[tauri::command]
pub fn setup_state(app: AppHandle, window: WebviewWindow) -> Result<serde_json::Value, String> {
    allowed(&app, &window)?;
    serde_json::to_value(snapshot(&app)).map_err(|e| e.to_string())
}

/// Set up NEO / Retry / Repair / Update the engine: download and install this app's engine.
#[tauri::command]
pub fn setup_start(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    allowed(&app, &window)?;
    start_install(&app, setup::Source::Download)
}

#[tauri::command]
pub fn setup_retry(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    allowed(&app, &window)?;
    start_install(&app, setup::Source::Download)
}

#[tauri::command]
pub fn setup_cancel(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    allowed(&app, &window)?;
    if let Some(t) = with(&app, |st| st.task.clone()) {
        setup::log("Cancel");
        t.cancel();
    }
    Ok(())
}

/// "Use a downloaded file…": the open panel (on the main thread, after this command has
/// answered), then an install from that file.
#[tauri::command]
pub fn setup_use_file(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    allowed(&app, &window)?;
    if with(&app, |st| st.task.is_some()) {
        return Err("busy".into());
    }
    let asset = setup::manifest().map(|m| m.asset_name()).unwrap_or_else(|| "NEO's engine".into());
    let handle = app.clone();
    app.run_on_main_thread(move || {
        let chosen = unsafe { crate::native::choose_file(&format!("Choose {asset} (from NEO's release page)"), "Use This File") };
        if let Some(path) = chosen {
            setup::log(&format!("using a downloaded file: {}", path.display()));
            if let Err(e) = start_install(&handle, setup::Source::File(path)) {
                core::log(&format!("Setup couldn't start from the file: {e}"));
            }
        }
    })
    .map_err(|e| e.to_string())
}

/// The page looks again: what macOS says of each permission, and what's on this Mac.
#[tauri::command]
pub fn setup_check(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    allowed(&app, &window)?;
    let core = app.state::<core::Core>();
    for what in core::PERMISSION_WHATS {
        core.permission(what, false);
    }
    refresh_here(&app);
    notify(&app, false);
    Ok(())
}

/// An optional download: start it (`on`) or cancel it.
#[tauri::command]
pub fn setup_optional(app: AppHandle, window: WebviewWindow, group: String, on: bool) -> Result<(), String> {
    allowed(&app, &window)?;
    let Some(g) = OPTIONAL.iter().copied().find(|g| *g == group) else { return Err("unknown".into()) };
    if on {
        run_fetch(&app, g);
    } else {
        cancel_fetch(&app, g);
    }
    Ok(())
}

/// The essential models again (after a failure).
#[tauri::command]
pub fn setup_models(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    allowed(&app, &window)?;
    run_fetch(&app, "essential");
    Ok(())
}

/// Open at login: the tickbox. Written now when setup is done (the checklist reopened from the
/// menu), else when it finishes — and only ever for a copy in Applications.
#[tauri::command]
pub fn setup_login(app: AppHandle, window: WebviewWindow, on: bool) -> Result<(), String> {
    allowed(&app, &window)?;
    with(&app, |st| st.login_on = on);
    if done() {
        let _ = core::set_launch_at_login(on);
        core::remember_login_choice();
        crate::login_item_changed(&app);
    }
    notify(&app, false);
    Ok(())
}

/// A permission's button: macOS's prompt (credited to NEO.app), and its pane of System Settings.
#[tauri::command]
pub fn setup_permission(app: AppHandle, window: WebviewWindow, what: String) -> Result<(), String> {
    allowed(&app, &window)?;
    match what.as_str() {
        "accessibility" => {
            perms::ask_accessibility();
            perms::open_settings("Privacy_Accessibility");
        }
        "screen_recording" => {
            perms::ask_screen_recording();
            perms::open_settings("Privacy_ScreenCapture");
        }
        "microphone" | "camera" => {
            let asked = app.state::<core::Core>().permission(&what, true);
            if !asked {
                return Err("not_ready".into());
            }
        }
        _ => return Err("unknown".into()),
    }
    notify(&app, false);
    Ok(())
}

/// The page's links: Brains & Keys, a pane of System Settings, the logs, Applications.
#[tauri::command]
pub fn setup_open(app: AppHandle, window: WebviewWindow, what: String) -> Result<(), String> {
    allowed(&app, &window)?;
    let open = |arg: &std::path::Path| {
        let _ = Command::new("/usr/bin/open").arg(arg).stdin(Stdio::null()).spawn();
    };
    match what.as_str() {
        "keys" => {
            let handle = app.clone();
            let _ = app.run_on_main_thread(move || keys::open(&handle));
            with(&app, |st| st.brain_opened = true);
        }
        "microphone_settings" => perms::open_settings("Privacy_Microphone"),
        "camera_settings" => perms::open_settings("Privacy_Camera"),
        "accessibility_settings" => perms::open_settings("Privacy_Accessibility"),
        "screen_settings" => perms::open_settings("Privacy_ScreenCapture"),
        "logs" => open(&core::logs_dir()),
        "applications" => open(std::path::Path::new("/Applications")),
        "releases" => {
            let _ = Command::new("/usr/bin/open").arg(update::RELEASES_LATEST).stdin(Stdio::null()).spawn();
        }
        _ => return Err("unknown".into()),
    }
    notify(&app, false);
    Ok(())
}

/// Done: setup is finished, and the window closes.
#[tauri::command]
pub fn setup_done(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    allowed(&app, &window)?;
    if core::launch_for(&core::decide_now()).is_some() {
        finish(&app);
    }
    let _ = window.close();
    Ok(())
}

/// The Move page's Quit (and the uninstalled app's).
#[tauri::command]
pub fn setup_quit(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    allowed(&app, &window)?;
    app.exit(0);
    Ok(())
}

/// Check for updates: `check` now, `skip` the offered version, `auto` on or off, `open` its page.
#[tauri::command]
pub fn setup_update(app: AppHandle, window: WebviewWindow, action: String, on: Option<bool>) -> Result<(), String> {
    allowed(&app, &window)?;
    let version = app.package_info().version.to_string();
    match action.as_str() {
        "check" => check_updates(&app),
        "skip" => {
            if let Some(v) = update::offer(&update::load(), &version) {
                update::skip(&v);
            }
            crate::update_offer(&app, None);
        }
        "auto" => update::set_auto(on.unwrap_or(true)),
        "open" => {
            let Some(v) = update::offer(&update::load(), &version) else { return Err("none".into()) };
            let _ = Command::new("/usr/bin/open").arg(update::page(&v)).stdin(Stdio::null()).spawn();
        }
        _ => return Err("unknown".into()),
    }
    notify(&app, false);
    Ok(())
}

/// A check now, off the main thread; the page and the menu hear what it found.
pub fn check_updates(app: &AppHandle) {
    if with(app, |st| std::mem::replace(&mut st.checking_update, true)) {
        return;
    }
    notify(app, false);
    let handle = app.clone();
    thread::spawn(move || {
        let version = handle.package_info().version.to_string();
        let res = update::check(&version);
        with(&handle, |st| {
            st.checking_update = false;
            st.update_error = res.as_ref().err().map(|e| {
                core::log(&format!("update check failed: {e}"));
                "NEO couldn't reach GitHub to check. Try again later.".to_string()
            });
        });
        if let Ok(offer) = res {
            crate::update_offer(&handle, offer);
        }
        notify(&handle, false);
    });
}

/// Uninstall: `list` (with sizes), or `run` with the chosen items.
#[tauri::command]
pub fn setup_uninstall(app: AppHandle, window: WebviewWindow, action: String, chosen: Option<Vec<String>>, reset: Option<bool>) -> Result<(), String> {
    allowed(&app, &window)?;
    match action.as_str() {
        "list" => list_uninstall(&app),
        "run" => {
            if with(&app, |st| std::mem::replace(&mut st.uninstalling, true)) {
                return Err("busy".into());
            }
            notify(&app, false);
            let (chosen, reset) = (chosen.unwrap_or_default(), reset.unwrap_or(false));
            let handle = app.clone();
            thread::spawn(move || {
                stop_all(&handle);
                if let Some(t) = with(&handle, |st| st.task.clone()) {
                    t.cancel();
                }
                handle.state::<core::Core>().stop(); // the core runs from the engine about to go
                let home = core::home();
                let bundle = core::bundle_path();
                let list = with(&handle, |st| st.uninstall_items.clone());
                let list = if list.is_empty() { uninstall::items(bundle.as_deref(), &home, core::BUNDLE_ID) } else { list };
                let outcome = uninstall::run(
                    &list,
                    &chosen,
                    reset,
                    bundle.as_deref(),
                    &home,
                    core::BUNDLE_ID,
                    &|p| crate::native::trash(p),
                    &|cmd| Command::new(&cmd[0]).args(&cmd[1..]).stdin(Stdio::null()).status().map(|s| s.success()).unwrap_or(false),
                );
                with(&handle, |st| {
                    st.uninstalling = false;
                    st.outcome = Some(outcome);
                });
                notify(&handle, false);
            });
        }
        _ => return Err("unknown".into()),
    }
    Ok(())
}

pub fn list_uninstall(app: &AppHandle) {
    if with(app, |st| std::mem::replace(&mut st.sizing, true)) {
        return;
    }
    let home = core::home();
    let bundle = core::bundle_path();
    let quick = uninstall::items(bundle.as_deref(), &home, core::BUNDLE_ID);
    with(app, |st| st.uninstall_items = quick.clone());
    notify(app, false);
    let handle = app.clone();
    thread::spawn(move || {
        let sized = uninstall::sized(quick);
        with(&handle, |st| {
            st.uninstall_items = sized;
            st.sizing = false;
        });
        notify(&handle, false);
    });
}

/// The menu's items for this window, by name.
pub fn from_menu(app: &AppHandle, what: &str) {
    match what {
        "setup_checklist" => {
            let d = core::decide_now();
            if core::launch_for(&d).is_some() {
                open_checklist(app);
            } else {
                open(app, view_for(&d));
            }
        }
        "repair_engine" => open(app, if setup::manifest().is_some() { View::Repair } else { View::NoEngine }),
        "updates" => {
            open(app, View::Updates);
            check_updates(app);
        }
        "uninstall" => {
            with(app, |st| st.outcome = None);
            open(app, View::Uninstall);
            list_uninstall(app);
        }
        _ => {}
    }
}

/// Every Setup command, by name (lib.rs registers exactly these; each checks `may_use` first).
#[cfg(test)]
pub const COMMANDS: [&str; 15] = [
    "setup_state",
    "setup_start",
    "setup_retry",
    "setup_cancel",
    "setup_use_file",
    "setup_check",
    "setup_optional",
    "setup_models",
    "setup_login",
    "setup_permission",
    "setup_open",
    "setup_done",
    "setup_quit",
    "setup_update",
    "setup_uninstall",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    #[test]
    fn only_the_shells_own_setup_window_uses_its_commands() {
        assert!(may_use("setup", true, Some(&u("tauri://localhost/index.html#/setup")), false));
        assert!(!may_use("setup", false, Some(&u("tauri://localhost/index.html#/setup")), false));
        assert!(!may_use("keys", true, Some(&u("tauri://localhost/index.html#/setup")), false));
        assert!(!may_use("setup", true, Some(&u("tauri://localhost/index.html#/keys")), false));
        assert!(!may_use("setup", true, Some(&u("https://example.com/#/setup")), false));
        assert!(!may_use("setup", true, None, false));
        assert!(may_use("setup", true, Some(&u("http://localhost:1420/index.html#/setup")), true));
        // Every command checks first, and is registered (lib.rs), and no other setup_ command exists.
        let src = include_str!("setupwin.rs");
        let lib = include_str!("lib.rs");
        let mut n = 0;
        for part in src.split("#[tauri::command]\npub fn ").skip(1) {
            let name = part.split('(').next().unwrap();
            let body = part.split("\n}\n").next().unwrap();
            assert!(body.lines().nth(1).unwrap().trim() == "allowed(&app, &window)?;", "{name} checks the window first");
            assert!(COMMANDS.contains(&name), "{name}");
            assert!(lib.contains(&format!("setupwin::{name},")) || lib.contains(&format!("setupwin::{name}\n")), "{name} is registered");
            n += 1;
        }
        assert_eq!(n, COMMANDS.len());
    }

    #[test]
    fn fetchs_json_lines_become_the_pages_progress() {
        let mut f = Fetch::default();
        let ev = |s: &str| serde_json::from_str::<serde_json::Value>(s).unwrap();
        apply(&mut f, &ev(r#"{"event":"plan","groups":["essential"],"size":1500,"items":[
            {"id":"laya","group":"essential","label":"Laya","source":"Hugging Face","size":1000,"disk":1000,"here":false,"skip":null},
            {"id":"wake","group":"essential","label":"Wake","source":"alphacephei.com","size":500,"disk":700,"here":false,"skip":null},
            {"id":"embedder","group":"essential","label":"MiniLM","source":"Hugging Face","size":90,"disk":90,"here":true,"skip":null}]}"#));
        assert_eq!(f.items.len(), 3);
        assert_eq!(f.items[2].state, "here");
        assert_eq!(f.items[2].done, 90);
        assert_eq!(f.total, 1500);
        apply(&mut f, &ev(r#"{"event":"start","id":"laya"}"#));
        assert_eq!(f.items[0].state, "running");
        apply(&mut f, &ev(r#"{"event":"progress","id":"laya","done":400,"total":1010,"all_done":400,"all_total":1510,"rate":100000,"eta_s":11.1}"#));
        assert_eq!((f.items[0].done, f.items[0].total, f.done, f.total), (400, 1010, 400, 1510));
        assert_eq!(f.eta_s, Some(11.1));
        apply(&mut f, &ev(r#"{"event":"item","id":"laya","state":"done"}"#));
        assert_eq!((f.items[0].state.as_str(), f.items[0].done), ("done", 1010));
        apply(&mut f, &ev(r#"{"event":"stage","id":"wake","stage":"unpacking"}"#));
        assert_eq!(f.items[1].stage.as_deref(), Some("unpacking"));
        apply(&mut f, &ev(r#"{"event":"item","id":"wake","state":"failed","code":"offline","message":"Can't reach the server — check the internet connection."}"#));
        assert_eq!(f.code.as_deref(), Some("offline"));
        apply(&mut f, &ev(r#"{"event":"done","ok":false,"failed":["wake"],"seconds":3}"#));
        assert_eq!(f.state, "failed");
        assert_eq!(f.error.as_deref(), Some("Can't reach the server — check the internet connection."));
        let mut c = Fetch::default();
        apply(&mut c, &ev(r#"{"event":"done","ok":false,"failed":[],"cancelled":true,"seconds":1}"#));
        assert_eq!(c.state, "cancelled");
        let mut e = Fetch::default();
        apply(&mut e, &ev(r#"{"event":"done","ok":false,"failed":[],"code":"failed","message":"boom","seconds":0}"#));
        assert_eq!((e.state.as_str(), e.error.as_deref()), ("failed", Some("boom")));
        apply(&mut e, &ev(r#"{"event":"unknown"}"#)); // ignored
        apply(&mut e, &ev("[]"));
    }

    #[test]
    fn a_status_plan_says_which_groups_are_here() {
        let plan: serde_json::Value = serde_json::from_str(r#"{"event":"plan","items":[
            {"id":"laya","group":"essential","here":true,"skip":null},
            {"id":"wake","group":"essential","here":false,"skip":null},
            {"id":"stt","group":"local_voice","here":true,"skip":null},
            {"id":"tts","group":"local_voice","here":true,"skip":null},
            {"id":"chromium","group":"browser","here":false,"skip":"the playwright package isn't installed"}]}"#).unwrap();
        assert_eq!(group_here(&plan, "essential"), Some(false));
        assert_eq!(group_here(&plan, "local_voice"), Some(true));
        assert_eq!(group_here(&plan, "browser"), Some(true)); // skipped by a setting: nothing to offer
        assert_eq!(group_here(&plan, "vision"), None);
    }

    #[test]
    fn a_launch_that_cant_run_a_core_opens_the_page_that_fixes_it() {
        assert_eq!(view_for(&Decision::Install), View::Welcome);
        assert_eq!(view_for(&Decision::NewEngine { from: "0123456789abcdef".into() }), View::NewEngine);
        assert_eq!(view_for(&Decision::NoEngine), View::NoEngine);
        assert_eq!(serde_json::to_value(View::NewEngine).unwrap(), "new_engine");
        assert_eq!(flag("local_voice"), Some("--local-voice"));
        assert_eq!(flag("../x"), None);
        assert!(OPTIONAL.iter().all(|g| flag(g).is_some()));
    }
}

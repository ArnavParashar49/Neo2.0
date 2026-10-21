//! Search Screen: the overlay that lets the user circle or drag over anything on screen and ask
//! about it (index.html#/lens, ui/src/lens/LensApp.tsx; the core's side is neo2/core/lens).
//!
//! The shell alone creates the overlay's window, as it does Brains & Keys' (keys.rs): one window
//! labelled `lens`, kept out of screenshots and screen shares (contentProtected), shown only on NEO's
//! own page at `#/lens`. Any other window with that label is closed as its page loads.
//!
//! How it opens:
//! - **Hold ⌥⌘** (chord.rs): `Warm` makes the window ready on the display under the pointer —
//!   on screen but see-through and click-through, painting nothing; `Preview` starts the glow;
//!   `Arm` mints the session's ticket and asks the core for its still (`lens_freeze`); `Commit`
//!   (let go cleanly) opens it; `Cancel` retracts the glow and lets the core drop the still.
//! - **The menu's "Search Screen…"**, the voice ("search my screen"), the agent's `search_screen`
//!   tool and a host (all `shell` action `lens` from the core): `open`, after a moment for the menu
//!   (or the notch's panel) to get out of the way.
//!
//! The window is made ahead of time, hidden, a moment after the core says Search Screen is on, and
//! kept while it is on: a hold never builds a web view on the main thread, where a slow ⌥⌘ tap's
//! summon would wait behind it. (Not made yet — the core has only just said hello — it is made at
//! Preview, never at Warm.)
//!
//! Open, the window is a non-activating panel (native.rs) above everything, the menu bar too, on
//! every Space: it takes the keys while the user's app stays the active one. It closes on Esc and
//! the page's other ways out (`lens_close`), on a summon (⌥⌘ tap, ⌘⇧Space, the menu-bar icon:
//! lib.rs `toggle`), when the core says so (`lens_close`: Erase, Stop, an action that opened
//! something), when the screen locks or its display changes (the watchdog), and when the core goes.
//! Its page may die (a script error, a stuck page): then nothing inside could close a window that
//! covers the display and takes every click, so the shell doesn't rely on it — the panel's own Esc
//! (a key the page didn't take) closes it, and so does the page falling silent (it says
//! `lens_painted` every second while open; silent for PAGE_SILENT, the watchdog closes it).
//!
//! The glow along the screen's edges is the shell's own (glow.rs: Core Animation in a view above the
//! page): each phase function below starts its curve the moment the phase begins, with no IPC and no
//! page; the page only says whether an area is selected (`lens_glow`). If the glow's view can't be
//! made, the phase events say `glow: "web"` and the page draws it as before.
//!
//! The ticket (32 bytes from /dev/urandom, 64 lowercase hex) admits the page to the core's websocket
//! as its one `lens` client. The core hears it over the socket only the two of them share
//! (`lens_freeze`, revoked by `lens_closed`); the page gets it from `lens_hello`, a command that
//! answers only this window. It is never in an event: Tauri hands every event to any page that
//! listens to all of them, whatever window it was meant for. The events (`neo://lens-phase`) say
//! only which phase the overlay is in.

use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::{atomic::Ordering, Arc};
use std::time::{Duration, Instant};

use objc2::runtime::AnyObject;
use serde::Serialize;
use tauri::webview::{Color, NewWindowResponse};
use tauri::{AppHandle, Emitter, Manager, Runtime, Url, WebviewUrl, WebviewWindow, WebviewWindowBuilder, WindowEvent};

use crate::chord::{HoldEvent, LensGate};
use crate::glow::{self, GlowPhase, PhaseIn};
use crate::{core, frame, keys, native, perms};

/// The window's label, and the route its page shows.
pub const LABEL: &str = "lens";
const ROUTE: &str = "/lens";
const PAGE: &str = "index.html#/lens";
/// The page's one event: which phase the overlay is in (never a ticket).
const EVENT: &str = "neo://lens-phase";
/// The page fades out in 160 ms; the window orders out just after.
const FADE: Duration = Duration::from_millis(170);
/// Closing because the area went into the notch (NotchHub shows the results): the crop's flight
/// (380 ms) plays out first.
const FADE_NOTCH: Duration = Duration::from_millis(440);
/// Opened from the menu: its own fade-out first (it must not be in the still). Voice and hosts:
/// the notch's panel folds away first (the core published `lens_opening`).
const OPEN_WAIT_MENU: Duration = Duration::from_millis(220);
const OPEN_WAIT: Duration = Duration::from_millis(150);
/// With Search Screen off (or no core), a window nobody used for this long is destroyed (its web
/// process with it). While it is on the window is kept, ready for the next hold.
const DESTROY_AFTER: Duration = Duration::from_secs(60);
/// The window is made this long after the core said Search Screen is on (NEO has just started).
const PREPARE_AFTER: Duration = Duration::from_secs(2);
/// Open, the page says it is alive every second (lens_painted); silent this long, it is closed.
const PAGE_SILENT: Duration = Duration::from_secs(5);
const WATCH_EVERY: Duration = Duration::from_millis(500);
/// The core went while the overlay was open: the page says so, then it closes.
const GONE_CLOSE: Duration = Duration::from_secs(3);
/// What `lens_closed` may say (the core logs anything else as "closed"). "notch": the area was handed
/// to NotchHub — the overlay is done, the area isn't.
const WHYS: [&str; 13] = ["esc", "cancel", "summon", "action", "lock", "display", "idle", "gone", "continue", "done", "erased", "stop", "notch"];
const ANNOUNCE: &str = "Search Screen. Drag over an area, or press Return to search the window under the pointer. Escape closes.";

/// The app, for the panel's own Esc (native.rs calls `native_escape` from AppKit, which has no handle).
static APP: OnceLock<AppHandle> = OnceLock::new();

/// A display, in global points with a top-left origin on the primary display.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Display {
    pub id: u32,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub scale: f64,
    /// The notch's height on a display that has one (the page keeps its hint below it).
    pub safe_top: f64,
}

/// The display the pointer is on — or, when it is on none of them, the nearest.
pub fn display_for_point(displays: &[Display], (x, y): (f64, f64)) -> Option<&Display> {
    let inside = |d: &&Display| x >= d.x && x < d.x + d.w && y >= d.y && y < d.y + d.h;
    let dist = |d: &Display| {
        let dx = (d.x - x).max(0.0).max(x - (d.x + d.w));
        let dy = (d.y - y).max(0.0).max(y - (d.y + d.h));
        let v = dx * dx + dy * dy;
        if v.is_finite() { v } else { f64::MAX }
    };
    displays.iter().find(inside).or_else(|| displays.iter().min_by(|a, b| dist(a).total_cmp(&dist(b))))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Hidden,
    Warm,
    Preview,
    /// Held long enough (or opening from the menu, the voice, a host): the ticket is minted.
    Armed,
    Open,
    Closing,
}

/// What opened it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Via {
    #[default]
    Hold,
    Menu,
    /// The core asked: the voice, the agent's tool, or a host (the core doesn't say which).
    Voice,
}

impl Via {
    fn as_str(self) -> &'static str {
        match self {
            Via::Hold => "hold",
            Via::Menu => "menu",
            Via::Voice => "voice",
        }
    }
}

/// What the core last said of Search Screen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Known {
    /// No core of this app's has said hello (it is starting, or gone).
    #[default]
    Unknown,
    /// A core from before Search Screen.
    Old,
    Off,
    On,
}

/// The menu's "Search Screen…" item: its text, and whether it can be chosen.
pub fn menu_item(chord: bool, known: Known) -> (&'static str, bool) {
    match known {
        Known::Off => ("Search Screen is off", false),
        Known::Old => ("Search Screen…", false),
        // Not known yet: it opens, and says NEO is starting.
        Known::On | Known::Unknown => (if chord { "Search Screen…   hold ⌥⌘" } else { "Search Screen…" }, true),
    }
}

/// Search Screen's state in the shell (managed by lib.rs), and the hold's gate (chord.rs reads it).
#[derive(Default)]
pub struct Lens {
    st: Mutex<St>,
    pub gate: Arc<LensGate>,
}

#[derive(Default)]
struct St {
    /// This shell's session number (the page's `g`): one more for each hold that warms, each open.
    sid: u64,
    /// The hold gesture (chord.rs's number) this session follows.
    hold: Option<u64>,
    phase: Phase,
    via: Via,
    ticket: Option<String>,
    /// The core heard `lens_freeze` for the ticket.
    told: bool,
    display: Option<Display>,
    /// The window this shell made (0: none).
    win: u64,
    made: u64,
    /// Its page has painted once (it is ordered front only then: no flash before it draws).
    painted: bool,
    /// It is the non-activating panel (else the fallback, which activates NEO).
    panel: bool,
    to_arm_ms: u32,
    since: Option<Instant>,
    why: &'static str,
    known: Known,
    /// Open: when the page last said it is alive (lens_painted).
    alive: Option<Instant>,
    /// Where the pointer was on the display (display-local points): Return picks the window under it.
    pointer: Option<(f64, f64)>,
}

fn ns(w: &WebviewWindow) -> Option<*mut AnyObject> {
    w.ns_window().ok().map(|p| p as *mut AnyObject).filter(|p| !p.is_null())
}

fn lock(app: &AppHandle) -> Option<std::sync::MutexGuard<'_, St>> {
    app.state::<Lens>().inner().st.lock().ok()
}

/// The overlay is open (a summon closes it rather than showing NEO).
pub fn is_open(app: &AppHandle) -> bool {
    lock(app).map(|s| s.phase == Phase::Open).unwrap_or(false)
}

/// Search Screen is anywhere between warming and fading out (esc.rs: an Esc then is its own). With
/// its state unreadable it counts as up: the careful answer.
pub fn up(app: &AppHandle) -> bool {
    lock(app).map(|s| s.phase != Phase::Hidden).unwrap_or(true)
}

/// What the core said of Search Screen (for the menu's item).
pub fn known(app: &AppHandle) -> Known {
    lock(app).map(|s| s.known).unwrap_or_default()
}

fn ours(app: &AppHandle) -> bool {
    lock(app).map(|s| s.win != 0).unwrap_or(false)
}

/// Whether a window may use Search Screen's commands: it is labelled `lens`, this shell created it,
/// and it shows NEO's own page at #/lens (as keys.rs's `may_get_ticket`).
pub fn may_use_lens(label: &str, ours: bool, url: Option<&Url>, dev: bool) -> bool {
    let Some(url) = url else { return false };
    label == LABEL && ours && keys::app_origin(url, dev) && url.fragment() == Some(ROUTE)
}

fn check(app: &AppHandle, window: &WebviewWindow) -> Result<(), String> {
    let url = window.url().ok();
    if may_use_lens(window.label(), ours(app), url.as_ref(), cfg!(debug_assertions)) {
        Ok(())
    } else {
        core::log("a window used a Search Screen command and was refused");
        Err("forbidden".into())
    }
}

fn phase_name(p: Phase) -> &'static str {
    match p {
        Phase::Hidden => "hidden",
        Phase::Warm => "warm",
        Phase::Preview => "preview",
        Phase::Armed => "armed",
        Phase::Open => "open",
        Phase::Closing => "closing",
    }
}

/// What the page needs besides the phase: the Mac's accessibility settings and the display.
fn extras(s: &St) -> serde_json::Value {
    let (motion, transparency, contrast, voiceover) = native::a11y_flags();
    serde_json::json!({
        "g": s.sid,
        "via": s.via.as_str(),
        "to_arm_ms": s.to_arm_ms,
        "why": s.why,
        "reduce_motion": motion,
        "reduce_transparency": transparency,
        "increase_contrast": contrast,
        "voiceover": voiceover,
        "safe_top": s.display.map(|d| d.safe_top).unwrap_or(0.0),
        "display": s.display.map(|d| serde_json::json!({"id": d.id, "x": d.x, "y": d.y, "w": d.w, "h": d.h, "scale": d.scale})),
        "glow": glow::mode(),
        // The non-activating panel (false: the fallback window, which activates NEO and gives the keys
        // back to the user's app as it goes — the page never hands an area to the notch from it).
        "panel": s.panel,
        "pointer": s.pointer.map(|(x, y)| serde_json::json!({"x": x, "y": y})),
    })
}

/// The page hears a phase (`warm`, `preview`, `armed`, `open`, `cancel`, `closing`, `gone`).
fn emit(app: &AppHandle, s: &St, phase: &str) {
    let mut v = extras(s);
    v["phase"] = phase.into();
    let _ = app.emit_to(LABEL, EVENT, v);
}

fn tell_freeze(app: &AppHandle, ticket: &str, d: &Display, via: Via) -> bool {
    app.state::<core::Core>().tell(&serde_json::json!({
        "t": "lens_freeze",
        "ticket": ticket,
        "display": {"id": d.id, "x": d.x, "y": d.y, "w": d.w, "h": d.h, "scale": d.scale},
        "via": via.as_str(),
    }))
}

fn tell_closed(app: &AppHandle, ticket: &str, why: &str) {
    let why = WHYS.iter().find(|w| **w == why).copied().unwrap_or("done");
    app.state::<core::Core>().tell(&serde_json::json!({"t": "lens_closed", "ticket": ticket, "why": why}));
}

/// The display under the pointer, and where the pointer is on it (display-local points, inside it).
/// Main thread.
fn display_now() -> Option<(Display, (f64, f64))> {
    let (displays, pointer) = unsafe { native::displays_and_pointer() };
    display_for_point(&displays, pointer).copied().map(|d| (d, local_point(&d, pointer)))
}

/// A global point as a point on the display `d`, kept inside it.
pub fn local_point(d: &Display, (x, y): (f64, f64)) -> (f64, f64) {
    let inside = |v: f64, size: f64| if v.is_finite() { v.clamp(0.0, size.max(0.0)) } else { size / 2.0 };
    (inside(x - d.x, d.w), inside(y - d.y, d.h))
}

fn place(w: &WebviewWindow, d: &Display) {
    if let Some(ns) = ns(w) {
        let _ = unsafe { frame::apply(ns, d.x, d.y, d.w, d.h) };
    }
    glow::resize(w.app_handle(), d);
}

/// The overlay's window, made if there is none (main thread). None: it can't be made now.
fn window(app: &AppHandle) -> Option<WebviewWindow> {
    if let Some(w) = app.get_webview_window(LABEL) {
        if ours(app) {
            return Some(w);
        }
        // Not this shell's: it goes, and the next open makes the real one.
        core::log("a Search Screen window the shell didn't create was closed");
        let _ = w.destroy();
        return None;
    }
    let id = {
        let mut s = lock(app)?;
        s.made += 1;
        s.win = s.made;
        s.painted = false;
        s.panel = false;
        s.made
    };
    let _ = APP.set(app.clone());
    let dev = cfg!(debug_assertions);
    let built = WebviewWindowBuilder::new(app, LABEL, WebviewUrl::App(PAGE.into()))
        .title("NEO Search Screen")
        .inner_size(640.0, 400.0)
        .transparent(true)
        .decorations(false)
        .shadow(false)
        .resizable(false)
        .skip_taskbar(true)
        .always_on_top(true)
        .visible_on_all_workspaces(true)
        .content_protected(true) // never in other apps' screenshots, recordings or screen shares
        .accept_first_mouse(true)
        .focused(false)
        .visible(false)
        .background_color(Color(0, 0, 0, 0))
        .on_navigation(move |url| keys::may_navigate(url, dev))
        .on_new_window(|_, _| NewWindowResponse::Deny)
        .build();
    let w = match built {
        Ok(w) => w,
        Err(e) => {
            if let Some(mut s) = lock(app) {
                if s.win == id {
                    s.win = 0;
                }
            }
            core::log(&format!("Search Screen's window couldn't be made: {e}"));
            return None;
        }
    };
    let handle = app.clone();
    w.on_window_event(move |e| {
        if let WindowEvent::Destroyed = e {
            destroyed(&handle, id);
        }
    });
    let (panel, glow) = match ns(&w) {
        Some(ns) => unsafe {
            let panel = native::lens_make_panel(ns);
            native::lens_setup(ns);
            (panel, glow::attach(ns))
        },
        None => (false, false),
    };
    if !panel {
        core::log("lens: no panel class — activating instead");
    }
    if glow {
        // Its strips baked now, for the display the pointer is on (the first hold finds them ready).
        if let Some((d, _)) = display_now() {
            glow::resize(app, &d);
        }
    } else {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, Ordering::SeqCst) {
            core::log("lens: no native glow — the page draws it");
        }
    }
    if let Some(mut s) = lock(app) {
        s.panel = panel;
    }
    Some(w)
}

/// The window went (destroyed after a while unused, or by someone else): if it was in use, the
/// session ends with it.
fn destroyed(app: &AppHandle, id: u64) {
    let ticket = {
        let Some(mut s) = lock(app) else { return };
        if s.win != id {
            return;
        }
        s.win = 0;
        s.painted = false;
        s.panel = false;
        let was = s.phase != Phase::Hidden;
        s.phase = Phase::Hidden;
        s.hold = None;
        if was { s.ticket.take() } else { None }
    };
    app.state::<Lens>().gate.open.store(false, Ordering::SeqCst);
    native::lens_not_keyable();
    glow::detach(app);
    if let Some(t) = ticket {
        tell_closed(app, &t, "gone");
    }
}

/// A step of the hold (chord.rs), on the main thread.
pub fn on_hold(app: &AppHandle, ev: HoldEvent) {
    match ev {
        HoldEvent::Warm(g) => warm(app, g),
        HoldEvent::Preview { g, to_arm_ms } => preview(app, g, to_arm_ms),
        HoldEvent::Arm(g) => arm(app, g),
        HoldEvent::Commit(g) => commit(app, g),
        HoldEvent::Cancel(g, why) => cancel(app, Some(g), why.as_str()),
    }
}

/// A session still fading out ends now (a new one is starting).
fn end_fade(app: &AppHandle) {
    let closing = lock(app).map(|s| s.phase == Phase::Closing).unwrap_or(false);
    if closing {
        if let Some(mut s) = lock(app) {
            s.phase = Phase::Hidden;
        }
        if let Some(ns) = app.get_webview_window(LABEL).as_ref().and_then(ns) {
            unsafe { native::lens_order_out(ns) };
        }
        glow::hide(app);
    }
}

/// The window, ready on the display under the pointer: click-through, and on screen once its page
/// has painted (main thread).
fn ready(w: &WebviewWindow, d: Option<Display>, painted: bool) {
    if let Some(d) = d {
        place(w, &d);
    }
    if let Some(ns) = ns(w) {
        unsafe {
            native::lens_set_click_through(ns, true);
            if painted {
                native::lens_order_front(ns);
            }
        }
    }
}

/// This shell's own overlay window, if it has been made (never made here).
fn made(app: &AppHandle) -> Option<WebviewWindow> {
    app.get_webview_window(LABEL).filter(|_| ours(app))
}

fn warm(app: &AppHandle, g: u64) {
    if lock(app).map(|s| s.phase != Phase::Hidden && s.phase != Phase::Closing).unwrap_or(true) {
        return; // open (the gate keeps a hold from warming then), or another session is starting
    }
    end_fade(app);
    // The window was made ahead (`prepare_soon`). Never made here: a slow tap let go of now still
    // summons NEO, and that summon must not wait behind a web view being built. (None yet: Preview.)
    let w = made(app);
    let d = display_now();
    let painted = {
        let Some(mut s) = lock(app) else { return };
        s.sid += 1;
        s.hold = Some(g);
        s.phase = Phase::Warm;
        s.via = Via::Hold;
        s.ticket = None;
        s.told = false;
        s.display = d.map(|x| x.0);
        s.pointer = d.map(|x| x.1);
        s.to_arm_ms = 0;
        s.why = "";
        s.since = Some(Instant::now());
        s.alive = None;
        emit(app, &s, "warm");
        s.painted
    };
    if let Some(w) = w {
        ready(&w, d.map(|x| x.0), painted);
    }
    glow::phase(app, PhaseIn::of(GlowPhase::Warm));
}

fn preview(app: &AppHandle, g: u64, to_arm_ms: u32) {
    let display = {
        let Some(mut s) = lock(app) else { return };
        if s.hold != Some(g) || s.phase != Phase::Warm {
            return;
        }
        s.phase = Phase::Preview;
        s.to_arm_ms = to_arm_ms;
        s.since = Some(Instant::now());
        emit(app, &s, "preview");
        s.display
    };
    if made(app).is_none() {
        // Not made ahead (the core has only just said hello): made now. Its page catches up with
        // lens_hello (the phase, and how long it has been on).
        if let Some(w) = window(app) {
            ready(&w, display, false);
        }
    }
    glow::phase(app, PhaseIn { phase: GlowPhase::Preview, to_arm_ms: to_arm_ms as f64, via_hold: true, notch: false });
}

fn arm(app: &AppHandle, g: u64) {
    let current = lock(app).map(|s| s.hold == Some(g) && s.phase == Phase::Preview).unwrap_or(false);
    if !current {
        return;
    }
    // The pointer may have moved to another display while the keys were held.
    let Some((d, pointer)) = display_now() else {
        cancel(app, Some(g), "paused");
        return;
    };
    let Some(ticket) = keys::mint() else {
        core::log("Search Screen didn't arm: no randomness for its ticket");
        cancel(app, Some(g), "paused");
        return;
    };
    let moved = {
        let Some(mut s) = lock(app) else { return };
        let moved = s.display != Some(d);
        s.display = Some(d);
        s.pointer = Some(pointer);
        s.told = tell_freeze(app, &ticket, &d, Via::Hold);
        s.ticket = Some(ticket);
        s.phase = Phase::Armed;
        s.since = Some(Instant::now());
        emit(app, &s, "armed");
        moved
    };
    if moved {
        if let Some(w) = app.get_webview_window(LABEL) {
            place(&w, &d);
        }
    }
    glow::phase(app, PhaseIn::of(GlowPhase::Armed));
    unsafe { native::lens_haptic() };
}

fn commit(app: &AppHandle, g: u64) {
    let current = lock(app).map(|s| s.hold == Some(g) && s.phase == Phase::Armed).unwrap_or(false);
    if current {
        show_open(app);
    }
}

/// Armed → open: the overlay takes the keys, the core starts smart pick's regions.
fn show_open(app: &AppHandle) {
    let Some(w) = app.get_webview_window(LABEL).filter(|_| ours(app)) else {
        cancel(app, None, "gone");
        return;
    };
    let (sid, panel, ticket, via) = {
        let Some(mut s) = lock(app) else { return };
        s.phase = Phase::Open;
        s.since = Some(Instant::now());
        s.alive = Some(Instant::now()); // (the page's first "alive" is due within PAGE_SILENT)
        (s.sid, s.panel, s.ticket.clone(), s.via)
    };
    glow::phase(app, PhaseIn { phase: GlowPhase::Open, to_arm_ms: 0.0, via_hold: via == Via::Hold, notch: false });
    app.state::<Lens>().gate.open.store(true, Ordering::SeqCst);
    if let Some(ns) = ns(&w) {
        unsafe { native::lens_make_key(ns, panel) };
    }
    if let Some(t) = ticket {
        app.state::<core::Core>().tell(&serde_json::json!({"t": "lens_commit", "ticket": t}));
    }
    let (_, _, _, voiceover) = native::a11y_flags();
    if voiceover {
        unsafe { native::lens_announce(ANNOUNCE) };
    }
    if let Some(s) = lock(app) {
        emit(app, &s, "open");
    }
    watch(app.clone(), sid);
}

/// A hold (or an opening) called off before it opened: the glow retracts, the window orders out.
fn cancel(app: &AppHandle, g: Option<u64>, why: &'static str) {
    let (sid, ticket) = {
        let Some(mut s) = lock(app) else { return };
        if g.is_some() && s.hold != g {
            return;
        }
        if !matches!(s.phase, Phase::Warm | Phase::Preview | Phase::Armed) {
            return;
        }
        s.phase = Phase::Hidden;
        s.why = why;
        emit(app, &s, "cancel");
        (s.sid, s.ticket.take())
    };
    glow::phase(app, PhaseIn::of(GlowPhase::Cancel));
    if let Some(t) = ticket {
        tell_closed(app, &t, "cancel");
    }
    after_fade(app, sid, FADE);
}

/// Search Screen closes (Esc, a summon, the core, the watchdog…). On the main thread.
pub fn close(app: &AppHandle, why: &str) {
    let why: &'static str = WHYS.iter().find(|w| **w == why).copied().unwrap_or("done");
    let phase = lock(app).map(|s| s.phase).unwrap_or_default();
    match phase {
        Phase::Hidden | Phase::Closing => {}
        Phase::Warm | Phase::Preview | Phase::Armed => cancel(app, None, why),
        Phase::Open => {
            let (sid, ticket) = {
                let Some(mut s) = lock(app) else { return };
                s.phase = Phase::Closing;
                s.why = why;
                s.since = Some(Instant::now());
                emit(app, &s, "closing");
                (s.sid, s.ticket.take())
            };
            glow::phase(app, PhaseIn { phase: GlowPhase::Closing, to_arm_ms: 0.0, via_hold: true, notch: why == "notch" });
            app.state::<Lens>().gate.open.store(false, Ordering::SeqCst);
            native::lens_not_keyable();
            if let Some(ns) = app.get_webview_window(LABEL).as_ref().and_then(ns) {
                unsafe { native::lens_set_click_through(ns, true) };
            }
            if let Some(t) = ticket {
                tell_closed(app, &t, why);
            }
            after_fade(app, sid, if why == "notch" { FADE_NOTCH } else { FADE });
        }
    }
}

/// Once the page has faded (`fade`: FADE, or FADE_NOTCH for the flight into the notch): the window
/// orders out (keys back to the user's app), and is destroyed a minute later if nothing used it again.
fn after_fade(app: &AppHandle, sid: u64, fade: Duration) {
    let handle = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(fade);
        let h = handle.clone();
        let _ = handle.run_on_main_thread(move || {
            let gone = match lock(&h) {
                Some(mut s) if s.sid == sid && matches!(s.phase, Phase::Hidden | Phase::Closing) => {
                    s.phase = Phase::Hidden;
                    true
                }
                _ => false,
            };
            if gone {
                if let Some(ns) = h.get_webview_window(LABEL).as_ref().and_then(ns) {
                    unsafe { native::lens_order_out(ns) };
                }
                glow::hide(&h);
            }
        });
        std::thread::sleep(DESTROY_AFTER);
        let h = handle.clone();
        let _ = handle.run_on_main_thread(move || {
            let unused = lock(&h).map(|s| s.sid == sid && s.phase == Phase::Hidden).unwrap_or(false);
            // While Search Screen is on the window stays, ready for the next hold (prepare_soon).
            let wanted = h.state::<Lens>().gate.on.load(Ordering::SeqCst);
            if unused && !wanted {
                if let Some(w) = h.get_webview_window(LABEL) {
                    let _ = w.destroy();
                }
            }
        });
    });
}

/// "Search Screen…" (the menu), the voice, the agent's tool, a host. On the main thread.
pub fn open(app: &AppHandle, via: Via) {
    let phase = lock(app).map(|s| s.phase).unwrap_or_default();
    match phase {
        Phase::Open => {
            // Already open: it keeps the keys.
            let panel = lock(app).map(|s| s.panel).unwrap_or(false);
            if let Some(ns) = app.get_webview_window(LABEL).as_ref().and_then(ns) {
                unsafe { native::lens_make_key(ns, panel) };
            }
            return;
        }
        Phase::Warm | Phase::Preview | Phase::Armed => return, // a hold (or an opening) is under way
        Phase::Closing => end_fade(app),
        Phase::Hidden => {}
    }
    let Some(w) = window(app) else { return };
    let dp = display_now();
    let d = dp.map(|x| x.0);
    let sid = {
        let Some(mut s) = lock(app) else { return };
        s.sid += 1;
        s.hold = None;
        s.phase = Phase::Armed;
        s.via = via;
        s.ticket = None;
        s.told = false;
        s.display = d;
        s.pointer = dp.map(|x| x.1);
        s.alive = None;
        s.to_arm_ms = 0;
        s.why = "";
        s.since = Some(Instant::now());
        s.sid
    };
    if let Some(d) = d {
        place(&w, &d);
    }
    if let Some(ns) = ns(&w) {
        unsafe { native::lens_set_click_through(ns, true) };
    }
    let wait = if via == Via::Menu { OPEN_WAIT_MENU } else { OPEN_WAIT };
    let handle = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(wait);
        let h = handle.clone();
        let _ = handle.run_on_main_thread(move || open_now(&h, sid));
    });
}

fn open_now(app: &AppHandle, sid: u64) {
    let (display, via) = match lock(app) {
        Some(s) if s.sid == sid && s.phase == Phase::Armed && s.hold.is_none() => (s.display, s.via),
        _ => return,
    };
    let Some(d) = display.or_else(|| display_now().map(|x| x.0)) else {
        cancel(app, None, "display");
        return;
    };
    let Some(ticket) = keys::mint() else {
        core::log("Search Screen didn't open: no randomness for its ticket");
        cancel(app, None, "cancel");
        return;
    };
    if let Some(mut s) = lock(app) {
        s.display = Some(d);
        s.told = tell_freeze(app, &ticket, &d, via);
        s.ticket = Some(ticket);
    }
    show_open(app);
}

/// Whether an open overlay's page has been silent too long (it says it is alive every second).
pub fn page_silent(alive: Option<Instant>, now: Instant) -> bool {
    alive.map(|t| now.saturating_duration_since(t) > PAGE_SILENT).unwrap_or(false)
}

/// While it is open: the screen locking, its display changing, or its page falling silent closes it.
fn watch(app: AppHandle, sid: u64) {
    std::thread::spawn(move || loop {
        std::thread::sleep(WATCH_EVERY);
        let (display, alive) = match lock(&app) {
            Some(s) if s.sid == sid && s.phase == Phase::Open => (s.display, s.alive),
            _ => return,
        };
        let why = if native::screen_locked() {
            Some("lock")
        } else if display.map(|d| !display_still_there(&d)).unwrap_or(false) {
            Some("display")
        } else if page_silent(alive, Instant::now()) {
            core::log("lens: the overlay's page went silent — closed");
            Some("gone")
        } else {
            None
        };
        if let Some(why) = why {
            let h = app.clone();
            let _ = app.run_on_main_thread(move || {
                if lock(&h).map(|s| s.sid == sid).unwrap_or(false) {
                    close(&h, why);
                }
            });
            return;
        }
    });
}

/// The display is still attached, where it was and the size it was (CoreGraphics: any thread).
fn display_still_there(d: &Display) -> bool {
    use core_graphics::display::CGDisplay;
    let Ok(ids) = CGDisplay::active_displays() else { return true };
    if !ids.contains(&d.id) {
        return d.id == 0; // (a display NSScreen gave no number: nothing to compare)
    }
    let b = CGDisplay::new(d.id).bounds();
    let near = |a: f64, b: f64| (a - b).abs() <= 1.0;
    near(b.origin.x, d.x) && near(b.origin.y, d.y) && near(b.size.width, d.w) && near(b.size.height, d.h)
}

/// The core said hello: whether Search Screen is on, and how long the hold is. A session the
/// old core knew of is over (the new one never heard its ticket). On the main thread.
pub fn core_hello(app: &AppHandle, cfg: Option<core::LensCfg>) {
    let gate = app.state::<Lens>().gate.clone();
    let on = cfg.map(|c| c.on).unwrap_or(false);
    gate.on.store(on, Ordering::SeqCst);
    if on {
        prepare_soon(app);
    } else if lock(app).map(|s| s.phase == Phase::Hidden).unwrap_or(false) {
        // Off (or a core from before Search Screen): the window made ahead goes, its web process with it.
        if let Some(w) = made(app) {
            let _ = w.destroy();
        }
    }
    if let Some(c) = cfg {
        gate.hold_ms.store(c.hold_ms, Ordering::SeqCst);
    }
    let stale = {
        let Some(mut s) = lock(app) else { return };
        s.known = match cfg {
            None => Known::Old,
            Some(c) if c.on => Known::On,
            Some(_) => Known::Off,
        };
        s.phase != Phase::Hidden && s.ticket.is_some()
    };
    if stale {
        close(app, "gone");
    }
}

/// Search Screen is on: its window is made a moment from now, hidden — its page loads and waits — so
/// the first hold never builds one (see the top).
fn prepare_soon(app: &AppHandle) {
    let handle = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(PREPARE_AFTER);
        let h = handle.clone();
        let _ = handle.run_on_main_thread(move || {
            let idle = lock(&h).map(|s| s.phase == Phase::Hidden).unwrap_or(false);
            if idle && h.state::<Lens>().gate.on.load(Ordering::SeqCst) && h.get_webview_window(LABEL).is_none() {
                let _ = window(&h);
            }
        });
    });
}

/// The panel's own Esc (native.rs): a key the page didn't take — its script died or is stuck —
/// still closes an open overlay. On the main thread (AppKit's keyDown:).
pub fn native_escape() {
    if let Some(app) = APP.get() {
        if is_open(app) {
            core::log("lens: Esc reached the panel itself — closed");
            close(app, "esc");
        }
    }
}

/// The core is gone: no hold until another says hello; an open overlay says so, then closes.
pub fn core_gone(app: &AppHandle) {
    app.state::<Lens>().gate.on.store(false, Ordering::SeqCst);
    let (phase, sid) = {
        let Some(mut s) = lock(app) else { return };
        s.known = Known::Unknown;
        if s.phase == Phase::Open {
            emit(app, &s, "gone");
        }
        (s.phase, s.sid)
    };
    match phase {
        Phase::Open => {
            let handle = app.clone();
            std::thread::spawn(move || {
                std::thread::sleep(GONE_CLOSE);
                let h = handle.clone();
                let _ = handle.run_on_main_thread(move || {
                    if lock(&h).map(|s| s.sid == sid).unwrap_or(false) {
                        close(&h, "gone");
                    }
                });
            });
        }
        Phase::Warm | Phase::Preview | Phase::Armed => cancel(app, None, "gone"),
        _ => {}
    }
}

/// Every page load, in every window: a `lens` window this shell didn't create is closed.
pub fn page_loaded<R: Runtime>(webview: &tauri::Webview<R>) {
    if webview.label() != LABEL {
        return;
    }
    let mine = webview.app_handle().state::<Lens>().st.lock().map(|s| s.win != 0).unwrap_or(false);
    if !mine {
        core::log("a Search Screen window the shell didn't create was closed");
        let _ = webview.window().destroy();
    }
}

/// The page, on load and at each phase: where things stand, and the ticket once there is one.
/// `core`: the core heard the ticket (false: none of this app's is listening — "NEO is starting").
#[tauri::command]
pub fn lens_hello(app: AppHandle, window: WebviewWindow) -> Result<serde_json::Value, String> {
    check(&app, &window)?;
    let s = lock(&app).ok_or("forbidden")?;
    let mut v = extras(&s);
    v["phase"] = phase_name(s.phase).into();
    v["ticket"] = s.ticket.clone().into();
    v["core"] = (s.told && app.state::<core::Core>().linked()).into();
    v["sc"] = perms::screen_recording().into();
    v["since_ms"] = s.since.map(|t| t.elapsed().as_millis() as u64).unwrap_or(0).into();
    v["hold_ms"] = app.state::<Lens>().gate.hold_ms.load(Ordering::SeqCst).into();
    Ok(v)
}

/// The page painted (it draws nothing until a phase asks): a window that is warming can come up.
/// Open, the page says it every second: it is alive (the watchdog closes a silent one).
#[tauri::command]
pub fn lens_painted(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    check(&app, &window)?;
    let phase = {
        let mut s = lock(&app).ok_or("forbidden")?;
        s.painted = true;
        if s.phase == Phase::Open {
            s.alive = Some(Instant::now());
        }
        s.phase
    };
    if matches!(phase, Phase::Warm | Phase::Preview | Phase::Armed) {
        if let Some(ns) = ns(&window) {
            unsafe { native::lens_order_front(ns) };
        }
    }
    Ok(())
}

/// Esc, ×, right-click, Not now, an action that opened something, the page's idle timer.
#[tauri::command]
pub fn lens_close(app: AppHandle, window: WebviewWindow, why: String) -> Result<(), String> {
    check(&app, &window)?;
    close(&app, &why);
    Ok(())
}

/// Live mode only (no still): the overlay steps out of the way while the core captures the area.
#[tauri::command]
pub fn lens_peek(app: AppHandle, window: WebviewWindow, hidden: bool) -> Result<(), String> {
    check(&app, &window)?;
    let (open, panel) = lock(&app).map(|s| (s.phase == Phase::Open, s.panel)).ok_or("forbidden")?;
    if !open {
        return Err("closed".into());
    }
    if let Some(ns) = ns(&window) {
        unsafe { native::lens_peek(ns, hidden, panel) };
    }
    Ok(())
}

/// The page selected an area (true) or let go of it (false): the glow narrows and dims, or comes back.
/// Only while open; at any other moment it changes nothing.
#[tauri::command]
pub fn lens_glow(app: AppHandle, window: WebviewWindow, selected: bool) -> Result<(), String> {
    check(&app, &window)?;
    if lock(&app).map(|s| s.phase == Phase::Open).unwrap_or(false) {
        glow::selected(&app, selected);
    }
    Ok(())
}

/// The permission card's "Open Privacy Settings": System Settings at Screen Recording, and close.
#[tauri::command]
pub fn lens_allow_sc(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    check(&app, &window)?;
    perms::open_settings("Privacy_ScreenCapture");
    close(&app, "action");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    fn d(id: u32, x: f64, y: f64, w: f64, h: f64, scale: f64) -> Display {
        Display { id, x, y, w, h, scale, safe_top: 0.0 }
    }

    #[test]
    fn the_display_is_the_one_under_the_pointer_or_the_nearest() {
        // A 1512×982 laptop (2×) with a 2560×1440 display (1×) to its right, top-aligned.
        let ds = [d(1, 0.0, 0.0, 1512.0, 982.0, 2.0), d(2, 1512.0, 0.0, 2560.0, 1440.0, 1.0)];
        assert_eq!(display_for_point(&ds, (100.0, 100.0)).map(|d| d.id), Some(1));
        assert_eq!(display_for_point(&ds, (1511.9, 981.0)).map(|d| d.id), Some(1));
        assert_eq!(display_for_point(&ds, (1512.0, 0.0)).map(|d| d.id), Some(2)); // the edge belongs to the right one
        assert_eq!(display_for_point(&ds, (3000.0, 1400.0)).map(|d| d.id), Some(2));
        // Below the laptop, beside nothing: the nearest.
        assert_eq!(display_for_point(&ds, (700.0, 1200.0)).map(|d| d.id), Some(1));
        assert_eq!(display_for_point(&ds, (2000.0, 1500.0)).map(|d| d.id), Some(2));
        // A display above the primary one (negative y), and one to its left (negative x).
        let ds = [d(1, 0.0, 0.0, 1440.0, 900.0, 2.0), d(3, -200.0, -1080.0, 1920.0, 1080.0, 1.0), d(4, -1920.0, 0.0, 1920.0, 1080.0, 1.0)];
        assert_eq!(display_for_point(&ds, (10.0, -10.0)).map(|d| d.id), Some(3));
        assert_eq!(display_for_point(&ds, (-10.0, 10.0)).map(|d| d.id), Some(4));
        assert!(display_for_point(&ds, (f64::NAN, 10.0)).is_some());
        assert_eq!(display_for_point(&[], (0.0, 0.0)), None);
    }

    #[test]
    fn only_the_shells_own_lens_window_at_its_route_uses_its_commands() {
        let page = u("tauri://localhost/index.html#/lens");
        assert!(may_use_lens("lens", true, Some(&page), false));
        assert!(may_use_lens("lens", true, Some(&u("http://tauri.localhost/index.html#/lens")), false));
        assert!(may_use_lens("lens", true, Some(&u("https://tauri.localhost/index.html#/lens")), false));
        assert!(!may_use_lens("lens", false, Some(&page), false));
        for label in ["main", "keys", "browser", "camera", "lens2"] {
            assert!(!may_use_lens(label, true, Some(&page), false), "{label}");
        }
        for route in ["", "#/keys", "#/lens/x", "#/browser"] {
            assert!(!may_use_lens("lens", true, Some(&u(&format!("tauri://localhost/index.html{route}"))), false), "{route}");
        }
        assert!(!may_use_lens("lens", true, None, false));
        for bad in [
            "https://example.com/#/lens",
            "https://tauri.localhost.example.com/#/lens",
            "tauri://evil/index.html#/lens",
            "file:///tmp/index.html#/lens",
            "http://localhost:1420/index.html#/lens",
        ] {
            assert!(!may_use_lens("lens", true, Some(&u(bad)), false), "{bad}");
        }
        // A `tauri dev` build's page comes from its dev server (port 1420 only).
        assert!(may_use_lens("lens", true, Some(&u("http://localhost:1420/index.html#/lens")), true));
        assert!(!may_use_lens("lens", true, Some(&u("http://localhost:1421/index.html#/lens")), true));
        assert_eq!(PAGE, format!("index.html#{ROUTE}"));
    }

    #[test]
    fn a_ticket_is_64_lowercase_hex_characters() {
        let t = keys::mint().unwrap();
        assert_eq!(t.len(), 64);
        assert!(t.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)), "{t}");
    }

    #[test]
    fn the_menus_item_follows_the_core_and_the_tap() {
        assert_eq!(menu_item(true, Known::On), ("Search Screen…   hold ⌥⌘", true));
        assert_eq!(menu_item(false, Known::On), ("Search Screen…", true)); // no Accessibility: no hold
        assert_eq!(menu_item(true, Known::Unknown), ("Search Screen…   hold ⌥⌘", true)); // it says NEO is starting
        assert_eq!(menu_item(true, Known::Off), ("Search Screen is off", false));
        assert_eq!(menu_item(true, Known::Old), ("Search Screen…", false));
    }

    #[test]
    fn the_window_is_a_panel_above_everything_and_kept_out_of_screenshots() {
        assert_eq!(native::NONACTIVATING_PANEL, 128);
        assert_eq!(native::LENS_SPACES, 262_481);
        assert_eq!(native::LENS_LEVEL, 1000);
        assert!(native::panel_fits(200, 200) && native::panel_fits(208, 200));
        assert!(!native::panel_fits(200, 208) && !native::panel_fits(200, 0));
        let src = include_str!("lens.rs");
        let build = src.split("WebviewWindowBuilder::new(").nth(1).unwrap().split(".build()").next().unwrap();
        for part in [".content_protected(true)", ".transparent(true)", ".focused(false)", ".visible(false)", "may_navigate", "NewWindowResponse::Deny"] {
            assert!(build.contains(part), "{part}");
        }
        // After the panel swap tao's own focus calls would touch memory the panel doesn't have.
        let body = src.split("#[cfg(test)]").next().unwrap();
        for call in [concat!(".set_", "focus("), concat!(".set_", "focusable(")] {
            assert!(!body.contains(call), "{call}");
        }
        // The ticket never rides an event: only the label-checked command hands it out.
        let emit = body.split("fn emit(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(!emit.contains("ticket"));
        let extras = body.split("fn extras(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(!extras.contains("ticket"));
        // The page hears whether it is the non-activating panel (the fallback never hands off).
        assert!(extras.contains("\"panel\": s.panel"));
    }

    #[test]
    fn the_pointer_is_a_point_on_its_display() {
        let laptop = d(1, 0.0, 0.0, 1512.0, 982.0, 2.0);
        let right = d(2, 1512.0, -200.0, 2560.0, 1440.0, 1.0);
        assert_eq!(local_point(&laptop, (100.0, 50.0)), (100.0, 50.0));
        assert_eq!(local_point(&right, (1600.0, -100.0)), (88.0, 100.0));
        assert_eq!(local_point(&laptop, (2000.0, 1200.0)), (1512.0, 982.0)); // (on no display: kept inside)
        assert_eq!(local_point(&laptop, (f64::NAN, -5.0)), (756.0, 0.0));
    }

    #[test]
    fn a_silent_page_is_closed_and_no_web_view_is_built_at_warm() {
        let now = Instant::now();
        assert!(!page_silent(None, now));
        assert!(!page_silent(Some(now), now + Duration::from_secs(4)));
        assert!(page_silent(Some(now), now + Duration::from_secs(6)));
        let src = include_str!("lens.rs");
        let body = src.split("#[cfg(test)]").next().unwrap();
        let warm = body.split("fn warm(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(!warm.contains("window(app)"), "warm must never build the window");
        let after = body.split("fn after_fade(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(after.contains("gate.on.load") && after.contains("!wanted"), "kept while Search Screen is on");
        let hello = body.split("fn core_hello(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(hello.contains("prepare_soon(app)") && hello.contains("w.destroy()"), "made when on, gone when off");
        let painted = body.split("fn lens_painted(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(painted.contains("s.alive = Some(Instant::now())"));
    }

    #[test]
    fn the_shell_drives_the_glow_at_every_phase_and_the_page_only_says_selected() {
        let src = include_str!("lens.rs");
        let body = src.split("#[cfg(test)]").next().unwrap();
        let f = |name: &str| body.split(&format!("fn {name}(")).nth(1).unwrap().split("\n}\n").next().unwrap().to_string();
        for (name, call) in [
            ("warm", "GlowPhase::Warm"),
            ("preview", "GlowPhase::Preview"),
            ("arm", "GlowPhase::Armed"),
            ("show_open", "GlowPhase::Open"),
            ("cancel", "GlowPhase::Cancel"),
            ("close", "GlowPhase::Closing"),
        ] {
            assert!(f(name).contains("glow::phase(app,") && f(name).contains(call), "{name}");
        }
        assert!(f("after_fade").contains("glow::hide(&h)"), "nothing ticks once it ordered out");
        assert!(f("end_fade").contains("glow::hide(app)"));
        assert!(f("destroyed").contains("glow::detach(app)"));
        assert!(f("place").contains("glow::resize("));
        assert!(f("window").contains("glow::attach(ns)"));
        // The page's one glow command answers only its own window, and only while open.
        let cmd = f("lens_glow");
        assert!(cmd.starts_with("app: AppHandle, window: WebviewWindow, selected: bool)"));
        assert!(cmd.contains("check(&app, &window)?;") && cmd.contains("Phase::Open") && cmd.contains("glow::selected(&app, selected)"));
        // The phase events say who draws the glow and which display it is on — still never the ticket.
        let extras = f("extras");
        assert!(extras.contains("\"glow\": glow::mode()") && extras.contains("\"id\": d.id"));
        assert!(!extras.contains("ticket"));
    }

    #[test]
    fn the_core_hears_only_the_words_it_knows() {
        assert!(WHYS.contains(&"esc") && WHYS.contains(&"summon") && WHYS.contains(&"lock"));
        assert!(WHYS.contains(&"notch"), "an area handed to the notch");
        let src = include_str!("lens.rs");
        let body = src.split("#[cfg(test)]").next().unwrap();
        let closed = body.split("fn tell_closed(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(closed.contains("unwrap_or(\"done\")"));
        // The page's `close("notch")` keeps its word all the way to the core.
        let close = body.split("pub fn close(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(close.contains("WHYS.iter().find(|w| **w == why)"));
    }

    #[test]
    fn closing_into_the_notch_waits_for_the_flight() {
        assert_eq!(FADE, Duration::from_millis(170));
        assert_eq!(FADE_NOTCH, Duration::from_millis(440));
        let src = include_str!("lens.rs");
        let body = src.split("#[cfg(test)]").next().unwrap();
        let close = body.split("pub fn close(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(close.contains("after_fade(app, sid, if why == \"notch\" { FADE_NOTCH } else { FADE })"));
        assert!(close.contains("notch: why == \"notch\""), "the glow's 200 ms fade");
        let cancel = body.split("fn cancel(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(cancel.contains("after_fade(app, sid, FADE)"));
        let after = body.split("fn after_fade(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(after.contains("std::thread::sleep(fade)"));
    }
}

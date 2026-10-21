//! Brains & Keys: the window where the user picks NEO's brains and pastes API keys
//! (index.html#/keys, ui/src/keys/KeysApp.tsx).
//!
//! The shell never handles a key: the core is the only writer of NEO's key and brain files, and
//! the window talks to the core over its websocket as the one client of kind `keys`
//! (neo2/core/server/ws.py, "Brains & Keys"). What the shell does:
//!
//! - It alone creates the window (`open`, from the menu's "Brains & Keys…" and a host's or the
//!   core's "settings"), kept out of screenshots and NEO's own screen tools (contentProtected),
//!   and it remembers that it did. Any other window labelled `keys` (the overlay may create
//!   webviews) is closed as soon as its page loads (`page_loaded`).
//! - It mints the window's ticket: 32 bytes from /dev/urandom, as 64 lowercase hex characters.
//!   The core hears it over the socket only the two of them share (`{"t":"keys_ticket"}`, again
//!   whenever a core says hello while the window is open) and admits the window with it. When the
//!   window goes, the ticket goes too (`{"t":"keys_closed"}`): the core revokes it.
//! - The page asks for the ticket with the `keys_ticket` command, which answers only that window:
//!   label `keys`, created here, showing NEO's own page at `#/keys` (`may_get_ticket`). Commands
//!   aren't gated by capability (build.rs has no app manifest), so the command checks for itself.
//! - `edit_env_file` opens ~/.neo/.env in TextEdit, from that window only (the menu has its own
//!   "Edit settings file…").

use std::fs::File;
use std::io::Read;
use std::sync::Mutex;
use std::time::Duration;

use tauri::webview::{Color, NewWindowResponse};
use tauri::{AppHandle, LogicalPosition, Manager, Runtime, TitleBarStyle, Url, WebviewUrl, WebviewWindow, WebviewWindowBuilder, WindowEvent};

use crate::core;

/// The window's label, and the route its page shows.
pub const LABEL: &str = "keys";
const ROUTE: &str = "/keys";
const PAGE: &str = "index.html#/keys";
/// A ticket's bytes (64 hex characters on the wire: the core accepts nothing else).
const TICKET_BYTES: usize = 32;
/// The page shows itself once it has painted (no flash of the wrong colour); if it hasn't by
/// then, the shell does.
const SHOW_ANYWAY: Duration = Duration::from_millis(1500);

/// The window this shell created, while it is open, and its ticket.
#[derive(Default)]
pub struct Keys(Mutex<State>);

#[derive(Default)]
struct State {
    open: Option<Open>,
    /// Each window this shell creates gets its own number: an old window's last event doesn't
    /// close a newer one.
    made: u64,
}

struct Open {
    id: u64,
    ticket: String,
}

/// Bytes as lowercase hex.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A fresh ticket, or None when the system's randomness can't be read (then no window opens:
/// a guessable ticket would be no ticket).
pub(crate) fn mint() -> Option<String> {
    let mut bytes = [0u8; TICKET_BYTES];
    File::open("/dev/urandom").ok()?.read_exact(&mut bytes).ok()?;
    if bytes.iter().all(|b| *b == 0) {
        return None;
    }
    Some(hex(&bytes))
}

/// NEO's own page: the bundled app's origin, or (`dev`: a `tauri dev` build) its dev server.
pub(crate) fn app_origin(url: &Url, dev: bool) -> bool {
    let host = url.host_str().unwrap_or("");
    match url.scheme() {
        "tauri" => host == "localhost",
        "http" | "https" if host == "tauri.localhost" => true,
        "http" => dev && (host == "localhost" || host == "127.0.0.1") && url.port() == Some(1420),
        _ => false,
    }
}

/// Whether a window may have the ticket: it is labelled `keys`, this shell created it (`ours`),
/// and it shows NEO's own page at #/keys.
pub fn may_get_ticket(label: &str, ours: bool, url: Option<&Url>, dev: bool) -> bool {
    let Some(url) = url else { return false };
    label == LABEL && ours && app_origin(url, dev) && url.fragment() == Some(ROUTE)
}

/// Where the window may navigate: only NEO's own page (a link opens in the user's browser,
/// through the opener, never in here).
pub fn may_navigate(url: &Url, dev: bool) -> bool {
    app_origin(url, dev)
}

/// Why the page can't have its ticket yet, from what the supervisor says of the core: "no_link"
/// (a core started outside the app: no socket to it, ever — the page shows the terminal
/// commands), "not_ready" (starting or restarting: the page asks again), "no_core" (not
/// installed, crashed, couldn't start).
pub fn ticket_error(core_status: &str) -> &'static str {
    match core_status {
        core::OUTSIDE => "no_link",
        "" | "starting" | "running" | "restarting" | "not responding — restarting" => "not_ready",
        _ => "no_core",
    }
}

fn ours(app: &AppHandle) -> bool {
    app.state::<Keys>().0.lock().map(|s| s.open.is_some()).unwrap_or(false)
}

fn ticket(app: &AppHandle) -> Option<String> {
    app.state::<Keys>().0.lock().ok()?.open.as_ref().map(|o| o.ticket.clone())
}

/// Hand the core the open window's ticket. False: no core of this app's is listening.
fn tell_ticket(app: &AppHandle) -> bool {
    let Some(t) = ticket(app) else { return false };
    app.state::<core::Core>().tell(&serde_json::json!({"t": "keys_ticket", "ticket": t}))
}

/// A core said hello on the socket: while the window is open, it hears the ticket again (a core
/// that restarted knows nothing of it).
pub fn core_said_hello(app: &AppHandle) {
    if ours(app) && tell_ticket(app) {
        core::log("Brains & Keys is open: its ticket went to the new core");
    }
}

/// The window is gone: its ticket is no good any more, and the core hears so.
fn closed(app: &AppHandle, id: u64) {
    let keys = app.state::<Keys>();
    let was = {
        let Ok(mut s) = keys.0.lock() else { return };
        match &s.open {
            Some(o) if o.id == id => s.open.take().is_some(),
            _ => false,
        }
    };
    if was {
        let told = app.state::<core::Core>().tell(&serde_json::json!({"t": "keys_closed"}));
        core::log(if told { "Brains & Keys closed: its ticket is revoked" } else { "Brains & Keys closed (no core to tell)" });
    }
}

fn front(w: &WebviewWindow) {
    let _ = w.unminimize();
    let _ = w.show();
    let _ = w.set_focus();
}

/// "Brains & Keys…": bring the window up, creating it if it isn't open. On the main thread.
pub fn open(app: &AppHandle) {
    open_once(app, true);
}

fn open_once(app: &AppHandle, again: bool) {
    if let Some(w) = app.get_webview_window(LABEL) {
        if ours(app) {
            front(&w);
            return;
        }
        // Not this shell's: it goes, and the real one opens once its label is free.
        core::log("a Brains & Keys window the shell didn't create was closed");
        let _ = w.destroy();
        if again {
            let handle = app.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                let h = handle.clone();
                let _ = handle.run_on_main_thread(move || open_once(&h, false));
            });
        }
        return;
    }
    let Some(t) = mint() else {
        core::log("Brains & Keys didn't open: no randomness for its ticket");
        return;
    };
    let keys = app.state::<Keys>();
    let id = {
        let Ok(mut s) = keys.0.lock() else { return };
        s.made += 1;
        s.open = Some(Open { id: s.made, ticket: t });
        s.made
    };
    let dev = cfg!(debug_assertions);
    let built = WebviewWindowBuilder::new(app, LABEL, WebviewUrl::App(PAGE.into()))
        .title("Brains & Keys")
        .inner_size(560.0, 700.0)
        .min_inner_size(460.0, 520.0)
        .resizable(true)
        .center()
        .decorations(true)
        .title_bar_style(TitleBarStyle::Overlay)
        .hidden_title(true)
        .traffic_light_position(LogicalPosition::new(16.0, 20.0))
        .transparent(false)
        .background_color(Color(0x11, 0x11, 0x14, 0xff))
        .always_on_top(false)
        .content_protected(true) // never in screenshots, screen recordings or NEO's own screen tools
        .visible(false)
        .focused(true)
        .on_navigation(move |url| may_navigate(url, dev))
        .on_new_window(|_, _| NewWindowResponse::Deny)
        .build();
    let w = match built {
        Ok(w) => w,
        Err(e) => {
            if let Ok(mut s) = app.state::<Keys>().0.lock() {
                if s.open.as_ref().map(|o| o.id) == Some(id) {
                    s.open = None;
                }
            }
            core::log(&format!("Brains & Keys didn't open: {e}"));
            return;
        }
    };
    let handle = app.clone();
    w.on_window_event(move |e| {
        if let WindowEvent::Destroyed = e {
            closed(&handle, id);
        }
    });
    if !tell_ticket(app) {
        core::log("Brains & Keys opened; the core will hear its ticket when the page asks");
    } else {
        core::log("Brains & Keys opened");
    }
    let handle = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(SHOW_ANYWAY);
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

/// Every page load, in every window: a `keys` window this shell didn't create is closed.
pub fn page_loaded<R: Runtime>(webview: &tauri::Webview<R>) {
    if webview.label() != LABEL {
        return;
    }
    let app = webview.app_handle();
    let mine = app.state::<Keys>().0.lock().map(|s| s.open.is_some()).unwrap_or(false);
    if !mine {
        core::log("a Brains & Keys window the shell didn't create was closed");
        let _ = webview.window().destroy();
    }
}

/// The window's page, for its ticket (see the module's doc). Errors: "forbidden" (not the window
/// this shell created), "no_link", "not_ready" or "no_core" (`ticket_error`).
#[tauri::command]
pub fn keys_ticket(app: AppHandle, window: WebviewWindow) -> Result<String, String> {
    let url = window.url().ok();
    if !may_get_ticket(window.label(), ours(&app), url.as_ref(), cfg!(debug_assertions)) {
        core::log("a window asked for the Brains & Keys ticket and was refused");
        return Err("forbidden".into());
    }
    let Some(t) = ticket(&app) else { return Err("forbidden".into()) };
    // Told again each time: the core may have restarted since the window opened.
    if app.state::<core::Core>().tell(&serde_json::json!({"t": "keys_ticket", "ticket": t})) {
        return Ok(t);
    }
    let status = app.state::<core::Core>().status.lock().map(|s| s.clone()).unwrap_or_default();
    Err(ticket_error(&status).into())
}

/// ~/.neo/.env in TextEdit (created with every setting explained if it doesn't exist yet).
pub fn edit_env() {
    if let Ok(path) = core::ensure_env_file() {
        let _ = std::process::Command::new("open").arg("-e").arg(path).spawn();
    }
}

/// The window's "Open ~/.neo/.env (advanced)" link: from that window only.
#[tauri::command]
pub fn edit_env_file(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    let url = window.url().ok();
    if !may_get_ticket(window.label(), ours(&app), url.as_ref(), cfg!(debug_assertions)) {
        core::log("a window asked to open the settings file and was refused");
        return Err("forbidden".into());
    }
    edit_env();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    #[test]
    fn hex_is_two_lowercase_digits_a_byte() {
        assert_eq!(hex(&[]), "");
        assert_eq!(hex(&[0, 1, 0xab, 0xff]), "0001abff");
    }

    #[test]
    fn a_ticket_is_64_hex_characters_and_new_each_time() {
        let (a, b) = (mint().unwrap(), mint().unwrap());
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)), "{a}");
        assert_ne!(a, b);
    }

    #[test]
    fn only_the_shells_own_keys_window_at_its_route_gets_the_ticket() {
        let page = u("tauri://localhost/index.html#/keys");
        assert!(may_get_ticket("keys", true, Some(&page), false));
        assert!(may_get_ticket("keys", true, Some(&u("http://tauri.localhost/index.html#/keys")), false));
        assert!(may_get_ticket("keys", true, Some(&u("https://tauri.localhost/index.html#/keys")), false));
        // Not the shell's window, another label, another route, no URL.
        assert!(!may_get_ticket("keys", false, Some(&page), false));
        assert!(!may_get_ticket("main", true, Some(&page), false));
        assert!(!may_get_ticket("browser", true, Some(&page), false));
        assert!(!may_get_ticket("keys", true, Some(&u("tauri://localhost/index.html")), false));
        assert!(!may_get_ticket("keys", true, Some(&u("tauri://localhost/index.html#/browser")), false));
        assert!(!may_get_ticket("keys", true, Some(&u("tauri://localhost/index.html#/keys/x")), false));
        assert!(!may_get_ticket("keys", true, None, false));
        // Anything that isn't NEO's own page.
        for bad in [
            "https://example.com/#/keys",
            "https://tauri.localhost.example.com/#/keys",
            "tauri://evil/index.html#/keys",
            "file:///tmp/index.html#/keys",
            "http://localhost:1420/index.html#/keys", // the dev server, in a release build
            "http://localhost:1421/index.html#/keys",
        ] {
            assert!(!may_get_ticket("keys", true, Some(&u(bad)), false), "{bad}");
        }
        // A `tauri dev` build's page comes from its dev server.
        assert!(may_get_ticket("keys", true, Some(&u("http://localhost:1420/index.html#/keys")), true));
        assert!(!may_get_ticket("keys", true, Some(&u("http://localhost:1421/index.html#/keys")), true));
    }

    #[test]
    fn the_window_never_navigates_away_from_neos_page() {
        assert!(may_navigate(&u("tauri://localhost/index.html#/keys"), false));
        assert!(!may_navigate(&u("https://console.anthropic.com/settings/keys"), false));
        assert!(!may_navigate(&u("http://localhost:1420/"), false));
        assert!(may_navigate(&u("http://localhost:1420/"), true));
    }

    #[test]
    fn why_the_page_cant_have_its_ticket_yet() {
        assert_eq!(ticket_error(core::OUTSIDE), "no_link");
        for s in ["", "starting", "running", "restarting", "not responding — restarting"] {
            assert_eq!(ticket_error(s), "not_ready", "{s}");
        }
        for s in ["not installed — run scripts/install-app.sh", "crashed repeatedly — see Open Logs; retrying in 10 min", "couldn't start: x"] {
            assert_eq!(ticket_error(s), "no_core", "{s}");
        }
    }

    #[test]
    fn the_window_is_kept_out_of_screenshots_and_shows_neos_page() {
        let src = include_str!("keys.rs");
        let build = src.split("WebviewWindowBuilder::new(").nth(1).unwrap().split(".build()").next().unwrap();
        assert!(build.contains(".content_protected(true)"));
        assert!(build.contains("on_new_window(|_, _| NewWindowResponse::Deny)"));
        assert!(build.contains("may_navigate"));
        assert_eq!(PAGE, format!("index.html#{ROUTE}"));
    }
}

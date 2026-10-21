//! Esc folds NEO's chat when the chat is on screen but the user's app has the keyboard.
//!
//! The page folds the chat on Esc wherever its focus is (ui/src/esc.ts), and in the notch NotchHub
//! folds its panel on Esc while the panel has the keys. Neither hears an Esc the user's app gets —
//! and the chat is often up without the keys: it opened by itself (in the notch a summon, a voice
//! turn, a login page; in NEO's window a login page, the camera, a song), or the user clicked back
//! into their app. Esc did nothing for it; only "−" folded it.
//!
//! The ⌥⌘ tap (chord.rs) sees every key go by and says when a plain Esc went down (no modifier, not a
//! repeat) — nothing else about any key. It never takes the key: the user's app gets its Esc as it
//! always did, so this can only ever add the fold, never stand in for the app's own Esc. That Esc is
//! passed on — at most one every MIN_GAP — only where it can be meant for NEO's chat:
//! - never while Search Screen is up (its overlay has the keys and its own Esc), nor while NEO.app is
//!   the active app (its window, Brains & Keys or a pop-out has the keys — and their own Esc);
//! - in NEO's own window: when the window is on screen and not the key window — to its page;
//! - in the notch: only while NEO's page there said it is on screen without the keys (the core tells
//!   the shell: `esc_watch`) — to the core, which hands it to that page alone.
//! The page decides (`awayEscFolds`): it folds only a chat that is open, not pinned, with nothing on
//! it that must stay in sight (a question, a login page, the camera, a song, the browser on stage),
//! and on screen for a moment already. In the notch not a word of any Esc leaves the shell unless
//! NEO's page there asked; in NEO's window it goes to that window's own page alone, which does
//! nothing with it unless the chat is up. Nothing is logged either way.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Emitter, Manager};

/// At most one Esc passed on this often (a hand that mashes Esc folds the chat once).
pub const MIN_GAP: Duration = Duration::from_millis(250);
/// What NEO's own window's page hears.
pub const EVENT: &str = "neo://esc";

#[derive(Default)]
pub struct EscWatch {
    /// NEO's page in the notch is on screen without the keys (the core said `esc_watch`).
    notch: AtomicBool,
    /// When an Esc was last passed on.
    last: Mutex<Option<Instant>>,
}

/// What the shell can see when a plain Esc goes by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Seen {
    /// NEO's UI is in the notch (this window stays hidden).
    pub notch: bool,
    /// …and its page there asked for an Esc pressed elsewhere.
    pub notch_watch: bool,
    /// Search Screen is up (anything from warming to fading out).
    pub lens_up: bool,
    /// NEO.app is the active app: one of its windows has the keys.
    pub neo_active: bool,
    /// NEO's own window is on screen…
    pub window_shown: bool,
    /// …and is the key window (its page has the key itself).
    pub window_key: bool,
}

/// Where a plain Esc goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Nowhere,
    /// NEO's own window's page (`neo://esc`).
    Page,
    /// The core, for NEO's page in the notch (`{"t":"esc"}`).
    Core,
}

pub fn route(s: Seen) -> Route {
    if s.lens_up || s.neo_active {
        return Route::Nowhere;
    }
    if s.notch {
        return if s.notch_watch { Route::Core } else { Route::Nowhere };
    }
    if s.window_shown && !s.window_key {
        Route::Page
    } else {
        Route::Nowhere
    }
}

/// Whether an Esc now is far enough from the last one passed on.
pub fn spaced(last: Option<Instant>, now: Instant) -> bool {
    last.map_or(true, |t| now.saturating_duration_since(t) >= MIN_GAP)
}

/// What the core says: NEO's page in the notch wants an Esc pressed elsewhere, or not (a core that
/// says hello, or is gone: not — the page says it again to a new one).
pub fn set_watch(app: &AppHandle, on: bool) {
    app.state::<EscWatch>().notch.store(on, Ordering::SeqCst);
}

/// On the tap's thread: a plain Esc went down. In the notch with no page asking, it ends here.
pub fn pressed(app: &AppHandle) {
    if crate::in_notch(app) && !app.state::<EscWatch>().notch.load(Ordering::SeqCst) {
        return;
    }
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || on_main(&handle));
}

fn on_main(app: &AppHandle) {
    let w = app.get_webview_window("main");
    let seen = Seen {
        notch: crate::in_notch(app),
        notch_watch: app.state::<EscWatch>().notch.load(Ordering::SeqCst),
        lens_up: crate::lens::up(app),
        neo_active: neo_active(),
        window_shown: w.as_ref().and_then(|w| w.is_visible().ok()).unwrap_or(false),
        window_key: w.as_ref().and_then(|w| w.is_focused().ok()).unwrap_or(false),
    };
    let to = route(seen);
    if to == Route::Nowhere {
        return;
    }
    {
        let watch = app.state::<EscWatch>();
        let Ok(mut last) = watch.last.lock() else { return };
        let now = Instant::now();
        if !spaced(*last, now) {
            return;
        }
        *last = Some(now);
    }
    match to {
        Route::Page => {
            let _ = app.emit_to("main", EVENT, ());
        }
        Route::Core => {
            let _ = app.state::<crate::core::Core>().tell(&serde_json::json!({"t": "esc"}));
        }
        Route::Nowhere => {}
    }
}

#[cfg(target_os = "macos")]
fn neo_active() -> bool {
    crate::native::app_active()
}

#[cfg(not(target_os = "macos"))]
fn neo_active() -> bool {
    true // (no tap, no Esc from elsewhere)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW_AWAY: Seen = Seen { notch: false, notch_watch: false, lens_up: false, neo_active: false, window_shown: true, window_key: false };
    const NOTCH_ASKED: Seen = Seen { notch: true, notch_watch: true, lens_up: false, neo_active: false, window_shown: false, window_key: false };

    #[test]
    fn an_esc_the_users_app_got_goes_to_neos_chat_where_it_is() {
        assert_eq!(route(WINDOW_AWAY), Route::Page);
        assert_eq!(route(NOTCH_ASKED), Route::Core);
    }

    #[test]
    fn never_where_its_own_esc_is_taken_care_of_or_no_chat_asked() {
        // NEO's window has the key itself (its page folds), or isn't on screen.
        assert_eq!(route(Seen { window_key: true, ..WINDOW_AWAY }), Route::Nowhere);
        assert_eq!(route(Seen { window_shown: false, ..WINDOW_AWAY }), Route::Nowhere);
        // NEO.app is the active app: its window, Brains & Keys or a pop-out has the keys.
        assert_eq!(route(Seen { neo_active: true, ..WINDOW_AWAY }), Route::Nowhere);
        assert_eq!(route(Seen { neo_active: true, ..NOTCH_ASKED }), Route::Nowhere);
        // Search Screen is up: its Esc closes it, and nothing else.
        assert_eq!(route(Seen { lens_up: true, ..WINDOW_AWAY }), Route::Nowhere);
        assert_eq!(route(Seen { lens_up: true, ..NOTCH_ASKED }), Route::Nowhere);
        // In the notch: only while NEO's page there asked — never to the hidden window's page.
        assert_eq!(route(Seen { notch_watch: false, ..NOTCH_ASKED }), Route::Nowhere);
        assert_eq!(route(Seen { notch_watch: false, window_shown: true, ..NOTCH_ASKED }), Route::Nowhere);
    }

    #[test]
    fn one_esc_in_a_quarter_second_at_most() {
        let t0 = Instant::now();
        assert!(spaced(None, t0));
        assert!(!spaced(Some(t0), t0));
        assert!(!spaced(Some(t0), t0 + MIN_GAP - Duration::from_millis(1)));
        assert!(spaced(Some(t0), t0 + MIN_GAP));
        assert!(!spaced(Some(t0 + Duration::from_secs(1)), t0)); // (a clock read out of order: not spaced)
    }

    #[test]
    fn the_tap_only_posts_and_a_quiet_notch_hears_nothing() {
        let src = include_str!("esc.rs");
        let pressed = src.split("pub fn pressed(").nth(1).unwrap().split("\n}\n").next().unwrap();
        // In the notch, unasked: not even a post to the main thread.
        let ask = pressed.find("crate::in_notch(app) && !app.state::<EscWatch>().notch.load").unwrap();
        assert!(ask < pressed.find("run_on_main_thread").unwrap());
        // The route is decided, then spaced, then said: the page's event or the core's line, nothing else.
        let main = src.split("fn on_main(").nth(1).unwrap().split("\n}\n").next().unwrap();
        let (r, s) = (main.find("route(seen)").unwrap(), main.find("spaced(").unwrap());
        assert!(r < s && s < main.find("emit_to(\"main\", EVENT").unwrap());
        assert!(main.contains(r#"json!({"t": "esc"})"#) && !main.contains("log("));
    }
}

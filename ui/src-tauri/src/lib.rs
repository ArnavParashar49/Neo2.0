//! NEO overlay shell: a transparent always-on-top window, a menu-bar icon, and a global
//! hotkey (⌘⇧Space) that summons NEO. All conversation logic lives in the Python core;
//! the window talks to it over a local websocket.
//!
//! With `NEO_UI_HOST=notch` (NotchHub hosting NEO's page in its island) the window stays hidden:
//! the core still runs, and the hotkeys still wake it. In the notch NEO has no menu-bar item
//! either: the host's Settings carry the menu. Everything the menu does is one function (`act`),
//! reached from the menu itself and — for a host — from the core (`FromCore::Shell`). What the
//! menu shows (the permissions, Open at Login, the version) goes the other way (`tell_status`),
//! for the host to read. The item is only hidden while a host is really there
//! (`FromCore::HostAttached`): a host that went away without handing NEO back must not leave an
//! app with no way to quit it.
//!
//! None of that goes through the hidden window's page. It used to — and nothing arrived: macOS
//! suspends the process of a web page whose window is not on screen, a few seconds after it is
//! hidden, so the page's websocket stayed open while no script ran to read it. The shell and the
//! core it starts talk over a socket of their own instead (core.rs: `Core::listen`, `Core::tell`).
//! The page keeps only what it does while it is on screen: in NEO's own window it expands and
//! sends the wake itself (`neo://wake`), it asks "Erase everything?" (`neo://erase`: the window
//! comes up for it), and it passes on a `ui_host` it hears (`ui_host_changed` — the only word of
//! where the UI lives when the core was started outside the app).
//!
//! "Brains & Keys…" (the menu's, and a host's or the core's "settings") opens NEO's own window
//! for picking brains and pasting keys (keys.rs). The shell creates it, mints its ticket and hands
//! that to the core; it never sees a key. "Edit settings file…" still opens ~/.neo/.env.
//!
//! Search Screen (lens.rs): holding ⌥⌘ a little longer than the tap (chord.rs), the menu's
//! "Search Screen…", the voice and a host open an overlay above everything for circling or dragging
//! over anything on screen. While it is open a summon (the ⌥⌘ tap, ⌘⇧Space, the menu-bar icon)
//! closes it and does nothing else.

mod chord;
mod core;
#[cfg(target_os = "macos")]
mod frame;
mod keys;
mod lens;
#[cfg(target_os = "macos")]
mod native;
mod perms;

use tauri::image::Image;
use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, RunEvent, Wry};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

/// Where NEO's UI lives right now: `true` while the notch (NotchHub) hosts it.
pub struct UiHost(pub Mutex<bool>);

pub fn in_notch(app: &AppHandle) -> bool {
    app.state::<UiHost>().0.lock().map(|n| *n).unwrap_or(false)
}

/// Whether the notch's host is connected to the core right now, as the core says (`None`: nothing
/// says — no core of this app's yet, or it is gone), and since when.
pub struct HostAttached(pub Mutex<(Option<bool>, Instant)>);

/// A host that left without a word (it crashed, or was never started): the menu-bar item returns.
const HOST_GONE_GRACE: Duration = Duration::from_secs(6);
/// Nothing known about the host (the core is starting, or down): the item returns after this long.
const HOST_UNKNOWN_GRACE: Duration = Duration::from_secs(45);

/// Whether NEO's menu-bar item shows. Never while the notch hosts NEO and its host is there; a
/// host that is gone (or a core that can't say) gets a grace period first, so the item doesn't
/// flash each time the core or the host restarts.
fn tray_visible(notch: bool, attached: Option<bool>, for_how_long: Duration) -> bool {
    if !notch {
        return true;
    }
    match attached {
        Some(true) => false,
        Some(false) => for_how_long > HOST_GONE_GRACE,
        None => for_how_long > HOST_UNKNOWN_GRACE,
    }
}

fn sync_tray(app: &AppHandle) {
    let (attached, since) = app.state::<HostAttached>().0.lock().map(|g| *g).unwrap_or((None, Instant::now()));
    let visible = tray_visible(in_notch(app), attached, since.elapsed());
    let shown = app.state::<Tray>();
    if shown.visible.swap(visible, Ordering::SeqCst) != visible {
        if let Some(tray) = app.tray_by_id("neo") {
            let _ = tray.set_visible(visible);
        }
        core::log(if visible { "menu-bar item: shown" } else { "menu-bar item: hidden (the notch's host carries the menu)" });
    }
}

fn set_host_attached(app: &AppHandle, attached: Option<bool>) {
    if let Ok(mut g) = app.state::<HostAttached>().0.lock() {
        if g.0 != attached {
            *g = (attached, Instant::now());
        }
    }
    if attached == Some(true) {
        sync_tray(app); // hidden at once; coming back waits out its grace (the status loop)
    }
}

/// Where NEO's UI lives changed. Into the notch: this window goes away. Back to the window
/// (NotchHub quit): NEO's own orb returns, and its menu-bar item.
fn set_ui_host(app: &AppHandle, notch: bool) {
    if let Ok(mut n) = app.state::<UiHost>().0.lock() {
        if *n == notch {
            return;
        }
        *n = notch;
    }
    core::log(if notch { "the UI lives in the notch: the window hides" } else { "the UI lives in NEO's own window again" });
    if let Some(w) = app.get_webview_window("main") {
        if notch {
            let _ = w.hide();
        } else {
            let _ = w.show();
        }
    }
    sync_tray(app);
}

/// What the core says over the socket it shares with the app (off the main thread).
fn from_core(app: &AppHandle, msg: core::FromCore) {
    match msg {
        core::FromCore::Hello { notch, host_attached, lens: cfg } => {
            set_ui_host(app, notch);
            set_host_attached(app, Some(host_attached));
            tell_status(app, true); // a new core knows nothing of the menu yet
            keys::core_said_hello(app); // …nor of an open Brains & Keys window's ticket
            // Search Screen: on or off, its hold time, the menu's item (an older core: no lens).
            let handle = app.clone();
            let _ = app.run_on_main_thread(move || {
                lens::core_hello(&handle, cfg);
                sync_lens_item(&handle);
            });
        }
        core::FromCore::HostAttached(attached) => set_host_attached(app, Some(attached)),
        core::FromCore::UiHost { notch } => set_ui_host(app, notch),
        core::FromCore::Shell { action, on } => {
            if !SHELL_ACTIONS.contains(&action.as_str()) {
                core::log(&format!("a host asked for something the menu doesn't have: {action:.32}"));
                return;
            }
            core::log(&format!("a host asked for: {action}"));
            let handle = app.clone();
            if action == "lens" {
                // The voice, the agent's search_screen tool or a host: the core can't say which.
                let _ = app.run_on_main_thread(move || lens::open(&handle, lens::Via::Voice));
                return;
            }
            let _ = app.run_on_main_thread(move || act(&handle, &action, on));
        }
        core::FromCore::LensClose { why } => {
            let handle = app.clone();
            let _ = app.run_on_main_thread(move || lens::close(&handle, &why));
        }
        core::FromCore::Gone => {
            set_host_attached(app, None); // no core to say whether the host is there
            let handle = app.clone();
            let _ = app.run_on_main_thread(move || {
                lens::core_gone(&handle); // no hold without a core; an open overlay says so, then closes
                sync_lens_item(&handle);
            });
        }
    }
}

/// The menu-bar item's parts that change after it is built.
struct Tray {
    visible: AtomicBool,
    login: Mutex<Option<CheckMenuItem<Wry>>>,
    /// ⌥⌘ works (it needs Accessibility); until then ⌘⇧Space is the way in.
    chord: AtomicBool,
    /// "Search Screen…": its text and whether it can be chosen follow the core and the tap.
    lens: Mutex<Option<MenuItem<Wry>>>,
    /// What the core was last told the menu shows (`tell_status`).
    told: Mutex<Option<ShellStatus>>,
}

/// What NEO's menu shows, for a host that carries the menu instead (the notch's Settings).
#[derive(Clone, PartialEq, serde::Serialize)]
pub struct ShellStatus {
    accessibility: bool,
    screen_recording: bool,
    open_at_login: bool,
    version: String,
    hotkey: &'static str,
}

fn status_now(app: &AppHandle) -> ShellStatus {
    ShellStatus {
        accessibility: perms::accessibility(),
        screen_recording: perms::screen_recording(),
        open_at_login: core::launch_at_login(),
        version: app.package_info().version.to_string(),
        hotkey: if app.state::<Tray>().chord.load(Ordering::SeqCst) { "⌥⌘" } else { "⌘⇧Space" },
    }
}

/// Tells the core what the menu shows, when it changed (or `anyway`: a core that just started).
/// With no core of this app's listening nothing is remembered as told, so the next look tries again.
fn tell_status(app: &AppHandle, anyway: bool) {
    let status = status_now(app);
    let tray = app.state::<Tray>();
    let Ok(mut told) = tray.told.lock() else { return };
    if !anyway && told.as_ref() == Some(&status) {
        return;
    }
    let mut msg = serde_json::to_value(&status).unwrap_or_default();
    if let Some(map) = msg.as_object_mut() {
        map.insert("t".into(), "status".into());
    }
    *told = if app.state::<core::Core>().tell(&msg) { Some(status) } else { None };
}

/// The menu's "Search Screen…" item, as the core and the tap say (lens::menu_item).
fn sync_lens_item(app: &AppHandle) {
    let chord = app.state::<Tray>().chord.load(Ordering::SeqCst);
    let (text, enabled) = lens::menu_item(chord, lens::known(app));
    if let Ok(item) = app.state::<Tray>().lens.lock() {
        if let Some(item) = item.as_ref() {
            let _ = item.set_text(text);
            let _ = item.set_enabled(enabled);
        }
    }
}

/// Everything NEO's menu does, by name — the menu's own items and a host's `shell` command
/// (`from_core`) both end here. On the main thread.
fn act(app: &AppHandle, action: &str, on: Option<bool>) {
    match action {
        "quit" => app.exit(0),
        "show" => show(app),
        // Search Screen… (the menu's item, a host's): the overlay, after the menu has gone.
        "lens" => lens::open(app, lens::Via::Menu),
        "allow_ax" => perms::open_settings("Privacy_Accessibility"),
        "allow_sc" => perms::open_settings("Privacy_ScreenCapture"),
        "restart" => {
            // Off the main thread: stopping the core can take a few seconds.
            let core = app.state::<core::Core>().inner().clone();
            std::thread::spawn(move || core.restart());
        }
        "open_logs" => {
            let _ = std::process::Command::new("open").arg(core::logs_dir()).spawn();
        }
        // Brains & Keys: NEO's own window (keys.rs) — the menu's item, a host's Settings, and the
        // core's `open_keys` (the chat's "Set up a brain", a request that needs a model) all end here.
        "settings" => keys::open(app),
        // The menu's "Edit settings file…": TextEdit on ~/.neo/.env (created with every setting
        // explained, mode 600). Not one a host may ask for.
        "edit_env" => keys::edit_env(),
        "erase_memory" => {
            // The overlay asks "Erase everything?" before anything is deleted — without
            // opening the microphone while the user reads it. In the notch too: this window
            // comes up for the question alone, and goes away again with it.
            reveal(app);
            let _ = app.emit("neo://erase", ());
        }
        "open_at_login" => {
            // The menu's item toggles; a host says which way.
            let on = on.unwrap_or_else(|| !core::launch_at_login());
            let _ = core::set_launch_at_login(on);
            core::remember_login_choice();
            if let Ok(item) = app.state::<Tray>().login.lock() {
                if let Some(item) = item.as_ref() {
                    let _ = item.set_checked(core::launch_at_login());
                }
            }
            tell_status(app, false); // the host's switch follows
        }
        _ => {}
    }
}

/// What a host may ask for (the core checks who is asking; see neo2/core/server/ws.py).
const SHELL_ACTIONS: [&str; 8] = ["show", "settings", "restart", "open_logs", "erase_memory", "open_at_login", "quit", "lens"];

fn show(app: &AppHandle) {
    if in_notch(app) {
        // The notch shows NEO; this window stays hidden, and its page asleep with it: the wake
        // goes to the core directly. With no core of this app's listening (it is starting, or was
        // started outside the app) the summon is dropped — handed to the sleeping page it would
        // be sent whenever that page next woke, and open the microphone then.
        if !app.state::<core::Core>().tell(&serde_json::json!({"t": "wake"})) {
            core::log("summoned, but no core of this app's is listening: nothing sent");
        }
        return;
    }
    if reveal(app) {
        let _ = app.emit("neo://wake", ()); // summoned: NEO starts listening
    }
}

/// Just bring the window up — no listening (e.g. to show the Erase dialog).
fn reveal(app: &AppHandle) -> bool {
    match app.get_webview_window("main") {
        Some(w) => {
            let _ = w.show();
            let _ = w.set_focus();
            true
        }
        None => false,
    }
}

fn toggle(app: &AppHandle) {
    // Summon wins: while Search Screen is open, ⌥⌘, ⌘⇧Space and the menu-bar icon close it.
    if lens::is_open(app) {
        lens::close(app, "summon");
        return;
    }
    if in_notch(app) {
        show(app);
        return;
    }
    if let Some(w) = app.get_webview_window("main") {
        match w.is_visible() {
            Ok(true) if w.is_focused().unwrap_or(false) => {
                let _ = w.hide();
            }
            _ => show(app),
        }
    }
}

/// The overlay's key to the core's websocket (see core::ws_token).
#[tauri::command]
fn ws_token() -> String {
    core::ws_token()
}

#[tauri::command]
fn hide_window(app: AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.hide();
    }
}

/// The page heard the core say where the UI lives, and passes it on. The core says so to the
/// shell directly too (`from_core`); this is the word that arrives when the core was started
/// outside the app, and has no socket to it.
#[tauri::command]
fn ui_host_changed(app: AppHandle, window: tauri::WebviewWindow, host: String) {
    if window.label() == "main" && (host == "notch" || host == "window") {
        set_ui_host(&app, host == "notch");
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[allow(unused_mut)]
    let mut app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        toggle(app);
                    }
                })
                .build(),
        )
        .invoke_handler(tauri::generate_handler![
            hide_window,
            ws_token,
            ui_host_changed,
            frame::set_frame,
            native::clipboard_write,
            native::show_quiet,
            keys::keys_ticket,
            keys::edit_env_file,
            lens::lens_hello,
            lens::lens_painted,
            lens::lens_close,
            lens::lens_peek,
            lens::lens_allow_sc
        ])
        // A Brains & Keys or Search Screen window the shell didn't create is closed as its page loads.
        .on_page_load(|webview, _| {
            keys::page_loaded(webview);
            lens::page_loaded(webview);
        })
        .manage(keys::Keys::default())
        .manage(lens::Lens::default())
        .manage(core::Core::default())
        .manage(UiHost(Mutex::new(core::ui_host_is_notch())))
        .manage(HostAttached(Mutex::new((None, Instant::now()))))
        .manage(Tray {
            visible: AtomicBool::new(true),
            login: Mutex::new(None),
            chord: AtomicBool::new(false),
            lens: Mutex::new(None),
            told: Mutex::new(None),
        })
        .setup(|app| {
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            // NEO's UI is in the notch (NotchHub): this window never shows.
            if in_notch(app.handle()) {
                if let Some(w) = app.get_webview_window("main") {
                    let _ = w.hide();
                }
            }

            // The brain: start NEO's Python core (or use one that's already running). What it
            // says to the shell (a host's menu action, where the UI lives) is heard here.
            let listener = app.handle().clone();
            app.state::<core::Core>().listen(move |msg| from_core(&listener, msg));
            app.state::<core::Core>().start();
            core::sync_launch_at_login();
            // Accessibility + Screen Recording belong to NEO.app: ask once per version, then the
            // menu shows what's missing.
            perms::ask_once(&app.package_info().version.to_string());
            let ax_item = MenuItem::with_id(app, "allow_ax", "Allow Accessibility…", !perms::accessibility(), None::<&str>)?;
            let sc_item = MenuItem::with_id(app, "allow_sc", "Allow Screen Recording…", !perms::screen_recording(), None::<&str>)?;

            let summon = Shortcut::new(Some(Modifiers::SUPER | Modifiers::SHIFT), Code::Space);
            app.global_shortcut().register(summon)?;
            let show_item = MenuItem::with_id(app, "show", "Show NEO   ⌘⇧Space", true, None::<&str>)?;
            let lens_item = MenuItem::with_id(app, "lens", "Search Screen…", true, None::<&str>)?;
            *app.state::<Tray>().lens.lock().unwrap() = Some(lens_item.clone());
            let status_item = MenuItem::with_id(app, "status", "NEO: starting…", false, None::<&str>)?;

            // ⌥⌘ (pressed together and released) summons NEO; held a little longer it opens Search
            // Screen. It needs Accessibility: until that's granted it keeps retrying, and ⌘⇧Space
            // (and the menu's Search Screen…) work meanwhile.
            let handle = app.handle().clone();
            let hold_handle = app.handle().clone();
            let (ready_handle, ready_item) = (app.handle().clone(), show_item.clone());
            let chord_ok = chord::watch(
                move || {
                    let h = handle.clone();
                    let _ = handle.run_on_main_thread(move || toggle(&h));
                },
                move || {
                    let item = ready_item.clone();
                    ready_handle.state::<Tray>().chord.store(true, Ordering::SeqCst);
                    let h = ready_handle.clone();
                    let _ = ready_handle.run_on_main_thread(move || {
                        let _ = item.set_text("Show NEO   ⌥⌘");
                        sync_lens_item(&h);
                    });
                },
                move |ev| {
                    let h = hold_handle.clone();
                    let _ = hold_handle.run_on_main_thread(move || lens::on_hold(&h, ev));
                },
                app.state::<lens::Lens>().gate.clone(),
            );
            if chord_ok {
                app.state::<Tray>().chord.store(true, Ordering::SeqCst);
                show_item.set_text("Show NEO   ⌥⌘")?;
            }
            sync_lens_item(app.handle());
            // The ids are the actions' names (`act`): a host's `shell` command uses the same ones.
            let restart_item = MenuItem::with_id(app, "restart", "Restart NEO", true, None::<&str>)?;
            let keys_item = MenuItem::with_id(app, "settings", "Brains & Keys…", true, None::<&str>)?;
            let env_item = MenuItem::with_id(app, "edit_env", "Edit settings file…", true, None::<&str>)?;
            let erase_item = MenuItem::with_id(app, "erase_memory", "Erase NEO's Memory…", true, None::<&str>)?;
            let logs_item = MenuItem::with_id(app, "open_logs", "Open Logs", true, None::<&str>)?;
            let login_item =
                CheckMenuItem::with_id(app, "open_at_login", "Open at Login", true, core::launch_at_login(), None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit NEO", true, None::<&str>)?;
            let sep = PredefinedMenuItem::separator(app)?;
            let sep2 = PredefinedMenuItem::separator(app)?;
            let menu = Menu::with_items(
                app,
                &[
                    &status_item, &ax_item, &sc_item, &show_item, &lens_item, &sep, &keys_item, &env_item, &restart_item,
                    &logs_item, &erase_item, &login_item, &sep2, &quit_item,
                ],
            )?;
            *app.state::<Tray>().login.lock().unwrap() = Some(login_item.clone());
            TrayIconBuilder::with_id("neo")
                .icon(Image::from_bytes(include_bytes!("../icons/tray.png"))?)
                .icon_as_template(true)
                .tooltip("NEO")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, e| act(app, e.id.as_ref(), None))
                .on_tray_icon_event(|tray, event| {
                    // One toggle per left click: only the button-up edge, and not the right button (menu).
                    if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                        toggle(tray.app_handle());
                    }
                })
                .build(app)?;
            // In the notch the host's Settings carry this menu: no item of NEO's own (it returns
            // if no host turns up — see `tray_visible`).
            sync_tray(app.handle());

            let (core, handle, item) = (app.state::<core::Core>().inner().clone(), app.handle().clone(), status_item.clone());
            let (ax_w, sc_w) = (ax_item.clone(), sc_item.clone());
            std::thread::spawn(move || {
                let mut last = String::new();
                loop {
                    let (ax, sc) = (perms::accessibility(), perms::screen_recording());
                    tell_status(&handle, false); // the host shows the same things: the core hears when one changed
                    sync_tray(&handle); // a host that went away: the item comes back
                    let core_status = core.status.lock().unwrap().clone();
                    let now = match (ax, sc) {
                        (false, _) => format!("{core_status} — needs Accessibility"),
                        (true, false) => format!("{core_status} — needs Screen Recording"),
                        _ => core_status,
                    };
                    let (a, s) = (ax_w.clone(), sc_w.clone());
                    let _ = handle.run_on_main_thread(move || {
                        let _ = a.set_enabled(!ax);
                        let _ = a.set_text(if ax { "Accessibility: allowed ✓" } else { "Allow Accessibility…" });
                        let _ = s.set_enabled(!sc);
                        let _ = s.set_text(if sc { "Screen Recording: allowed ✓" } else { "Allow Screen Recording…" });
                    });
                    if now != last {
                        last = now.clone();
                        let (h, item) = (handle.clone(), item.clone());
                        let _ = handle.run_on_main_thread(move || {
                            let _ = item.set_text(format!("NEO: {now}"));
                            if let Some(tray) = h.tray_by_id("neo") {
                                let _ = tray.set_tooltip(Some(format!("NEO — {now}")));
                            }
                        });
                    }
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building NEO");
    // Menu-bar only from the very first moment. Set in `setup` it came too late: the window
    // framework makes the app a regular (Dock) app while launching, the Dock sees that instant
    // and lists NEO under recent apps.
    #[cfg(target_os = "macos")]
    app.set_activation_policy(tauri::ActivationPolicy::Accessory);
    app.run(|app, event| {
            if let RunEvent::Exit = event {
                app.state::<core::Core>().stop(); // the core leaves with the app
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_menu_bar_item_is_hidden_only_while_the_notchs_host_is_there() {
        let s = Duration::from_secs;
        // NEO's own window: always its own menu-bar item.
        assert!(tray_visible(false, Some(true), s(0)));
        assert!(tray_visible(false, None, s(0)));
        // In the notch, with the host connected: none.
        assert!(!tray_visible(true, Some(true), s(0)));
        assert!(!tray_visible(true, Some(true), s(3600)));
        // The host went away without handing NEO back: after a grace period the item returns…
        assert!(!tray_visible(true, Some(false), s(1)));
        assert!(tray_visible(true, Some(false), HOST_GONE_GRACE + s(1)));
        // …and so it does when nothing can say (the core never came up).
        assert!(!tray_visible(true, None, s(10)));
        assert!(tray_visible(true, None, HOST_UNKNOWN_GRACE + s(1)));
    }

    #[test]
    fn a_host_reaches_exactly_the_menus_actions() {
        // Every action a host may ask for is one `act` knows (the menu's items carry the same ids).
        let src = include_str!("lib.rs");
        let act = src.split("fn act(").nth(1).unwrap().split("\n}\n").next().unwrap();
        for a in SHELL_ACTIONS {
            assert!(act.contains(&format!("\"{a}\" =>")), "{a}");
            assert!(src.contains(&format!("with_id(app, \"{a}\"")), "{a}");
        }
        // The panes of System Settings are the menu's own; a host opens them itself.
        assert!(!SHELL_ACTIONS.contains(&"allow_ax") && !SHELL_ACTIONS.contains(&"allow_sc"));
    }

    #[test]
    fn search_screen_is_in_the_menu_and_its_commands_are_registered() {
        let src = include_str!("lib.rs");
        let act = src.split("fn act(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(act.contains("\"lens\" => lens::open(app, lens::Via::Menu)"));
        assert_eq!(SHELL_ACTIONS.len(), 8);
        assert!(SHELL_ACTIONS.contains(&"lens"));
        // Right after "Show NEO" in the menu.
        assert!(src.contains("&show_item, &lens_item,"));
        for cmd in ["lens_hello", "lens_painted", "lens_close", "lens_peek", "lens_allow_sc"] {
            assert!(src.contains(&format!("lens::{cmd}")), "{cmd}");
        }
        // A lens window the shell didn't create goes as its page loads.
        let load = src.split(".on_page_load(").nth(1).unwrap().split("})").next().unwrap();
        assert!(load.contains("lens::page_loaded(webview)") && load.contains("keys::page_loaded(webview)"));
        // Summon wins: toggle (⌥⌘, ⌘⇧Space, the menu-bar icon) closes an open overlay first.
        let toggle = src.split("fn toggle(").nth(1).unwrap().split("\n}\n").next().unwrap();
        let first = toggle.find("lens::is_open(app)").unwrap();
        assert!(first < toggle.find("in_notch(app)").unwrap());
        assert!(toggle.contains("lens::close(app, \"summon\")"));
        // The hold reaches the shell, and the core's hello and its going set the gate.
        assert!(src.contains("lens::on_hold(&h, ev)"));
        let from_core = src.split("fn from_core(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(from_core.contains("lens::core_hello(&handle, cfg)") && from_core.contains("lens::core_gone(&handle)"));
        assert!(from_core.contains("core::FromCore::LensClose"));
    }

    #[test]
    fn brains_and_keys_is_the_window_and_the_settings_file_has_its_own_item() {
        let src = include_str!("lib.rs");
        let act = src.split("fn act(").nth(1).unwrap().split("\n}\n").next().unwrap();
        // "settings" (the menu's item, a host's row, the core's open_keys) opens the window…
        assert!(act.contains("\"settings\" => keys::open(app)"));
        assert!(src.contains("with_id(app, \"settings\", \"Brains & Keys…\""));
        // …and the old TextEdit door is the menu's alone: no host may ask for it.
        assert!(act.contains("\"edit_env\" => keys::edit_env()"));
        assert!(src.contains("with_id(app, \"edit_env\", \"Edit settings file…\""));
        assert!(!SHELL_ACTIONS.contains(&"edit_env"));
        assert!(!src.contains(concat!("\"Settings & API", " Keys…\"")), "the old item's name is gone");
        // The window's two commands are registered, and the core hears its ticket after every hello.
        assert!(src.contains("keys::keys_ticket,") && src.contains("keys::edit_env_file"));
        let from_core = src.split("fn from_core(").nth(1).unwrap().split("\n}\n").next().unwrap();
        let hello = from_core.split("core::FromCore::Hello {").nth(1).unwrap().split("core::FromCore::HostAttached").next().unwrap();
        assert!(hello.contains("keys::core_said_hello(app)"));
    }
}

//! NEO overlay shell: a transparent always-on-top window, a menu-bar icon, and a global
//! hotkey (⌘⇧Space) that summons NEO. All conversation logic lives in the Python core;
//! the window talks to it over a local websocket.
//!
//! With `NEO_UI_HOST=notch` (NotchHub hosting NEO's page in its island) the window stays hidden:
//! the core still runs, and the hotkeys still send the wake — through this window's own page,
//! which keeps its socket to the core while hidden. The core tells the page when the host changes
//! (`ui_host`), and the page relays it here (`ui_host_changed`).
//!
//! In the notch NEO has no menu-bar item either: the host's Settings carry the menu. Everything
//! the menu does is one function (`act`), reached from the menu itself and — for a host — from
//! the core's `shell` event, which the hidden page hands to `shell_action`. What the menu shows
//! (the permissions, Open at Login, the version) goes the other way: `shell_status`, which the
//! page passes to the core for the host to read. The item is only hidden while a host is really
//! there (`host_attached`): a host that went away without handing NEO back must not leave an
//! app with no way to quit it.

mod chord;
mod core;
#[cfg(target_os = "macos")]
mod frame;
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

/// Whether the notch's host is connected to the core right now, as far as the page knows
/// (`None`: it doesn't — no core yet, or its socket is down), and since when.
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
    }
}

/// The menu-bar item's parts that change after it is built.
struct Tray {
    visible: AtomicBool,
    login: Mutex<Option<CheckMenuItem<Wry>>>,
    /// ⌥⌘ works (it needs Accessibility); until then ⌘⇧Space is the way in.
    chord: AtomicBool,
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

/// Everything NEO's menu does, by name — the menu's own items and a host's `shell` command
/// (`shell_action`) both end here. On the main thread.
fn act(app: &AppHandle, action: &str, on: Option<bool>) {
    match action {
        "quit" => app.exit(0),
        "show" => show(app),
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
        "settings" => {
            // TextEdit on ~/.neo/.env (created with every setting explained, mode 600).
            if let Ok(path) = core::ensure_env_file() {
                let _ = std::process::Command::new("open").arg("-e").arg(path).spawn();
            }
        }
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
            let _ = app.emit("neo://shell-status", ()); // the host's switch follows
        }
        _ => {}
    }
}

/// What a host may ask for (the core checks who is asking; see neo2/core/server/ws.py).
const SHELL_ACTIONS: [&str; 7] = ["show", "settings", "restart", "open_logs", "erase_memory", "open_at_login", "quit"];

/// A host's `shell` command, relayed by the core to this window's page: the menu's own action.
#[tauri::command]
fn shell_action(app: AppHandle, window: tauri::WebviewWindow, action: String, on: Option<bool>) -> Result<(), String> {
    if window.label() != "main" {
        return Err("only NEO's own window relays the shell".into());
    }
    if !SHELL_ACTIONS.contains(&action.as_str()) {
        return Err(format!("unknown shell action: {action}"));
    }
    let handle = app.clone();
    app.run_on_main_thread(move || act(&handle, &action, on)).map_err(|e| e.to_string())
}

/// What the menu shows (permissions, Open at Login, the version), for the page to pass to the core.
#[tauri::command]
fn shell_status(app: AppHandle) -> ShellStatus {
    status_now(&app)
}

/// The page says whether the notch's host is connected to the core (`null`: it can't tell).
#[tauri::command]
fn host_attached(app: AppHandle, attached: Option<bool>) {
    if let Ok(mut g) = app.state::<HostAttached>().0.lock() {
        if g.0 != attached {
            *g = (attached, Instant::now());
        }
    }
    if attached == Some(true) {
        sync_tray(&app); // hidden at once; coming back waits out its grace (the status loop)
    }
}

fn show(app: &AppHandle) {
    if in_notch(app) {
        // The notch shows NEO; this window stays hidden. Its page still carries the wake to the core.
        let _ = app.emit("neo://wake", ());
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

/// The core said where the UI lives (the page relays its `ui_host` event). Into the notch: this
/// window goes away. Back to the window (NotchHub quit): NEO's own orb returns.
#[tauri::command]
fn ui_host_changed(app: AppHandle, host: String) {
    let notch = host == "notch";
    if let Ok(mut n) = app.state::<UiHost>().0.lock() {
        if *n == notch {
            return;
        }
        *n = notch;
    }
    if let Some(w) = app.get_webview_window("main") {
        if notch {
            let _ = w.hide();
        } else {
            let _ = w.show();
        }
    }
    sync_tray(&app); // back in its own window, NEO has its menu-bar item again
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
            host_attached,
            shell_action,
            shell_status,
            frame::set_frame,
            native::clipboard_write,
            native::show_quiet
        ])
        .manage(core::Core::default())
        .manage(UiHost(Mutex::new(core::ui_host_is_notch())))
        .manage(HostAttached(Mutex::new((None, Instant::now()))))
        .manage(Tray { visible: AtomicBool::new(true), login: Mutex::new(None), chord: AtomicBool::new(false) })
        .setup(|app| {
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            // NEO's UI is in the notch (NotchHub): this window never shows.
            if in_notch(app.handle()) {
                if let Some(w) = app.get_webview_window("main") {
                    let _ = w.hide();
                }
            }

            // The brain: start NEO's Python core (or use one that's already running).
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
            let status_item = MenuItem::with_id(app, "status", "NEO: starting…", false, None::<&str>)?;

            // ⌥⌘ (pressed together and released) summons NEO. It needs Accessibility: until
            // that's granted it keeps retrying, and ⌘⇧Space works meanwhile.
            let handle = app.handle().clone();
            let (ready_handle, ready_item) = (app.handle().clone(), show_item.clone());
            let chord_ok = chord::watch(
                move || {
                    let h = handle.clone();
                    let _ = handle.run_on_main_thread(move || toggle(&h));
                },
                move || {
                    let item = ready_item.clone();
                    ready_handle.state::<Tray>().chord.store(true, Ordering::SeqCst);
                    let _ = ready_handle.run_on_main_thread(move || {
                        let _ = item.set_text("Show NEO   ⌥⌘");
                    });
                },
            );
            if chord_ok {
                app.state::<Tray>().chord.store(true, Ordering::SeqCst);
                show_item.set_text("Show NEO   ⌥⌘")?;
            }
            // The ids are the actions' names (`act`): a host's `shell` command uses the same ones.
            let restart_item = MenuItem::with_id(app, "restart", "Restart NEO", true, None::<&str>)?;
            let keys_item = MenuItem::with_id(app, "settings", "Settings & API Keys…", true, None::<&str>)?;
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
                    &status_item, &ax_item, &sc_item, &show_item, &sep, &keys_item, &restart_item, &logs_item,
                    &erase_item, &login_item, &sep2, &quit_item,
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
                let mut told: Option<ShellStatus> = None;
                loop {
                    let (ax, sc) = (perms::accessibility(), perms::screen_recording());
                    // The host shows the same things: tell the page when one of them changed.
                    let status = status_now(&handle);
                    if told.as_ref() != Some(&status) {
                        told = Some(status);
                        let _ = handle.emit("neo://shell-status", ());
                    }
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
}

//! NEO overlay shell: a transparent always-on-top window, a menu-bar icon, and a global
//! hotkey (⌘⇧Space) that summons NEO. All conversation logic lives in the Python core;
//! the window talks to it over a local websocket.

mod chord;
mod core;
mod perms;

use tauri::image::Image;
use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, RunEvent};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

fn show(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.set_focus();
        let _ = app.emit("neo://wake", ());
    }
}

fn toggle(app: &AppHandle) {
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
        .invoke_handler(tauri::generate_handler![hide_window, ws_token])
        .manage(core::Core::default())
        .setup(|app| {
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

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
                    let _ = ready_handle.run_on_main_thread(move || {
                        let _ = item.set_text("Show NEO   ⌥⌘");
                    });
                },
            );
            if chord_ok {
                show_item.set_text("Show NEO   ⌥⌘")?;
            }
            let restart_item = MenuItem::with_id(app, "restart", "Restart NEO", true, None::<&str>)?;
            let logs_item = MenuItem::with_id(app, "logs", "Open Logs", true, None::<&str>)?;
            let login_item =
                CheckMenuItem::with_id(app, "login", "Open at Login", true, core::launch_at_login(), None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit NEO", true, None::<&str>)?;
            let sep = PredefinedMenuItem::separator(app)?;
            let sep2 = PredefinedMenuItem::separator(app)?;
            let menu = Menu::with_items(
                app,
                &[&status_item, &ax_item, &sc_item, &show_item, &sep, &restart_item, &logs_item, &login_item, &sep2, &quit_item],
            )?;
            let login_check = login_item.clone();
            TrayIconBuilder::with_id("neo")
                .icon(Image::from_bytes(include_bytes!("../icons/tray.png"))?)
                .icon_as_template(true)
                .tooltip("NEO")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(move |app, e| match e.id.as_ref() {
                    "quit" => app.exit(0),
                    "show" => show(app),
                    "allow_ax" => perms::open_settings("Privacy_Accessibility"),
                    "allow_sc" => perms::open_settings("Privacy_ScreenCapture"),
                    "restart" => {
                        // Off the main thread: stopping the core can take a few seconds.
                        let core = app.state::<core::Core>().inner().clone();
                        std::thread::spawn(move || core.restart());
                    }
                    "logs" => {
                        let _ = std::process::Command::new("open").arg(core::logs_dir()).spawn();
                    }
                    "login" => {
                        let on = !core::launch_at_login();
                        let _ = core::set_launch_at_login(on);
                        core::remember_login_choice();
                        let _ = login_check.set_checked(core::launch_at_login());
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    // One toggle per left click: only the button-up edge, and not the right button (menu).
                    if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                        toggle(tray.app_handle());
                    }
                })
                .build(app)?;

            let (core, handle, item) = (app.state::<core::Core>().inner().clone(), app.handle().clone(), status_item.clone());
            let (ax_w, sc_w) = (ax_item.clone(), sc_item.clone());
            std::thread::spawn(move || {
                let mut last = String::new();
                loop {
                    let (ax, sc) = (perms::accessibility(), perms::screen_recording());
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

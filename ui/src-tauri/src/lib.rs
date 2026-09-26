//! NEO overlay shell: a transparent always-on-top window, a menu-bar icon, and a global
//! hotkey (⌘⇧Space) that summons NEO. All conversation logic lives in the Python core;
//! the window talks to it over a local websocket.

mod chord;
mod core;

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

#[tauri::command]
fn hide_window(app: AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.hide();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
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
        .invoke_handler(tauri::generate_handler![hide_window])
        .manage(core::Core::default())
        .setup(|app| {
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            // The brain: start NEO's Python core (or connect to one that's already running).
            app.state::<core::Core>().start();
            core::default_launch_at_login();

            // ⌥⌘ (pressed together and released) summons NEO. It needs Accessibility; until that's
            // granted, ⌘⇧Space still works.
            let handle = app.handle().clone();
            let chord_ok = chord::watch(move || {
                let h = handle.clone();
                let _ = handle.run_on_main_thread(move || toggle(&h));
            });
            let summon = Shortcut::new(Some(Modifiers::SUPER | Modifiers::SHIFT), Code::Space);
            app.global_shortcut().register(summon)?;

            let label = if chord_ok { "Show NEO   ⌥⌘" } else { "Show NEO   ⌘⇧Space" };
            let show_item = MenuItem::with_id(app, "show", label, true, None::<&str>)?;
            let restart_item = MenuItem::with_id(app, "restart", "Restart NEO", true, None::<&str>)?;
            let logs_item = MenuItem::with_id(app, "logs", "Open Logs", true, None::<&str>)?;
            let login_item =
                CheckMenuItem::with_id(app, "login", "Open at Login", true, core::launch_at_login(), None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit NEO", true, None::<&str>)?;
            let sep = PredefinedMenuItem::separator(app)?;
            let sep2 = PredefinedMenuItem::separator(app)?;
            let menu = Menu::with_items(app, &[&show_item, &sep, &restart_item, &logs_item, &login_item, &sep2, &quit_item])?;
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
                    "restart" => app.state::<core::Core>().restart(),
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
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building NEO")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                app.state::<core::Core>().stop(); // the core leaves with the app
            }
        });
}

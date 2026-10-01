//! Two small AppKit calls the browser card needs, made directly (no plugin, no new crates):
//!
//! - `clipboard_write`: ⌘C / ⌘X on the page. The page's selection comes back from the core
//!   after the keypress is over, when WebKit no longer lets the page write the clipboard; the
//!   shell writes the general pasteboard instead (plain text only, capped).
//! - `show_quiet`: a login / CAPTCHA / 2FA prompt must surface the overlay without taking focus
//!   from the app the user is in (`orderFrontRegardless`: shown, not made key, app not activated).

use objc2::runtime::{AnyObject, Bool};
use objc2::{class, msg_send, MainThreadMarker};
use std::ffi::c_void;

const MAX_CLIP: usize = 400_000; // bytes of UTF-8 (the UI caps at 100 000 characters)
const NS_UTF8_STRING_ENCODING: usize = 4;

/// A +1 retained `NSString` from UTF-8 bytes (no NUL needed); the caller releases it.
unsafe fn ns_string(s: &str) -> *mut AnyObject {
    let obj: *mut AnyObject = msg_send![class!(NSString), alloc];
    msg_send![obj, initWithBytes: s.as_ptr() as *const c_void, length: s.len(), encoding: NS_UTF8_STRING_ENCODING]
}

/// # Safety
/// Must run on the main thread.
unsafe fn write_pasteboard(text: &str) -> bool {
    let pb: *mut AnyObject = msg_send![class!(NSPasteboard), generalPasteboard];
    if pb.is_null() {
        return false;
    }
    let value = ns_string(text);
    if value.is_null() {
        return false;
    }
    // NSPasteboardTypeString is the UTI "public.utf8-plain-text".
    let kind = ns_string("public.utf8-plain-text");
    let _: isize = msg_send![pb, clearContents];
    let ok: Bool = msg_send![pb, setString: value, forType: kind];
    let _: () = msg_send![value, release];
    let _: () = msg_send![kind, release];
    ok.as_bool()
}

#[tauri::command]
pub fn clipboard_write(window: tauri::WebviewWindow, text: String) -> Result<(), String> {
    let mut end = text.len().min(MAX_CLIP);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let text = text[..end].to_string();
    if MainThreadMarker::new().is_some() {
        return if unsafe { write_pasteboard(&text) } { Ok(()) } else { Err("pasteboard refused".into()) };
    }
    let (tx, rx) = std::sync::mpsc::channel();
    window
        .run_on_main_thread(move || {
            let _ = tx.send(unsafe { write_pasteboard(&text) });
        })
        .map_err(|e| e.to_string())?;
    match rx.recv_timeout(std::time::Duration::from_secs(2)) {
        Ok(true) => Ok(()),
        _ => Err("pasteboard refused".into()),
    }
}

/// # Safety
/// `ns_window` must be a live `NSWindow*`, and this must run on the main thread.
unsafe fn order_front(ns_window: *mut AnyObject) {
    let _: () = msg_send![ns_window, orderFrontRegardless];
}

#[tauri::command]
pub fn show_quiet(app: tauri::AppHandle, window: tauri::WebviewWindow) -> Result<(), String> {
    if crate::in_notch(&app) {
        return Ok(()); // the notch shows NEO's cards; this window stays hidden
    }
    let ns = window.ns_window().map_err(|e| e.to_string())? as usize;
    if MainThreadMarker::new().is_some() {
        unsafe { order_front(ns as *mut AnyObject) };
        return Ok(());
    }
    window
        .run_on_main_thread(move || unsafe { order_front(ns as *mut AnyObject) })
        .map_err(|e| e.to_string())
}

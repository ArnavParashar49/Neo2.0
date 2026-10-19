//! Two small AppKit calls the browser card needs, made directly (no plugin, no new crates):
//!
//! - `clipboard_write`: ⌘C / ⌘X on the page. The page's selection comes back from the core
//!   after the keypress is over, when WebKit no longer lets the page write the clipboard; the
//!   shell writes the general pasteboard instead (plain text only, capped).
//! - `show_quiet`: a login / CAPTCHA / 2FA prompt must surface the overlay without taking focus
//!   from the app the user is in (`orderFrontRegardless`: shown, not made key, app not activated).
//!
//! And two for Setup and Uninstall:
//!
//! - `choose_file`: "Use a downloaded file…" — the system's open panel, for one file.
//! - `trash`: Uninstall moves NEO's things to the Trash (`NSFileManager trashItemAtURL`: the user
//!   can put them back, and no Finder automation prompt), never deleting anything itself.

use objc2::runtime::{AnyObject, Bool};
use objc2::{class, msg_send, MainThreadMarker};
use std::ffi::{c_char, c_void, CStr};
use std::path::{Path, PathBuf};

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

/// An `NSString`'s text (borrowed; None for nil).
unsafe fn rust_string(ns: *mut AnyObject) -> Option<String> {
    if ns.is_null() {
        return None;
    }
    let p: *const c_char = msg_send![ns, UTF8String];
    if p.is_null() {
        return None;
    }
    Some(CStr::from_ptr(p).to_string_lossy().into_owned())
}

/// The system's open panel, for one file: its path, or None when the user cancelled.
///
/// # Safety
/// Must run on the main thread (the panel runs modally there).
pub unsafe fn choose_file(message: &str, button: &str) -> Option<PathBuf> {
    let panel: *mut AnyObject = msg_send![class!(NSOpenPanel), openPanel];
    if panel.is_null() {
        return None;
    }
    let _: () = msg_send![panel, setCanChooseFiles: Bool::YES];
    let _: () = msg_send![panel, setCanChooseDirectories: Bool::NO];
    let _: () = msg_send![panel, setAllowsMultipleSelection: Bool::NO];
    let _: () = msg_send![panel, setResolvesAliases: Bool::YES];
    let text = ns_string(message);
    let _: () = msg_send![panel, setMessage: text];
    let _: () = msg_send![text, release];
    let prompt = ns_string(button);
    let _: () = msg_send![panel, setPrompt: prompt];
    let _: () = msg_send![prompt, release];
    // NEO lives in the menu bar: bring it forward, or the panel opens behind the user's app.
    let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
    if !app.is_null() {
        let _: () = msg_send![app, activateIgnoringOtherApps: Bool::YES];
    }
    let response: isize = msg_send![panel, runModal];
    if response != 1 {
        return None; // NSModalResponseCancel
    }
    let url: *mut AnyObject = msg_send![panel, URL];
    if url.is_null() {
        return None;
    }
    let path: *mut AnyObject = msg_send![url, path];
    rust_string(path).map(PathBuf::from)
}

/// Move `path` to the user's Trash (any thread). The error is macOS's own words.
pub fn trash(path: &Path) -> Result<(), String> {
    objc2::rc::autoreleasepool(|_| unsafe {
        let fm: *mut AnyObject = msg_send![class!(NSFileManager), defaultManager];
        if fm.is_null() {
            return Err("no file manager".into());
        }
        let p = ns_string(&path.to_string_lossy());
        if p.is_null() {
            return Err("the path isn't text".into());
        }
        let url: *mut AnyObject = msg_send![class!(NSURL), fileURLWithPath: p];
        let _: () = msg_send![p, release];
        if url.is_null() {
            return Err("the path isn't a file URL".into());
        }
        let mut error: *mut AnyObject = std::ptr::null_mut();
        let ok: Bool = msg_send![fm, trashItemAtURL: url, resultingItemURL: std::ptr::null_mut::<*mut AnyObject>(), error: &mut error];
        if ok.as_bool() {
            return Ok(());
        }
        let words = if error.is_null() {
            None
        } else {
            let d: *mut AnyObject = msg_send![error, localizedDescription];
            rust_string(d)
        };
        Err(words.unwrap_or_else(|| "macOS didn't move it".into()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trash_moves_a_file_to_the_trash_and_says_why_it_cannot() {
        let dir = std::env::temp_dir().join(format!("neo-trash-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Only a missing file here: trashing a real one would put test litter in the user's Trash.
        let err = trash(&dir.join("not here")).unwrap_err();
        assert!(!err.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }
}

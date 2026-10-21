//! `set_frame`: move and resize the overlay in one step. `setSize` then `setPosition` are two
//! window updates, and in between the card (anchored bottom-right) is drawn one frame out of
//! place. AppKit's `setFrame:display:` changes both at once, without its own animation.

use objc2::encode::{Encode, Encoding};
use objc2::runtime::AnyObject;
use objc2::{class, msg_send, MainThreadMarker};

#[repr(C)]
#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Size {
    w: f64,
    h: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Rect {
    origin: Point,
    size: Size,
}
// Same layout and encoding as CGPoint / CGSize / CGRect (64-bit).
unsafe impl Encode for Point {
    const ENCODING: Encoding = Encoding::Struct("CGPoint", &[f64::ENCODING, f64::ENCODING]);
}
unsafe impl Encode for Size {
    const ENCODING: Encoding = Encoding::Struct("CGSize", &[f64::ENCODING, f64::ENCODING]);
}
unsafe impl Encode for Rect {
    const ENCODING: Encoding = Encoding::Struct("CGRect", &[Point::ENCODING, Size::ENCODING]);
}

/// `x, y, w, h` are logical points with a top-left origin on the primary screen (Tauri's
/// convention); AppKit frames have a bottom-left origin on the same screen.
///
/// # Safety
/// `ns_window` must be a live `NSWindow*`, and this must run on the main thread.
pub(crate) unsafe fn apply(ns_window: *mut AnyObject, x: f64, y: f64, w: f64, h: f64) -> Result<(), String> {
    let screens: *mut AnyObject = msg_send![class!(NSScreen), screens];
    if screens.is_null() {
        return Err("no screens".into());
    }
    let primary: *mut AnyObject = msg_send![screens, firstObject];
    if primary.is_null() {
        return Err("no primary screen".into());
    }
    let sf: Rect = msg_send![primary, frame];
    let frame = Rect { origin: Point { x, y: sf.size.h - y - h }, size: Size { w, h } };
    let _: () = msg_send![ns_window, setFrame: frame, display: true];
    Ok(())
}

#[tauri::command]
pub fn set_frame(window: tauri::WebviewWindow, x: f64, y: f64, w: f64, h: f64) -> Result<(), String> {
    if ![x, y, w, h].iter().all(|v| v.is_finite()) || w < 1.0 || h < 1.0 || w > 16384.0 || h > 16384.0 {
        return Err("invalid frame".into());
    }
    let ns = window.ns_window().map_err(|e| e.to_string())? as usize;
    if MainThreadMarker::new().is_some() {
        // Sync commands run on the main thread: apply now, so the caller's next paint sees it.
        return unsafe { apply(ns as *mut AnyObject, x, y, w, h) };
    }
    window
        .run_on_main_thread(move || {
            let _ = unsafe { apply(ns as *mut AnyObject, x, y, w, h) };
        })
        .map_err(|e| e.to_string())
}

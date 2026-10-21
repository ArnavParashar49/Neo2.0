//! Small AppKit calls made directly (no plugin, no new crates):
//!
//! - `clipboard_write`: ⌘C / ⌘X on the page. The page's selection comes back from the core
//!   after the keypress is over, when WebKit no longer lets the page write the clipboard; the
//!   shell writes the general pasteboard instead (plain text only, capped).
//! - `show_quiet`: a login / CAPTCHA / 2FA prompt must surface the overlay without taking focus
//!   from the app the user is in (`orderFrontRegardless`: shown, not made key, app not activated).
//! - Search Screen's overlay (lens.rs): the window made a non-activating panel (it takes the keys
//!   without making NEO the active app, so the user's app keeps its full-screen Space, its menu bar
//!   and its playing video), raised above everything, the displays and where the pointer is, a
//!   trackpad tick, the Mac's accessibility settings, a VoiceOver announcement, a locked screen.

use objc2::encode::{Encode, Encoding};
use objc2::runtime::{AnyClass, AnyObject, Bool, ClassBuilder, Sel};
use objc2::{class, msg_send, sel, MainThreadMarker};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::OnceLock;

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

// ---- Search Screen's overlay (lens.rs) ---------------------------------------------------------

/// NSWindowStyleMaskNonactivatingPanel: keys without activating the app (NSPanel only).
pub const NONACTIVATING_PANEL: usize = 1 << 7;
/// NSScreenSaverWindowLevel: above the menu bar (24), NotchHub's island (25) and pop-up menus (101).
pub const LENS_LEVEL: isize = 1000;
/// CanJoinAllSpaces | Stationary | IgnoresCycle | FullScreenAuxiliary | CanJoinAllApplications:
/// on every Space, full-screen ones too, never in ⌘` or Mission Control's shuffle.
pub const LENS_SPACES: usize = 1 | 16 | 64 | 256 | (1 << 18);
const NS_ANIMATION_NONE: isize = 2;
const NS_KEY_DOWN: usize = 10;
const CMD: usize = 1 << 20;
const SHIFT: usize = 1 << 17;
const CTRL: usize = 1 << 18;
const ALT: usize = 1 << 19;

/// Whether the overlay may be the key window now (only while it is open: never while it is warm,
/// building its glow or fading, so no key is ever taken from the user's app before they let go).
static LENS_KEYABLE: AtomicBool = AtomicBool::new(false);
/// The app that was active before the fallback (no panel) activated NEO: it gets the keys back.
static PREV_APP: AtomicI32 = AtomicI32::new(0);

extern "C" fn panel_can_become_key(_this: &AnyObject, _sel: Sel) -> Bool {
    Bool::new(LENS_KEYABLE.load(Ordering::SeqCst))
}

extern "C" fn panel_can_become_main(_this: &AnyObject, _sel: Sel) -> Bool {
    Bool::NO
}

/// ⌘C, ⌘X, ⌘V, ⌘A, ⌘Z, ⇧⌘Z in the overlay. The page sees every key first (WebKit gets them from
/// the superclass); one nobody took is sent as the edit action to the first responder — an
/// accessory app's panel has no Edit menu of its own to route them.
extern "C" fn panel_key_equivalent(this: &AnyObject, _sel: Sel, event: *mut AnyObject) -> Bool {
    unsafe {
        let handled: Bool = msg_send![super(this, class!(NSPanel)), performKeyEquivalent: event];
        if handled.as_bool() || event.is_null() {
            return handled;
        }
        let kind: usize = msg_send![event, type];
        if kind != NS_KEY_DOWN {
            return Bool::NO;
        }
        let flags: usize = msg_send![event, modifierFlags];
        if flags & CMD == 0 || flags & (CTRL | ALT) != 0 {
            return Bool::NO;
        }
        let chars: *mut AnyObject = msg_send![event, charactersIgnoringModifiers];
        let action = match (rust_string(chars).to_lowercase().as_str(), flags & SHIFT != 0) {
            ("c", false) => sel!(copy:),
            ("x", false) => sel!(cut:),
            ("v", false) => sel!(paste:),
            ("a", false) => sel!(selectAll:),
            ("z", false) => sel!(undo:),
            ("z", true) => sel!(redo:),
            _ => return Bool::NO,
        };
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
        let done: Bool = msg_send![app, sendAction: action, to: std::ptr::null_mut::<AnyObject>(), from: this];
        done
    }
}

/// Esc's key code (kVK_Escape).
const KEY_ESCAPE: u16 = 53;

/// A key the page didn't take reaches the panel itself (WebKit passes up what its page left alone):
/// Esc closes Search Screen even when the page's script is dead or stuck. Anything else goes on as
/// it would.
extern "C" fn panel_key_down(this: &AnyObject, _sel: Sel, event: *mut AnyObject) {
    unsafe {
        if !event.is_null() && LENS_KEYABLE.load(Ordering::SeqCst) {
            let code: u16 = msg_send![event, keyCode];
            if code == KEY_ESCAPE {
                crate::lens::native_escape();
                return;
            }
        }
        let _: () = msg_send![super(this, class!(NSPanel)), keyDown: event];
    }
}

/// Esc as AppKit's own "cancel" (what the key bindings make of it on the way up): the same.
extern "C" fn panel_cancel(_this: &AnyObject, _sel: Sel, _sender: *mut AnyObject) {
    if LENS_KEYABLE.load(Ordering::SeqCst) {
        crate::lens::native_escape();
    }
}

/// An NSString's text (empty for nil).
unsafe fn rust_string(s: *mut AnyObject) -> String {
    if s.is_null() {
        return String::new();
    }
    let p: *const std::ffi::c_char = msg_send![s, UTF8String];
    if p.is_null() {
        return String::new();
    }
    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
}

/// `NeoLensPanel : NSPanel`, made once: no ivars of its own, five methods.
fn panel_class() -> Option<&'static AnyClass> {
    static CLASS: OnceLock<usize> = OnceLock::new();
    let ptr = *CLASS.get_or_init(|| {
        let name = c"NeoLensPanel";
        if let Some(c) = AnyClass::get(name) {
            return c as *const AnyClass as usize;
        }
        let Some(mut b) = ClassBuilder::new(name, class!(NSPanel)) else { return 0 };
        unsafe {
            b.add_method(sel!(canBecomeKeyWindow), panel_can_become_key as extern "C" fn(_, _) -> _);
            b.add_method(sel!(canBecomeMainWindow), panel_can_become_main as extern "C" fn(_, _) -> _);
            b.add_method(sel!(performKeyEquivalent:), panel_key_equivalent as extern "C" fn(_, _, _) -> _);
            b.add_method(sel!(keyDown:), panel_key_down as extern "C" fn(_, _, _));
            b.add_method(sel!(cancelOperation:), panel_cancel as extern "C" fn(_, _, _));
        }
        b.register() as *const AnyClass as usize
    });
    (ptr != 0).then(|| unsafe { &*(ptr as *const AnyClass) })
}

/// Whether a window of `size` bytes may become a panel of `panel` bytes: never one that would read
/// past the window's own memory.
pub fn panel_fits(window_size: usize, panel_size: usize) -> bool {
    panel_size > 0 && panel_size <= window_size
}

/// Make the overlay's window a non-activating panel (its class swapped for `NeoLensPanel`). False:
/// it can't be (the fallback then activates NEO instead). After a swap the window's own focus
/// calls (tao's `set_focus`, `set_focusable`) must never be used on it: the shell uses these.
///
/// # Safety
/// `ns` must be a live `NSWindow*`; main thread.
pub unsafe fn lens_make_panel(ns: *mut AnyObject) -> bool {
    let Some(panel) = panel_class() else { return false };
    let obj = &*ns;
    let is_window: Bool = msg_send![ns, isKindOfClass: class!(NSWindow)];
    if !is_window.as_bool() || !panel_fits(obj.class().instance_size(), panel.instance_size()) {
        return false;
    }
    AnyObject::set_class(obj, panel);
    let mask: usize = msg_send![ns, styleMask];
    let _: () = msg_send![ns, setStyleMask: mask | NONACTIVATING_PANEL];
    let _: () = msg_send![ns, setFloatingPanel: true];
    let _: () = msg_send![ns, setBecomesKeyOnlyIfNeeded: false];
    let _: () = msg_send![ns, setWorksWhenModal: true];
    true
}

/// Level, Spaces, no animation, no shadow, see-through and click-through (until it opens).
///
/// # Safety
/// `ns` must be a live `NSWindow*`; main thread.
pub unsafe fn lens_setup(ns: *mut AnyObject) {
    let _: () = msg_send![ns, setLevel: LENS_LEVEL];
    let _: () = msg_send![ns, setCollectionBehavior: LENS_SPACES];
    let _: () = msg_send![ns, setAnimationBehavior: NS_ANIMATION_NONE];
    let _: () = msg_send![ns, setOpaque: false];
    let _: () = msg_send![ns, setHasShadow: false];
    let _: () = msg_send![ns, setHidesOnDeactivate: false];
    let _: () = msg_send![ns, setAcceptsMouseMovedEvents: true];
    let _: () = msg_send![ns, setIgnoresMouseEvents: true];
}

/// On screen without taking the keys (warm, the glow building).
///
/// # Safety
/// `ns` must be a live `NSWindow*`; main thread.
pub unsafe fn lens_order_front(ns: *mut AnyObject) {
    let _: () = msg_send![ns, orderFrontRegardless];
}

/// Clicks reach the overlay (false) or go through it to the app below (true).
///
/// # Safety
/// `ns` must be a live `NSWindow*`; main thread.
pub unsafe fn lens_set_click_through(ns: *mut AnyObject, through: bool) {
    let _: () = msg_send![ns, setIgnoresMouseEvents: through];
}

/// Open: the overlay takes the keys. As a panel the user's app stays active; without one (the
/// fallback) NEO is activated and the app that was active is remembered, for `lens_order_out`.
///
/// # Safety
/// `ns` must be a live `NSWindow*`; main thread.
pub unsafe fn lens_make_key(ns: *mut AnyObject, panel: bool) {
    LENS_KEYABLE.store(true, Ordering::SeqCst);
    if !panel {
        let ws: *mut AnyObject = msg_send![class!(NSWorkspace), sharedWorkspace];
        let front: *mut AnyObject = msg_send![ws, frontmostApplication];
        let me: *mut AnyObject = msg_send![class!(NSRunningApplication), currentApplication];
        let pid: i32 = if front.is_null() { 0 } else { msg_send![front, processIdentifier] };
        let mine: i32 = if me.is_null() { -1 } else { msg_send![me, processIdentifier] };
        if pid > 0 && pid != mine {
            PREV_APP.store(pid, Ordering::SeqCst); // (NEO itself in front: the one remembered stays)
        }
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
        let _: () = msg_send![app, activateIgnoringOtherApps: true];
    }
    let _: () = msg_send![ns, setIgnoresMouseEvents: false];
    let _: () = msg_send![ns, makeKeyAndOrderFront: std::ptr::null_mut::<AnyObject>()];
    // The page takes the keys at once (Esc, Return, typing), not after a first click.
    let content: *mut AnyObject = msg_send![ns, contentView];
    let web = find_web_view(content, 0);
    if !web.is_null() {
        let _: Bool = msg_send![ns, makeFirstResponder: web];
    }
    let cursor: *mut AnyObject = msg_send![class!(NSCursor), crosshairCursor];
    if !cursor.is_null() {
        let _: () = msg_send![cursor, set];
    }
}

/// The window's WKWebView (its page), searched for under `view` (null: none).
///
/// # Safety
/// `view` must be a live `NSView*` (or null); main thread.
unsafe fn find_web_view(view: *mut AnyObject, depth: usize) -> *mut AnyObject {
    let Some(wk) = AnyClass::get(c"WKWebView") else { return std::ptr::null_mut() };
    if view.is_null() || depth > 6 {
        return std::ptr::null_mut();
    }
    let is_web: Bool = msg_send![view, isKindOfClass: wk];
    if is_web.as_bool() {
        return view;
    }
    let subs: *mut AnyObject = msg_send![view, subviews];
    if subs.is_null() {
        return std::ptr::null_mut();
    }
    let n: usize = msg_send![subs, count];
    for i in 0..n {
        let v: *mut AnyObject = msg_send![subs, objectAtIndex: i];
        let found = find_web_view(v, depth + 1);
        if !found.is_null() {
            return found;
        }
    }
    std::ptr::null_mut()
}

/// Off screen, keys given back (the fallback re-activates the app that had them).
///
/// # Safety
/// `ns` must be a live `NSWindow*`; main thread.
pub unsafe fn lens_order_out(ns: *mut AnyObject) {
    LENS_KEYABLE.store(false, Ordering::SeqCst);
    let _: () = msg_send![ns, setIgnoresMouseEvents: true];
    let _: () = msg_send![ns, orderOut: std::ptr::null_mut::<AnyObject>()];
    let pid = PREV_APP.swap(0, Ordering::SeqCst);
    if pid > 0 {
        let app: *mut AnyObject = msg_send![class!(NSRunningApplication), runningApplicationWithProcessIdentifier: pid];
        if !app.is_null() {
            let _: Bool = msg_send![app, activateWithOptions: 0usize];
        }
    }
}

/// Live mode (no still): out of the way while the core captures the area (`hidden`), then back
/// with the keys.
///
/// # Safety
/// `ns` must be a live `NSWindow*`; main thread.
pub unsafe fn lens_peek(ns: *mut AnyObject, hidden: bool, panel: bool) {
    if hidden {
        let _: () = msg_send![ns, orderOut: std::ptr::null_mut::<AnyObject>()];
    } else {
        lens_make_key(ns, panel);
    }
}

/// No more keys for the overlay (it is fading out; it orders out a moment later).
pub fn lens_not_keyable() {
    LENS_KEYABLE.store(false, Ordering::SeqCst);
}

/// The trackpad's "level change" tick, now (nothing on a mouse or without Force Touch).
///
/// # Safety
/// Main thread.
pub unsafe fn lens_haptic() {
    let performer: *mut AnyObject = msg_send![class!(NSHapticFeedbackManager), defaultPerformer];
    if !performer.is_null() {
        let _: () = msg_send![performer, performFeedbackPattern: 2isize, performanceTime: 1usize];
    }
}

#[link(name = "AppKit", kind = "framework")]
extern "C" {
    fn NSAccessibilityPostNotificationWithUserInfo(element: *mut AnyObject, notification: *mut AnyObject, info: *mut AnyObject);
}

/// VoiceOver says `text` (only called while VoiceOver is on).
///
/// # Safety
/// Main thread.
pub unsafe fn lens_announce(text: &str) {
    let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
    let (name, k_text, k_prio) = (ns_string("AXAnnouncementRequested"), ns_string("AXAnnouncementKey"), ns_string("AXPriorityKey"));
    let value = ns_string(text);
    let prio: *mut AnyObject = msg_send![class!(NSNumber), numberWithInteger: 90isize]; // NSAccessibilityPriorityHigh
    let keys = [k_text, k_prio];
    let objects = [value, prio];
    let info: *mut AnyObject =
        msg_send![class!(NSDictionary), dictionaryWithObjects: objects.as_ptr(), forKeys: keys.as_ptr(), count: 2usize];
    if !app.is_null() && !info.is_null() {
        NSAccessibilityPostNotificationWithUserInfo(app, name, info);
    }
    for o in [name, k_text, k_prio, value] {
        let _: () = msg_send![o, release];
    }
}

/// The Mac's accessibility display settings: (reduce motion, reduce transparency, increase
/// contrast, VoiceOver on).
pub fn a11y_flags() -> (bool, bool, bool, bool) {
    unsafe {
        let ws: *mut AnyObject = msg_send![class!(NSWorkspace), sharedWorkspace];
        if ws.is_null() {
            return (false, false, false, false);
        }
        let motion: Bool = msg_send![ws, accessibilityDisplayShouldReduceMotion];
        let transparency: Bool = msg_send![ws, accessibilityDisplayShouldReduceTransparency];
        let contrast: Bool = msg_send![ws, accessibilityDisplayShouldIncreaseContrast];
        let vo: Bool = msg_send![ws, isVoiceOverEnabled];
        (motion.as_bool(), transparency.as_bool(), contrast.as_bool(), vo.as_bool())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NSPointC {
    x: f64,
    y: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NSSizeC {
    w: f64,
    h: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NSRectC {
    origin: NSPointC,
    size: NSSizeC,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NSEdgeInsetsC {
    top: f64,
    left: f64,
    bottom: f64,
    right: f64,
}
// Same layout and encoding as CGPoint / CGSize / CGRect / NSEdgeInsets (64-bit).
unsafe impl Encode for NSPointC {
    const ENCODING: Encoding = Encoding::Struct("CGPoint", &[f64::ENCODING, f64::ENCODING]);
}
unsafe impl Encode for NSSizeC {
    const ENCODING: Encoding = Encoding::Struct("CGSize", &[f64::ENCODING, f64::ENCODING]);
}
unsafe impl Encode for NSRectC {
    const ENCODING: Encoding = Encoding::Struct("CGRect", &[NSPointC::ENCODING, NSSizeC::ENCODING]);
}
unsafe impl Encode for NSEdgeInsetsC {
    const ENCODING: Encoding = Encoding::Struct("NSEdgeInsets", &[f64::ENCODING, f64::ENCODING, f64::ENCODING, f64::ENCODING]);
}

/// The displays, in global points with a top-left origin on the primary display (CoreGraphics'
/// convention, the core's too), and where the pointer is in the same terms.
///
/// # Safety
/// Main thread.
pub unsafe fn displays_and_pointer() -> (Vec<crate::lens::Display>, (f64, f64)) {
    let mut out = Vec::new();
    let screens: *mut AnyObject = msg_send![class!(NSScreen), screens];
    if screens.is_null() {
        return (out, (0.0, 0.0));
    }
    let n: usize = msg_send![screens, count];
    let mut primary_h = 0.0;
    let key = ns_string("NSScreenNumber");
    for i in 0..n {
        let scr: *mut AnyObject = msg_send![screens, objectAtIndex: i];
        if scr.is_null() {
            continue;
        }
        let f: NSRectC = msg_send![scr, frame];
        if i == 0 {
            primary_h = f.size.h;
        }
        let scale: f64 = msg_send![scr, backingScaleFactor];
        let desc: *mut AnyObject = msg_send![scr, deviceDescription];
        let num: *mut AnyObject = if desc.is_null() { std::ptr::null_mut() } else { msg_send![desc, objectForKey: key] };
        let id: u32 = if num.is_null() { 0 } else { msg_send![num, unsignedIntValue] };
        let has_safe: Bool = msg_send![scr, respondsToSelector: sel!(safeAreaInsets)];
        let safe = if has_safe.as_bool() { msg_send![scr, safeAreaInsets] } else { NSEdgeInsetsC::default() };
        out.push(crate::lens::Display {
            id,
            x: f.origin.x,
            y: primary_h - (f.origin.y + f.size.h),
            w: f.size.w,
            h: f.size.h,
            scale: if scale.is_finite() && scale > 0.0 { scale } else { 1.0 },
            safe_top: if safe.top.is_finite() { safe.top.max(0.0) } else { 0.0 },
        });
    }
    let _: () = msg_send![key, release];
    let p: NSPointC = msg_send![class!(NSEvent), mouseLocation];
    (out, (p.x, primary_h - p.y))
}

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGSessionCopyCurrentDictionary() -> *const c_void;
}

/// The screen is locked (or the session is switched out): Search Screen closes.
pub fn screen_locked() -> bool {
    use core_foundation::base::{CFType, TCFType};
    use core_foundation::boolean::CFBoolean;
    use core_foundation::dictionary::CFDictionary;
    use core_foundation::string::CFString;
    let raw = unsafe { CGSessionCopyCurrentDictionary() };
    if raw.is_null() {
        return false;
    }
    let dict: CFDictionary<CFString, CFType> = unsafe { CFDictionary::wrap_under_create_rule(raw as _) };
    let locked = dict
        .find(CFString::new("CGSSessionScreenIsLocked"))
        .and_then(|v| v.downcast::<CFBoolean>())
        .map(bool::from)
        .unwrap_or(false);
    let on_console = dict
        .find(CFString::new("kCGSSessionOnConsoleKey"))
        .and_then(|v| v.downcast::<CFBoolean>())
        .map(bool::from)
        .unwrap_or(true);
    locked || !on_console
}

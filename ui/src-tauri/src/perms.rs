//! The permissions NEO needs from macOS, owned by the app (the process macOS attributes them to).
//!
//! NEO.app checks them itself, asks at most once per app version (never on every start — the
//! system prompt is a nag, not a status), and shows what's missing in its menu with a shortcut to
//! the right pane of System Settings. The core never prompts when the app launched it.

use std::fs;
use std::path::PathBuf;

use core_foundation::base::TCFType;
use core_foundation::boolean::CFBoolean;
use core_foundation::dictionary::{CFDictionary, CFDictionaryRef};
use core_foundation::string::CFString;

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn AXIsProcessTrusted() -> bool;
    fn AXIsProcessTrustedWithOptions(options: CFDictionaryRef) -> bool;
}

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGPreflightScreenCaptureAccess() -> bool;
    fn CGRequestScreenCaptureAccess() -> bool;
}

pub fn accessibility() -> bool {
    unsafe { AXIsProcessTrusted() }
}

pub fn screen_recording() -> bool {
    unsafe { CGPreflightScreenCaptureAccess() }
}

fn marker() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
        .join(".neo")
        .join("permissions-asked")
}

/// Show macOS's own prompts for whatever is missing — once per app version.
pub fn ask_once(version: &str) {
    if accessibility() && screen_recording() {
        return;
    }
    if fs::read_to_string(marker()).map(|v| v.trim() == version).unwrap_or(false) {
        return; // asked for this version already: the menu shows what's still missing
    }
    let _ = fs::create_dir_all(marker().parent().unwrap());
    let _ = fs::write(marker(), version);
    if !accessibility() {
        let opts = CFDictionary::from_CFType_pairs(&[(
            CFString::new("AXTrustedCheckOptionPrompt"),
            CFBoolean::true_value().as_CFType(),
        )]);
        unsafe {
            AXIsProcessTrustedWithOptions(opts.as_concrete_TypeRef());
        }
    }
    if !screen_recording() {
        unsafe {
            CGRequestScreenCaptureAccess();
        }
    }
}

/// Open the matching pane of System Settings (Privacy & Security).
pub fn open_settings(pane: &str) {
    let url = format!("x-apple.systempreferences:com.apple.preference.security?{pane}");
    let _ = std::process::Command::new("open").arg(url).spawn();
}

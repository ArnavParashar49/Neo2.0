//! The permissions NEO needs from macOS, owned by the app (the process macOS attributes them to).
//!
//! NEO.app checks them itself, asks at most once per app version (never on every start — the
//! system prompt is a nag, not a status), and shows what's missing in its menu with a shortcut to
//! the right pane of System Settings. The core never prompts when the app launched it.
//!
//! On a new Mac nothing is asked at launch until Setup is done: its checklist has a button for each
//! (`ask_accessibility`, `ask_screen_recording`), and once it's done this version counts as asked.

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
    crate::core::home().join(".neo").join("permissions-asked")
}

/// This version has asked (Setup's checklist did): `ask_once` won't again.
pub fn mark_asked(version: &str) {
    let _ = fs::create_dir_all(marker().parent().unwrap());
    let _ = fs::write(marker(), version);
}

/// macOS's Accessibility prompt (it offers System Settings).
pub fn ask_accessibility() {
    let opts = CFDictionary::from_CFType_pairs(&[(CFString::new("AXTrustedCheckOptionPrompt"), CFBoolean::true_value().as_CFType())]);
    unsafe {
        AXIsProcessTrustedWithOptions(opts.as_concrete_TypeRef());
    }
}

/// macOS's Screen Recording prompt (granting it asks to Quit & Reopen NEO).
pub fn ask_screen_recording() {
    unsafe {
        CGRequestScreenCaptureAccess();
    }
}

/// Show macOS's own prompts for whatever is missing — once per app version.
pub fn ask_once(version: &str) {
    if accessibility() && screen_recording() {
        return;
    }
    if fs::read_to_string(marker()).map(|v| v.trim() == version).unwrap_or(false) {
        return; // asked for this version already: the menu shows what's still missing
    }
    mark_asked(version);
    if !accessibility() {
        ask_accessibility();
    }
    if !screen_recording() {
        ask_screen_recording();
    }
}

/// Open the matching pane of System Settings (Privacy & Security).
pub fn open_settings(pane: &str) {
    let url = format!("x-apple.systempreferences:com.apple.preference.security?{pane}");
    let _ = std::process::Command::new("open").arg(url).spawn();
}

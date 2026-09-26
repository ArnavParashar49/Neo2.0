//! ⌥⌘ on its own summons NEO: press Option and Command together and let go, with nothing else
//! in between. macOS's hotkey API needs a real key, so this watches the keyboard with a
//! listen-only event tap (it needs the Accessibility permission NEO already asks for).
//!
//! A *gesture* runs from the first of ⌥/⌘ going down until both are up again. It fires only if
//! both were held together, released within MAX_HOLD, and nothing else happened during the whole
//! gesture: no key (⌥⌘Esc, ⌘⇥ then ⌥), no click (⌥⌘-click), no ⇧ or ⌃ (⇧⌥⌘ shortcuts).

use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use core_foundation::base::TCFType;
use core_foundation::runloop::{kCFRunLoopCommonModes, CFRunLoop};
use core_graphics::event::{
    CGEventFlags, CGEventTap, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement, CGEventType, CallbackResult,
};

const MAX_HOLD: Duration = Duration::from_millis(900); // a longer hold is something else
const RETRY: Duration = Duration::from_secs(3); // until Accessibility is granted

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGEventTapEnable(tap: *mut c_void, enable: bool);
}

#[derive(Default)]
struct State {
    active: bool, // ⌥ or ⌘ has been held since the last time both were up
    armed_at: Option<Instant>,
    dirty: bool, // something else happened during this gesture: it's a shortcut, not the chord
}

/// Pure decision logic, one event at a time (unit-tested below). True = toggle NEO now.
fn step(st: &mut State, kind: CGEventType, flags: CGEventFlags, now: Instant) -> bool {
    match kind {
        CGEventType::FlagsChanged => {
            let cmd = flags.contains(CGEventFlags::CGEventFlagCommand);
            let alt = flags.contains(CGEventFlags::CGEventFlagAlternate);
            // Caps Lock, Fn and the keypad flag are states, not modifiers someone is pressing.
            let other = flags.intersects(CGEventFlags::CGEventFlagShift | CGEventFlags::CGEventFlagControl);
            if !cmd && !alt {
                let fire = st.active
                    && !st.dirty
                    && !other
                    && st.armed_at.map(|t| now.duration_since(t) <= MAX_HOLD).unwrap_or(false);
                *st = State::default();
                return fire;
            }
            if !st.active {
                *st = State { active: true, ..State::default() };
            }
            if other {
                st.dirty = true; // stays dirty until ⌥ and ⌘ are both up
            }
            if cmd && alt && st.armed_at.is_none() {
                st.armed_at = Some(now);
            }
            false
        }
        CGEventType::KeyDown
        | CGEventType::LeftMouseDown
        | CGEventType::RightMouseDown
        | CGEventType::OtherMouseDown
        | CGEventType::ScrollWheel => {
            if st.active {
                st.dirty = true;
            }
            false
        }
        _ => false,
    }
}

fn install(state: Arc<Mutex<State>>, on_chord: Arc<dyn Fn() + Send + Sync>) -> Option<CGEventTap<'static>> {
    let port: Arc<AtomicPtr<c_void>> = Arc::new(AtomicPtr::new(std::ptr::null_mut()));
    let port_cb = port.clone();
    let tap = CGEventTap::new(
        CGEventTapLocation::Session,
        // Head of the chain: see every key before another tap can swallow it (listen-only, so
        // nothing is changed or delayed).
        CGEventTapPlacement::HeadInsertEventTap,
        CGEventTapOptions::ListenOnly,
        vec![
            CGEventType::FlagsChanged,
            CGEventType::KeyDown,
            CGEventType::LeftMouseDown,
            CGEventType::RightMouseDown,
            CGEventType::OtherMouseDown,
            CGEventType::ScrollWheel,
        ],
        move |_proxy, kind, event| {
            match kind {
                // macOS pauses taps (e.g. while a password field has secure input): resume.
                CGEventType::TapDisabledByTimeout | CGEventType::TapDisabledByUserInput => {
                    let p = port_cb.load(Ordering::SeqCst);
                    if !p.is_null() {
                        unsafe { CGEventTapEnable(p, true) };
                    }
                }
                _ => {
                    if step(&mut state.lock().unwrap(), kind, event.get_flags(), Instant::now()) {
                        on_chord();
                    }
                }
            }
            CallbackResult::Keep
        },
    )
    .ok()?;
    port.store(tap.mach_port().as_concrete_TypeRef() as *mut c_void, Ordering::SeqCst);
    Some(tap)
}

/// Watch for ⌥⌘ on its own thread. `on_chord` runs each time it completes. If the tap can't be
/// created yet (no Accessibility), it keeps retrying and calls `on_ready` once it works. Returns
/// whether it is working right away.
pub fn watch(on_chord: impl Fn() + Send + Sync + 'static, on_ready: impl Fn() + Send + 'static) -> bool {
    let (tx, rx) = std::sync::mpsc::channel::<bool>();
    let on_chord: Arc<dyn Fn() + Send + Sync> = Arc::new(on_chord);
    thread::spawn(move || {
        let state = Arc::new(Mutex::new(State::default()));
        let mut first = true;
        let tap = loop {
            if let Some(tap) = install(state.clone(), on_chord.clone()) {
                break tap;
            }
            if first {
                let _ = tx.send(false);
                first = false;
            }
            thread::sleep(RETRY);
        };
        let source = tap.mach_port().create_runloop_source(0).expect("run loop source");
        CFRunLoop::get_current().add_source(&source, unsafe { kCFRunLoopCommonModes });
        tap.enable();
        if first {
            let _ = tx.send(true);
        } else {
            on_ready();
        }
        CFRunLoop::run_current(); // keeps `tap` alive for the life of the app
    });
    rx.recv_timeout(Duration::from_secs(2)).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(bits: u64) -> CGEventFlags {
        CGEventFlags::from_bits_truncate(bits)
    }
    const CMD: u64 = 0x0010_0000;
    const ALT: u64 = 0x0008_0000;
    const SHIFT: u64 = 0x0002_0000;
    const CTRL: u64 = 0x0004_0000;
    const CAPS: u64 = 0x0001_0000;
    const MS: fn(u64) -> Duration = Duration::from_millis;

    fn run(events: &[(CGEventType, u64, u64)]) -> Vec<bool> {
        let (mut st, t0) = (State::default(), Instant::now());
        events.iter().map(|&(k, fl, ms)| step(&mut st, k, f(fl), t0 + MS(ms))).collect()
    }
    use CGEventType::{FlagsChanged as F, KeyDown as K, LeftMouseDown as M};

    #[test]
    fn press_and_release_option_command_fires_once() {
        assert_eq!(run(&[(F, ALT, 0), (F, ALT | CMD, 10), (F, CMD, 200), (F, 0, 220)]), [false, false, false, true]);
    }

    #[test]
    fn caps_lock_on_does_not_matter() {
        assert_eq!(run(&[(F, CAPS | ALT | CMD, 0), (F, CAPS, 150)]), [false, true]);
    }

    #[test]
    fn a_real_shortcut_does_not_fire() {
        assert_eq!(run(&[(F, ALT | CMD, 0), (K, ALT | CMD, 20), (F, 0, 100)]), [false, false, false]); // ⌥⌘D, ⌥⌘Esc
    }

    #[test]
    fn a_key_before_the_second_modifier_does_not_fire() {
        // ⌘⇥ (switch app) then ⌥ (restore a minimised window) then let go
        assert_eq!(run(&[(F, CMD, 0), (K, CMD, 10), (F, CMD | ALT, 30), (F, 0, 80)]), [false, false, false, false]);
    }

    #[test]
    fn shift_or_control_anywhere_in_the_gesture_does_not_fire() {
        // ⇧ released first, then ⌥⌘: still the ⇧⌥⌘ shortcut
        assert_eq!(run(&[(F, SHIFT | ALT | CMD, 0), (F, ALT | CMD, 30), (F, 0, 60)]), [false, false, false]);
        assert_eq!(run(&[(F, ALT | CMD, 0), (F, ALT | CMD | CTRL, 30), (F, ALT | CMD, 40), (F, 0, 60)]), [false, false, false, false]);
    }

    #[test]
    fn option_command_click_or_a_long_hold_does_not_fire() {
        assert_eq!(run(&[(F, ALT | CMD, 0), (M, ALT | CMD, 10), (F, 0, 50)]), [false, false, false]);
        assert_eq!(run(&[(F, ALT | CMD, 0), (F, 0, 2000)]), [false, false]);
    }

    #[test]
    fn the_next_gesture_starts_clean() {
        assert_eq!(
            run(&[(F, ALT | CMD, 0), (K, ALT | CMD, 10), (F, 0, 40), (F, ALT | CMD, 500), (F, 0, 600)]),
            [false, false, false, false, true]
        );
    }
}

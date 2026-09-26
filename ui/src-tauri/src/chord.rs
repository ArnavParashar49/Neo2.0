//! ⌥⌘ on its own summons NEO: press Option and Command together and let go, with no other key
//! in between. macOS's hotkey API needs a real key, so this watches the keyboard with a
//! listen-only event tap (it needs the Accessibility permission NEO already asks for). A real
//! shortcut that uses the same modifiers — ⌥⌘Esc, ⌥⌘D, ⌥⌘-click — never triggers it.

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

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGEventTapEnable(tap: *mut c_void, enable: bool);
}

#[derive(Default)]
struct State {
    armed_at: Option<Instant>,
    dirty: bool, // another key, button or modifier joined in: it's a shortcut, not the chord
}

/// Pure decision logic, one event at a time (kept separate so it can be unit-tested).
/// Returns true when the chord was completed and NEO should toggle.
fn step(st: &mut State, kind: CGEventType, flags: CGEventFlags, now: Instant) -> bool {
    match kind {
        CGEventType::FlagsChanged => {
            let cmd = flags.contains(CGEventFlags::CGEventFlagCommand);
            let alt = flags.contains(CGEventFlags::CGEventFlagAlternate);
            let other = flags.intersects(CGEventFlags::CGEventFlagShift | CGEventFlags::CGEventFlagControl);
            if cmd && alt && !other {
                if st.armed_at.is_none() {
                    *st = State { armed_at: Some(now), dirty: false };
                }
                false
            } else if cmd && alt {
                st.dirty = true; // ⌥⌘ plus ⇧/⌃
                false
            } else if let Some(t) = st.armed_at.take() {
                let fire = !st.dirty && now.duration_since(t) <= MAX_HOLD;
                st.dirty = false;
                fire // one of the two was released: that's the moment it counts
            } else {
                false
            }
        }
        CGEventType::KeyDown
        | CGEventType::LeftMouseDown
        | CGEventType::RightMouseDown
        | CGEventType::OtherMouseDown
        | CGEventType::ScrollWheel => {
            if st.armed_at.is_some() {
                st.dirty = true;
            }
            false
        }
        _ => false,
    }
}

/// Start watching on its own thread. `on_chord` runs (off the main thread) each time the chord
/// completes. Returns false if the tap couldn't be created (no Accessibility permission yet).
pub fn watch(on_chord: impl Fn() + Send + Sync + 'static) -> bool {
    let (tx, rx) = std::sync::mpsc::channel::<bool>();
    let on_chord = Arc::new(on_chord);
    thread::spawn(move || {
        let state = Arc::new(Mutex::new(State::default()));
        let port: Arc<AtomicPtr<c_void>> = Arc::new(AtomicPtr::new(std::ptr::null_mut()));
        let port_cb = port.clone();
        let cb = on_chord.clone();
        let tap = CGEventTap::new(
            CGEventTapLocation::Session,
            CGEventTapPlacement::TailAppendEventTap,
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
                        let fire = step(&mut state.lock().unwrap(), kind, event.get_flags(), Instant::now());
                        if fire {
                            cb();
                        }
                    }
                }
                CallbackResult::Keep
            },
        );
        let Ok(tap) = tap else {
            let _ = tx.send(false);
            return;
        };
        port.store(tap.mach_port().as_concrete_TypeRef() as *mut c_void, Ordering::SeqCst);
        let source = tap.mach_port().create_runloop_source(0).expect("run loop source");
        CFRunLoop::get_current().add_source(&source, unsafe { kCFRunLoopCommonModes });
        tap.enable();
        let _ = tx.send(true);
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

    #[test]
    fn press_and_release_option_command_fires() {
        let (mut st, t0) = (State::default(), Instant::now());
        assert!(!step(&mut st, CGEventType::FlagsChanged, f(ALT), t0));
        assert!(!step(&mut st, CGEventType::FlagsChanged, f(ALT | CMD), t0));
        assert!(step(&mut st, CGEventType::FlagsChanged, f(CMD), t0 + Duration::from_millis(200)));
        assert!(!step(&mut st, CGEventType::FlagsChanged, f(0), t0 + Duration::from_millis(250)));
    }

    #[test]
    fn a_real_shortcut_does_not_fire() {
        let (mut st, t0) = (State::default(), Instant::now());
        step(&mut st, CGEventType::FlagsChanged, f(ALT | CMD), t0);
        step(&mut st, CGEventType::KeyDown, f(ALT | CMD), t0); // ⌥⌘D, ⌥⌘Esc…
        assert!(!step(&mut st, CGEventType::FlagsChanged, f(0), t0 + Duration::from_millis(100)));
    }

    #[test]
    fn extra_modifiers_or_a_long_hold_do_not_fire() {
        let (mut st, t0) = (State::default(), Instant::now());
        step(&mut st, CGEventType::FlagsChanged, f(ALT | CMD), t0);
        step(&mut st, CGEventType::FlagsChanged, f(ALT | CMD | SHIFT), t0);
        assert!(!step(&mut st, CGEventType::FlagsChanged, f(0), t0 + Duration::from_millis(100)));
        let mut st = State::default();
        step(&mut st, CGEventType::FlagsChanged, f(ALT | CMD), t0);
        assert!(!step(&mut st, CGEventType::FlagsChanged, f(0), t0 + Duration::from_secs(2)));
    }

    #[test]
    fn option_command_click_does_not_fire() {
        let (mut st, t0) = (State::default(), Instant::now());
        step(&mut st, CGEventType::FlagsChanged, f(ALT | CMD), t0);
        step(&mut st, CGEventType::LeftMouseDown, f(ALT | CMD), t0);
        assert!(!step(&mut st, CGEventType::FlagsChanged, f(0), t0 + Duration::from_millis(100)));
    }
}

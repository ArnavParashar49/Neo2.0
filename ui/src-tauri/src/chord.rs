//! ⌥⌘ on its own summons NEO: press Option and Command together and let go, with nothing else
//! in between. macOS's hotkey API needs a real key, so this watches the keyboard with an event
//! tap. It is a pass-through tap (every event is returned untouched), which macOS covers with the
//! Accessibility permission NEO already has — a listen-only tap would need a second permission,
//! Input Monitoring, and its own prompt.
//!
//! A *gesture* runs from the first of ⌥/⌘ going down until both are up again. It fires only if
//! both were held together, released within MAX_HOLD, and nothing else happened during the whole
//! gesture: no key (⌥⌘Esc, ⌘⇥ then ⌥), no click (⌥⌘-click), no ⇧ or ⌃ (⇧⌥⌘ shortcuts).
//!
//! Holding the same two keys a little longer is Search Screen (lens.rs): a glow builds along the
//! screen's edges while they are held, and once they have been held for the hold time
//! (NEO_LENS_HOLD_MS, 1 s unless the core says otherwise) letting go opens the overlay. The quick
//! tap keeps its rule exactly: with the default hold it is the same 900 ms; a shorter hold shrinks
//! the tap's window to `hold − 100` ms, so the two never overlap. The hold is a second reading of
//! the same events, never a different handling of them: nothing is swallowed, nothing becomes key
//! until the user lets go cleanly, and any key, click, scroll, ⇧ or ⌃ during the hold calls it off
//! (it was a shortcut — ⌥⌘Esc, ⌥⌘D, NotchHub's ⌥⌘V — and the key reaches the app as it always did).
//!
//! What the hold says goes out as `HoldEvent`s: Warm (make the overlay ready, invisible), Preview
//! (the glow starts building), Arm (held long enough: the still is taken), Commit (let go cleanly:
//! open) or Cancel (and why). Time-based steps come from a second thread (`chord-hold`) that sleeps
//! until the next deadline; the tap's callback only takes a lock, steps the state and wakes it.
//! It never says which key was pressed, only that something happened.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use core_foundation::base::TCFType;
use core_foundation::runloop::{kCFRunLoopCommonModes, CFRunLoop};
use core_graphics::event::{
    CGEventFlags, CGEventTap, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement, CGEventType, CallbackResult,
};

const MAX_HOLD: Duration = Duration::from_millis(900); // a longer hold is something else
const RETRY: Duration = Duration::from_secs(3); // until Accessibility is granted
/// Search Screen's hold time: the core's NEO_LENS_HOLD_MS, clamped here too.
pub const HOLD_MS: (u32, u32) = (600, 3000);
pub const DEFAULT_HOLD_MS: u32 = 1000;
/// Still holding this long after the hold armed: stuck keys, Sticky Keys, a cat. It gives up.
const GIVE_UP_AFTER: Duration = Duration::from_millis(4000);
/// The timer thread's sleep when nothing is due.
const IDLE_WAIT: Duration = Duration::from_secs(3600);

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGEventTapEnable(tap: *mut c_void, enable: bool);
    fn CGEventSourceKeyState(state: i32, key: u16) -> bool;
}

#[link(name = "Carbon", kind = "framework")]
extern "C" {
    fn IsSecureEventInputEnabled() -> u8;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFPreferencesCopyAppValue(key: *const c_void, app: *const c_void) -> *const c_void;
}

/// When each step of a hold happens, measured from the instant ⌥ and ⌘ are both down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timing {
    /// A clean release within this summons NEO (the quick tap).
    pub tap: Duration,
    /// The overlay is made ready (invisible, click-through).
    pub warm: Duration,
    /// The glow starts building.
    pub preview: Duration,
    /// Held long enough: armed (full glow, the still is taken); letting go now opens it.
    pub arm: Duration,
    /// Still held: given up.
    pub give_up: Duration,
    /// Whether a hold can happen at all (false: the quick tap only).
    pub hold: bool,
}

impl Timing {
    /// The quick tap alone, exactly as before Search Screen.
    pub fn tap_only() -> Self {
        Timing { tap: MAX_HOLD, warm: Duration::MAX, preview: Duration::MAX, arm: Duration::MAX, give_up: Duration::MAX, hold: false }
    }

    /// The tap and the hold for a hold time (clamped to 600–3000 ms).
    pub fn for_hold(hold_ms: u32) -> Self {
        let h = hold_ms.clamp(HOLD_MS.0, HOLD_MS.1) as u64;
        let ms = Duration::from_millis;
        let preview = 450.min(h - 300);
        Timing {
            tap: ms(900.min(h - 100)),
            warm: ms(150.max(preview - 200)),
            preview: ms(preview),
            arm: ms(h),
            give_up: ms(h) + GIVE_UP_AFTER,
            hold: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Idle,
    /// Both down, clean, nothing shown yet.
    Held,
    Warm,
    Preview,
    Armed,
    /// No hold for the rest of this gesture.
    Off,
}

/// Why a hold that was showing was called off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    Key,
    Click,
    Scroll,
    /// ⇧ or ⌃ joined in (or was down at the release).
    Modifier,
    /// Let go before it armed (or one of the two let go early).
    Released,
    TooLong,
    /// The keys weren't physically down when it would arm (latched by Sticky Keys).
    Sticky,
    /// The tap was paused by macOS (events may have been missed), or Search Screen can't open now.
    Paused,
}

impl Why {
    pub fn as_str(self) -> &'static str {
        match self {
            Why::Key => "key",
            Why::Click => "click",
            Why::Scroll => "scroll",
            Why::Modifier => "modifier",
            Why::Released => "released",
            Why::TooLong => "too_long",
            Why::Sticky => "sticky",
            Why::Paused => "paused",
        }
    }
}

/// What a hold says, each with its gesture's number: the shell acts only on the current gesture's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldEvent {
    Warm(u64),
    Preview { g: u64, to_arm_ms: u32 },
    Arm(u64),
    Commit(u64),
    /// Only ever for a gesture that got `Warm`.
    Cancel(u64, Why),
}

/// What the timer thread knows when a step is due (never read in the tap's callback).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gate {
    /// Search Screen is on, a core of this app's is linked and said so, and it isn't open already.
    pub on: bool,
    /// Secure Event Input is on (a password field): keys may not reach the tap, so no hold.
    pub secure: bool,
    /// ⌥ and ⌘ are really down (not latched).
    pub physical: bool,
}

/// What the shell tells the hold (lens.rs sets it from the core's hello; lib.rs owns it).
#[derive(Debug)]
pub struct LensGate {
    /// The core said `lens.on`, and is still linked.
    pub on: AtomicBool,
    pub hold_ms: AtomicU32,
    /// Search Screen is open now.
    pub open: AtomicBool,
}

impl Default for LensGate {
    fn default() -> Self {
        LensGate { on: AtomicBool::new(false), hold_ms: AtomicU32::new(DEFAULT_HOLD_MS), open: AtomicBool::new(false) }
    }
}

impl LensGate {
    /// The timing for a gesture that starts now: the quick tap alone while Search Screen is off.
    pub fn timing(&self) -> Timing {
        if self.on.load(Ordering::SeqCst) {
            Timing::for_hold(self.hold_ms.load(Ordering::SeqCst))
        } else {
            Timing::tap_only()
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct State {
    active: bool, // ⌥ or ⌘ has been held since the last time both were up
    armed_at: Option<Instant>,
    dirty: bool, // something else happened during this gesture: it's a shortcut, not the chord
    phase: Phase,
    g: u64,         // the gesture's number: one more at each both-down
    timing: Timing, // taken at both-down
}

impl Default for State {
    fn default() -> Self {
        State { active: false, armed_at: None, dirty: false, phase: Phase::Idle, g: 0, timing: Timing::tap_only() }
    }
}

impl State {
    /// No hold for the rest of this gesture; a Cancel if one was showing.
    fn stop(&mut self, why: Why) -> Option<HoldEvent> {
        let showing = matches!(self.phase, Phase::Warm | Phase::Preview | Phase::Armed);
        self.phase = Phase::Off;
        showing.then_some(HoldEvent::Cancel(self.g, why))
    }
}

/// Pure decision logic, one event at a time (unit-tested below). True = toggle NEO now. (The tap
/// alone, as it was before Search Screen: the app uses `step_hold`, which is this plus the hold.)
#[cfg(test)]
fn step(st: &mut State, kind: CGEventType, flags: CGEventFlags, now: Instant) -> bool {
    step_hold(st, kind, flags, now, &Timing::tap_only()).0
}

/// The tap's decision and the hold's, one event at a time. `fresh`: the timing a gesture starting
/// now gets. Returns whether to toggle NEO now (the quick tap), and what the hold says, if anything.
fn step_hold(st: &mut State, kind: CGEventType, flags: CGEventFlags, now: Instant, fresh: &Timing) -> (bool, Option<HoldEvent>) {
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
                    && st.armed_at.map(|t| now.duration_since(t) <= st.timing.tap).unwrap_or(false);
                let ev = match st.phase {
                    Phase::Armed if !st.dirty && !other => Some(HoldEvent::Commit(st.g)),
                    Phase::Armed => Some(HoldEvent::Cancel(st.g, Why::Modifier)),
                    Phase::Warm | Phase::Preview => Some(HoldEvent::Cancel(st.g, Why::Released)),
                    _ => None,
                };
                *st = State { g: st.g, ..State::default() };
                return (fire, ev);
            }
            if !st.active {
                *st = State { active: true, g: st.g, ..State::default() };
            }
            let mut ev = None;
            if other {
                st.dirty = true; // stays dirty until ⌥ and ⌘ are both up
                ev = st.stop(Why::Modifier);
            }
            if cmd && alt {
                if st.armed_at.is_none() {
                    st.armed_at = Some(now);
                    st.g += 1;
                    st.timing = *fresh;
                    st.phase = if !st.dirty && !other && fresh.hold { Phase::Held } else { Phase::Off };
                }
            } else if st.armed_at.is_some() && st.phase != Phase::Armed {
                // One of the two let go while the other is still held, before the hold armed. Once
                // armed it waits: the commit comes when both are up.
                ev = ev.or(st.stop(Why::Released));
            }
            (false, ev)
        }
        CGEventType::KeyDown
        | CGEventType::LeftMouseDown
        | CGEventType::RightMouseDown
        | CGEventType::OtherMouseDown
        | CGEventType::ScrollWheel => {
            if !st.active {
                return (false, None);
            }
            st.dirty = true;
            let why = match kind {
                CGEventType::KeyDown => Why::Key,
                CGEventType::ScrollWheel => Why::Scroll,
                _ => Why::Click,
            };
            (false, st.stop(why))
        }
        _ => (false, None),
    }
}

/// The hold's time-based steps (the timer thread): at most one phase per call — call again until
/// it says nothing.
fn tick(st: &mut State, now: Instant, gate: &Gate) -> Option<HoldEvent> {
    let at = st.armed_at?;
    if !st.active {
        return None;
    }
    let (t, g) = (st.timing, st.g);
    let held = now.saturating_duration_since(at);
    match st.phase {
        Phase::Held if held >= t.warm => {
            if gate.on && !gate.secure {
                st.phase = Phase::Warm;
                Some(HoldEvent::Warm(g))
            } else {
                st.phase = Phase::Off; // nothing was shown: nothing to call off
                None
            }
        }
        Phase::Warm if held >= t.preview => {
            st.phase = Phase::Preview;
            let to_arm_ms = t.arm.saturating_sub(held).as_millis().min(u32::MAX as u128) as u32;
            Some(HoldEvent::Preview { g, to_arm_ms })
        }
        Phase::Preview if held >= t.arm => {
            if !gate.on || gate.secure {
                st.stop(Why::Paused)
            } else if !gate.physical {
                st.stop(Why::Sticky)
            } else {
                st.phase = Phase::Armed;
                Some(HoldEvent::Arm(g))
            }
        }
        Phase::Armed if held >= t.give_up => st.stop(Why::TooLong),
        _ => None,
    }
}

/// When the hold's next time-based step is due (None: nothing is waiting on the clock).
fn next_deadline(st: &State) -> Option<Instant> {
    let at = st.armed_at?;
    if !st.active {
        return None;
    }
    let t = st.timing;
    match st.phase {
        Phase::Held => Some(at + t.warm),
        Phase::Warm => Some(at + t.preview),
        Phase::Preview => Some(at + t.arm),
        Phase::Armed => Some(at + t.give_up),
        _ => None,
    }
}

/// macOS paused the tap: events may have been missed, so a hold that was showing is called off.
fn paused(st: &mut State) -> Option<HoldEvent> {
    if st.active {
        st.stop(Why::Paused)
    } else {
        None
    }
}

/// Secure Event Input (a password field has the keyboard): keys may not reach the tap.
fn secure_input() -> bool {
    unsafe { IsSecureEventInputEnabled() != 0 }
}

/// Whether Sticky Keys is on (Accessibility → Keyboard). Unreadable: taken as off.
fn sticky_keys_on() -> bool {
    use core_foundation::base::CFType;
    use core_foundation::boolean::CFBoolean;
    use core_foundation::string::CFString;
    let (key, app) = (CFString::new("stickyKey"), CFString::new("com.apple.universalaccess"));
    let v = unsafe { CFPreferencesCopyAppValue(key.as_concrete_TypeRef() as *const c_void, app.as_concrete_TypeRef() as *const c_void) };
    if v.is_null() {
        return false;
    }
    let v = unsafe { CFType::wrap_under_create_rule(v as _) };
    v.downcast::<CFBoolean>().map(bool::from).unwrap_or(false)
}

/// ⌥ and ⌘ are really held (the keyboard's own state, not the event's flags). Only Sticky Keys can
/// make the two differ — it latches a modifier that was let go — so without it the flags are trusted.
fn physically_down() -> bool {
    const HID: i32 = 1; // kCGEventSourceStateHIDSystemState
    let down = |keys: [u16; 2]| keys.iter().any(|k| unsafe { CGEventSourceKeyState(HID, *k) });
    (down([55, 54]) && down([58, 61])) || !sticky_keys_on() // ⌘ left/right, ⌥ left/right
}

type Shared = Arc<(Mutex<State>, Condvar)>;
type OnHold = Arc<dyn Fn(HoldEvent) + Send + Sync>;

fn install(shared: Shared, gate: Arc<LensGate>, on_chord: Arc<dyn Fn() + Send + Sync>, on_hold: OnHold) -> Option<CGEventTap<'static>> {
    let port: Arc<AtomicPtr<c_void>> = Arc::new(AtomicPtr::new(std::ptr::null_mut()));
    let port_cb = port.clone();
    let tap = CGEventTap::new(
        CGEventTapLocation::Session,
        // Head of the chain: see every key before another tap can swallow it (listen-only, so
        // nothing is changed or delayed).
        CGEventTapPlacement::HeadInsertEventTap,
        CGEventTapOptions::Default, // pass-through: needs Accessibility only (see top)
        vec![
            CGEventType::FlagsChanged,
            CGEventType::KeyDown,
            CGEventType::LeftMouseDown,
            CGEventType::RightMouseDown,
            CGEventType::OtherMouseDown,
            CGEventType::ScrollWheel,
        ],
        move |_proxy, kind, event| {
            let (lock, wake) = &*shared;
            match kind {
                // macOS pauses taps (e.g. while a password field has secure input): resume.
                CGEventType::TapDisabledByTimeout | CGEventType::TapDisabledByUserInput => {
                    let p = port_cb.load(Ordering::SeqCst);
                    if !p.is_null() {
                        unsafe { CGEventTapEnable(p, true) };
                    }
                    if let Ok(mut st) = lock.lock() {
                        if let Some(ev) = paused(&mut st) {
                            on_hold(ev);
                        }
                    }
                }
                _ => {
                    let fresh = gate.timing();
                    if let Ok(mut st) = lock.lock() {
                        let (tap, ev) = step_hold(&mut st, kind, event.get_flags(), Instant::now(), &fresh);
                        // Said under the lock, so the shell hears the hold's events in the order they
                        // happened (the timer thread says its own under it too). Both only post to the
                        // main thread: nothing here waits on it.
                        if let Some(ev) = ev {
                            on_hold(ev);
                        }
                        if tap {
                            on_chord();
                        }
                    }
                    wake.notify_one();
                }
            }
            CallbackResult::Keep
        },
    )
    .ok()?;
    port.store(tap.mach_port().as_concrete_TypeRef() as *mut c_void, Ordering::SeqCst);
    Some(tap)
}

/// The hold's clock: sleeps until the next step is due (or the tap's callback wakes it), then says it.
fn timer(shared: Shared, gate: Arc<LensGate>, on_hold: OnHold) {
    let (lock, wake) = &*shared;
    let Ok(mut st) = lock.lock() else { return };
    loop {
        let now = Instant::now();
        match next_deadline(&st) {
            None => {
                st = match wake.wait_timeout(st, IDLE_WAIT) {
                    Ok((g, _)) => g,
                    Err(_) => return,
                };
                continue;
            }
            Some(due) if due > now => {
                st = match wake.wait_timeout(st, due - now) {
                    Ok((g, _)) => g,
                    Err(_) => return,
                };
                continue;
            }
            Some(_) => {}
        }
        // Due. The gate's system calls are made with the lock let go, so the tap never waits on them.
        let arming = st.phase == Phase::Preview;
        drop(st);
        let now_gate = Gate {
            on: gate.on.load(Ordering::SeqCst) && !gate.open.load(Ordering::SeqCst),
            secure: secure_input(),
            physical: !arming || physically_down(),
        };
        st = match lock.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let now = Instant::now();
        let before = next_deadline(&st);
        while let Some(ev) = tick(&mut st, now, &now_gate) {
            on_hold(ev);
        }
        if next_deadline(&st) == before && before.map(|d| d <= now).unwrap_or(false) {
            // (Never: a due step always moves the phase on. Belt and braces against a busy loop.)
            st.phase = Phase::Off;
        }
    }
}

/// Watch for ⌥⌘ on its own thread. `on_chord` runs each time it completes, `on_hold` with each step
/// of a hold (Search Screen; `gate` says whether one may happen). If the tap can't be created yet
/// (no Accessibility), it keeps retrying and calls `on_ready` once it works. Returns whether it is
/// working right away.
pub fn watch(
    on_chord: impl Fn() + Send + Sync + 'static,
    on_ready: impl Fn() + Send + 'static,
    on_hold: impl Fn(HoldEvent) + Send + Sync + 'static,
    gate: Arc<LensGate>,
) -> bool {
    let (tx, rx) = std::sync::mpsc::channel::<bool>();
    let on_chord: Arc<dyn Fn() + Send + Sync> = Arc::new(on_chord);
    let on_hold: OnHold = Arc::new(on_hold);
    let shared: Shared = Arc::new((Mutex::new(State::default()), Condvar::new()));
    {
        let (shared, gate, on_hold) = (shared.clone(), gate.clone(), on_hold.clone());
        let _ = thread::Builder::new().name("chord-hold".into()).spawn(move || timer(shared, gate, on_hold));
    }
    thread::spawn(move || {
        let mut first = true;
        let tap = loop {
            // Only once Accessibility is granted: creating a tap without it can make macOS prompt,
            // and this loop must never nag.
            if crate::perms::accessibility() {
                if let Some(tap) = install(shared.clone(), gate.clone(), on_chord.clone(), on_hold.clone()) {
                    break tap;
                }
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

    // ---- Search Screen's hold (step_hold / tick on a fake clock) ----------------------------------

    use super::HoldEvent::{Arm, Cancel, Commit, Preview, Warm};
    use CGEventType::{RightMouseDown as R, ScrollWheel as S};

    const ON: Gate = Gate { on: true, secure: false, physical: true };

    #[derive(Debug, PartialEq)]
    enum Out {
        Tap,
        H(HoldEvent),
    }

    /// A gesture's events (kind, flags, ms) with hold time `h`, the timer thread ticking at every
    /// deadline up to each event and then up to `until`. What came out, and when.
    fn hold(h: u32, gate: Gate, events: &[(CGEventType, u64, u64)], until: u64) -> Vec<(u64, Out)> {
        let (mut st, t0, fresh) = (State::default(), Instant::now(), Timing::for_hold(h));
        let mut out = Vec::new();
        let drain = |st: &mut State, upto: u64, out: &mut Vec<(u64, Out)>| {
            while let Some(due) = next_deadline(st) {
                let ms = due.duration_since(t0).as_millis() as u64;
                if ms > upto {
                    break;
                }
                while let Some(ev) = tick(st, t0 + MS(ms), &gate) {
                    out.push((ms, Out::H(ev)));
                }
                assert_ne!(next_deadline(st), Some(due), "a due step moves the phase on");
            }
        };
        for &(k, fl, ms) in events {
            drain(&mut st, ms, &mut out);
            let (tap, ev) = step_hold(&mut st, k, f(fl), t0 + MS(ms), &fresh);
            if let Some(ev) = ev {
                out.push((ms, Out::H(ev)));
            }
            if tap {
                out.push((ms, Out::Tap));
            }
        }
        drain(&mut st, until, &mut out);
        out
    }

    fn taps(out: &[(u64, Out)]) -> usize {
        out.iter().filter(|(_, o)| *o == Out::Tap).count()
    }

    fn commits(out: &[(u64, Out)]) -> usize {
        out.iter().filter(|(_, o)| matches!(o, Out::H(Commit(_)))).count()
    }

    #[test]
    fn a_quick_tap_still_summons_with_search_screen_on() {
        for r in [150, 600, 899] {
            let out = hold(1000, ON, &[(F, ALT | CMD, 0), (F, 0, r)], 6000);
            assert_eq!(taps(&out), 1, "{r}: {out:?}");
            assert_eq!(commits(&out), 0, "{r}");
            assert_eq!(out.last(), Some(&(r, Out::Tap)), "{r}");
        }
        // Released before the overlay was even made ready: nothing of the hold at all.
        assert_eq!(hold(1000, ON, &[(F, ALT | CMD, 0), (F, 0, 150)], 6000), [(150, Out::Tap)]);
        // Released while the glow builds: the tap, and the glow retracts.
        assert_eq!(
            hold(1000, ON, &[(F, ALT, 0), (F, ALT | CMD, 10), (F, CMD, 600), (F, 0, 620)], 6000),
            [(260, Out::H(Warm(1))), (460, Out::H(Preview { g: 1, to_arm_ms: 550 })), (600, Out::H(Cancel(1, Why::Released))), (620, Out::Tap)]
        );
    }

    #[test]
    fn letting_go_between_the_tap_and_the_hold_does_nothing() {
        let out = hold(1000, ON, &[(F, ALT | CMD, 0), (F, 0, 950)], 6000);
        assert_eq!(out, [(250, Out::H(Warm(1))), (450, Out::H(Preview { g: 1, to_arm_ms: 550 })), (950, Out::H(Cancel(1, Why::Released)))]);
    }

    #[test]
    fn a_clean_hold_warms_builds_arms_and_opens_on_release() {
        let out = hold(1000, ON, &[(F, CMD, 0), (F, ALT | CMD, 5), (F, ALT, 1200), (F, 0, 1230)], 9000);
        assert_eq!(
            out,
            [
                (255, Out::H(Warm(1))),
                (455, Out::H(Preview { g: 1, to_arm_ms: 550 })),
                (1005, Out::H(Arm(1))),
                (1230, Out::H(Commit(1))),
            ]
        );
        // Caps Lock on changes nothing.
        let out = hold(1000, ON, &[(F, CAPS | ALT | CMD, 0), (F, CAPS, 1500)], 9000);
        assert_eq!(out.last(), Some(&(1500, Out::H(Commit(1)))));
        assert_eq!(taps(&out), 0);
    }

    #[test]
    fn a_key_during_the_hold_cancels_and_reaches_the_app() {
        // ⌥⌘ then a key while the glow builds (⌥⌘Esc after a pause, NotchHub's ⌥⌘V)…
        let out = hold(1000, ON, &[(F, ALT | CMD, 0), (K, ALT | CMD, 700), (F, 0, 1200)], 9000);
        assert_eq!(out[2..], [(700, Out::H(Cancel(1, Why::Key)))]);
        // …and once it armed: called off, no commit, no tap.
        let out = hold(1000, ON, &[(F, ALT | CMD, 0), (K, ALT | CMD, 1100), (F, 0, 1300)], 9000);
        assert_eq!(out.last(), Some(&(1100, Out::H(Cancel(1, Why::Key)))));
        assert_eq!((taps(&out), commits(&out)), (0, 0));
        // A key before the overlay was made ready: nothing was shown, nothing to call off.
        assert_eq!(hold(1000, ON, &[(F, ALT | CMD, 0), (K, ALT | CMD, 100), (F, 0, 3000)], 9000), []);
    }

    #[test]
    fn a_click_or_a_scroll_cancels() {
        let out = hold(1000, ON, &[(F, ALT | CMD, 0), (M, ALT | CMD, 500), (F, 0, 1500)], 9000);
        assert_eq!(out.last(), Some(&(500, Out::H(Cancel(1, Why::Click)))));
        let out = hold(1000, ON, &[(F, ALT | CMD, 0), (R, ALT | CMD, 1200), (F, 0, 1500)], 9000);
        assert_eq!(out.last(), Some(&(1200, Out::H(Cancel(1, Why::Click)))));
        let out = hold(1000, ON, &[(F, ALT | CMD, 0), (S, ALT | CMD, 1200), (F, 0, 1500)], 9000);
        assert_eq!(out.last(), Some(&(1200, Out::H(Cancel(1, Why::Scroll)))));
        assert_eq!(commits(&out), 0);
    }

    #[test]
    fn shift_or_control_cancels() {
        let out = hold(1000, ON, &[(F, ALT | CMD, 0), (F, ALT | CMD | SHIFT, 600), (F, ALT | CMD, 700), (F, 0, 1500)], 9000);
        assert_eq!(out.last(), Some(&(600, Out::H(Cancel(1, Why::Modifier)))));
        let out = hold(1000, ON, &[(F, ALT | CMD, 0), (F, ALT | CMD | CTRL, 1100), (F, 0, 1500)], 9000);
        assert_eq!(out.last(), Some(&(1100, Out::H(Cancel(1, Why::Modifier)))));
        // ⇧ held from the start: the ⇧⌥⌘ shortcut, never a hold.
        assert_eq!(hold(1000, ON, &[(F, SHIFT | ALT | CMD, 0), (F, 0, 1500)], 9000), []);
        assert_eq!(hold(1000, ON, &[(F, SHIFT, 0), (F, SHIFT | ALT | CMD, 10), (F, 0, 1500)], 9000), []);
    }

    #[test]
    fn one_key_let_go_before_the_hold_cancels_after_it_waits_for_both() {
        let out = hold(1000, ON, &[(F, ALT | CMD, 0), (F, CMD, 600), (F, ALT | CMD, 700), (F, 0, 1500)], 9000);
        assert_eq!(out.last(), Some(&(600, Out::H(Cancel(1, Why::Released)))));
        assert_eq!(commits(&out), 0);
        // Armed: letting go of one waits for the other.
        let out = hold(1000, ON, &[(F, ALT | CMD, 0), (F, CMD, 1100), (F, 0, 1300)], 9000);
        assert_eq!(out.last(), Some(&(1300, Out::H(Commit(1)))));
        // …and a key in between still calls it off (⌘C with ⌥ let go).
        let out = hold(1000, ON, &[(F, ALT | CMD, 0), (F, CMD, 1100), (K, CMD, 1200), (F, 0, 1300)], 9000);
        assert_eq!(out.last(), Some(&(1200, Out::H(Cancel(1, Why::Key)))));
    }

    #[test]
    fn holding_too_long_gives_up() {
        let out = hold(1000, ON, &[(F, ALT | CMD, 0), (F, 0, 5200)], 9000);
        assert_eq!(out[3..], [(5000, Out::H(Cancel(1, Why::TooLong)))]);
        assert_eq!((taps(&out), commits(&out)), (0, 0));
    }

    #[test]
    fn secure_input_never_warms_but_the_tap_works() {
        let gate = Gate { secure: true, ..ON };
        assert_eq!(hold(1000, gate, &[(F, ALT | CMD, 0), (F, 0, 1500)], 9000), []);
        assert_eq!(hold(1000, gate, &[(F, ALT | CMD, 0), (F, 0, 600)], 9000), [(600, Out::Tap)]);
    }

    #[test]
    fn latched_modifiers_dont_arm() {
        let gate = Gate { physical: false, ..ON };
        let out = hold(1000, gate, &[(F, ALT | CMD, 0), (F, 0, 1500)], 9000);
        assert_eq!(out.last(), Some(&(1000, Out::H(Cancel(1, Why::Sticky)))));
        assert_eq!(commits(&out), 0);
    }

    #[test]
    fn search_screen_off_or_no_core_means_no_hold_events_at_all() {
        let off = Gate { on: false, ..ON };
        assert_eq!(hold(1000, off, &[(F, ALT | CMD, 0), (F, 0, 1500)], 9000), []);
        assert_eq!(hold(1000, off, &[(F, ALT | CMD, 0), (F, 0, 800)], 9000), [(800, Out::Tap)]);
        // Off for the shell (LensGate): the quick tap alone, with today's 900 ms.
        let gate = LensGate::default();
        assert_eq!(gate.timing(), Timing::tap_only());
        gate.hold_ms.store(600, Ordering::SeqCst);
        assert_eq!(gate.timing().tap, MS(900));
        gate.on.store(true, Ordering::SeqCst);
        assert_eq!(gate.timing(), Timing::for_hold(600));
        // The tap-only timing never warms, whatever the gate says.
        let (mut st, t0) = (State::default(), Instant::now());
        assert_eq!(step_hold(&mut st, F, f(ALT | CMD), t0, &Timing::tap_only()), (false, None));
        assert_eq!(next_deadline(&st), None);
        assert_eq!(tick(&mut st, t0 + MS(5000), &ON), None);
    }

    #[test]
    fn timing_for_hold() {
        let t = Timing::for_hold(1000);
        assert_eq!((t.tap, t.warm, t.preview, t.arm, t.give_up), (MS(900), MS(250), MS(450), MS(1000), MS(5000)));
        assert!(t.hold);
        let t = Timing::for_hold(600);
        assert_eq!((t.tap, t.warm, t.preview, t.arm), (MS(500), MS(150), MS(300), MS(600)));
        let t = Timing::for_hold(3000);
        assert_eq!((t.tap, t.warm, t.preview, t.give_up), (MS(900), MS(250), MS(450), MS(7000)));
        assert_eq!(Timing::for_hold(100), Timing::for_hold(600));
        assert_eq!(Timing::for_hold(9999), Timing::for_hold(3000));
        assert!(!Timing::tap_only().hold);
        assert_eq!(Timing::tap_only().tap, MAX_HOLD);
        // A shorter hold shortens the tap, so the two never overlap.
        let out = hold(600, ON, &[(F, ALT | CMD, 0), (F, 0, 550)], 9000);
        assert_eq!(taps(&out), 0);
        assert_eq!(out.last(), Some(&(550, Out::H(Cancel(1, Why::Released)))));
        assert_eq!(hold(600, ON, &[(F, ALT | CMD, 0), (F, 0, 450)], 9000).last(), Some(&(450, Out::Tap)));
        assert_eq!(hold(600, ON, &[(F, ALT | CMD, 0), (F, 0, 700)], 9000).last(), Some(&(700, Out::H(Commit(1)))));
    }

    #[test]
    fn an_old_gestures_tick_does_nothing() {
        let (mut st, t0, fresh) = (State::default(), Instant::now(), Timing::for_hold(1000));
        step_hold(&mut st, F, f(ALT | CMD), t0, &fresh);
        assert_eq!(tick(&mut st, t0 + MS(300), &ON), Some(Warm(1)));
        assert_eq!(step_hold(&mut st, F, f(0), t0 + MS(320), &fresh), (true, Some(Cancel(1, Why::Released))));
        // The timer thread wakes late for the gesture that has ended: nothing.
        assert_eq!(tick(&mut st, t0 + MS(2000), &ON), None);
        assert_eq!(next_deadline(&st), None);
        // The next gesture has the next number.
        step_hold(&mut st, F, f(ALT | CMD), t0 + MS(3000), &fresh);
        assert_eq!(tick(&mut st, t0 + MS(3300), &ON), Some(Warm(2)));
        // A late wake catches up one phase at a time, in order.
        let mut evs = Vec::new();
        while let Some(ev) = tick(&mut st, t0 + MS(4100), &ON) {
            evs.push(ev);
        }
        assert_eq!(evs, [Preview { g: 2, to_arm_ms: 0 }, Arm(2)]);
    }

    #[test]
    fn next_deadline_follows_the_phase() {
        let (mut st, t0, fresh) = (State::default(), Instant::now(), Timing::for_hold(1000));
        assert_eq!(next_deadline(&st), None);
        step_hold(&mut st, F, f(ALT), t0, &fresh);
        assert_eq!(next_deadline(&st), None); // only one of the two
        step_hold(&mut st, F, f(ALT | CMD), t0 + MS(20), &fresh);
        let t1 = t0 + MS(20);
        assert_eq!(next_deadline(&st), Some(t1 + MS(250)));
        tick(&mut st, t1 + MS(250), &ON);
        assert_eq!(next_deadline(&st), Some(t1 + MS(450)));
        tick(&mut st, t1 + MS(450), &ON);
        assert_eq!(next_deadline(&st), Some(t1 + MS(1000)));
        tick(&mut st, t1 + MS(1000), &ON);
        assert_eq!(next_deadline(&st), Some(t1 + MS(5000)));
        step_hold(&mut st, K, f(ALT | CMD), t1 + MS(1100), &fresh);
        assert_eq!(next_deadline(&st), None);
    }

    #[test]
    fn a_paused_tap_cancels_a_showing_hold() {
        let (mut st, t0, fresh) = (State::default(), Instant::now(), Timing::for_hold(1000));
        assert_eq!(paused(&mut st), None);
        step_hold(&mut st, F, f(ALT | CMD), t0, &fresh);
        assert_eq!(paused(&mut st), None); // nothing shown yet…
        assert_eq!(tick(&mut st, t0 + MS(300), &ON), None); // …and no hold after it
        let mut st = State::default();
        step_hold(&mut st, F, f(ALT | CMD), t0, &fresh);
        tick(&mut st, t0 + MS(300), &ON);
        assert_eq!(paused(&mut st), Some(Cancel(1, Why::Paused)));
        assert_eq!(step_hold(&mut st, F, f(0), t0 + MS(1500), &fresh), (false, None));
        // Search Screen opened (or the core went) while it built: called off when it would arm.
        let mut st = State::default();
        step_hold(&mut st, F, f(ALT | CMD), t0, &fresh);
        tick(&mut st, t0 + MS(300), &ON);
        tick(&mut st, t0 + MS(500), &ON);
        assert_eq!(tick(&mut st, t0 + MS(1000), &Gate { on: false, ..ON }), Some(Cancel(1, Why::Paused)));
    }
}

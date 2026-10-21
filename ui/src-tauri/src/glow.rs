//! Search Screen's glow along the screen's edges, drawn by the shell with Core Animation in the
//! overlay's panel (lens.rs drives it; the page draws it only when this can't: `glow: "web"`).
//!
//! v1 drew the glow in the page: display-sized opacity groups, masks and tiles that WebKit painted
//! and composited on its main thread, started late by a page that was frozen while hidden. Here it
//! is 19 layers in a layer-hosting view above the web view (native.rs `NeoLensGlowView`, which
//! never takes a click): four soft bands and four crisp rims, each a clip holding one long strip of
//! NEO's five colours, baked once into a small image per edge, that slides along its edge by
//! `transform`. With the strips' phases set from the screen's perimeter, the colours travel round
//! the screen as one ring, clockwise. No masks, filters, shadows or group opacity; only transforms
//! and opacities animate, on the render server, at up to 120 Hz; nothing runs on the main thread per
//! frame, and nothing at all while hidden (every animation is removed).
//!
//! The curves are ported verbatim from ui/src/lens/glow.ts (which the web fallback still uses); the
//! fixture ui/tests/fixtures/lens-glow.json, made once from glow.ts, pins the two together. Each
//! phase (lens.rs: warm, preview, armed, open, cancel, closing, hidden) becomes keyframe animations
//! sampled from its curve, starting from wherever the glow is at that instant, the moment lens.rs
//! enters the phase — no IPC, no page.
//!
//! Main thread only: the layers live in a thread-local; every public entry hops there first.

use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use objc2::encode::{Encode, Encoding};
use objc2::runtime::{AnyObject, Bool};
use objc2::{class, msg_send, sel};
use tauri::AppHandle;

// ---- the curves (verbatim from ui/src/lens/glow.ts) ---------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlowPhase {
    Idle,
    Warm,
    Preview,
    Armed,
    Open,
    Cancel,
    Closing,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Look {
    /// The soft band's width, in points.
    pub band: f64,
    pub opacity: f64,
    /// Seconds for the colours to travel once round the screen (0: still).
    pub period: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ctx {
    pub to_arm_ms: f64,
    pub reduced: bool,
    /// Opened by the hold (its glow is built already); the menu, the voice and a host bloom it.
    pub via_hold: bool,
    pub from: Option<Look>,
    pub notch: bool,
}

pub const BAND_MAX: f64 = 56.0;
const BAND_PREVIEW: (f64, f64) = (8.0, 28.0);
const BAND_ARMED: (f64, f64) = (28.0, 56.0);
pub const BAND_OPEN: f64 = 36.0;
pub const BAND_SELECTED: f64 = 20.0;
const OPACITY_PREVIEW: f64 = 0.65;
pub const OPACITY_SELECTED: f64 = 0.55;
const BREATHE_LOW: f64 = 0.85;
pub const PERIOD_BUILDING: f64 = 2.4;
pub const PERIOD_OPEN: f64 = 8.0;
const DUR_ARM_UP: f64 = 140.0;
const DUR_ARM_SETTLE: f64 = 420.0;
const DUR_BLOOM: f64 = 260.0;
const DUR_SETTLE: f64 = 200.0;
const DUR_CANCEL: f64 = 140.0;
const DUR_CLOSING: f64 = 160.0;
const DUR_CLOSING_NOTCH: f64 = 200.0;
pub const DUR_REDUCED: f64 = 120.0;
const DUR_BREATHE: f64 = 3200.0;
pub const DUR_SELECT: f64 = 200.0;

const NONE: Look = Look { band: 0.0, opacity: 0.0, period: 0.0 };

fn clamp01(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        1.0
    }
}
pub fn ease_in(t: f64) -> f64 {
    clamp01(t).powi(3)
}
pub fn ease_out(t: f64) -> f64 {
    1.0 - (1.0 - clamp01(t)).powi(3)
}
/// A spring settling on 1 with one small overshoot.
pub fn spring(t: f64) -> f64 {
    let x = clamp01(t);
    1.0 - (-6.0 * x).exp() * (9.0 * x).cos() * (1.0 - x)
}
fn mix(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// How long a phase's curve runs before it holds still (ms).
pub fn glow_span(phase: GlowPhase, ctx: &Ctx) -> f64 {
    use GlowPhase::*;
    if ctx.reduced {
        return if matches!(phase, Preview | Armed | Open | Cancel | Closing) { DUR_REDUCED } else { 0.0 };
    }
    match phase {
        Preview => ctx.to_arm_ms.max(0.0),
        Armed => DUR_ARM_UP + DUR_ARM_SETTLE,
        Open => {
            if ctx.from.is_some() {
                DUR_SETTLE
            } else if ctx.via_hold {
                0.0
            } else {
                DUR_BLOOM
            }
        }
        Cancel => DUR_CANCEL,
        Closing => {
            if ctx.notch {
                DUR_CLOSING_NOTCH
            } else {
                DUR_CLOSING
            }
        }
        _ => 0.0,
    }
}

/// The glow `ms` into a phase.
pub fn glow_at(phase: GlowPhase, ms: f64, ctx: &Ctx) -> Look {
    use GlowPhase::*;
    let t = ms.max(0.0);
    let from = ctx.from;
    if ctx.reduced {
        let k = clamp01(t / DUR_REDUCED);
        let start = from.map(|f| f.opacity).unwrap_or(0.0);
        return match phase {
            Preview => Look { band: 28.0, opacity: mix(start, 0.4, k), period: 0.0 },
            Armed | Open => Look { band: BAND_OPEN, opacity: mix(start, 1.0, k), period: 0.0 },
            Cancel | Closing => Look { band: from.map(|f| f.band).unwrap_or(BAND_OPEN), opacity: mix(start, 0.0, k), period: 0.0 },
            _ => NONE,
        };
    }
    match phase {
        Preview => {
            let p = if ctx.to_arm_ms > 0.0 { clamp01(t / ctx.to_arm_ms) } else { 1.0 };
            Look { band: mix(BAND_PREVIEW.0, BAND_PREVIEW.1, p), opacity: OPACITY_PREVIEW * ease_in(p), period: PERIOD_BUILDING }
        }
        Armed => {
            if t < DUR_ARM_UP {
                return Look { band: mix(BAND_ARMED.0, BAND_ARMED.1, ease_out(t / DUR_ARM_UP)), opacity: 1.0, period: PERIOD_BUILDING };
            }
            let s = spring((t - DUR_ARM_UP) / DUR_ARM_SETTLE);
            Look { band: mix(BAND_ARMED.1, BAND_OPEN, s), opacity: 1.0, period: PERIOD_BUILDING }
        }
        Open => {
            if let Some(f) = from {
                let k = ease_out(t / DUR_SETTLE);
                return Look { band: mix(f.band, BAND_OPEN, k), opacity: mix(f.opacity, 1.0, k), period: PERIOD_OPEN };
            }
            if ctx.via_hold {
                return Look { band: BAND_OPEN, opacity: 1.0, period: PERIOD_OPEN };
            }
            let k = ease_out(t / DUR_BLOOM);
            Look { band: BAND_OPEN * k, opacity: k, period: PERIOD_OPEN }
        }
        Cancel => {
            let f = from.unwrap_or(Look { band: BAND_PREVIEW.1, opacity: 0.3, period: PERIOD_BUILDING });
            Look { band: f.band, opacity: f.opacity * (1.0 - ease_in(t / DUR_CANCEL)), period: f.period }
        }
        Closing => {
            let f = from.unwrap_or(Look { band: BAND_OPEN, opacity: 1.0, period: PERIOD_OPEN });
            let d = if ctx.notch { DUR_CLOSING_NOTCH } else { DUR_CLOSING };
            Look { band: f.band, opacity: f.opacity * (1.0 - clamp01(t / d)), period: f.period }
        }
        _ => NONE,
    }
}

/// Whether the glow breathes (open, nothing selected, motion allowed).
pub fn breathes(phase: GlowPhase, reduced: bool, selected: bool) -> bool {
    phase == GlowPhase::Open && !reduced && !selected
}

/// How fast the strips slide for a period (one length of colours in 4 s is speed 1).
pub fn slide_speed(period: f64) -> f64 {
    if period > 0.0 && period.is_finite() {
        PERIOD_OPEN / period
    } else {
        0.0
    }
}

// ---- the strips (pure) ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    Top,
    Right,
    Bottom,
    Left,
}
pub const EDGES: [Edge; 4] = [Edge::Top, Edge::Right, Edge::Bottom, Edge::Left];

/// NEO's five colours round the ring (the page's --lens-1…5, then the first again), sRGB 0..1.
const RING: [[f64; 3]; 6] = [
    [0x6f as f64 / 255.0, 0xb6 as f64 / 255.0, 1.0],
    [0xa5 as f64 / 255.0, 0x8b as f64 / 255.0, 1.0],
    [1.0, 0x7a as f64 / 255.0, 0xc3 as f64 / 255.0],
    [1.0, 0xb3 as f64 / 255.0, 0x6b as f64 / 255.0],
    [0x5f as f64 / 255.0, 0xe0 as f64 / 255.0, 0xc8 as f64 / 255.0],
    [0x6f as f64 / 255.0, 0xb6 as f64 / 255.0, 1.0],
];

/// One length of the colours: half the screen's perimeter (they go round twice), at least 200 pt.
pub fn ring_len(w: f64, h: f64) -> f64 {
    (w + h).max(200.0)
}

/// The colour `t` points along the ring (stops every fifth of `l`, mixed linearly in sRGB, as CSS).
pub fn ring(t: f64, l: f64) -> [f64; 3] {
    let f = t.rem_euclid(l) / l * 5.0;
    let i = (f.floor() as usize).min(4);
    let k = (f - i as f64).clamp(0.0, 1.0);
    let (a, b) = (RING[i], RING[i + 1]);
    [mix(a[0], b[0], k), mix(a[1], b[1], k), mix(a[2], b[2], k)]
}

/// Where on the ring a strip's point `u` (points from the strip's start) is, so that the four edges
/// meet at the corners (EdgeGlow.tsx's background offsets, as numbers).
pub fn edge_phase(e: Edge, u: f64, w: f64, h: f64) -> f64 {
    match e {
        Edge::Top => u,
        Edge::Right => u + w,
        Edge::Bottom => 2.0 * w + h - u,
        Edge::Left => -u,
    }
}

/// Where a strip starts along its edge (it reaches one length of colours before the screen on the
/// side it slides from) and which way it slides: top +x, right +y, bottom −x, left −y.
pub fn strip_origin(e: Edge, l: f64) -> f64 {
    match e {
        Edge::Top | Edge::Right => -l,
        Edge::Bottom | Edge::Left => 0.0,
    }
}
pub fn slide_dir(e: Edge) -> f64 {
    match e {
        Edge::Top | Edge::Right => 1.0,
        Edge::Bottom | Edge::Left => -1.0,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
/// The colour on screen at `pos` points along edge `e` (x for top and bottom, y for the sides) when
/// the strips have slid `s` points.
pub fn edge_colour(e: Edge, pos: f64, s: f64, w: f64, h: f64) -> [f64; 3] {
    let l = ring_len(w, h);
    let u = pos - strip_origin(e, l) - slide_dir(e) * s;
    ring(edge_phase(e, u, w, h), l)
}

/// The band's falloff from the screen's edge inwards (`t` 0..1 across it): the page's mask.
pub fn falloff(t: f64) -> f64 {
    const P: [(f64, f64); 4] = [(0.0, 1.0), (0.14, 0.62), (0.5, 0.22), (1.0, 0.0)];
    let t = clamp01(t);
    for w in P.windows(2) {
        let ((t0, a0), (t1, a1)) = (w[0], w[1]);
        if t <= t1 {
            return mix(a0, a1, (t - t0) / (t1 - t0));
        }
    }
    0.0
}

/// A strip's sizes for a display: one length of colours is exactly `period_px` pixels (so the image
/// repeats every length), about 1 px per 2 pt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub w: f64,
    pub h: f64,
    pub l: f64,
    pub period_px: usize,
    /// Points per pixel along a strip.
    pub step: f64,
    /// Pixels along the top and bottom strips, and along the side ones.
    pub along_tb: usize,
    pub along_lr: usize,
}

impl Geometry {
    pub fn new(w: f64, h: f64) -> Geometry {
        let l = ring_len(w, h);
        let period_px = ((l / 2.0).round() as usize).max(1);
        let step = l / period_px as f64;
        let along = |len: f64| ((len / step).ceil() as usize).max(1);
        Geometry { w, h, l, period_px, step, along_tb: along(w + l), along_lr: along(h + l) }
    }
    pub fn along(&self, e: Edge) -> usize {
        match e {
            Edge::Top | Edge::Bottom => self.along_tb,
            Edge::Right | Edge::Left => self.along_lr,
        }
    }
}

/// Pixels across a band's image (its 56 pt are scaled to the band's width).
pub const BAND_PX: usize = 64;

/// A baked strip: premultiplied BGRA (little-endian 32-bit ARGB, Core Animation's own), `width` ×
/// `height` pixels, sRGB.
#[derive(Clone, Debug)]
pub struct Strip {
    pub width: usize,
    pub height: usize,
    pub bytes: Vec<u8>,
}

impl Strip {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn px(&self, x: usize, y: usize) -> [u8; 4] {
        let i = (y * self.width + x) * 4;
        [self.bytes[i], self.bytes[i + 1], self.bytes[i + 2], self.bytes[i + 3]]
    }
}

/// The image of edge `e`'s strip: a band (BAND_PX across, falling off from the screen's edge) or a
/// rim (1 px across, solid). Pure.
pub fn bake(e: Edge, g: &Geometry, band: bool) -> Strip {
    let along = g.along(e);
    let across = if band { BAND_PX } else { 1 };
    let colours: Vec<[f64; 3]> = (0..along).map(|i| ring(edge_phase(e, (i as f64 + 0.5) * g.step, g.w, g.h), g.l)).collect();
    // Alpha across, indexed from the screen's edge inwards.
    let alpha: Vec<f64> = (0..across).map(|j| if band && across > 1 { falloff(j as f64 / (across - 1) as f64) } else { 1.0 }).collect();
    let horizontal = matches!(e, Edge::Top | Edge::Bottom);
    let (width, height) = if horizontal { (along, across) } else { (across, along) };
    let mut bytes = vec![0u8; width * height * 4];
    let q = |v: f64| (v * 255.0).round().clamp(0.0, 255.0) as u8;
    for y in 0..height {
        for x in 0..width {
            let (i, j) = if horizontal { (x, y) } else { (y, x) };
            // The screen's edge: the image's top row (top band), bottom row, left or right column.
            let from_edge = match e {
                Edge::Top | Edge::Left => j,
                Edge::Bottom | Edge::Right => across - 1 - j,
            };
            let a = alpha[from_edge];
            let c = colours[i];
            let o = (y * width + x) * 4;
            bytes[o] = q(c[2] * a);
            bytes[o + 1] = q(c[1] * a);
            bytes[o + 2] = q(c[0] * a);
            bytes[o + 3] = q(a);
        }
    }
    Strip { width, height, bytes }
}

/// A layer's timing (Core Animation's: local = (parent − beginTime) × speed + timeOffset).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    pub time_offset: f64,
    pub begin_time: f64,
    pub speed: f64,
}

/// A new speed for a layer whose local time is `local_now` while its parent's is `parent_now`, so
/// its local time — where its colours are — carries on from exactly there (no jump).
pub fn rebase(local_now: f64, parent_now: f64, speed: f64) -> Timing {
    Timing { time_offset: local_now, begin_time: parent_now, speed }
}
#[cfg_attr(not(test), allow(dead_code))]
pub fn local_time(t: &Timing, parent: f64) -> f64 {
    (parent - t.begin_time) * t.speed + t.time_offset
}

/// Keyframe samples for a curve: every 8 ms, at most 128 steps (the values at i / n of `span`).
pub fn sample_count(span_ms: f64) -> usize {
    ((span_ms / 8.0).ceil().max(1.0) as usize).min(128)
}

// ---- Core Animation (raw objc2: no new crate) ----------------------------------------------------

type Id = *mut AnyObject;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CGPoint {
    x: f64,
    y: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CGSize {
    w: f64,
    h: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CGRect {
    origin: CGPoint,
    size: CGSize,
}
unsafe impl Encode for CGPoint {
    const ENCODING: Encoding = Encoding::Struct("CGPoint", &[f64::ENCODING, f64::ENCODING]);
}
unsafe impl Encode for CGSize {
    const ENCODING: Encoding = Encoding::Struct("CGSize", &[f64::ENCODING, f64::ENCODING]);
}
unsafe impl Encode for CGRect {
    const ENCODING: Encoding = Encoding::Struct("CGRect", &[CGPoint::ENCODING, CGSize::ENCODING]);
}

/// macOS 12+: an animation's frame rate (each one asks for 120 Hz on ProMotion).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CAFrameRateRange {
    pub minimum: f32,
    pub maximum: f32,
    pub preferred: f32,
}
unsafe impl Encode for CAFrameRateRange {
    const ENCODING: Encoding = Encoding::Struct("CAFrameRateRange", &[f32::ENCODING, f32::ENCODING, f32::ENCODING]);
}
pub const FRAME_RATE: CAFrameRateRange = CAFrameRateRange { minimum: 80.0, maximum: 120.0, preferred: 120.0 };

#[link(name = "QuartzCore", kind = "framework")]
extern "C" {
    fn CACurrentMediaTime() -> f64;
    static kCAMediaTimingFunctionLinear: Id;
    static kCAMediaTimingFunctionEaseInEaseOut: Id;
    static kCAFillModeForwards: Id;
    static kCAAnimationLinear: Id;
    static kCAGravityResize: Id;
}

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    static kCGColorSpaceSRGB: *const c_void;
    fn CGColorSpaceCreateWithName(name: *const c_void) -> *mut c_void;
    fn CGColorSpaceRelease(space: *mut c_void);
    fn CGDataProviderCreateWithCFData(data: *const c_void) -> *mut c_void;
    fn CGDataProviderRelease(provider: *mut c_void);
    #[allow(clippy::too_many_arguments)]
    fn CGImageCreate(
        width: usize,
        height: usize,
        bits_per_component: usize,
        bits_per_pixel: usize,
        bytes_per_row: usize,
        space: *mut c_void,
        bitmap_info: u32,
        provider: *mut c_void,
        decode: *const f64,
        should_interpolate: bool,
        intent: i32,
    ) -> *mut c_void;
    fn CGImageRelease(image: *mut c_void);
}
/// kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little: BGRA in memory.
const BGRA_PREMULTIPLIED: u32 = 2 | (2 << 12);

/// An autoreleased NSString.
unsafe fn nsstr(s: &str) -> Id {
    let o: Id = msg_send![class!(NSString), alloc];
    let o: Id = msg_send![o, initWithBytes: s.as_ptr() as *const c_void, length: s.len(), encoding: 4usize];
    msg_send![o, autorelease]
}
unsafe fn nsnum(v: f64) -> Id {
    msg_send![class!(NSNumber), numberWithDouble: v]
}
unsafe fn nsarray(items: &[Id]) -> Id {
    msg_send![class!(NSArray), arrayWithObjects: items.as_ptr(), count: items.len()]
}
fn yes(b: bool) -> Bool {
    Bool::new(b)
}

unsafe fn set_frame(l: Id, x: f64, y: f64, w: f64, h: f64) {
    let r = CGRect { origin: CGPoint { x, y }, size: CGSize { w, h } };
    let _: () = msg_send![l, setFrame: r];
}
unsafe fn set_num(l: Id, path: &str, v: f64) {
    let _: () = msg_send![l, setValue: nsnum(v), forKeyPath: nsstr(path)];
}
unsafe fn get_num(l: Id, path: &str) -> f64 {
    let v: Id = msg_send![l, valueForKeyPath: nsstr(path)];
    if v.is_null() {
        return f64::NAN;
    }
    msg_send![v, doubleValue]
}
unsafe fn now_media() -> f64 {
    CACurrentMediaTime()
}

/// The one way an animation is added: it asks for 120 Hz first (where the OS can), then goes on.
unsafe fn add_anim(l: Id, anim: Id, key: &str) {
    let can: Bool = msg_send![anim, respondsToSelector: sel!(setPreferredFrameRateRange:)];
    if can.as_bool() {
        let _: () = msg_send![anim, setPreferredFrameRateRange: FRAME_RATE];
    }
    let _: () = msg_send![l, addAnimation: anim, forKey: nsstr(key)];
}
unsafe fn remove_anim(l: Id, key: &str) {
    let _: () = msg_send![l, removeAnimationForKey: nsstr(key)];
}

/// A curve sampled into a linear keyframe animation that holds its end.
unsafe fn keyframes(path: &str, values: &[f64], dur_ms: f64) -> Id {
    let a: Id = msg_send![class!(CAKeyframeAnimation), animationWithKeyPath: nsstr(path)];
    let n = values.len().max(2);
    let vals: Vec<Id> = values.iter().map(|v| nsnum(*v)).collect();
    let times: Vec<Id> = (0..values.len()).map(|i| nsnum(i as f64 / (n - 1) as f64)).collect();
    let _: () = msg_send![a, setValues: nsarray(&vals)];
    let _: () = msg_send![a, setKeyTimes: nsarray(&times)];
    let _: () = msg_send![a, setDuration: (dur_ms / 1000.0).max(0.001)];
    let _: () = msg_send![a, setCalculationMode: kCAAnimationLinear];
    let _: () = msg_send![a, setFillMode: kCAFillModeForwards];
    let _: () = msg_send![a, setRemovedOnCompletion: Bool::NO];
    a
}

/// An endless animation from `from` to `to` (a slide, or the breathing).
unsafe fn endless(path: &str, from: f64, to: f64, dur_s: f64, autoreverses: bool, timing: Id) -> Id {
    let a: Id = msg_send![class!(CABasicAnimation), animationWithKeyPath: nsstr(path)];
    let _: () = msg_send![a, setFromValue: nsnum(from)];
    let _: () = msg_send![a, setToValue: nsnum(to)];
    let _: () = msg_send![a, setDuration: dur_s];
    let _: () = msg_send![a, setRepeatCount: f32::INFINITY];
    let _: () = msg_send![a, setAutoreverses: yes(autoreverses)];
    // Never "finished" — not even by a layer tree that isn't on screen at that moment (the window is
    // ordered out between sessions): only hide() takes it away.
    let _: () = msg_send![a, setRemovedOnCompletion: Bool::NO];
    let tf: Id = msg_send![class!(CAMediaTimingFunction), functionWithName: timing];
    let _: () = msg_send![a, setTimingFunction: tf];
    a
}

struct Tx;
impl Tx {
    /// A transaction with no implicit animations (every change here is either instant or explicit).
    unsafe fn begin() -> Tx {
        let _: () = msg_send![class!(CATransaction), begin];
        let _: () = msg_send![class!(CATransaction), setDisableActions: Bool::YES];
        Tx
    }
}
impl Drop for Tx {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![class!(CATransaction), commit];
        }
    }
}

/// An sRGB image of a baked strip (+1 retained; the caller releases it).
unsafe fn cg_image(s: &Strip) -> *mut c_void {
    use core_foundation::base::TCFType;
    use core_foundation::data::CFData;
    let data = CFData::from_buffer(&s.bytes);
    let provider = CGDataProviderCreateWithCFData(data.as_concrete_TypeRef() as *const c_void);
    let space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    let img = if provider.is_null() || space.is_null() {
        std::ptr::null_mut()
    } else {
        CGImageCreate(s.width, s.height, 8, 32, s.width * 4, space, BGRA_PREMULTIPLIED, provider, std::ptr::null(), true, 0)
    };
    if !provider.is_null() {
        CGDataProviderRelease(provider);
    }
    if !space.is_null() {
        CGColorSpaceRelease(space);
    }
    img
}

// ---- the layers -----------------------------------------------------------------------------------

/// The scale key path of an edge's band (across the edge).
fn scale_path(e: Edge) -> &'static str {
    match e {
        Edge::Top | Edge::Bottom => "transform.scale.y",
        Edge::Right | Edge::Left => "transform.scale.x",
    }
}
fn slide_path(e: Edge) -> &'static str {
    match e {
        Edge::Top | Edge::Bottom => "transform.translation.x",
        Edge::Right | Edge::Left => "transform.translation.y",
    }
}
fn band_scale(band: f64) -> f64 {
    (band / BAND_MAX).max(0.001)
}
/// Whether a sampled curve changes at all (a flat one needs no animation).
fn moves(values: &[f64]) -> bool {
    values.len() > 1 && values.iter().any(|v| (v - values[0]).abs() > 1e-9)
}

#[derive(Clone, Copy, Debug)]
struct Last {
    phase: GlowPhase,
    at: Instant,
    ctx: Ctx,
}

#[derive(Clone, Copy, Debug)]
struct Selection {
    on: bool,
    at: Instant,
    from_band: f64,
    from_dim: f64,
    reduced: bool,
}

impl Selection {
    fn k(&self, now: Instant) -> f64 {
        let t = now.saturating_duration_since(self.at).as_secs_f64() * 1000.0;
        if self.reduced {
            clamp01(t / DUR_REDUCED)
        } else {
            ease_out(t / DUR_SELECT)
        }
    }
    fn band(&self, now: Instant) -> f64 {
        mix(self.from_band, if self.on { BAND_SELECTED } else { BAND_OPEN }, self.k(now))
    }
    fn dim(&self, now: Instant) -> f64 {
        mix(self.from_dim, if self.on { OPACITY_SELECTED } else { 1.0 }, self.k(now))
    }
}

/// What a phase needs to know besides the Mac's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseIn {
    pub phase: GlowPhase,
    pub to_arm_ms: f64,
    pub via_hold: bool,
    pub notch: bool,
}

impl PhaseIn {
    pub fn of(phase: GlowPhase) -> PhaseIn {
        PhaseIn { phase, to_arm_ms: 0.0, via_hold: true, notch: false }
    }
}

/// The glow's layers (all +1 retained here), and where the glow is.
pub struct Native {
    view: Id,
    root: Id,
    dim: Id,
    breathe: Id,
    band_clips: [Id; 4],
    band_strips: [Id; 4],
    rim_clips: [Id; 4],
    rim_strips: [Id; 4],
    size: (f64, f64),
    rim: f64,
    baked: Option<(i64, i64)>,
    last: Option<Last>,
    sel: Option<Selection>,
    sliding: bool,
    speed: f64,
    breathing: bool,
}

unsafe fn new_layer() -> Id {
    let l: Id = msg_send![class!(CALayer), new];
    let _: () = msg_send![l, setAllowsGroupOpacity: Bool::NO];
    l
}
unsafe fn add_sub(parent: Id, child: Id) {
    let _: () = msg_send![parent, addSublayer: child];
}

impl Native {
    /// The tree, empty and hidden (no view: the caller hosts `root`).
    ///
    /// # Safety
    /// Any thread, inside an autorelease pool (the main thread for one shown on screen).
    pub unsafe fn build() -> Native {
        let _tx = Tx::begin();
        let root = new_layer();
        let dim = new_layer();
        let breathe = new_layer();
        add_sub(root, dim);
        add_sub(dim, breathe);
        let mk = || -> ([Id; 4], [Id; 4]) {
            let mut clips = [std::ptr::null_mut(); 4];
            let mut strips = [std::ptr::null_mut(); 4];
            for i in 0..4 {
                clips[i] = new_layer();
                strips[i] = new_layer();
                let _: () = msg_send![clips[i], setMasksToBounds: Bool::YES];
                let _: () = msg_send![strips[i], setContentsGravity: kCAGravityResize];
                add_sub(clips[i], strips[i]);
                add_sub(breathe, clips[i]);
            }
            (clips, strips)
        };
        let (band_clips, band_strips) = mk();
        let (rim_clips, rim_strips) = mk();
        let _: () = msg_send![root, setHidden: Bool::YES];
        set_num(root, "opacity", 0.0);
        let n = Native {
            view: std::ptr::null_mut(),
            root,
            dim,
            breathe,
            band_clips,
            band_strips,
            rim_clips,
            rim_strips,
            size: (0.0, 0.0),
            rim: 0.0,
            baked: None,
            last: None,
            sel: None,
            sliding: false,
            speed: 0.0,
            breathing: false,
        };
        n.orient();
        n
    }

    fn all(&self) -> Vec<Id> {
        let mut v = vec![self.root, self.dim, self.breathe];
        v.extend(self.band_clips);
        v.extend(self.band_strips);
        v.extend(self.rim_clips);
        v.extend(self.rim_strips);
        v
    }

    /// Top-left coordinates, like the page's: whatever the layers above say, the root's own flip
    /// makes y grow downwards (`contentsAreFlipped` is exactly "an odd number of flips up to the
    /// window"). Images show the right way up either way. Checked again at every layout (each
    /// session), in case AppKit changed a layer above.
    unsafe fn orient(&self) {
        let _: () = msg_send![self.root, setGeometryFlipped: Bool::NO];
        let down: Bool = msg_send![self.root, contentsAreFlipped];
        if !down.as_bool() {
            let _: () = msg_send![self.root, setGeometryFlipped: Bool::YES];
        }
    }

    /// Laid out for a display of `w` × `h` points (re-baked only when the size changed).
    ///
    /// # Safety
    /// As `build`.
    pub unsafe fn layout(&mut self, w: f64, h: f64, contrast: bool) {
        if !(w > 0.0 && h > 0.0 && w.is_finite() && h.is_finite()) {
            return;
        }
        let rim = if contrast { 4.0 } else { 3.0 };
        if self.size == (w, h) && self.rim == rim {
            return;
        }
        let _tx = Tx::begin();
        self.orient();
        let g = Geometry::new(w, h);
        set_frame(self.root, 0.0, 0.0, w, h);
        set_frame(self.dim, 0.0, 0.0, w, h);
        set_frame(self.breathe, 0.0, 0.0, w, h);
        let l = g.l;
        for (i, e) in EDGES.iter().enumerate() {
            let len = g.along(*e) as f64 * g.step;
            let o = strip_origin(*e, l);
            // A band's clip is scaled across from the screen's edge (its anchor), never resized.
            let (cl, st) = (self.band_clips[i], self.band_strips[i]);
            let (anchor, pos, bounds) = match e {
                Edge::Top => (CGPoint { x: 0.5, y: 0.0 }, CGPoint { x: w / 2.0, y: 0.0 }, (w, BAND_MAX)),
                Edge::Right => (CGPoint { x: 1.0, y: 0.5 }, CGPoint { x: w, y: h / 2.0 }, (BAND_MAX, h)),
                Edge::Bottom => (CGPoint { x: 0.5, y: 1.0 }, CGPoint { x: w / 2.0, y: h }, (w, BAND_MAX)),
                Edge::Left => (CGPoint { x: 0.0, y: 0.5 }, CGPoint { x: 0.0, y: h / 2.0 }, (BAND_MAX, h)),
            };
            let b = CGRect { origin: CGPoint::default(), size: CGSize { w: bounds.0, h: bounds.1 } };
            let _: () = msg_send![cl, setBounds: b];
            let _: () = msg_send![cl, setAnchorPoint: anchor];
            let _: () = msg_send![cl, setPosition: pos];
            match e {
                Edge::Top | Edge::Bottom => set_frame(st, o, 0.0, len, BAND_MAX),
                Edge::Right | Edge::Left => set_frame(st, 0.0, o, BAND_MAX, len),
            }
            // A rim: a crisp edge, never scaled.
            let (cl, st) = (self.rim_clips[i], self.rim_strips[i]);
            match e {
                Edge::Top => set_frame(cl, 0.0, 0.0, w, rim),
                Edge::Right => set_frame(cl, w - rim, 0.0, rim, h),
                Edge::Bottom => set_frame(cl, 0.0, h - rim, w, rim),
                Edge::Left => set_frame(cl, 0.0, 0.0, rim, h),
            }
            match e {
                Edge::Top | Edge::Bottom => set_frame(st, o, 0.0, len, rim),
                Edge::Right | Edge::Left => set_frame(st, 0.0, o, rim, len),
            }
        }
        let key = (w.round() as i64, h.round() as i64);
        if self.baked != Some(key) {
            for (i, e) in EDGES.iter().enumerate() {
                for (layer, band) in [(self.band_strips[i], true), (self.rim_strips[i], false)] {
                    let img = cg_image(&bake(*e, &g, band));
                    let _: () = msg_send![layer, setContents: img as Id];
                    if !img.is_null() {
                        CGImageRelease(img);
                    }
                }
            }
            self.baked = Some(key);
        }
        self.size = (w, h);
        self.rim = rim;
        // A slide's length changed with the display: the slides start again at the new length.
        if self.sliding {
            for (i, e) in EDGES.iter().enumerate() {
                for st in [self.band_strips[i], self.rim_strips[i]] {
                    remove_anim(st, "slide");
                    add_anim(st, endless(slide_path(*e), 0.0, slide_dir(*e) * l, 4.0, false, kCAMediaTimingFunctionLinear), "slide");
                }
            }
        }
    }

    /// Where the glow is now (the phase's curve, and a selection's change of band while open).
    fn look_now(&self, now: Instant) -> Look {
        let Some(last) = self.last else { return NONE };
        let ms = now.saturating_duration_since(last.at).as_secs_f64() * 1000.0;
        let mut look = glow_at(last.phase, ms, &last.ctx);
        if last.phase == GlowPhase::Open {
            if let Some(s) = self.sel {
                look.band = s.band(now);
            }
        }
        look
    }
    fn dim_now(&self, now: Instant) -> f64 {
        self.sel.map(|s| s.dim(now)).unwrap_or(1.0)
    }
    fn selected(&self) -> bool {
        self.sel.map(|s| s.on).unwrap_or(false)
    }

    /// The bands' width: set to `values`' last at once, and animated through them over `ms` (no
    /// animation for a curve that doesn't change).
    unsafe fn bands(&self, values: &[f64], ms: f64) {
        let scales: Vec<f64> = values.iter().map(|b| band_scale(*b)).collect();
        let end = *scales.last().unwrap_or(&0.001);
        for (i, e) in EDGES.iter().enumerate() {
            let cl = self.band_clips[i];
            set_num(cl, scale_path(*e), end);
            if ms > 0.0 && moves(&scales) {
                add_anim(cl, keyframes(scale_path(*e), &scales, ms), "band");
            } else {
                remove_anim(cl, "band");
            }
        }
    }
    unsafe fn fade(&self, layer: Id, key: &str, values: &[f64], ms: f64) {
        set_num(layer, "opacity", *values.last().unwrap_or(&0.0));
        if ms > 0.0 && moves(values) {
            add_anim(layer, keyframes("opacity", values, ms), key);
        } else {
            remove_anim(layer, key);
        }
    }

    /// The colours' speed (0: still — Reduce Motion, or no glow). A change keeps every strip where it
    /// is (`rebase`), all eight from the same instant.
    unsafe fn slide(&mut self, speed: f64) {
        if speed <= 0.0 {
            if self.sliding {
                for st in self.band_strips.iter().chain(self.rim_strips.iter()) {
                    remove_anim(*st, "slide");
                }
            }
            self.sliding = false;
            self.speed = 0.0;
            return;
        }
        if self.sliding && (self.speed - speed).abs() < 1e-9 {
            return;
        }
        let now = now_media();
        let l = ring_len(self.size.0.max(1.0), self.size.1.max(1.0));
        for (i, e) in EDGES.iter().enumerate() {
            for st in [self.band_strips[i], self.rim_strips[i]] {
                let parent: Id = msg_send![st, superlayer];
                let local: f64 = msg_send![st, convertTime: now, fromLayer: std::ptr::null_mut::<AnyObject>()];
                let pnow: f64 = if parent.is_null() { now } else { msg_send![parent, convertTime: now, fromLayer: std::ptr::null_mut::<AnyObject>()] };
                let t = rebase(local, pnow, speed);
                let _: () = msg_send![st, setTimeOffset: t.time_offset];
                let _: () = msg_send![st, setBeginTime: t.begin_time];
                let _: () = msg_send![st, setSpeed: t.speed as f32];
                if !self.sliding {
                    add_anim(st, endless(slide_path(*e), 0.0, slide_dir(*e) * l, 4.0, false, kCAMediaTimingFunctionLinear), "slide");
                }
            }
        }
        self.sliding = true;
        self.speed = speed;
    }

    unsafe fn breathe(&mut self, on: bool) {
        if on == self.breathing {
            return;
        }
        if on {
            remove_anim(self.breathe, "breathe_out");
            let a = endless("opacity", 1.0, BREATHE_LOW, DUR_BREATHE / 1000.0, true, kCAMediaTimingFunctionEaseInEaseOut);
            add_anim(self.breathe, a, "breathe");
        } else {
            // Back to full from wherever the breath was (a short fade, not a jump).
            let p: Id = msg_send![self.breathe, presentationLayer];
            let at = if p.is_null() { 1.0 } else { get_num(p, "opacity") };
            remove_anim(self.breathe, "breathe");
            set_num(self.breathe, "opacity", 1.0);
            if at.is_finite() && at < 0.999 {
                add_anim(self.breathe, keyframes("opacity", &[at, 1.0], DUR_SELECT), "breathe_out");
            }
        }
        self.breathing = on;
    }

    /// lens.rs entered a phase, now.
    ///
    /// # Safety
    /// As `build`.
    pub unsafe fn phase(&mut self, inp: PhaseIn, reduced: bool, now: Instant) {
        use GlowPhase::*;
        if inp.phase == Idle {
            self.hide();
            return;
        }
        let was = self.look_now(now);
        let prev = self.last.map(|l| l.phase);
        let carry = matches!(inp.phase, Cancel | Closing) || (inp.phase == Open && prev == Some(Armed));
        let ctx = Ctx { to_arm_ms: inp.to_arm_ms, reduced, via_hold: inp.via_hold, from: if carry { Some(was) } else { None }, notch: inp.notch };
        self.last = Some(Last { phase: inp.phase, at: now, ctx });
        let span = glow_span(inp.phase, &ctx);
        let n = sample_count(span);
        let looks: Vec<Look> = (0..=n).map(|i| glow_at(inp.phase, span * i as f64 / n as f64, &ctx)).collect();
        let end = *looks.last().unwrap_or(&NONE);
        let _tx = Tx::begin();
        let _: () = msg_send![self.root, setHidden: Bool::NO];
        if matches!(inp.phase, Warm | Open) {
            // A new session, or the overlay opening: nothing selected yet, full brightness.
            self.sel = None;
            self.fade(self.dim, "dim", &[1.0], 0.0);
        }
        let ms = if span > 0.0 { span } else { 0.0 };
        self.fade(self.root, "phase", &looks.iter().map(|l| l.opacity).collect::<Vec<_>>(), ms);
        self.bands(&looks.iter().map(|l| l.band).collect::<Vec<_>>(), ms);
        // Warm: the colours already travel at the building speed (the glow is still invisible).
        let speed = if inp.phase == Warm && !reduced { slide_speed(PERIOD_BUILDING) } else { slide_speed(end.period) };
        self.slide(speed);
        self.breathe(breathes(inp.phase, reduced, self.selected()));
    }

    /// An area was selected (or let go of) while open: the band narrows to 20 pt and dims to 0.55
    /// (or comes back), from wherever it is, and the breathing stops (or resumes).
    ///
    /// # Safety
    /// As `build`.
    pub unsafe fn select(&mut self, on: bool, reduced: bool, now: Instant) {
        let Some(last) = self.last else { return };
        if last.phase != GlowPhase::Open || self.selected() == on {
            return;
        }
        let from_band = self.look_now(now).band;
        let from_dim = self.dim_now(now);
        let s = Selection { on, at: now, from_band, from_dim, reduced };
        self.sel = Some(s);
        let span = if reduced { DUR_REDUCED } else { DUR_SELECT };
        let n = sample_count(span);
        let at = |i: usize| now + std::time::Duration::from_secs_f64(span * i as f64 / n as f64 / 1000.0);
        let bands: Vec<f64> = (0..=n).map(|i| s.band(at(i))).collect();
        let dims: Vec<f64> = (0..=n).map(|i| s.dim(at(i))).collect();
        let _tx = Tx::begin();
        self.bands(&bands, span);
        self.fade(self.dim, "dim", &dims, span);
        self.breathe(breathes(GlowPhase::Open, reduced, on));
    }

    /// Hidden: every animation gone (no clock ticks at idle), the root hidden.
    ///
    /// # Safety
    /// As `build`.
    pub unsafe fn hide(&mut self) {
        let _tx = Tx::begin();
        for l in self.all() {
            let _: () = msg_send![l, removeAllAnimations];
        }
        let _: () = msg_send![self.root, setHidden: Bool::YES];
        set_num(self.root, "opacity", 0.0);
        set_num(self.breathe, "opacity", 1.0);
        set_num(self.dim, "opacity", 1.0);
        self.last = None;
        self.sel = None;
        self.sliding = false;
        self.speed = 0.0;
        self.breathing = false;
    }
}

impl Drop for Native {
    fn drop(&mut self) {
        unsafe {
            if !self.view.is_null() {
                crate::native::lens_glow_detach(self.view);
                self.view = std::ptr::null_mut();
            }
            let _: () = msg_send![self.root, removeFromSuperlayer];
            for l in self.all() {
                let _: () = msg_send![l, release];
            }
        }
    }
}

// ---- the shell's entries (main thread) -----------------------------------------------------------

thread_local! {
    static NATIVE: RefCell<Option<Native>> = const { RefCell::new(None) };
}
/// The native glow is up (the page is told `glow: "native"`; else it draws the glow itself).
static NATIVE_OK: AtomicBool = AtomicBool::new(false);

/// "native" or "web": who draws the glow (lens.rs puts it in every phase event and the hello).
pub fn mode() -> &'static str {
    if NATIVE_OK.load(Ordering::SeqCst) {
        "native"
    } else {
        "web"
    }
}

fn is_main() -> bool {
    unsafe {
        let b: Bool = msg_send![class!(NSThread), isMainThread];
        b.as_bool()
    }
}

/// `f` with the glow, on the main thread (hopping there if needed); nothing without one.
fn with(app: &AppHandle, f: impl FnOnce(&mut Native) + Send + 'static) {
    let run = move || {
        objc2::rc::autoreleasepool(|_| {
            NATIVE.with(|c| {
                if let Some(n) = c.borrow_mut().as_mut() {
                    f(n);
                }
            })
        })
    };
    if is_main() {
        run();
    } else {
        let _ = app.run_on_main_thread(run);
    }
}

/// Makes the glow in the overlay's window (above its web view). False: it can't be made, and the
/// page draws the glow instead.
///
/// # Safety
/// `ns_window` must be a live `NSWindow*`; main thread.
pub unsafe fn attach(ns_window: *mut AnyObject) -> bool {
    if !is_main() {
        return false;
    }
    let ok = objc2::rc::autoreleasepool(|_| {
        let mut n = Native::build();
        // The view hosts a plain layer (whatever AppKit does to its geometry is its own), and the
        // glow's root is inside it, oriented by `orient` relative to everything above.
        let host = new_layer();
        add_sub(host, n.root);
        let view = crate::native::lens_glow_attach(ns_window, host);
        let _: () = msg_send![host, release];
        if view.is_null() {
            return false;
        }
        n.view = view;
        n.orient();
        NATIVE.with(|c| *c.borrow_mut() = Some(n));
        true
    });
    NATIVE_OK.store(ok, Ordering::SeqCst);
    ok
}

/// The overlay's window is gone: its glow with it.
pub fn detach(app: &AppHandle) {
    let run = || {
        NATIVE.with(|c| c.borrow_mut().take());
        NATIVE_OK.store(false, Ordering::SeqCst);
    };
    if is_main() {
        run();
    } else {
        let _ = app.run_on_main_thread(run);
    }
}

/// The overlay is on `d`: the glow fits it (and stays the top view in the window).
pub fn resize(app: &AppHandle, d: &crate::lens::Display) {
    let (w, h) = (d.w, d.h);
    with(app, move |n| unsafe {
        let (_, _, contrast, _) = crate::native::a11y_flags();
        if !n.view.is_null() {
            crate::native::lens_glow_front(n.view);
        }
        n.layout(w, h, contrast);
        n.orient();
    });
}

/// lens.rs entered a phase.
pub fn phase(app: &AppHandle, inp: PhaseIn) {
    let at = Instant::now();
    with(app, move |n| unsafe {
        let (reduced, _, contrast, _) = crate::native::a11y_flags();
        if n.size.0 > 0.0 {
            n.layout(n.size.0, n.size.1, contrast);
        }
        n.phase(inp, reduced, at);
    });
}

/// The page selected an area (or let go of one): `lens_glow {selected}`.
pub fn selected(app: &AppHandle, on: bool) {
    let at = Instant::now();
    with(app, move |n| unsafe {
        let (reduced, _, _, _) = crate::native::a11y_flags();
        n.select(on, reduced, at);
    });
}

/// The overlay ordered out: no glow, no animation left running.
pub fn hide(app: &AppHandle) {
    with(app, |n| unsafe { n.hide() });
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2::runtime::Sel;

    #[derive(serde::Deserialize)]
    struct FixCtx {
        #[serde(rename = "toArmMs")]
        to_arm_ms: f64,
        reduced: bool,
        via: String,
        from: Option<FixLook>,
        notch: Option<bool>,
    }
    #[derive(serde::Deserialize, Clone, Copy)]
    struct FixLook {
        band: f64,
        opacity: f64,
        period: f64,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        phases: Vec<String>,
        ctxs: Vec<FixCtx>,
        ms: Vec<f64>,
        spans: Vec<f64>,
        looks: Vec<[f64; 3]>,
    }

    fn phase_of(s: &str) -> GlowPhase {
        match s {
            "idle" => GlowPhase::Idle,
            "warm" => GlowPhase::Warm,
            "preview" => GlowPhase::Preview,
            "armed" => GlowPhase::Armed,
            "open" => GlowPhase::Open,
            "cancel" => GlowPhase::Cancel,
            "closing" => GlowPhase::Closing,
            _ => panic!("{s}"),
        }
    }

    #[test]
    fn the_glow_curves_match_the_pages() {
        let f: Fixture = serde_json::from_str(include_str!("../../tests/fixtures/lens-glow.json")).unwrap();
        let mut k = 0;
        let mut s = 0;
        assert!(f.looks.len() > 500);
        for p in &f.phases {
            for c in &f.ctxs {
                let ctx = Ctx {
                    to_arm_ms: c.to_arm_ms,
                    reduced: c.reduced,
                    via_hold: c.via == "hold",
                    from: c.from.map(|l| Look { band: l.band, opacity: l.opacity, period: l.period }),
                    notch: c.notch.unwrap_or(false),
                };
                assert!((glow_span(phase_of(p), &ctx) - f.spans[s]).abs() < 1e-9, "span {p} {s}");
                s += 1;
                for ms in &f.ms {
                    let got = glow_at(phase_of(p), *ms, &ctx);
                    let want = f.looks[k];
                    for (a, b, what) in [(got.band, want[0], "band"), (got.opacity, want[1], "opacity"), (got.period, want[2], "period")] {
                        assert!((a - b).abs() < 1e-9, "{p} at {ms} ms ({what}): {a} vs {b}");
                    }
                    k += 1;
                }
            }
        }
        assert_eq!(k, f.looks.len());
    }

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1.0 / 255.0)
    }

    #[test]
    fn the_strips_meet_at_every_corner_and_repeat_every_l() {
        for (w, h) in [(1512.0, 982.0), (1728.0, 1117.0), (2560.0, 1440.0), (5120.0, 2880.0)] {
            let l = ring_len(w, h);
            for k in 0..8 {
                let s = l * k as f64 / 8.0;
                let c = |e: Edge, pos: f64| edge_colour(e, pos, s, w, h);
                assert!(close(c(Edge::Top, w), c(Edge::Right, 0.0)), "top-right {w}x{h} at {s}");
                assert!(close(c(Edge::Right, h), c(Edge::Bottom, w)), "bottom-right {w}x{h} at {s}");
                assert!(close(c(Edge::Bottom, 0.0), c(Edge::Left, h)), "bottom-left {w}x{h} at {s}");
                assert!(close(c(Edge::Left, 0.0), c(Edge::Top, 0.0)), "top-left {w}x{h} at {s}");
                // A whole length slid is where it started: the slide loops without a seam.
                assert!(close(edge_colour(Edge::Top, 300.0, s + l, w, h), c(Edge::Top, 300.0)));
                assert!(close(edge_colour(Edge::Left, 300.0, s + l, w, h), c(Edge::Left, 300.0)));
            }
            // Clockwise: sliding moves a colour right along the top and down the right side.
            assert!(close(edge_colour(Edge::Top, 500.0, 40.0, w, h), edge_colour(Edge::Top, 460.0, 0.0, w, h)));
            assert!(close(edge_colour(Edge::Right, 500.0, 40.0, w, h), edge_colour(Edge::Right, 460.0, 0.0, w, h)));
            let g = Geometry::new(w, h);
            // Each strip covers its edge however far it has slid (one length either side).
            assert!(g.along_tb as f64 * g.step >= w + l && g.along_lr as f64 * g.step >= h + l);
            for e in EDGES {
                let img = bake(e, &g, true);
                let along = g.along(e);
                assert!(along < 16_384, "under the texture limit");
                for i in [0, 1, 17, (along - g.period_px) / 2, along - g.period_px - 1] {
                    let (a, b) = match e {
                        Edge::Top | Edge::Bottom => (img.px(i, 0), img.px(i + g.period_px, 0)),
                        _ => (img.px(0, i), img.px(0, i + g.period_px)),
                    };
                    let d = a.iter().zip(b).map(|(x, y)| (*x as i32 - y as i32).abs()).max().unwrap();
                    assert!(d <= 1, "{e:?} px {i} vs {}: {a:?} {b:?}", i + g.period_px);
                }
            }
        }
        // A 16-inch top band is about 2.3K × 64 px, and all eight about 2.3 MB.
        let g = Geometry::new(1728.0, 1117.0);
        let total: usize = EDGES.iter().map(|e| bake(*e, &g, true).bytes.len() + bake(*e, &g, false).bytes.len()).sum();
        assert!((2200..2400).contains(&g.along_tb), "{}", g.along_tb);
        assert!(total < 3_000_000, "{total}");
    }

    #[test]
    fn the_band_falls_off_as_the_mask_did() {
        let q = |v: f64| (v * 255.0).round() as i32;
        for (t, want) in [(0.0, 255), (0.14, 158), (0.5, 56), (1.0, 0)] {
            assert!((q(falloff(t)) - want).abs() <= 1, "{t}: {}", q(falloff(t)));
        }
        let g = Geometry::new(1512.0, 982.0);
        // The screen's edge is opaque, the inner edge clear, for every band; premultiplied.
        let top = bake(Edge::Top, &g, true);
        assert_eq!((top.width, top.height), (g.along_tb, BAND_PX));
        assert_eq!(top.px(10, 0)[3], 255);
        assert_eq!(top.px(10, BAND_PX - 1)[3], 0);
        assert_eq!(top.px(10, BAND_PX - 1), [0, 0, 0, 0]);
        let bottom = bake(Edge::Bottom, &g, true);
        assert_eq!(bottom.px(10, BAND_PX - 1)[3], 255);
        assert_eq!(bottom.px(10, 0)[3], 0);
        let left = bake(Edge::Left, &g, true);
        assert_eq!((left.width, left.height), (BAND_PX, g.along_lr));
        assert_eq!(left.px(0, 10)[3], 255);
        let right = bake(Edge::Right, &g, true);
        assert_eq!(right.px(BAND_PX - 1, 10)[3], 255);
        assert_eq!(right.px(0, 10)[3], 0);
        let mid = top.px(10, BAND_PX / 2);
        assert!(mid[0] <= mid[3] && mid[1] <= mid[3] && mid[2] <= mid[3], "premultiplied");
        // A rim is one pixel across, solid.
        let rim = bake(Edge::Right, &g, false);
        assert_eq!((rim.width, rim.height), (1, g.along_lr));
        assert!((0..rim.height).step_by(97).all(|y| rim.px(0, y)[3] == 255));
        // The colours are NEO's: the first pixel of the top strip is the ring's start (sky blue).
        let c = ring(0.0, g.l);
        assert!((c[0] - 0x6f as f64 / 255.0).abs() < 1e-9 && c[2] == 1.0);
    }

    #[test]
    fn a_speed_change_keeps_the_colours_where_they_are() {
        let mut t = Timing { time_offset: 0.0, begin_time: 0.0, speed: 1.0 };
        let mut parent = 100.0;
        for speed in [slide_speed(PERIOD_BUILDING), slide_speed(PERIOD_OPEN), 0.0, 1.0] {
            let before = local_time(&t, parent);
            t = rebase(before, parent, speed);
            assert!((local_time(&t, parent) - before).abs() < 1e-12, "no jump into {speed}");
            parent += 0.5;
            assert!((local_time(&t, parent) - (before + 0.5 * speed)).abs() < 1e-12, "then {speed}× time");
        }
        assert!((slide_speed(PERIOD_BUILDING) - 3.3333333333).abs() < 1e-6);
        assert_eq!(slide_speed(PERIOD_OPEN), 1.0);
        assert_eq!(slide_speed(0.0), 0.0);
        assert_eq!(sample_count(0.0), 1);
        assert_eq!(sample_count(560.0), 70);
        assert_eq!(sample_count(5000.0), 128);
    }

    unsafe fn count(l: Id) -> usize {
        let subs: Id = msg_send![l, sublayers];
        let n: usize = if subs.is_null() { 0 } else { msg_send![subs, count] };
        let mut total = 1;
        for i in 0..n {
            let s: Id = msg_send![subs, objectAtIndex: i];
            total += count(s);
        }
        total
    }
    unsafe fn keys(l: Id) -> Vec<String> {
        let ks: Id = msg_send![l, animationKeys];
        if ks.is_null() {
            return vec![];
        }
        let n: usize = msg_send![ks, count];
        (0..n)
            .map(|i| {
                let k: Id = msg_send![ks, objectAtIndex: i];
                let p: *const std::ffi::c_char = msg_send![k, UTF8String];
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            })
            .collect()
    }
    unsafe fn rate(l: Id, key: &str) -> Option<CAFrameRateRange> {
        let a: Id = msg_send![l, animationForKey: nsstr(key)];
        assert!(!a.is_null(), "{key}");
        let can: Bool = msg_send![a, respondsToSelector: sel!(preferredFrameRateRange)];
        can.as_bool().then(|| msg_send![a, preferredFrameRateRange])
    }

    #[test]
    fn the_layers_build_headless() {
        objc2::rc::autoreleasepool(|_| unsafe {
            let mut n = Native::build();
            n.layout(1512.0, 982.0, false);
            assert_eq!(count(n.root), 19, "root, dim, breathe, and a clip and a strip for each band and rim");
            let flipped: Bool = msg_send![n.root, contentsAreFlipped];
            assert!(flipped.as_bool(), "top-left coordinates");
            let hidden: Bool = msg_send![n.root, isHidden];
            assert!(hidden.as_bool());
            // A band's clip is the full 56 pt, scaled across; its strip reaches one length past the screen.
            let f: CGRect = msg_send![n.band_strips[0], frame];
            assert_eq!(f.origin.x, -ring_len(1512.0, 982.0));
            let contents: Id = msg_send![n.band_strips[0], contents];
            assert!(!contents.is_null(), "the strip's image is in");
            let t0 = Instant::now();
            let ms = |v: u64| t0 + std::time::Duration::from_millis(v);
            n.phase(PhaseIn::of(GlowPhase::Warm), false, ms(0));
            let hidden: Bool = msg_send![n.root, isHidden];
            assert!(!hidden.as_bool());
            assert!(keys(n.band_strips[0]).contains(&"slide".to_string()), "the colours travel from warm");
            assert!((n.speed - slide_speed(PERIOD_BUILDING)).abs() < 1e-9);
            n.phase(PhaseIn { phase: GlowPhase::Preview, to_arm_ms: 550.0, via_hold: true, notch: false }, false, ms(150));
            assert!(keys(n.root).contains(&"phase".to_string()));
            for i in 0..4 {
                assert!(keys(n.band_clips[i]).contains(&"band".to_string()));
                assert!(keys(n.rim_strips[i]).contains(&"slide".to_string()));
            }
            // Every animation asks for 120 Hz, where this Mac can.
            for (l, k) in [(n.root, "phase"), (n.band_strips[0], "slide"), (n.rim_strips[3], "slide"), (n.band_clips[3], "band")] {
                if let Some(r) = rate(l, k) {
                    assert_eq!(r, FRAME_RATE, "{k}");
                }
            }
            n.phase(PhaseIn::of(GlowPhase::Armed), false, ms(700));
            assert!(keys(n.root).is_empty(), "armed: full brightness, no fade");
            assert!(keys(n.band_clips[1]).contains(&"band".to_string()), "the swell and the spring");
            n.phase(PhaseIn::of(GlowPhase::Open), false, ms(1300));
            assert!((n.speed - 1.0).abs() < 1e-9, "open: one turn per 8 s");
            assert!(keys(n.breathe).contains(&"breathe".to_string()));
            assert!((get_num(n.root, "opacity") - 1.0).abs() < 1e-6);
            n.select(true, false, ms(2000));
            assert!(keys(n.dim).contains(&"dim".to_string()));
            assert!((get_num(n.dim, "opacity") - OPACITY_SELECTED).abs() < 1e-6);
            assert!((get_num(n.band_clips[0], "transform.scale.y") - BAND_SELECTED / BAND_MAX).abs() < 1e-6);
            assert!(!keys(n.breathe).contains(&"breathe".to_string()), "no breathing over an area");
            n.select(false, false, ms(2500));
            assert!(keys(n.breathe).contains(&"breathe".to_string()));
            assert!((get_num(n.band_clips[2], "transform.scale.y") - BAND_OPEN / BAND_MAX).abs() < 1e-6);
            n.select(true, false, ms(2600));
            n.phase(PhaseIn { phase: GlowPhase::Closing, to_arm_ms: 0.0, via_hold: true, notch: true }, false, ms(3000));
            assert!(keys(n.root).contains(&"phase".to_string()));
            assert!(get_num(n.root, "opacity").abs() < 1e-6, "fades to nothing");
            assert!((get_num(n.band_clips[3], "transform.scale.x") - BAND_SELECTED / BAND_MAX).abs() < 1e-6, "the band held");
            for (l, k) in [(n.root, "phase"), (n.dim, "dim")] {
                if let Some(r) = rate(l, k) {
                    assert_eq!(r, FRAME_RATE, "{k}");
                }
            }
            n.phase(PhaseIn::of(GlowPhase::Idle), false, ms(3300));
            for l in n.all() {
                assert!(keys(l).is_empty(), "nothing ticking at idle");
            }
            let hidden: Bool = msg_send![n.root, isHidden];
            assert!(hidden.as_bool());
            // Reduce Motion: still colours, no breathing, fades of 120 ms.
            n.phase(PhaseIn::of(GlowPhase::Warm), true, ms(4000));
            assert!(!n.sliding && keys(n.band_strips[0]).is_empty());
            n.phase(PhaseIn { phase: GlowPhase::Open, to_arm_ms: 0.0, via_hold: false, notch: false }, true, ms(4100));
            assert!(!keys(n.breathe).contains(&"breathe".to_string()));
            assert!(keys(n.root).contains(&"phase".to_string()));
            // A bigger display: laid out again, the strips re-baked; Increase Contrast: a 4 pt rim.
            n.layout(2560.0, 1440.0, true);
            let f: CGRect = msg_send![n.rim_clips[1], frame];
            assert_eq!((f.origin.x, f.size.w, f.size.h), (2556.0, 4.0, 1440.0));
            assert_eq!(n.baked, Some((2560, 1440)));
            drop(n);
        });
    }

    #[test]
    fn the_glows_view_never_takes_a_click() {
        let cls = crate::native::glow_view_class().expect("NeoLensGlowView");
        assert_eq!(cls.name().to_str().unwrap(), "NeoLensGlowView");
        assert_eq!(cls.superclass().map(|c| c.name().to_str().unwrap().to_string()), Some("NSView".into()));
        let own = |name: Sel| cls.instance_methods().iter().any(|m| m.name() == name);
        for sel in [sel!(hitTest:), sel!(isOpaque), sel!(acceptsFirstResponder)] {
            assert!(own(sel), "{sel:?} is its own");
        }
        // hitTest: keeps NSView's signature: a point in, an object (always nil) out.
        let m = cls.instance_method(sel!(hitTest:)).unwrap();
        assert_eq!(m.return_type().to_str().unwrap(), "@");
        assert!(m.argument_type(2).unwrap().to_str().unwrap().starts_with("{CGPoint="));
        let src = include_str!("native.rs");
        let hit = src.split("extern \"C\" fn glow_hit_test(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(hit.contains("std::ptr::null_mut()"));
        let attach = src.split("pub unsafe fn lens_glow_attach(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(attach.find("setLayer: layer").unwrap() < attach.find("setWantsLayer: true").unwrap(), "layer-hosting");
        assert!(attach.contains("positioned: NS_WINDOW_ABOVE, relativeTo: web"), "above the page");
    }

    #[test]
    fn the_glow_never_paints_on_the_main_thread_per_frame() {
        let src = include_str!("glow.rs");
        let body = src.split("#[cfg(test)]").next().unwrap();
        for bad in [concat!("setNeeds", "Display"), concat!("NS", "Timer"), concat!("CVDisplay", "Link"), concat!("thread::", "spawn"), concat!("setMa", "sk:"), concat!("setFil", "ters:"), concat!("setSha", "dow")] {
            assert!(!body.contains(bad), "{bad}");
        }
        // Every animation goes through the one helper that asks for 120 Hz.
        assert_eq!(body.matches(concat!("addAnim", "ation:")).count(), 1);
        let helper = body.split("unsafe fn add_anim(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(helper.contains("setPreferredFrameRateRange") && helper.contains("respondsToSelector"));
        // No group opacity on the three containers (every layer is made without it).
        let new = body.split("unsafe fn new_layer(").nth(1).unwrap().split("\n}\n").next().unwrap();
        assert!(new.contains("setAllowsGroupOpacity: Bool::NO"));
        let build = body.split("pub unsafe fn build(").nth(1).unwrap().split("\n    }\n").next().unwrap();
        for l in ["let root = new_layer()", "let dim = new_layer()", "let breathe = new_layer()"] {
            assert!(build.contains(l), "{l}");
        }
    }
}

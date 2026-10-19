//! NEO's engine: the Python (and every package) the core runs with, how it gets onto a Mac, and
//! which one a launch uses.
//!
//! A downloaded NEO.app carries no Python. Its Resources hold `engine/manifest.json` (written by
//! scripts/stage_resources.sh: which engine this NEO runs, where it is published, its SHA-256 and
//! sizes, and the neo2 wheel beside it), `engine/neo2-X.Y.Z-py3-none-any.whl` and `ui/` (the page,
//! for NotchHub). On a new Mac the Setup window (setupwin.rs) installs the engine:
//!
//! 1. room: the unpacked engine, the models and a gigabyte to spare must fit on the disk;
//! 2. download: `/usr/bin/curl` into `.downloads/<id>.part`, resumed where it stopped (`-C -`),
//!    progress measured as the file grows against the manifest's byte count; Cancel kills curl
//!    and keeps the part for Retry. Or "Use a downloaded file…": an archive from anywhere;
//! 3. integrity: its SHA-256 against the one in the manifest (in the signed bundle);
//! 4. unpack: `/usr/bin/tar` into `.staging-<id>`, refusing `..` and absolute names and writes
//!    through links (bsdtar's defaults), with no extended attributes (no quarantine flag). Then
//!    every entry is checked: only folders, plain files and links that stay inside, nothing setuid;
//! 5. switch in: the staged folder becomes `engines/<id>` (one rename; a repaired engine's old copy
//!    waits as `.replaced-<id>` until the new one works);
//! 6. compile the bytecode (`compileall`, shown as "Optimizing"), install the bundled neo2 wheel
//!    offline, and a smoke test: the engine imports NEO's main libraries and says its neo2 version;
//! 7. `engines/<id>/.neo-ready`, then `engine.json` (through a temporary file and a rename) points
//!    at it. That write is the one moment the Mac switches engines; anything that stopped before it
//!    is undone at the next launch (`cleanup`).
//!
//! `engine.json` holds no paths: `{"schema":1,"source":"release"|"dev","engine_id","neo2_sha256"}`.
//! The interpreter is always one of two fixed places — `engines/<16 hex>/bin/python3` or `.venv/bin/
//! python` (scripts/install-app.sh's developer engine) — so no file can point NEO.app, and the
//! permissions macOS gave it, at other code. Nothing ever writes inside NEO.app, and NEO never
//! edits a Mach-O file it unpacked.
//!
//! What a launch does with what it finds is `decide` (the design's cases a–f). Each step is logged
//! to ~/.neo/logs/setup.log.

use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::core;

/// The manifest's own format version: a NEO.app only reads its own.
pub const MANIFEST_SCHEMA: u64 = 1;
const POINTER_SCHEMA: u64 = 1;
/// What every first start downloads besides the engine (`neo2 fetch --essential`), when the
/// manifest doesn't say.
pub const MODELS_BYTES: u64 = 1_700_000_000;
/// Room left on the disk after the engine and the models.
pub const HEADROOM: u64 = 1_000_000_000;
/// The smoke test: what NEO's core can't start without, when the manifest doesn't say.
pub const SMOKE: [&str; 7] = ["neo2", "torch", "mlx.core", "laya", "sounddevice", "Quartz", "AVFoundation"];
const SMOKE_TIMEOUT: Duration = Duration::from_secs(180);
const COMPILE_TIMEOUT: Duration = Duration::from_secs(1200);
const PIP_TIMEOUT: Duration = Duration::from_secs(300);
const POLL: Duration = Duration::from_millis(100);
/// The interpreter children of the installer get: macOS's own tools only.
const SYSTEM_PATH: &str = "/usr/bin:/bin:/usr/sbin:/sbin";

// ---- the manifest (in the bundle) -----------------------------------------------------------------

/// The engine a NEO.app runs, as published beside its release.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineAsset {
    /// 16 lowercase hex characters (scripts/engine_id.py): its folder is engines/<id>.
    pub id: String,
    pub url: String,
    pub sha256: String,
    /// The archive's size, exactly.
    pub bytes: u64,
    /// About how much it takes once unpacked.
    pub unpacked_bytes: u64,
    /// The archive's entries (`tar -t | wc -l`): what unpacking counts against.
    pub files: u64,
    /// The interpreter's version, X.Y.Z (its packages live in lib/pythonX.Y).
    pub python: String,
    #[serde(default)]
    pub min_macos: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: u64,
    pub app_version: String,
    pub neo2_version: String,
    /// The wheel's file name, beside the manifest in Resources/engine.
    pub wheel: String,
    pub wheel_sha256: String,
    pub engine: EngineAsset,
    /// What the essential models download (Setup's first screen says it before anything starts).
    #[serde(default)]
    pub models_bytes: u64,
    /// What the smoke test imports (module names only).
    #[serde(default)]
    pub smoke: Vec<String>,
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// An engine's id: what may name a folder in engines/.
pub fn valid_id(id: &str) -> bool {
    is_hex(id, 16)
}

fn valid_version(v: &str) -> bool {
    let parts: Vec<&str> = v.split('.').collect();
    parts.len() == 3 && parts.iter().all(|p| !p.is_empty() && p.len() <= 6 && p.bytes().all(|b| b.is_ascii_digit()))
}

fn valid_module(m: &str) -> bool {
    !m.is_empty()
        && m.len() <= 64
        && m.split('.').all(|part| {
            let mut b = part.bytes();
            matches!(b.next(), Some(c) if c.is_ascii_alphabetic() || c == b'_') && b.all(|c| c.is_ascii_alphanumeric() || c == b'_')
        })
}

/// Where an engine may come from: GitHub's https in NEO.app; a test build (`sandbox`) also takes a
/// local file or a server on this Mac.
pub fn url_allowed(url: &str, sandbox: bool) -> bool {
    if url.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return false;
    }
    if let Some(rest) = url.strip_prefix("https://") {
        return !rest.is_empty() && !rest.starts_with('/');
    }
    sandbox && (url.starts_with("file:///") || url.starts_with("http://127.0.0.1:") || url.starts_with("http://localhost:"))
}

impl Manifest {
    pub fn parse(text: &str, sandbox: bool) -> Result<Manifest, String> {
        let m: Manifest = serde_json::from_str(text).map_err(|e| format!("not a manifest: {e}"))?;
        if m.schema != MANIFEST_SCHEMA {
            return Err(format!("manifest schema {} (this NEO reads {MANIFEST_SCHEMA})", m.schema));
        }
        let e = &m.engine;
        if !valid_id(&e.id) {
            return Err("the engine's id isn't 16 hex characters".into());
        }
        if !is_hex(&e.sha256, 64) || !is_hex(&m.wheel_sha256, 64) {
            return Err("a SHA-256 isn't 64 lowercase hex characters".into());
        }
        if !url_allowed(&e.url, sandbox) {
            return Err(format!("the engine's URL isn't one NEO downloads from: {:.80}", e.url));
        }
        if e.bytes == 0 || e.unpacked_bytes == 0 || e.files == 0 {
            return Err("the engine's sizes are missing".into());
        }
        if !valid_version(&e.python) || !e.python.starts_with("3.") {
            return Err(format!("the engine's Python version isn't X.Y.Z: {:.20}", e.python));
        }
        if !valid_version(&m.neo2_version) {
            return Err("neo2's version isn't X.Y.Z".into());
        }
        let wheel_ok = m.wheel.starts_with("neo2-")
            && m.wheel.ends_with("-py3-none-any.whl")
            && m.wheel.len() < 80
            && m.wheel.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b));
        if !wheel_ok {
            return Err("the wheel's name isn't neo2-X.Y.Z-py3-none-any.whl".into());
        }
        if let Some(bad) = m.smoke.iter().find(|s| !valid_module(s)) {
            return Err(format!("the smoke test names something that isn't a module: {bad:.40}"));
        }
        Ok(m)
    }

    /// The modules the smoke test imports.
    pub fn smoke_modules(&self) -> Vec<String> {
        if self.smoke.is_empty() {
            SMOKE.iter().map(|s| s.to_string()).collect()
        } else {
            self.smoke.clone()
        }
    }

    pub fn models_bytes(&self) -> u64 {
        if self.models_bytes > 0 {
            self.models_bytes
        } else {
            MODELS_BYTES
        }
    }

    /// `lib/python3.12`, from the engine's Python version.
    pub fn lib_dir(&self) -> String {
        let mut it = self.engine.python.split('.');
        format!("lib/python{}.{}", it.next().unwrap_or("3"), it.next().unwrap_or("12"))
    }

    /// The archive's file name, as its URL ends.
    pub fn asset_name(&self) -> String {
        let tail = self.engine.url.rsplit('/').next().unwrap_or("");
        let tail = tail.split(['?', '#']).next().unwrap_or("");
        if tail.is_empty() {
            format!("neo-engine-{}.tar.xz", self.engine.id)
        } else {
            tail.replace("%2B", "+")
        }
    }

    /// The host an engine is downloaded from, for the words on the screen ("github.com").
    pub fn host(&self) -> String {
        let url = &self.engine.url;
        if url.starts_with("file://") {
            return "this Mac".into();
        }
        let rest = url.split("://").nth(1).unwrap_or("");
        rest.split(['/', ':']).next().unwrap_or("").to_string()
    }
}

/// What this NEO.app carries: its Resources folder, and the manifest in it (when it has one —
/// `tauri dev` and builds without staged resources don't; they run developer engines only).
pub struct Bundle {
    pub resources: PathBuf,
    pub manifest: Option<Manifest>,
    pub problem: Option<String>,
}

impl Bundle {
    pub fn load(resources: PathBuf, sandbox: bool) -> Bundle {
        let path = resources.join("engine").join("manifest.json");
        match fs::read_to_string(&path) {
            Err(_) => Bundle { resources, manifest: None, problem: None },
            Ok(text) => match Manifest::parse(&text, sandbox) {
                Ok(m) => Bundle { resources, manifest: Some(m), problem: None },
                Err(e) => Bundle { resources, manifest: None, problem: Some(e) },
            },
        }
    }

    pub fn wheel(&self) -> Option<PathBuf> {
        self.manifest.as_ref().map(|m| self.resources.join("engine").join(&m.wheel))
    }

    pub fn ui(&self) -> PathBuf {
        self.resources.join("ui")
    }
}

/// This app's bundle, read once (None when it isn't running from a .app).
pub fn bundle() -> Option<&'static Bundle> {
    static B: OnceLock<Option<Bundle>> = OnceLock::new();
    B.get_or_init(|| {
        let app = core::bundle_path()?;
        let b = Bundle::load(app.join("Contents").join("Resources"), core::SANDBOX);
        if let Some(p) = &b.problem {
            log(&format!("this NEO.app's engine manifest can't be used: {p}"));
        }
        Some(b)
    })
    .as_ref()
}

pub fn manifest() -> Option<&'static Manifest> {
    bundle().and_then(|b| b.manifest.as_ref())
}

// ---- where things are ---------------------------------------------------------------------------

/// NEO's folder in Application Support, and what lives in it.
#[derive(Debug, Clone)]
pub struct Paths {
    pub root: PathBuf,
}

impl Paths {
    pub fn new(root: impl Into<PathBuf>) -> Paths {
        Paths { root: root.into() }
    }

    /// This Mac's (or a sandbox build's) ~/Library/Application Support/NEO.
    pub fn here() -> Paths {
        Paths::new(core::runtime_dir())
    }

    pub fn engines(&self) -> PathBuf {
        self.root.join("engines")
    }
    pub fn engine(&self, id: &str) -> PathBuf {
        self.engines().join(id)
    }
    pub fn python(&self, id: &str) -> PathBuf {
        self.engine(id).join("bin").join("python3")
    }
    fn ready(&self, id: &str) -> PathBuf {
        self.engine(id).join(".neo-ready")
    }
    pub fn downloads(&self) -> PathBuf {
        self.root.join(".downloads")
    }
    pub fn part(&self, id: &str) -> PathBuf {
        self.downloads().join(format!("{id}.part"))
    }
    fn staging(&self, id: &str) -> PathBuf {
        self.root.join(format!(".staging-{id}"))
    }
    fn replaced(&self, id: &str) -> PathBuf {
        self.root.join(format!(".replaced-{id}"))
    }
    pub fn pointer(&self) -> PathBuf {
        self.root.join("engine.json")
    }
    pub fn ui(&self) -> PathBuf {
        self.root.join("ui")
    }
    fn lock(&self) -> PathBuf {
        self.root.join(".setup.lock")
    }
}

// ---- engine.json ------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pointer {
    pub schema: u64,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neo2_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Points {
    /// No engine.json (this Mac before 2.2, or a fresh one).
    Missing,
    /// One that can't be read as one: never trusted, never acted on beyond "not set up".
    Invalid,
    /// scripts/install-app.sh's engine (.venv).
    Dev,
    Release { id: String, wheel: String },
}

pub fn read_pointer(paths: &Paths) -> Points {
    let Ok(text) = fs::read_to_string(paths.pointer()) else { return Points::Missing };
    let Ok(p) = serde_json::from_str::<Pointer>(&text) else { return Points::Invalid };
    if p.schema != POINTER_SCHEMA {
        return Points::Invalid;
    }
    match p.source.as_str() {
        "dev" => Points::Dev,
        "release" => match (p.engine_id, p.neo2_sha256) {
            (Some(id), Some(sha)) if valid_id(&id) && is_hex(&sha, 64) => Points::Release { id, wheel: sha },
            _ => Points::Invalid,
        },
        _ => Points::Invalid,
    }
}

/// engine.json, through a temporary file and a rename: whoever reads it sees the old pointer or the
/// new one, never half of one.
pub fn write_pointer(paths: &Paths, p: &Pointer) -> std::io::Result<()> {
    fs::create_dir_all(&paths.root)?;
    let tmp = paths.root.join(format!("engine.json.tmp-{}", std::process::id()));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(serde_json::to_string(p).map_err(std::io::Error::other)?.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_all()?;
    }
    fs::rename(&tmp, paths.pointer())
}

fn release_pointer(id: &str, wheel_sha: &str) -> Pointer {
    Pointer { schema: POINTER_SCHEMA, source: "release".into(), engine_id: Some(id.into()), neo2_sha256: Some(wheel_sha.into()) }
}

// ---- which engine a launch runs -------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// (a) no engine.json but a .venv, (b) engine.json says dev: the developer engine, as before 2.2.
    Dev,
    /// (d) the release engine this app was built with, ready.
    Release { id: String },
    /// (e) the same engine, another neo2: reinstall the bundled wheel (offline, seconds), then run.
    UpdateWheel { id: String },
    /// (c) nothing installed, or the engine is gone or half there: Setup installs it.
    Install,
    /// (f) an engine this app wasn't built with: Setup downloads this app's alongside, then switches.
    NewEngine { from: String },
    /// Nothing to run and nothing this build could install (no manifest: a developer build).
    NoEngine,
}

impl Decision {
    /// Whether Setup has to run before a core can.
    pub fn needs_setup(&self) -> bool {
        matches!(self, Decision::Install | Decision::NewEngine { .. } | Decision::NoEngine)
    }
}

/// What this launch runs. `dev_ok`: a developer engine's interpreter is there (`.venv/bin/python`,
/// or a debug build's own checkout).
pub fn decide(paths: &Paths, manifest: Option<&Manifest>, dev_ok: bool) -> Decision {
    let fallback = || {
        if dev_ok {
            Decision::Dev
        } else if manifest.is_some() {
            Decision::Install
        } else {
            Decision::NoEngine
        }
    };
    match read_pointer(paths) {
        Points::Missing | Points::Invalid | Points::Dev => fallback(),
        Points::Release { id, wheel } => {
            let ready = paths.python(&id).is_file() && paths.ready(&id).is_file();
            match manifest {
                // A build without a manifest can't check the engine; it runs a ready one as it is.
                None => {
                    if ready {
                        Decision::Release { id }
                    } else {
                        fallback()
                    }
                }
                Some(m) if m.engine.id != id => Decision::NewEngine { from: id },
                Some(_) if !ready => Decision::Install,
                Some(m) if m.wheel_sha256 != wheel => Decision::UpdateWheel { id },
                Some(_) => Decision::Release { id },
            }
        }
    }
}

/// The release engine engine.json points at, if any (its id).
pub fn current_release(paths: &Paths) -> Option<String> {
    match read_pointer(paths) {
        Points::Release { id, .. } => Some(id),
        _ => None,
    }
}

// ---- where NEO.app runs from ------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Place {
    Ok,
    /// macOS runs a quarantined app from a random read-only copy (App Translocation).
    Translocated,
    /// A disk image (or another volume's loose copy, not in an Applications folder).
    Volume,
}

/// Where this binary runs from, from its path alone (`read_only`: the volume it is on can't be
/// written). An app opened from the DMG or straight from Downloads is not where it will stay: its
/// login item would point at a path that goes away.
pub fn placement(exe: &Path, read_only: bool) -> Place {
    let s = exe.to_string_lossy();
    if s.contains("/AppTranslocation/") {
        return Place::Translocated;
    }
    if let Some(rest) = s.strip_prefix("/Volumes/") {
        let in_apps = rest.split('/').skip(1).any(|c| c == "Applications");
        if read_only || !in_apps {
            return Place::Volume;
        }
    }
    Place::Ok
}

fn read_only_volume(path: &Path) -> bool {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = CString::new(path.as_os_str().as_bytes()) else { return false };
    let mut st: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statfs(c.as_ptr(), &mut st) } != 0 {
        return false;
    }
    (st.f_flags & libc::MNT_RDONLY as u32) != 0
}

/// Where this NEO.app runs from now (Ok when it isn't a bundle: `tauri dev`).
pub fn placement_now() -> Place {
    let Ok(exe) = std::env::current_exe() else { return Place::Ok };
    if core::bundle_path().is_none() {
        return Place::Ok;
    }
    placement(&exe, read_only_volume(&exe))
}

/// Whether NEO may write its login item for a copy at `app`: only in /Applications or
/// ~/Applications (anything else may move or vanish), never from a disk image or translocated.
pub fn login_allowed(app: &Path, home: &Path) -> bool {
    if placement(app, false) != Place::Ok {
        return false;
    }
    let normal = !app.components().any(|c| matches!(c, Component::ParentDir | Component::CurDir));
    normal && (app.starts_with("/Applications") || app.starts_with(home.join("Applications")))
}

/// The macOS version this runs on, X.Y (from the kernel).
pub fn macos_version() -> Option<(u32, u32)> {
    let mut buf = [0u8; 32];
    let mut len = buf.len();
    let name = c"kern.osproductversion";
    let rc = unsafe { libc::sysctlbyname(name.as_ptr(), buf.as_mut_ptr() as *mut _, &mut len, std::ptr::null_mut(), 0) };
    if rc != 0 {
        return None;
    }
    let s = std::str::from_utf8(&buf[..len.saturating_sub(1).min(buf.len())]).ok()?;
    let mut it = s.trim_end_matches('\0').split('.');
    Some((it.next()?.parse().ok()?, it.next().map(|m| m.parse().unwrap_or(0)).unwrap_or(0)))
}

// ---- the disk -------------------------------------------------------------------------------------

/// Free bytes on the volume `path` is on (or would be, once made).
pub fn free_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let mut p = path.to_path_buf();
    while !p.exists() {
        p = p.parent()?.to_path_buf();
    }
    let c = CString::new(p.as_os_str().as_bytes()).ok()?;
    let mut st: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statfs(c.as_ptr(), &mut st) } != 0 {
        return None;
    }
    Some(st.f_bavail.saturating_mul(st.f_bsize as u64))
}

/// What has to be free before an install starts: the engine unpacked, the models, the rest of
/// the download, and room to spare.
pub fn bytes_needed(m: &Manifest, downloaded: u64, from_file: bool) -> u64 {
    let rest = if from_file { 0 } else { m.engine.bytes.saturating_sub(downloaded) };
    m.engine.unpacked_bytes + m.models_bytes() + HEADROOM + rest
}

pub fn gb(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1} GB", n as f64 / 1e9)
    } else {
        format!("{} MB", (n as f64 / 1e6).round() as u64)
    }
}

fn file_len(p: &Path) -> u64 {
    fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

// ---- setup.log -------------------------------------------------------------------------------------

pub fn log_path() -> PathBuf {
    core::logs_dir().join("setup.log")
}

pub fn log(msg: &str) {
    let _ = fs::create_dir_all(core::logs_dir());
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log_path()) {
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let _ = writeln!(f, "[{secs}] {msg}");
    }
}

// ---- progress and failures ------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    #[default]
    Idle,
    Preparing,
    Downloading,
    Verifying,
    Unpacking,
    Optimizing,
    Installing,
    Testing,
    Finishing,
    Done,
    Failed,
    Cancelled,
}

/// Why an install stopped, in fixed words: `code` for the window, `message` for the user.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Failure {
    pub code: &'static str,
    pub message: String,
}

impl Failure {
    fn new(code: &'static str, message: impl Into<String>) -> Failure {
        Failure { code, message: message.into() }
    }
    fn cancelled() -> Failure {
        Failure::new("cancelled", "Setup was cancelled. Retry picks up where it stopped.")
    }
    pub fn is_cancel(&self) -> bool {
        self.code == "cancelled"
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct Progress {
    pub stage: Stage,
    pub done: u64,
    pub total: u64,
    /// Bytes a second (downloading, checking).
    pub rate: f64,
    pub eta_s: Option<u64>,
    pub error: Option<Failure>,
}

impl Progress {
    fn at(stage: Stage, done: u64, total: u64) -> Progress {
        Progress { stage, done, total, ..Progress::default() }
    }
}

/// Bytes over the last few seconds, for a rate and an ETA that don't jump about.
struct Rate {
    window: Duration,
    samples: VecDeque<(Instant, u64)>,
}

impl Rate {
    fn new() -> Rate {
        Rate { window: Duration::from_secs(6), samples: VecDeque::new() }
    }
    fn push(&mut self, n: u64) {
        let now = Instant::now();
        self.samples.push_back((now, n));
        while self.samples.len() > 2 && now.duration_since(self.samples[0].0) > self.window {
            self.samples.pop_front();
        }
    }
    fn rate(&self) -> f64 {
        let (Some(a), Some(b)) = (self.samples.front(), self.samples.back()) else { return 0.0 };
        let dt = b.0.duration_since(a.0).as_secs_f64();
        if dt < 0.5 {
            return 0.0;
        }
        b.1.saturating_sub(a.1) as f64 / dt
    }
    fn progress(&self, stage: Stage, done: u64, total: u64) -> Progress {
        let rate = self.rate();
        let eta = if rate > 1.0 && total > done { Some(((total - done) as f64 / rate).ceil() as u64) } else { None };
        Progress { stage, done, total, rate, eta_s: eta, error: None }
    }
}

// ---- running a program -----------------------------------------------------------------------------

/// One install's handle for the window: Cancel sets the flag and stops the program running now.
#[derive(Clone, Default)]
pub struct Task {
    cancel: Arc<AtomicBool>,
    pid: Arc<Mutex<Option<u32>>>,
}

impl Task {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
        if let Some(pid) = *self.pid.lock().unwrap() {
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
        }
    }
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

struct Ran {
    code: Option<i32>,
    stdout: String,
    tail: Vec<String>,
    timed_out: bool,
    cancelled: bool,
}

impl Ran {
    fn ok(&self) -> bool {
        self.code == Some(0) && !self.timed_out && !self.cancelled
    }
    fn said(&self) -> String {
        let t = self.tail.iter().rev().take(3).rev().cloned().collect::<Vec<_>>().join(" / ");
        if t.is_empty() {
            format!("exit status {}", self.code.map(|c| c.to_string()).unwrap_or_else(|| "signal".into()))
        } else {
            t
        }
    }
}

/// Run `cmd` to its end — or until the task is cancelled, or `timeout` — calling `tick` every
/// tenth of a second (with the number of stderr lines so far: tar's file count). Its output is
/// kept (stdout up to 64 KB, stderr's last 40 lines) and stderr goes to setup.log.
fn run(cmd: &mut Command, task: &Task, timeout: Option<Duration>, mut tick: impl FnMut(u64)) -> std::io::Result<Ran> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    *task.pid.lock().unwrap() = Some(child.id());
    let lines = Arc::new(AtomicU64::new(0));
    let tail = Arc::new(Mutex::new(VecDeque::<String>::new()));
    let out = child.stdout.take();
    let err = child.stderr.take();
    let reader_out = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut o) = out {
            let _ = o.by_ref().take(64 * 1024).read_to_end(&mut buf);
            let _ = std::io::copy(&mut o, &mut std::io::sink());
        }
        String::from_utf8_lossy(&buf).into_owned()
    });
    let (lines2, tail2) = (lines.clone(), tail.clone());
    let reader_err = thread::spawn(move || {
        if let Some(e) = err {
            for line in BufReader::new(e).split(b'\n').map_while(Result::ok) {
                lines2.fetch_add(1, Ordering::Relaxed);
                let s = String::from_utf8_lossy(&line).into_owned();
                let mut t = tail2.lock().unwrap();
                t.push_back(s);
                if t.len() > 40 {
                    t.pop_front();
                }
            }
        }
    });
    let started = Instant::now();
    let (mut timed_out, mut cancelled) = (false, false);
    let mut term_at: Option<Instant> = None;
    let status = loop {
        if let Some(st) = child.try_wait()? {
            break st;
        }
        if term_at.is_none() {
            if task.cancelled() {
                cancelled = true;
            } else if timeout.is_some_and(|t| started.elapsed() > t) {
                timed_out = true;
            }
            if cancelled || timed_out {
                unsafe {
                    libc::kill(child.id() as i32, libc::SIGTERM);
                }
                term_at = Some(Instant::now());
            }
        } else if term_at.is_some_and(|t| t.elapsed() > Duration::from_secs(3)) {
            let _ = child.kill();
        }
        tick(lines.load(Ordering::Relaxed));
        thread::sleep(POLL);
    };
    *task.pid.lock().unwrap() = None;
    tick(lines.load(Ordering::Relaxed));
    let stdout = reader_out.join().unwrap_or_default();
    let _ = reader_err.join();
    let tail: Vec<String> = tail.lock().unwrap().iter().cloned().collect();
    Ok(Ran { code: status.code(), stdout, tail, timed_out, cancelled: cancelled || task.cancelled() })
}

/// What installing needs to know besides the manifest.
#[derive(Clone)]
pub struct Opts {
    /// A test build's rules (more URL schemes).
    pub sandbox: bool,
    /// HOME for the programs the installer runs (a sandbox build's own home).
    pub home: PathBuf,
    /// Extra curl arguments (the tests slow a download down to cancel it).
    pub curl_extra: Vec<String>,
    pub smoke_timeout: Duration,
}

impl Opts {
    pub fn here() -> Opts {
        Opts { sandbox: core::SANDBOX, home: core::home(), curl_extra: Vec::new(), smoke_timeout: SMOKE_TIMEOUT }
    }
}

/// The engine's interpreter, run with nothing of the user's environment: `-I` (no PYTHON*
/// variables, no user site-packages, no current folder on the path), macOS's PATH, a UTF-8 locale.
fn python(engine: &Path, opts: &Opts) -> Command {
    let mut cmd = Command::new(engine.join("bin").join("python3"));
    cmd.env_clear()
        .env("HOME", &opts.home)
        .env("PATH", SYSTEM_PATH)
        .env("LANG", "en_US.UTF-8")
        .env("LC_ALL", "en_US.UTF-8")
        .arg("-I")
        .current_dir(engine);
    if let Some(t) = std::env::var_os("TMPDIR") {
        cmd.env("TMPDIR", t);
    }
    cmd
}

// ---- 1. room ----------------------------------------------------------------------------------------

fn check_room(paths: &Paths, m: &Manifest, from_file: bool) -> Result<(), Failure> {
    let downloaded = file_len(&paths.part(&m.engine.id));
    let need = bytes_needed(m, downloaded, from_file);
    let Some(free) = free_bytes(&paths.root) else { return Ok(()) };
    if free < need {
        log(&format!("not enough room: {} needed, {} free", gb(need), gb(free)));
        return Err(Failure::new(
            "disk_full",
            format!("NEO needs {} free on this Mac to set up, and {} is free. Make some room, then Retry.", gb(need), gb(free)),
        ));
    }
    Ok(())
}

// ---- 2. download -------------------------------------------------------------------------------------

/// What curl's exit status (and the HTTP status it saw) means for the user.
pub fn curl_failure(code: i32, http: u16, host: &str, no_room: bool) -> Failure {
    match code {
        6 | 7 | 28 | 35 | 52 | 55 | 56 | 18 | 16 | 92 => {
            Failure::new("offline", format!("Can't reach {host} — check your connection, then Retry."))
        }
        22 if http == 404 || http == 410 => Failure::new(
            "not_found",
            "This NEO's engine isn't on GitHub any more. Download the newest NEO from its releases page.",
        ),
        22 => Failure::new("failed", format!("{host} answered with HTTP {http}. Try again in a few minutes.")),
        37 | 78 => Failure::new("not_found", "The engine's file isn't there."),
        23 if no_room => Failure::new("disk_full", "The disk filled up during the download. Make some room, then Retry."),
        23 => Failure::new("failed", "The download couldn't be written to disk."),
        60 | 77 | 51 | 58 | 59 | 83 | 90 | 91 => Failure::new(
            "failed",
            format!("The connection to {host} isn't trusted (a certificate problem). A network that inspects traffic can cause this; \"Use a downloaded file…\" works around it."),
        ),
        _ => Failure::new("failed", format!("The download stopped (curl exit {code}). Retry picks up where it stopped.")),
    }
}

/// curl's exit statuses for a transfer that broke off (partial file, a receive or send that failed,
/// an empty reply, an HTTP/2 stream error, too slow for a minute): resumed from where it got to.
const RESUMABLE: [i32; 7] = [18, 56, 52, 55, 92, 16, 28];

fn curl_protocols(sandbox: bool) -> [&'static str; 4] {
    if sandbox {
        ["--proto", "=https,http,file", "--proto-redir", "=https,http"]
    } else {
        ["--proto", "=https", "--proto-redir", "=https"]
    }
}

/// The archive in .downloads, whole (not yet checked). Resumes a part already there.
fn download(paths: &Paths, m: &Manifest, opts: &Opts, task: &Task, report: &dyn Fn(&Progress)) -> Result<PathBuf, Failure> {
    let id = &m.engine.id;
    let total = m.engine.bytes;
    let part = paths.part(id);
    fs::create_dir_all(paths.downloads()).map_err(|e| Failure::new("failed", format!("Couldn't make NEO's download folder: {e}")))?;
    if file_len(&part) > total {
        let _ = fs::remove_file(&part);
    }
    let mut restarted = false;
    let mut stalls = 0;
    while file_len(&part) < total {
        if task.cancelled() {
            return Err(Failure::cancelled());
        }
        let from = file_len(&part);
        log(&format!("downloading {} from {} ({} of {} here)", m.asset_name(), m.engine.url, from, total));
        let mut cmd = Command::new("/usr/bin/curl");
        // curl's own retries are for a server that is busy (timeouts, 5xx); a transfer that breaks
        // off is resumed below, from what has arrived — curl's --retry would start each try over
        // from where this run began.
        cmd.arg("-q") // never a ~/.curlrc
            .args(["--fail", "--location", "--silent", "--show-error"])
            .args(["--retry", "3", "--retry-delay", "2"])
            .args(["--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "60"])
            .args(curl_protocols(opts.sandbox))
            .args(["--continue-at", "-", "--write-out", "%{http_code}", "--output"])
            .arg(&part)
            .args(&opts.curl_extra)
            .arg(&m.engine.url)
            .env("HOME", &opts.home);
        let mut rate = Rate::new();
        let ran = run(&mut cmd, task, None, |_| {
            let n = file_len(&part);
            rate.push(n);
            report(&rate.progress(Stage::Downloading, n, total));
        })
        .map_err(|e| Failure::new("failed", format!("curl didn't start: {e}")))?;
        if ran.cancelled {
            log("download cancelled");
            return Err(Failure::cancelled());
        }
        let grew = file_len(&part) > from;
        if grew {
            stalls = 0;
        }
        if ran.ok() {
            if !grew {
                stalls += 1;
                if stalls >= 3 {
                    break; // the server says it's done and sends nothing more: the size check says what's wrong
                }
            }
            continue;
        }
        let code = ran.code.unwrap_or(-1);
        let http: u16 = ran.stdout.trim().parse().unwrap_or(0);
        log(&format!("curl exit {code} (HTTP {http}): {}", ran.said()));
        if (code == 33 || http == 416) && !restarted {
            restarted = true; // a part the server won't resume: start it over, once
            let _ = fs::remove_file(&part);
            continue;
        }
        // The connection broke off part-way (or stalled): carry on from what arrived. Three tries in
        // a row that bring nothing are an error the user sees (offline, most likely).
        if RESUMABLE.contains(&code) {
            if grew {
                continue;
            }
            stalls += 1;
            if stalls < 3 {
                for _ in 0..20 {
                    if task.cancelled() {
                        return Err(Failure::cancelled());
                    }
                    thread::sleep(POLL);
                }
                continue;
            }
        }
        let no_room = free_bytes(&paths.root).is_some_and(|f| f < 64 << 20) || ran.said().contains("No space left");
        return Err(curl_failure(code, http, &m.host(), no_room));
    }
    let n = file_len(&part);
    if n != total {
        log(&format!("the download is {n} bytes, the manifest says {total}"));
        let _ = fs::remove_file(&part);
        return Err(Failure::new("checksum", "The download was damaged (it isn't the size it should be). Retry downloads it again."));
    }
    report(&Progress::at(Stage::Downloading, n, total));
    Ok(part)
}

// ---- 3. integrity -----------------------------------------------------------------------------------

/// A file's SHA-256 (lowercase hex), read in 1 MB pieces; None when cancelled.
pub fn sha256_file(path: &Path, task: &Task, mut each: impl FnMut(u64)) -> std::io::Result<Option<String>> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        if task.cancelled() {
            return Ok(None);
        }
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        done += n as u64;
        each(done);
    }
    Ok(Some(crate::keys::hex(&h.finalize())))
}

fn verify(archive: &Path, m: &Manifest, task: &Task, report: &dyn Fn(&Progress), from_file: bool) -> Result<(), Failure> {
    let total = file_len(archive);
    let mut rate = Rate::new();
    let mut last = Instant::now() - Duration::from_secs(1);
    let got = sha256_file(archive, task, |n| {
        if last.elapsed() >= Duration::from_millis(150) {
            last = Instant::now();
            rate.push(n);
            report(&rate.progress(Stage::Verifying, n, total));
        }
    })
    .map_err(|e| Failure::new("failed", format!("The engine's file couldn't be read: {e}")))?;
    let Some(got) = got else { return Err(Failure::cancelled()) };
    if got != m.engine.sha256 {
        log(&format!("SHA-256 {got} isn't the manifest's {}", m.engine.sha256));
        return Err(if from_file {
            Failure::new(
                "checksum",
                format!("That file isn't this NEO's engine (its checksum doesn't match). NEO {} needs {}.", m.app_version, m.asset_name()),
            )
        } else {
            Failure::new("checksum", "The download was damaged (it didn't match its checksum). Retry downloads it again.")
        });
    }
    log("the engine's SHA-256 matches");
    Ok(())
}

// ---- 4. unpack ----------------------------------------------------------------------------------------

/// Each entry under `root`: folders, plain files (none setuid or setgid) and links whose target
/// stays inside `root`. Returns how many there are.
pub fn check_tree(root: &Path) -> Result<u64, String> {
    fn inside(rel_parent: &Path, target: &Path) -> bool {
        if target.is_absolute() {
            return false;
        }
        let mut depth: i64 = 0;
        for c in rel_parent.components().chain(target.components()) {
            match c {
                Component::Normal(_) => depth += 1,
                Component::ParentDir => {
                    depth -= 1;
                    if depth < 0 {
                        return false;
                    }
                }
                Component::CurDir => {}
                _ => return false,
            }
        }
        true
    }
    let mut count = 0u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for e in entries {
            let e = e.map_err(|e| e.to_string())?;
            let path = e.path();
            let meta = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
            let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
            count += 1;
            let ft = meta.file_type();
            if ft.is_symlink() {
                let target = fs::read_link(&path).map_err(|e| e.to_string())?;
                if !inside(rel.parent().unwrap_or(Path::new("")), &target) {
                    return Err(format!("{} links outside the engine ({})", rel.display(), target.display()));
                }
            } else if ft.is_dir() {
                stack.push(path);
            } else if ft.is_file() {
                if meta.permissions().mode() & 0o6000 != 0 {
                    return Err(format!("{} is setuid or setgid", rel.display()));
                }
            } else {
                return Err(format!("{} isn't a file, a folder or a link", rel.display()));
            }
        }
    }
    Ok(count)
}

/// The engine inside the staging folder: its root, or the one folder the archive put there.
fn engine_root(staging: &Path) -> Option<PathBuf> {
    if staging.join("bin").join("python3").exists() {
        return Some(staging.to_path_buf());
    }
    let mut entries = fs::read_dir(staging).ok()?.flatten().filter(|e| e.file_name() != ".DS_Store");
    let only = entries.next()?;
    if entries.next().is_some() {
        return None;
    }
    let p = only.path();
    (fs::symlink_metadata(&p).ok()?.is_dir() && p.join("bin").join("python3").exists()).then_some(p)
}

fn unpack(paths: &Paths, m: &Manifest, archive: &Path, task: &Task, report: &dyn Fn(&Progress)) -> Result<PathBuf, Failure> {
    let staging = paths.staging(&m.engine.id);
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging).map_err(|e| Failure::new("failed", format!("Couldn't make the unpacking folder: {e}")))?;
    let total = m.engine.files;
    log(&format!("unpacking into {}", staging.display()));
    let mut cmd = Command::new("/usr/bin/tar");
    cmd.args(["-x", "-v", "-J", "-f"])
        .arg(archive)
        .arg("-C")
        .arg(&staging)
        // No owners, ACLs, flags, extended attributes or AppleDouble metadata from the archive (so
        // no quarantine flag either). Names with "..", absolute names and writes through a link are
        // refused by bsdtar's own defaults (no -P).
        .args(["--no-same-owner", "--no-acls", "--no-fflags", "--no-xattrs", "--no-mac-metadata"])
        .env_clear()
        .env("PATH", SYSTEM_PATH)
        .env("LANG", "en_US.UTF-8")
        .env("LC_ALL", "en_US.UTF-8");
    let mut last = Instant::now() - Duration::from_secs(1);
    let ran = run(&mut cmd, task, None, |n| {
        if last.elapsed() >= Duration::from_millis(150) {
            last = Instant::now();
            report(&Progress::at(Stage::Unpacking, n.min(total), total));
        }
    })
    .map_err(|e| Failure::new("failed", format!("tar didn't start: {e}")))?;
    let fail = |f: Failure| {
        let _ = fs::remove_dir_all(&staging);
        Err(f)
    };
    if ran.cancelled {
        return fail(Failure::cancelled());
    }
    if !ran.ok() {
        log(&format!("tar failed: {}", ran.said()));
        let full = ran.tail.iter().any(|l| l.contains("No space left"));
        return fail(if full {
            Failure::new("disk_full", "The disk filled up while unpacking. Make some room, then Retry.")
        } else {
            Failure::new("unpack", "The engine couldn't be unpacked (the archive is damaged). Retry downloads it again.")
        });
    }
    match check_tree(&staging) {
        Ok(n) => log(&format!("unpacked {n} entries (the manifest counts {total})")),
        Err(e) => {
            log(&format!("refused the archive: {e}"));
            return fail(Failure::new("unpack", "The engine's archive holds something NEO won't install. Retry downloads it again."));
        }
    }
    match engine_root(&staging) {
        Some(root) => Ok(root),
        None => fail(Failure::new("unpack", "The archive has no Python in it.")),
    }
}

// ---- 5–7. switch in, prepare, point at it ---------------------------------------------------------

fn compile(engine: &Path, dir: &str, opts: &Opts, task: &Task) -> Result<(), Failure> {
    let mut cmd = python(engine, opts);
    cmd.args(["-m", "compileall", "-q", "-j", "0"]).arg(engine.join(dir));
    let ran = run(&mut cmd, task, Some(COMPILE_TIMEOUT), |_| {}).map_err(|e| Failure::new("smoke", format!("The engine's Python didn't start: {e}")))?;
    if ran.cancelled {
        return Err(Failure::cancelled());
    }
    if !ran.ok() {
        // A package may carry a file that isn't Python 3 on purpose (a template, a test fixture):
        // it stays uncompiled, as it would anywhere.
        log(&format!("compileall {dir}: {}", ran.said()));
    }
    Ok(())
}

fn install_wheel(engine: &Path, m: &Manifest, wheel: &Path, opts: &Opts, task: &Task) -> Result<(), Failure> {
    let got = sha256_file(wheel, task, |_| {}).map_err(|e| Failure::new("failed", format!("NEO's own package isn't in the app ({e}). Download NEO again.")))?;
    match got {
        None => return Err(Failure::cancelled()),
        Some(sha) if sha != m.wheel_sha256 => {
            log(&format!("the bundled wheel's SHA-256 {sha} isn't the manifest's"));
            return Err(Failure::new("failed", "NEO's own package in the app is damaged. Download NEO again."));
        }
        Some(_) => {}
    }
    let mut cmd = python(engine, opts);
    cmd.args(["-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", "--no-compile"])
        .args(["--break-system-packages", "--disable-pip-version-check", "--no-warn-script-location", "--quiet"])
        .arg(wheel);
    let ran = run(&mut cmd, task, Some(PIP_TIMEOUT), |_| {}).map_err(|e| Failure::new("smoke", format!("The engine's Python didn't start: {e}")))?;
    if ran.cancelled {
        return Err(Failure::cancelled());
    }
    if !ran.ok() {
        log(&format!("pip install {}: {}", m.wheel, ran.said()));
        return Err(Failure::new("smoke", "NEO couldn't install itself into its engine. The setup log has the details."));
    }
    log(&format!("installed {}", m.wheel));
    let neo2 = format!("{}/site-packages/neo2", m.lib_dir());
    compile(engine, &neo2, opts, task)
}

/// The engine imports what the core needs and says its neo2 version. Also pays each library's
/// first load (and macOS's first look at each of its files) now, not inside the core's start.
fn smoke(engine: &Path, m: &Manifest, modules: &[String], opts: &Opts, task: &Task) -> Result<(), Failure> {
    let code = format!("import {}\nimport neo2\nprint('neo2', neo2.__version__)", modules.join(", "));
    let mut cmd = python(engine, opts);
    cmd.arg("-c").arg(&code);
    let ran = run(&mut cmd, task, Some(opts.smoke_timeout), |_| {}).map_err(|e| Failure::new("smoke", format!("The engine's Python didn't start: {e}")))?;
    if ran.cancelled {
        return Err(Failure::cancelled());
    }
    let said = ran.stdout.lines().rev().find(|l| l.starts_with("neo2 ")).map(|l| l.trim().to_string());
    if !ran.ok() || said.as_deref() != Some(&format!("neo2 {}", m.neo2_version)) {
        log(&format!("smoke test: {} (stdout {:?})", if ran.timed_out { "timed out".into() } else { ran.said() }, said));
        return Err(Failure::new("smoke", "NEO's engine didn't start on this Mac. The setup log has the details; Retry installs it again."));
    }
    log(&format!("smoke test passed: {}", modules.join(", ")));
    Ok(())
}

/// Where an install's archive comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    Download,
    /// "Use a downloaded file…": copied into NEO's folder and checked like a download; the user's
    /// own file is never moved or changed.
    File(PathBuf),
}

/// An exclusive hold on the installer (two NEO.apps, or a repair while an install runs).
struct Hold(#[allow(dead_code)] File);

fn hold(paths: &Paths) -> Result<Hold, Failure> {
    use std::os::fd::AsRawFd;
    fs::create_dir_all(&paths.root).map_err(|e| Failure::new("failed", format!("Couldn't make NEO's folder: {e}")))?;
    let f = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(paths.lock())
        .map_err(|e| Failure::new("failed", format!("Couldn't open {}: {e}", paths.lock().display())))?;
    // A program this app is starting holds a copy of every descriptor between its fork and its
    // exec, this one too: a lock that is busy for a moment isn't another install. One holds it for
    // minutes, so a second of trying tells them apart.
    for _ in 0..40 {
        if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return Ok(Hold(f));
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err(Failure::new("busy", "NEO's setup is already running."))
}

/// Install `m`'s engine, start to end (see the module's doc). `pause` runs just before the new
/// engine takes its folder (a core running from the old copy of the same engine stops there); the
/// caller starts the core again afterwards, whatever happened.
pub fn install(
    paths: &Paths,
    m: &Manifest,
    wheel: &Path,
    source: &Source,
    opts: &Opts,
    task: &Task,
    report: &dyn Fn(&Progress),
    pause: &dyn Fn(),
) -> Result<(), Failure> {
    let _hold = hold(paths)?;
    let from_file = matches!(source, Source::File(_));
    let id = m.engine.id.clone();
    log(&format!(
        "installing engine {id} for NEO {} ({})",
        m.app_version,
        match source {
            Source::Download => format!("from {}", m.engine.url),
            Source::File(p) => format!("from {}", p.display()),
        }
    ));
    report(&Progress::at(Stage::Preparing, 0, 0));
    check_room(paths, m, from_file)?;
    let archive = match source {
        Source::Download => download(paths, m, opts, task, report)?,
        Source::File(p) => {
            if file_len(p) != m.engine.bytes {
                return Err(Failure::new(
                    "checksum",
                    format!("That file isn't this NEO's engine (it isn't the right size). NEO {} needs {}.", m.app_version, m.asset_name()),
                ));
            }
            // A copy of it in NEO's own folder (a clone on the same disk): what the user chose is
            // read here, by NEO.app, and never touched — tar then reads only NEO's own copy (a
            // file in Downloads may be one macOS lets only the app that was given it read).
            report(&Progress::at(Stage::Preparing, 0, 0));
            let part = paths.part(&id);
            let copied = fs::create_dir_all(paths.downloads()).and_then(|_| {
                let _ = fs::remove_file(&part);
                fs::copy(p, &part)
            });
            if let Err(e) = copied {
                let _ = fs::remove_file(&part);
                log(&format!("couldn't copy {}: {e}", p.display()));
                let full = e.raw_os_error() == Some(libc::ENOSPC);
                return Err(if full {
                    Failure::new("disk_full", "The disk is full. Make some room, then try again.")
                } else {
                    Failure::new("failed", format!("NEO couldn't read that file: {e}"))
                });
            }
            part
        }
    };
    if let Err(e) = verify(&archive, m, task, report, from_file) {
        if e.code == "checksum" {
            let _ = fs::remove_file(&archive); // a damaged download, or a copy of the wrong file
        }
        return Err(e);
    }
    let staged = unpack(paths, m, &archive, task, report)?;
    if task.cancelled() {
        let _ = fs::remove_dir_all(paths.staging(&id));
        return Err(Failure::cancelled());
    }

    // Switch in: the old copy of this engine (a repair) waits aside until the new one works.
    report(&Progress::at(Stage::Optimizing, 0, 0));
    pause();
    let (dest, replaced) = (paths.engine(&id), paths.replaced(&id));
    let _ = fs::remove_dir_all(&replaced);
    let had_old = dest.exists();
    let moved = (|| -> std::io::Result<()> {
        fs::create_dir_all(paths.engines())?;
        if had_old {
            fs::rename(&dest, &replaced)?;
        }
        fs::rename(&staged, &dest)
    })();
    let _ = fs::remove_dir_all(paths.staging(&id)); // (empty, or what held the archive's one folder)
    if let Err(e) = moved {
        log(&format!("couldn't move the engine into place: {e}"));
        if had_old && !dest.exists() {
            let _ = fs::rename(&replaced, &dest);
        }
        return Err(Failure::new("failed", format!("The engine couldn't be moved into place: {e}")));
    }
    let undo = |f: Failure| -> Result<(), Failure> {
        log(&format!("undoing the switch: {}", f.message));
        let _ = fs::remove_dir_all(&dest);
        if had_old {
            let _ = fs::rename(&replaced, &dest);
        }
        Err(f)
    };
    if let Err(f) = compile(&dest, &m.lib_dir(), opts, task) {
        return undo(f);
    }
    report(&Progress::at(Stage::Installing, 0, 0));
    if let Err(f) = install_wheel(&dest, m, wheel, opts, task) {
        return undo(f);
    }
    report(&Progress::at(Stage::Testing, 0, 0));
    if let Err(f) = smoke(&dest, m, &m.smoke_modules(), opts, task) {
        return undo(f);
    }
    report(&Progress::at(Stage::Finishing, 0, 0));
    let pointed = fs::write(paths.ready(&id), format!("{}\n", m.app_version)).and_then(|_| write_pointer(paths, &release_pointer(&id, &m.wheel_sha256)));
    if let Err(e) = pointed {
        return undo(Failure::new("failed", format!("engine.json couldn't be written: {e}")));
    }
    log(&format!("engine.json points at {id}"));
    let _ = fs::remove_dir_all(&replaced);
    let _ = fs::remove_file(&archive); // (NEO's own: the download, or its copy of the chosen file)
    report(&Progress::at(Stage::Done, 1, 1));
    Ok(())
}

/// Case (e): the same engine with another neo2 — the bundled wheel, offline, then a quick check.
pub fn update_wheel(paths: &Paths, m: &Manifest, wheel: &Path, opts: &Opts) -> Result<(), Failure> {
    let _hold = hold(paths)?;
    let id = &m.engine.id;
    let engine = paths.engine(id);
    let task = Task::default();
    log(&format!("updating neo2 in engine {id} to {}", m.neo2_version));
    install_wheel(&engine, m, wheel, opts, &task)?;
    smoke(&engine, m, &["neo2".to_string()], opts, &task)?;
    write_pointer(paths, &release_pointer(id, &m.wheel_sha256)).map_err(|e| Failure::new("failed", format!("engine.json couldn't be written: {e}")))?;
    log(&format!("neo2 {} is installed in engine {id}", m.neo2_version));
    Ok(())
}

// ---- at launch --------------------------------------------------------------------------------------

/// What an earlier launch left: unpacking folders, a half-done switch (undone, unless the new engine
/// was ready), temporary files, downloads of engines this app doesn't need, and engines engine.json
/// no longer points at. `keep_part`: the engine this app installs (its part resumes). Nothing is
/// touched while another install holds the lock.
pub fn cleanup(paths: &Paths, keep_part: Option<&str>) {
    let Ok(_hold) = hold(paths) else { return };
    let Ok(entries) = fs::read_dir(&paths.root) else { return };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let path = e.path();
        if name.starts_with(".staging-") || name.starts_with("engine.json.tmp") || name == "ui.new" || name == "ui.old" {
            log(&format!("removing what an earlier setup left: {name}"));
            let _ = fs::remove_dir_all(&path).or_else(|_| fs::remove_file(&path));
        } else if let Some(id) = name.strip_prefix(".replaced-") {
            let dest = paths.engine(id);
            if valid_id(id) && !paths.ready(id).exists() {
                log(&format!("a repair of {id} stopped half-way: the old engine goes back"));
                let _ = fs::remove_dir_all(&dest);
                let _ = fs::rename(&path, &dest);
            } else {
                let _ = fs::remove_dir_all(&path);
            }
        }
    }
    if let Ok(entries) = fs::read_dir(paths.downloads()) {
        for e in entries.flatten() {
            let keep = keep_part.is_some_and(|id| e.file_name().to_string_lossy() == format!("{id}.part"));
            if !keep {
                let _ = fs::remove_dir_all(e.path()).or_else(|_| fs::remove_file(e.path()));
            }
        }
    }
    // Engines nothing points at (an update's old one, or an install that never finished). Only when
    // engine.json is clearly readable: a pointer that can't be read deletes nothing.
    let current = match read_pointer(paths) {
        Points::Invalid => return,
        Points::Release { id, .. } => Some(id),
        Points::Missing | Points::Dev => None,
    };
    sweep_engines(paths, current.as_deref());
}

/// Every engine but `keep` (folders named like an engine id only).
pub fn sweep_engines(paths: &Paths, keep: Option<&str>) {
    let Ok(entries) = fs::read_dir(paths.engines()) else { return };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if valid_id(&name) && Some(name.as_str()) != keep && e.path().is_dir() {
            log(&format!("removing engine {name}: nothing runs it any more"));
            let _ = fs::remove_dir_all(e.path());
        }
    }
}

/// NotchHub shows NEO's page from Application Support/NEO/ui: copy this app's (Resources/ui) there
/// whenever its build id differs, through ui.new and a swap. True when it copied.
pub fn sync_ui(bundled: &Path, paths: &Paths) -> Result<bool, String> {
    let read = |p: PathBuf| fs::read_to_string(p).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let Some(want) = read(bundled.join(".build-id")) else { return Ok(false) };
    if read(paths.ui().join(".build-id")).as_deref() == Some(want.as_str()) {
        return Ok(false);
    }
    let (new, old) = (paths.root.join("ui.new"), paths.root.join("ui.old"));
    let _ = fs::remove_dir_all(&new);
    let _ = fs::remove_dir_all(&old);
    fs::create_dir_all(&paths.root).map_err(|e| e.to_string())?;
    let st = Command::new("/usr/bin/ditto").arg(bundled).arg(&new).stdin(Stdio::null()).status().map_err(|e| e.to_string())?;
    if !st.success() {
        let _ = fs::remove_dir_all(&new);
        return Err(format!("ditto: {st}"));
    }
    if paths.ui().exists() {
        fs::rename(paths.ui(), &old).map_err(|e| e.to_string())?;
    }
    if let Err(e) = fs::rename(&new, paths.ui()) {
        let _ = fs::rename(&old, paths.ui());
        return Err(e.to_string());
    }
    let _ = fs::remove_dir_all(&old);
    log(&format!("NEO's page for NotchHub is now build {want}"));
    Ok(true)
}

/// Bytes and entries an archive will unpack into (for tests and scripts: `tar -t`).
#[cfg(test)]
fn tar_entries(archive: &Path) -> u64 {
    let out = Command::new("/usr/bin/tar").args(["-t", "-f"]).arg(archive).output().unwrap();
    String::from_utf8_lossy(&out.stdout).lines().count() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    /// A fresh folder for one test, under $TMPDIR (the test runs set it to the scratchpad).
    fn scratch(name: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!("neo-setup-{}-{}-{name}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn sha_of(p: &Path) -> String {
        sha256_file(p, &Task::default(), |_| {}).unwrap().unwrap()
    }

    /// A stand-in engine: `bin/python3` is a shell script that answers the installer's three
    /// questions (compileall, pip install, the smoke test) the way Python would, and records each.
    const STUB: &str = r#"#!/bin/sh
here="$(cd "$(dirname "$0")/.." && pwd)"
printf '%s' "$*" | tr '\n' ' ' >> "$here/calls.log"; echo >> "$here/calls.log"
[ "$1" = "-I" ] || { echo "not isolated" >&2; exit 9; }
case "$2" in
  -m)
    case "$3" in
      compileall) exit 0 ;;
      pip)
        for last in "$@"; do :; done
        mkdir -p "$here/lib/python3.12/site-packages/neo2"
        v="$(basename "$last" | sed -e 's/^neo2-//' -e 's/-py3-none-any.whl$//')"
        echo "$v" > "$here/lib/python3.12/site-packages/neo2/VERSION"
        exit 0 ;;
    esac ;;
  -c)
    [ -f "$here/fail-smoke" ] && { echo "ImportError: no module named torch" >&2; exit 1; }
    echo "neo2 $(cat "$here/lib/python3.12/site-packages/neo2/VERSION")"
    exit 0 ;;
esac
exit 3
"#;

    struct Fixture {
        dir: PathBuf,
        archive: PathBuf,
        wheel: PathBuf,
        manifest: Manifest,
    }

    /// An engine archive (made by the system's tar, as the build makes it: its files at the root)
    /// with `extra` files of `size` bytes, a wheel, and the manifest for them.
    fn fixture(name: &str, version: &str, extra: usize, size: usize) -> Fixture {
        let dir = scratch(name);
        let src = dir.join("src");
        fs::create_dir_all(src.join("bin")).unwrap();
        fs::create_dir_all(src.join("lib/python3.12/site-packages")).unwrap();
        fs::write(src.join("bin/python3"), STUB).unwrap();
        fs::set_permissions(src.join("bin/python3"), fs::Permissions::from_mode(0o755)).unwrap();
        symlink("python3", src.join("bin/python")).unwrap();
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15 ^ (name.len() as u64);
        for i in 0..extra {
            // xorshift: bytes xz can't shrink, so a download of them takes as long as it should
            let bytes: Vec<u8> = (0..size)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x >> 24) as u8
                })
                .collect();
            fs::write(src.join(format!("lib/python3.12/blob{i}.bin")), bytes).unwrap();
        }
        let archive = dir.join("neo-engine-0123456789abcdef-macos14-arm64.tar.xz");
        let st = Command::new("/usr/bin/tar").arg("-cJf").arg(&archive).arg("-C").arg(&src).arg(".").status().unwrap();
        assert!(st.success());
        let wheel = dir.join(format!("neo2-{version}-py3-none-any.whl"));
        fs::write(&wheel, format!("a wheel for {version}")).unwrap();
        let manifest = Manifest {
            schema: 1,
            app_version: version.into(),
            neo2_version: version.into(),
            wheel: format!("neo2-{version}-py3-none-any.whl"),
            wheel_sha256: sha_of(&wheel),
            engine: EngineAsset {
                id: "0123456789abcdef".into(),
                url: url_for(&archive),
                sha256: sha_of(&archive),
                bytes: file_len(&archive),
                unpacked_bytes: 10_000,
                files: tar_entries(&archive),
                python: "3.12.14".into(),
                min_macos: "14.0".into(),
            },
            models_bytes: 1,
            smoke: vec!["neo2".into()],
        };
        Fixture { dir, archive, wheel, manifest }
    }

    fn url_for(p: &Path) -> String {
        // curl takes a file URL's path as it is, percent-escapes included
        let mut s = String::from("file://");
        for b in p.to_string_lossy().bytes() {
            if b.is_ascii_alphanumeric() || b"/-._~".contains(&b) {
                s.push(b as char);
            } else {
                s.push_str(&format!("%{b:02X}"));
            }
        }
        s
    }

    fn opts(home: &Path) -> Opts {
        Opts { sandbox: true, home: home.to_path_buf(), curl_extra: Vec::new(), smoke_timeout: Duration::from_secs(30) }
    }

    fn quiet() -> impl Fn(&Progress) {
        |_| {}
    }

    #[test]
    fn a_manifest_says_exactly_what_to_install() {
        let f = fixture("manifest", "2.2.0", 0, 0);
        let text = serde_json::to_string(&f.manifest).unwrap();
        assert_eq!(Manifest::parse(&text, true).unwrap(), f.manifest);
        // A release build downloads from https only; a test build may use a file or this Mac.
        assert!(Manifest::parse(&text, false).unwrap_err().contains("URL"));
        assert!(url_allowed("https://github.com/ArnavParashar49/Neo2.0/releases/download/v2.2.0/x.tar.xz", false));
        for bad in ["http://github.com/x", "file:///tmp/x", "ftp://x", "https:///x", "https://a b", "https://x\n", "javascript:x"] {
            assert!(!url_allowed(bad, false), "{bad}");
        }
        assert!(url_allowed("http://127.0.0.1:8123/x.tar.xz", true) && !url_allowed("http://10.0.0.2:8123/x", true));
        let edit = |f2: &dyn Fn(&mut serde_json::Value)| {
            let mut v = serde_json::to_value(&f.manifest).unwrap();
            f2(&mut v);
            Manifest::parse(&v.to_string(), true)
        };
        assert!(edit(&|v| v["engine"]["id"] = "../../etc".into()).is_err());
        assert!(edit(&|v| v["engine"]["id"] = "0123456789ABCDEF".into()).is_err());
        assert!(edit(&|v| v["engine"]["sha256"] = "00".into()).is_err());
        assert!(edit(&|v| v["schema"] = 2.into()).is_err());
        assert!(edit(&|v| v["wheel"] = "../neo2-2.2.0-py3-none-any.whl".into()).is_err());
        assert!(edit(&|v| v["wheel"] = "evil-2.2.0-py3-none-any.whl".into()).is_err());
        assert!(edit(&|v| v["smoke"] = serde_json::json!(["neo2; import os"])).is_err());
        assert!(edit(&|v| v["engine"]["python"] = "3.12".into()).is_err());
        assert!(edit(&|v| v["engine"]["bytes"] = 0.into()).is_err());
        let m = edit(&|v| {
            v["smoke"] = serde_json::json!([]);
            v["models_bytes"] = 0.into();
        })
        .unwrap();
        assert_eq!(m.smoke_modules(), SMOKE.map(String::from).to_vec());
        assert_eq!(m.models_bytes(), MODELS_BYTES);
        assert_eq!(m.lib_dir(), "lib/python3.12");
        assert_eq!(f.manifest.asset_name(), "neo-engine-0123456789abcdef-macos14-arm64.tar.xz");
        let gh = edit(&|v| v["engine"]["url"] = "https://github.com/A/B/releases/download/v2.2.0/neo-engine-x.tar.xz".into()).unwrap();
        assert_eq!(gh.host(), "github.com");
        let _ = fs::remove_dir_all(f.dir);
    }

    #[test]
    fn engine_json_holds_no_paths_and_is_replaced_whole() {
        let dir = scratch("pointer");
        let paths = Paths::new(&dir);
        assert_eq!(read_pointer(&paths), Points::Missing);
        write_pointer(&paths, &release_pointer("0123456789abcdef", &"a".repeat(64))).unwrap();
        let text = fs::read_to_string(paths.pointer()).unwrap();
        assert_eq!(text, format!("{{\"schema\":1,\"source\":\"release\",\"engine_id\":\"0123456789abcdef\",\"neo2_sha256\":\"{}\"}}\n", "a".repeat(64)));
        assert_eq!(read_pointer(&paths), Points::Release { id: "0123456789abcdef".into(), wheel: "a".repeat(64) });
        assert!(fs::read_dir(&dir).unwrap().flatten().all(|e| !e.file_name().to_string_lossy().contains("tmp")));
        for (bad, why) in [
            (r#"{"schema":1,"source":"release","engine_id":"../../x","neo2_sha256":"aa"}"#, "a path for an id"),
            (r#"{"schema":1,"source":"release","engine_id":"0123456789abcdef"}"#, "no wheel"),
            (r#"{"schema":2,"source":"dev"}"#, "another schema"),
            (r#"{"schema":1,"source":"/usr/bin/python3"}"#, "a path for a source"),
            ("not json", "not json"),
        ] {
            fs::write(paths.pointer(), bad).unwrap();
            assert_eq!(read_pointer(&paths), Points::Invalid, "{why}");
        }
        fs::write(paths.pointer(), r#"{"schema":1,"source":"dev"}"#).unwrap();
        assert_eq!(read_pointer(&paths), Points::Dev);
        let _ = fs::remove_dir_all(dir);
    }

    /// The design's cases (a)–(f), and the build with no manifest.
    #[test]
    fn the_decision_table() {
        let f = fixture("decide", "2.2.0", 0, 0);
        let m = &f.manifest;
        let id = m.engine.id.clone();
        let root = f.dir.join("NEO");
        let paths = Paths::new(&root);
        fs::create_dir_all(&root).unwrap();
        let ready = |id: &str| {
            fs::create_dir_all(paths.engine(id).join("bin")).unwrap();
            fs::write(paths.python(id), "").unwrap();
            fs::write(paths.ready(id), "").unwrap();
        };
        let point = |id: &str, sha: &str| write_pointer(&paths, &release_pointer(id, sha)).unwrap();

        // (a) this Mac before 2.2: a .venv and no engine.json — the developer engine, as always.
        assert_eq!(decide(&paths, Some(m), true), Decision::Dev);
        // (c) nothing at all: Setup installs.
        assert_eq!(decide(&paths, Some(m), false), Decision::Install);
        assert!(Decision::Install.needs_setup());
        // …and a build with nothing to install says so.
        assert_eq!(decide(&paths, None, false), Decision::NoEngine);
        // (b) engine.json says dev.
        write_pointer(&paths, &Pointer { schema: 1, source: "dev".into(), engine_id: None, neo2_sha256: None }).unwrap();
        assert_eq!(decide(&paths, Some(m), true), Decision::Dev);
        assert_eq!(decide(&paths, Some(m), false), Decision::Install); // its .venv is gone
        // (d) this app's engine, ready, with this app's neo2.
        point(&id, &m.wheel_sha256);
        assert_eq!(decide(&paths, Some(m), false), Decision::Install); // not there yet (or half there)
        fs::create_dir_all(paths.engine(&id).join("bin")).unwrap();
        fs::write(paths.python(&id), "").unwrap();
        assert_eq!(decide(&paths, Some(m), true), Decision::Install); // no .neo-ready: never finished
        ready(&id);
        assert_eq!(decide(&paths, Some(m), true), Decision::Release { id: id.clone() });
        assert!(!Decision::Release { id: id.clone() }.needs_setup());
        assert_eq!(decide(&paths, None, false), Decision::Release { id: id.clone() }); // a build that can't check
        // (e) the same engine, another neo2.
        point(&id, &"b".repeat(64));
        assert_eq!(decide(&paths, Some(m), true), Decision::UpdateWheel { id: id.clone() });
        // (f) another engine: Setup downloads this app's alongside.
        point("fedcba9876543210", &m.wheel_sha256);
        ready("fedcba9876543210");
        assert_eq!(decide(&paths, Some(m), true), Decision::NewEngine { from: "fedcba9876543210".into() });
        assert!(Decision::NewEngine { from: String::new() }.needs_setup());
        // An engine.json that isn't one is never acted on: as if there were none.
        fs::write(paths.pointer(), "{").unwrap();
        assert_eq!(decide(&paths, Some(m), true), Decision::Dev);
        assert_eq!(decide(&paths, Some(m), false), Decision::Install);
        let _ = fs::remove_dir_all(f.dir);
    }

    #[test]
    fn a_disk_image_or_a_translocated_copy_is_not_where_neo_stays() {
        let p = |s: &str| placement(Path::new(s), false);
        assert_eq!(p("/Applications/NEO.app/Contents/MacOS/neo-ui"), Place::Ok);
        assert_eq!(p("/Users/a/Applications/NEO.app/Contents/MacOS/neo-ui"), Place::Ok);
        assert_eq!(p("/Users/a/Downloads/NEO.app/Contents/MacOS/neo-ui"), Place::Ok);
        assert_eq!(
            p("/private/var/folders/x7/abc/T/AppTranslocation/3F2A9C1E-0B7D-4C1A/d/NEO.app/Contents/MacOS/neo-ui"),
            Place::Translocated
        );
        assert_eq!(p("/Volumes/NEO 2.2.0/NEO.app/Contents/MacOS/neo-ui"), Place::Volume);
        assert_eq!(p("/Volumes/Work SSD/Applications/NEO.app/Contents/MacOS/neo-ui"), Place::Ok); // apps kept on another disk
        assert_eq!(placement(Path::new("/Volumes/X/Applications/NEO.app/Contents/MacOS/neo-ui"), true), Place::Volume); // read-only: a disk image
        let home = Path::new("/Users/a");
        assert!(login_allowed(Path::new("/Applications/NEO.app"), home));
        assert!(login_allowed(Path::new("/Applications/Utilities/NEO.app"), home));
        assert!(login_allowed(Path::new("/Users/a/Applications/NEO.app"), home));
        for no in [
            "/Users/a/Downloads/NEO.app",
            "/Volumes/NEO 2.2.0/NEO.app",
            "/private/var/folders/x7/T/AppTranslocation/U/d/NEO.app",
            "/Applications/../tmp/NEO.app",
            "/ApplicationsX/NEO.app",
            "/Users/b/Applications/NEO.app",
        ] {
            assert!(!login_allowed(Path::new(no), home), "{no}");
        }
    }

    #[test]
    fn what_curl_says_becomes_words() {
        for code in [6, 7, 28, 56] {
            let f = curl_failure(code, 0, "github.com", false);
            assert_eq!(f.code, "offline", "{code}");
            assert!(f.message.contains("Can't reach github.com"), "{}", f.message);
        }
        assert_eq!(curl_failure(22, 404, "github.com", false).code, "not_found");
        assert_eq!(curl_failure(22, 503, "github.com", false).code, "failed");
        assert_eq!(curl_failure(23, 0, "github.com", true).code, "disk_full");
        assert_eq!(curl_failure(37, 0, "this Mac", false).code, "not_found");
        assert!(curl_failure(60, 0, "github.com", false).message.contains("certificate"));
    }

    /// A whole install from a file:// URL into a folder whose path has spaces and non-ASCII letters,
    /// resuming a part that an earlier (stopped) download left.
    #[test]
    fn an_install_resumes_a_download_and_switches_in_atomically() {
        let f = fixture("install", "2.2.0", 3, 300_000);
        let root = f.dir.join("home dir/Library/Application Support/NÉO ünï 名前");
        let paths = Paths::new(&root);
        // An earlier download stopped a third of the way.
        fs::create_dir_all(paths.downloads()).unwrap();
        let whole = fs::read(&f.archive).unwrap();
        fs::write(paths.part(&f.manifest.engine.id), &whole[..whole.len() / 3]).unwrap();
        let seen = Mutex::new(Vec::<Stage>::new());
        let report = |p: &Progress| {
            let mut s = seen.lock().unwrap();
            if s.last() != Some(&p.stage) {
                s.push(p.stage);
            }
        };
        let paused = AtomicBool::new(false);
        let res = install(&paths, &f.manifest, &f.wheel, &Source::Download, &opts(&f.dir), &Task::default(), &report, &|| paused.store(true, Ordering::SeqCst));
        assert_eq!(res, Ok(()));
        assert!(paused.load(Ordering::SeqCst));
        let id = &f.manifest.engine.id;
        assert_eq!(decide(&paths, Some(&f.manifest), false), Decision::Release { id: id.clone() });
        let calls = fs::read_to_string(paths.engine(id).join("calls.log")).unwrap();
        assert!(calls.contains("-m compileall -q -j 0"), "{calls}");
        assert!(calls.contains("-m pip install --no-index --no-deps --force-reinstall --no-compile"), "{calls}");
        assert!(calls.lines().last().unwrap().starts_with("-I -c import neo2"), "{calls}");
        assert!(calls.lines().all(|l| l.starts_with("-I ")), "every run is isolated: {calls}");
        // The archive and everything temporary are gone; the engine's files are as packed.
        assert!(!paths.part(id).exists());
        assert!(fs::read_dir(&root).unwrap().flatten().all(|e| !e.file_name().to_string_lossy().starts_with(".staging")));
        assert_eq!(fs::read_link(paths.engine(id).join("bin/python")).unwrap(), PathBuf::from("python3"));
        assert_eq!(fs::read(paths.engine(id).join("lib/python3.12/blob2.bin")).unwrap().len(), 300_000);
        let stages = seen.lock().unwrap().clone();
        for s in [Stage::Preparing, Stage::Downloading, Stage::Verifying, Stage::Unpacking, Stage::Optimizing, Stage::Installing, Stage::Testing, Stage::Done] {
            assert!(stages.contains(&s), "{s:?} in {stages:?}");
        }
        // Nothing the archive carried, or the download, put a quarantine flag on the interpreter.
        let x = Command::new("/usr/bin/xattr").arg(paths.python(id)).output().unwrap();
        assert!(!String::from_utf8_lossy(&x.stdout).contains("quarantine"));
        let _ = fs::remove_dir_all(f.dir);
    }

    #[test]
    fn a_damaged_download_is_thrown_away_and_retry_starts_over() {
        let f = fixture("damaged", "2.2.0", 2, 200_000);
        let mut bytes = fs::read(&f.archive).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x40; // one byte flipped: the same size, another hash
        let bad = f.dir.join("damaged.tar.xz");
        fs::write(&bad, &bytes).unwrap();
        let mut m = f.manifest.clone();
        m.engine.url = url_for(&bad);
        let paths = Paths::new(f.dir.join("NEO"));
        let res = install(&paths, &m, &f.wheel, &Source::Download, &opts(&f.dir), &Task::default(), &quiet(), &|| {});
        let err = res.unwrap_err();
        assert_eq!(err.code, "checksum");
        assert!(err.message.contains("damaged"), "{}", err.message);
        assert!(!paths.part(&m.engine.id).exists(), "a damaged part is never resumed");
        assert_eq!(decide(&paths, Some(&m), false), Decision::Install);
        // Retry, with the good file back at the URL.
        m.engine.url = url_for(&f.archive);
        assert_eq!(install(&paths, &m, &f.wheel, &Source::Download, &opts(&f.dir), &Task::default(), &quiet(), &|| {}), Ok(()));
        let _ = fs::remove_dir_all(f.dir);
    }

    #[test]
    fn a_downloaded_file_is_checked_and_left_where_it_is() {
        let f = fixture("usefile", "2.2.0", 1, 1000);
        let paths = Paths::new(f.dir.join("NEO"));
        let other = f.dir.join("other.tar.xz");
        fs::write(&other, vec![7u8; f.manifest.engine.bytes as usize]).unwrap();
        let err = install(&paths, &f.manifest, &f.wheel, &Source::File(other.clone()), &opts(&f.dir), &Task::default(), &quiet(), &|| {}).unwrap_err();
        assert_eq!(err.code, "checksum");
        assert!(err.message.contains("isn't this NEO's engine") && err.message.contains(&f.manifest.asset_name()), "{}", err.message);
        assert!(other.exists(), "the user's own file is never deleted");
        assert!(!paths.part(&f.manifest.engine.id).exists(), "the copy of a wrong file goes");
        let res = install(&paths, &f.manifest, &f.wheel, &Source::File(f.archive.clone()), &opts(&f.dir), &Task::default(), &quiet(), &|| {});
        assert_eq!(res, Ok(()));
        assert!(f.archive.exists());
        assert_eq!(sha_of(&f.archive), f.manifest.engine.sha256, "…and the chosen file is as it was");
        assert!(!paths.part(&f.manifest.engine.id).exists());
        let _ = fs::remove_dir_all(f.dir);
    }

    #[test]
    fn cancel_stops_curl_and_keeps_the_part_for_retry() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::OpenOptionsExt;
        let f = fixture("cancel", "2.2.0", 4, 400_000);
        let paths = Paths::new(f.dir.join("NEO"));
        // A slow server: the archive's bytes through a named pipe, 32 KB every 50 ms.
        let fifo = f.dir.join("slow.fifo");
        let c = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
        let bytes = fs::read(&f.archive).unwrap();
        let fifo2 = fifo.clone();
        let writer = thread::spawn(move || {
            let Ok(mut w) = OpenOptions::new().write(true).open(&fifo2) else { return 0 };
            let mut sent = 0;
            for chunk in bytes.chunks(32 * 1024) {
                if w.write_all(chunk).is_err() {
                    break; // curl went away
                }
                sent += chunk.len();
                thread::sleep(Duration::from_millis(50));
            }
            sent
        });
        let mut slow = f.manifest.clone();
        slow.engine.url = url_for(&fifo);
        let task = Task::default();
        let t2 = task.clone();
        let canceller = thread::spawn(move || {
            thread::sleep(Duration::from_millis(700));
            t2.cancel();
        });
        let started = Instant::now();
        let err = install(&paths, &slow, &f.wheel, &Source::Download, &opts(&f.dir), &task, &quiet(), &|| {}).unwrap_err();
        canceller.join().unwrap();
        assert!(err.is_cancel(), "{err:?}");
        assert!(started.elapsed() < Duration::from_secs(5), "curl was stopped, not waited for");
        // (a writer still waiting for a reader is let go)
        let _ = OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK).open(&fifo);
        let sent = writer.join().unwrap();
        let part = file_len(&paths.part(&f.manifest.engine.id));
        assert!(part > 0 && part < f.manifest.engine.bytes && part as usize <= sent, "{part} of {sent}");
        // No curl left behind.
        let ps = Command::new("/bin/ps").args(["-ax", "-o", "command="]).output().unwrap();
        assert!(!String::from_utf8_lossy(&ps.stdout).contains(&slow.engine.url));
        // Retry resumes where it stopped, and finishes.
        let seen = Mutex::new(None::<u64>);
        let report = |p: &Progress| {
            let mut s = seen.lock().unwrap();
            if p.stage == Stage::Downloading && s.is_none() {
                *s = Some(p.done);
            }
        };
        assert_eq!(install(&paths, &f.manifest, &f.wheel, &Source::Download, &opts(&f.dir), &Task::default(), &report, &|| {}), Ok(()));
        assert!(seen.lock().unwrap().is_some_and(|first| first >= part), "it began from the part, not from zero");
        let _ = fs::remove_dir_all(f.dir);
    }

    /// A server on this Mac that answers Range requests but drops every connection after `cut`
    /// bytes — a network that keeps breaking. Returns its port and how many requests it has had.
    fn serve_cutting(bytes: Vec<u8>, cut: usize) -> (u16, Arc<AtomicU64>) {
        use std::io::Write as _;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let hits = Arc::new(AtomicU64::new(0));
        let h2 = hits.clone();
        thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut s) = conn else { continue };
                h2.fetch_add(1, Ordering::SeqCst);
                let mut head = Vec::new();
                let mut b = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") && s.read(&mut b).unwrap_or(0) == 1 {
                    head.push(b[0]);
                }
                let head = String::from_utf8_lossy(&head).to_lowercase();
                let start = head
                    .lines()
                    .find_map(|l| l.strip_prefix("range: bytes="))
                    .and_then(|r| r.split('-').next()?.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if start >= bytes.len() {
                    let _ = write!(s, "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{}\r\nContent-Length: 0\r\n\r\n", bytes.len());
                    continue;
                }
                let body = &bytes[start..];
                let status = if start > 0 {
                    format!("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{}/{}\r\n", bytes.len() - 1, bytes.len())
                } else {
                    "HTTP/1.1 200 OK\r\n".to_string()
                };
                let _ = write!(s, "{status}Content-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n", body.len());
                let _ = s.write_all(&body[..body.len().min(cut)]);
            }
        });
        (port, hits)
    }

    #[test]
    fn a_connection_that_keeps_breaking_still_finishes_by_resuming() {
        let f = fixture("flaky", "2.2.0", 4, 400_000);
        let (port, hits) = serve_cutting(fs::read(&f.archive).unwrap(), 300_000);
        let mut m = f.manifest.clone();
        m.engine.url = format!("http://127.0.0.1:{port}/{}", m.asset_name());
        let paths = Paths::new(f.dir.join("NEO"));
        assert_eq!(install(&paths, &m, &f.wheel, &Source::Download, &opts(&f.dir), &Task::default(), &quiet(), &|| {}), Ok(()));
        let n = hits.load(Ordering::SeqCst);
        assert!(n as u64 >= m.engine.bytes / 300_000, "{n} requests: each resumed where the last broke off");
        let _ = fs::remove_dir_all(f.dir);
    }

    #[test]
    fn a_server_that_cant_be_reached_is_offline() {
        // A port nothing listens on (bound, then let go).
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let f = fixture("refused", "2.2.0", 0, 0);
        let mut m = f.manifest.clone();
        m.engine.url = format!("http://127.0.0.1:{port}/{}", m.asset_name());
        let paths = Paths::new(f.dir.join("NEO"));
        let t0 = Instant::now();
        let err = install(&paths, &m, &f.wheel, &Source::Download, &opts(&f.dir), &Task::default(), &quiet(), &|| {}).unwrap_err();
        assert_eq!(err.code, "offline", "{err:?}");
        assert!(err.message.starts_with("Can't reach 127.0.0.1"), "{}", err.message);
        assert!(t0.elapsed() < Duration::from_secs(20));
        let _ = fs::remove_dir_all(f.dir);
    }

    /// The app killed mid-unpack (or mid-switch): the next launch removes the staging folder, puts a
    /// repaired engine's old copy back, keeps this app's part, and drops what nothing needs.
    #[test]
    fn what_an_interrupted_setup_left_is_cleaned_at_the_next_launch() {
        let dir = scratch("leftovers");
        let paths = Paths::new(&dir);
        let id = "0123456789abcdef";
        fs::create_dir_all(dir.join(format!(".staging-{id}/lib/python3.12"))).unwrap();
        fs::write(dir.join(format!(".staging-{id}/lib/python3.12/half.py")), "x").unwrap();
        fs::create_dir_all(paths.downloads()).unwrap();
        fs::write(paths.part(id), "resume me").unwrap();
        fs::write(paths.part("fedcba9876543210"), "an old app's").unwrap();
        fs::write(dir.join("engine.json.tmp-123"), "{").unwrap();
        fs::create_dir_all(dir.join("ui.new")).unwrap();
        // A repair stopped half-way: the new copy isn't ready, the old one waits aside.
        fs::create_dir_all(paths.engine(id).join("bin")).unwrap();
        fs::create_dir_all(dir.join(format!(".replaced-{id}/bin"))).unwrap();
        fs::write(dir.join(format!(".replaced-{id}/bin/python3")), "old").unwrap();
        fs::write(dir.join(format!(".replaced-{id}/.neo-ready")), "").unwrap();
        // An engine nothing points at, and something that isn't an engine.
        fs::create_dir_all(paths.engine("aaaaaaaaaaaaaaaa")).unwrap();
        fs::create_dir_all(paths.engines().join("notes")).unwrap();
        write_pointer(&paths, &release_pointer(id, &"c".repeat(64))).unwrap();
        cleanup(&paths, Some(id));
        assert!(!dir.join(format!(".staging-{id}")).exists());
        assert!(!dir.join("engine.json.tmp-123").exists() && !dir.join("ui.new").exists());
        assert_eq!(fs::read_to_string(paths.part(id)).unwrap(), "resume me");
        assert!(!paths.part("fedcba9876543210").exists());
        assert_eq!(fs::read_to_string(paths.python(id)).unwrap(), "old");
        assert!(!dir.join(format!(".replaced-{id}")).exists());
        assert!(!paths.engine("aaaaaaaaaaaaaaaa").exists());
        assert!(paths.engines().join("notes").exists(), "only engine folders are swept");
        // An engine.json that can't be read deletes no engine.
        fs::create_dir_all(paths.engine("bbbbbbbbbbbbbbbb")).unwrap();
        fs::write(paths.pointer(), "garbage").unwrap();
        cleanup(&paths, None);
        assert!(paths.engine("bbbbbbbbbbbbbbbb").exists() && paths.engine(id).exists());
        let _ = fs::remove_dir_all(dir);
    }

    /// Killed while tar runs: the staging folder is all there is, and it goes.
    #[test]
    fn a_cancelled_unpack_leaves_nothing() {
        let f = fixture("unpack-cancel", "2.2.0", 8, 200_000);
        let paths = Paths::new(f.dir.join("NEO"));
        let task = Task::default();
        let t2 = task.clone();
        // Stopped as soon as tar has started (the app quit, or was killed, mid-unpack).
        let report = move |p: &Progress| {
            if p.stage == Stage::Unpacking {
                t2.cancel();
            }
        };
        let err = install(&paths, &f.manifest, &f.wheel, &Source::File(f.archive.clone()), &opts(&f.dir), &task, &report, &|| {}).unwrap_err();
        assert!(err.is_cancel(), "{err:?}");
        assert!(!paths.root.join(format!(".staging-{}", f.manifest.engine.id)).exists());
        assert!(!paths.engine(&f.manifest.engine.id).exists());
        assert_eq!(read_pointer(&paths), Points::Missing);
        let _ = fs::remove_dir_all(f.dir);
    }

    #[test]
    fn an_archive_that_reaches_outside_is_refused() {
        let dir = scratch("evil");
        let tree = dir.join("tree");
        fs::create_dir_all(tree.join("bin")).unwrap();
        fs::write(tree.join("bin/python3"), STUB).unwrap();
        // a link out of the engine
        symlink("../../../../../etc/hosts", tree.join("bin/hosts")).unwrap();
        assert!(check_tree(&tree).unwrap_err().contains("links outside"));
        fs::remove_file(tree.join("bin/hosts")).unwrap();
        symlink("/etc/hosts", tree.join("bin/abs")).unwrap();
        assert!(check_tree(&tree).is_err());
        fs::remove_file(tree.join("bin/abs")).unwrap();
        symlink("../bin/python3", tree.join("bin/ok")).unwrap();
        assert!(check_tree(&tree).is_ok());
        fs::set_permissions(tree.join("bin/python3"), fs::Permissions::from_mode(0o4755)).unwrap();
        assert!(check_tree(&tree).unwrap_err().contains("setuid"));
        // And tar itself refuses names with "..": made with bsdtar's -s rewrite, as an attacker could.
        fs::set_permissions(tree.join("bin/python3"), fs::Permissions::from_mode(0o755)).unwrap();
        let archive = dir.join("evil.tar.xz");
        let st = Command::new("/usr/bin/tar")
            .arg("-cJf")
            .arg(&archive)
            .args(["-s", ",^bin/python3$,../../escaped,", "-C"])
            .arg(&tree)
            .arg("bin/python3")
            .status()
            .unwrap();
        assert!(st.success());
        let m = Manifest {
            schema: 1,
            app_version: "2.2.0".into(),
            neo2_version: "2.2.0".into(),
            wheel: "neo2-2.2.0-py3-none-any.whl".into(),
            wheel_sha256: "0".repeat(64),
            engine: EngineAsset {
                id: "0123456789abcdef".into(),
                url: url_for(&archive),
                sha256: sha_of(&archive),
                bytes: file_len(&archive),
                unpacked_bytes: 1,
                files: 1,
                python: "3.12.14".into(),
                min_macos: String::new(),
            },
            models_bytes: 1,
            smoke: vec![],
        };
        let root = dir.join("a/b/NEO");
        let paths = Paths::new(&root);
        let err = install(&paths, &m, &dir.join("none.whl"), &Source::File(archive), &opts(&dir), &Task::default(), &quiet(), &|| {}).unwrap_err();
        assert_eq!(err.code, "unpack");
        assert!(!dir.join("a/escaped").exists() && !root.join("escaped").exists() && !dir.join("escaped").exists());
        assert!(!root.join(".staging-0123456789abcdef").exists());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_smoke_test_that_fails_undoes_the_switch_and_keeps_the_old_engine() {
        let f = fixture("smoke", "2.2.0", 1, 100);
        let paths = Paths::new(f.dir.join("NEO"));
        let id = f.manifest.engine.id.clone();
        assert_eq!(install(&paths, &f.manifest, &f.wheel, &Source::File(f.archive.clone()), &opts(&f.dir), &Task::default(), &quiet(), &|| {}), Ok(()));
        let pointer = fs::read_to_string(paths.pointer()).unwrap();
        // A repair whose new copy won't start: the old one, and engine.json, stay as they were.
        let mut m = f.manifest.clone();
        m.neo2_version = "9.9.9".into(); // the stub says 2.2.0
        let err = install(&paths, &m, &f.wheel, &Source::File(f.archive.clone()), &opts(&f.dir), &Task::default(), &quiet(), &|| {}).unwrap_err();
        assert_eq!(err.code, "smoke");
        assert!(paths.python(&id).exists() && paths.ready(&id).exists());
        assert_eq!(fs::read_to_string(paths.pointer()).unwrap(), pointer);
        assert!(!paths.replaced(&id).exists());
        let _ = fs::remove_dir_all(f.dir);
    }

    /// Case (e): an update that kept the engine — the wheel only, no download.
    #[test]
    fn a_neo2_only_update_reinstalls_the_wheel_offline() {
        let f = fixture("wheel", "2.2.0", 1, 100);
        let paths = Paths::new(f.dir.join("NEO"));
        assert_eq!(install(&paths, &f.manifest, &f.wheel, &Source::File(f.archive.clone()), &opts(&f.dir), &Task::default(), &quiet(), &|| {}), Ok(()));
        let id = f.manifest.engine.id.clone();
        // NEO 2.2.1: the same engine id, a new wheel.
        let wheel = f.dir.join("neo2-2.2.1-py3-none-any.whl");
        fs::write(&wheel, "a wheel for 2.2.1").unwrap();
        let mut m = f.manifest.clone();
        m.app_version = "2.2.1".into();
        m.neo2_version = "2.2.1".into();
        m.wheel = "neo2-2.2.1-py3-none-any.whl".into();
        m.wheel_sha256 = sha_of(&wheel);
        m.engine.url = "file:///nonexistent/never-downloaded.tar.xz".into();
        assert_eq!(decide(&paths, Some(&m), false), Decision::UpdateWheel { id: id.clone() });
        let t0 = Instant::now();
        assert_eq!(update_wheel(&paths, &m, &wheel, &opts(&f.dir)), Ok(()));
        assert!(t0.elapsed() < Duration::from_secs(10));
        assert_eq!(decide(&paths, Some(&m), false), Decision::Release { id: id.clone() });
        let calls = fs::read_to_string(paths.engine(&id).join("calls.log")).unwrap();
        assert!(calls.contains("neo2-2.2.1-py3-none-any.whl"), "{calls}");
        assert!(!paths.downloads().join(format!("{id}.part")).exists());
        // A wheel that isn't the manifest's is refused.
        fs::write(&wheel, "tampered").unwrap();
        assert_eq!(update_wheel(&paths, &m, &wheel, &opts(&f.dir)).unwrap_err().code, "failed");
        let _ = fs::remove_dir_all(f.dir);
    }

    /// A real engine (Python itself, not the stand-in): compileall, pip and the smoke test as they
    /// run on a user's Mac, into a folder whose path has a space. Run by hand:
    /// `NEO_TEST_ENGINE=<neo-engine-….tar.xz> NEO_TEST_WHEEL=<neo2 wheel> cargo test -- --ignored a_real_engine`
    /// (scripts/sandbox_build.sh --fake-engine makes a small one; NEO_TEST_SMOKE=neo2,torch,… for a full one).
    #[test]
    #[ignore]
    fn a_real_engine_installs_and_runs_neo2_where_it_lands() {
        let (Ok(engine), Ok(wheel)) = (std::env::var("NEO_TEST_ENGINE"), std::env::var("NEO_TEST_WHEEL")) else {
            eprintln!("skipped: NEO_TEST_ENGINE and NEO_TEST_WHEEL name an engine archive and a neo2 wheel");
            return;
        };
        let (engine, wheel) = (PathBuf::from(engine), PathBuf::from(wheel));
        let name = engine.file_name().unwrap().to_string_lossy().into_owned();
        let id = name.trim_start_matches("neo-engine-").chars().take(16).collect::<String>();
        let version = wheel.file_name().unwrap().to_string_lossy().trim_start_matches("neo2-").trim_end_matches("-py3-none-any.whl").to_string();
        let smoke = std::env::var("NEO_TEST_SMOKE").unwrap_or_else(|_| "neo2".into());
        let m = Manifest {
            schema: 1,
            app_version: version.clone(),
            neo2_version: version.clone(),
            wheel: wheel.file_name().unwrap().to_string_lossy().into_owned(),
            wheel_sha256: sha_of(&wheel),
            engine: EngineAsset {
                id: id.clone(),
                url: url_for(&engine),
                sha256: sha_of(&engine),
                bytes: file_len(&engine),
                unpacked_bytes: 1,
                files: tar_entries(&engine),
                python: "3.12.14".into(),
                min_macos: "14.0".into(),
            },
            models_bytes: 1,
            smoke: smoke.split(',').map(String::from).collect(),
        };
        let dir = scratch("real-engine");
        let paths = Paths::new(dir.join("home/Library/Application Support/NEO"));
        let mut o = opts(&dir.join("home"));
        o.smoke_timeout = Duration::from_secs(600);
        let t0 = Instant::now();
        assert_eq!(install(&paths, &m, &wheel, &Source::Download, &o, &Task::default(), &quiet(), &|| {}), Ok(()));
        eprintln!("installed in {:.1} s", t0.elapsed().as_secs_f64());
        let out = Command::new(paths.python(&id)).args(["-I", "-c", "import neo2, sys; print(neo2.__file__); print(sys.prefix)"]).output().unwrap();
        let said = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(out.status.success(), "{said} {}", String::from_utf8_lossy(&out.stderr));
        assert!(said.lines().all(|l| l.starts_with(&paths.engine(&id).to_string_lossy().into_owned())), "{said}");
        // Nothing points at where it was unpacked.
        let grep = Command::new("/usr/bin/grep").args(["-rIl", ".staging-"]).arg(paths.engine(&id)).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&grep.stdout), "", "files naming the staging folder");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_second_install_waits_for_the_first() {
        let dir = scratch("busy");
        let paths = Paths::new(&dir);
        let held = hold(&paths).unwrap();
        assert_eq!(hold(&paths).err().map(|f| f.code), Some("busy"));
        drop(held);
        assert!(hold(&paths).is_ok());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn notchhubs_page_is_copied_when_the_build_changes() {
        let dir = scratch("ui");
        let bundled = dir.join("Resources/ui");
        fs::create_dir_all(bundled.join("assets")).unwrap();
        fs::write(bundled.join("index.html"), "v1").unwrap();
        fs::write(bundled.join("assets/a.js"), "js1").unwrap();
        fs::write(bundled.join(".build-id"), "build-1\n").unwrap();
        let paths = Paths::new(dir.join("NEO"));
        assert_eq!(sync_ui(&bundled, &paths), Ok(true));
        assert_eq!(fs::read_to_string(paths.ui().join("assets/a.js")).unwrap(), "js1");
        assert_eq!(sync_ui(&bundled, &paths), Ok(false)); // the same build: nothing to do
        fs::write(bundled.join("index.html"), "v2").unwrap();
        fs::remove_file(bundled.join("assets/a.js")).unwrap();
        fs::write(bundled.join(".build-id"), "build-2").unwrap();
        fs::write(paths.ui().join("stale.js"), "x").unwrap();
        assert_eq!(sync_ui(&bundled, &paths), Ok(true));
        assert_eq!(fs::read_to_string(paths.ui().join("index.html")).unwrap(), "v2");
        assert!(!paths.ui().join("stale.js").exists() && !paths.ui().join("assets/a.js").exists());
        assert!(!dir.join("NEO/ui.old").exists() && !dir.join("NEO/ui.new").exists());
        // A bundle with no page (a developer build) leaves install-app.sh's copy alone.
        assert_eq!(sync_ui(&dir.join("nothing"), &paths), Ok(false));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn room_is_counted_before_anything_downloads() {
        let f = fixture("room", "2.2.0", 0, 0);
        let mut m = f.manifest.clone();
        m.engine.unpacked_bytes = 1_475_000_000;
        m.models_bytes = 1_700_000_000;
        m.engine.bytes = 225_000_000;
        assert_eq!(bytes_needed(&m, 0, false), 1_475_000_000 + 1_700_000_000 + HEADROOM + 225_000_000);
        assert_eq!(bytes_needed(&m, 100_000_000, false), 1_475_000_000 + 1_700_000_000 + HEADROOM + 125_000_000);
        assert_eq!(bytes_needed(&m, 0, true), 1_475_000_000 + 1_700_000_000 + HEADROOM);
        assert_eq!(gb(225_000_000), "225 MB");
        assert_eq!(gb(4_375_000_000), "4.4 GB");
        assert!(free_bytes(&f.dir.join("not/yet/made")).is_some());
        let _ = fs::remove_dir_all(f.dir);
    }

    /// A small disk (an APFS image this user attaches, no sudo): too little room is said before
    /// anything downloads, and a disk that fills up during the download says so too.
    #[test]
    fn a_full_disk_is_said_in_words() {
        if !Path::new("/usr/bin/hdiutil").exists() {
            eprintln!("skipped: no hdiutil");
            return;
        }
        let dir = scratch("small-disk");
        let image = dir.join("small.sparseimage");
        let mount = dir.join("mnt");
        fs::create_dir_all(&mount).unwrap();
        let made = Command::new("/usr/bin/hdiutil")
            .args(["create", "-quiet", "-size", "700m", "-fs", "APFS", "-type", "SPARSE", "-volname", "neotest"])
            .arg(&image)
            .status()
            .unwrap();
        assert!(made.success());
        let attached = Command::new("/usr/bin/hdiutil")
            .args(["attach", "-quiet", "-nobrowse", "-noautoopen", "-mountpoint"])
            .arg(&mount)
            .arg(&image)
            .status()
            .unwrap();
        assert!(attached.success());
        struct Detach(PathBuf);
        impl Drop for Detach {
            fn drop(&mut self) {
                let _ = Command::new("/usr/bin/hdiutil").args(["detach", "-quiet", "-force"]).arg(&self.0).status();
            }
        }
        let _detach = Detach(mount.clone());
        let free = free_bytes(&mount).unwrap();
        assert!(free < 700 << 20, "{free}");

        let f = fixture("small-disk-src", "2.2.0", 0, 0);
        let paths = Paths::new(mount.join("Library/Application Support/NEO"));
        // Before anything: the engine wouldn't fit.
        let mut m = f.manifest.clone();
        m.engine.unpacked_bytes = 2_000_000_000;
        let err = install(&paths, &m, &f.wheel, &Source::Download, &opts(&f.dir), &Task::default(), &quiet(), &|| {}).unwrap_err();
        assert_eq!(err.code, "disk_full");
        assert!(err.message.contains("free on this Mac") && err.message.contains("Make some room"), "{}", err.message);
        assert!(!paths.part(&m.engine.id).exists());
        // During the download: a manifest that undercounts (the room check passes), and a file
        // bigger than the disk.
        let big = f.dir.join("big.tar.xz");
        let st = Command::new("/usr/sbin/mkfile").args(["-n", "800m"]).arg(&big).status().unwrap();
        assert!(st.success());
        let mut m = f.manifest.clone();
        m.engine.url = url_for(&big);
        m.engine.bytes = 800 << 20;
        m.engine.unpacked_bytes = 1;
        m.models_bytes = 1;
        let mut o = opts(&f.dir);
        o.curl_extra = vec![];
        // (the room check counts HEADROOM: give this small disk a manifest that fits on paper)
        let room = free_bytes(&mount).unwrap();
        assert!(bytes_needed(&m, 0, false) > room);
        let err = (|| -> Result<(), Failure> {
            let _hold = hold(&paths)?;
            download(&paths, &m, &o, &Task::default(), &quiet()).map(|_| ())
        })()
        .unwrap_err();
        assert_eq!(err.code, "disk_full", "{err:?}");
        let _ = fs::remove_dir_all(f.dir);
        drop(_detach);
        let _ = fs::remove_dir_all(dir);
    }
}

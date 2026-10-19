//! Runs NEO's Python core as a child of the app and keeps it running.
//!
//! NEO.app is the process macOS asks for permissions (Accessibility, Microphone, Screen
//! Recording): the core it starts inherits them. The core also watches its parent and exits if
//! the app goes away without stopping it (force quit, crash), so a core with the mic open can't
//! be left behind; if one ever is, the next launch recognises it by its pidfile and replaces it.
//! A core started some other way (a developer's `python -m neo2`) is used as-is.
//!
//! The app and the core it started also talk directly, over an anonymous socket pair made here
//! before the core starts (the core inherits its end: `NEO_APP_FD`; neo2/core/server/shell_link.py
//! is the other side, and says what goes over it). The menu's actions for a host, the summon keys,
//! what the menu shows, whether a host is there and where the UI lives all go this way — none of
//! it through the window's page: macOS suspends the process of a page whose window is hidden
//! (runningboardd: "Suspending task", seconds after), and in the notch that window always is.
//! So do Setup's microphone and camera rows: the core uses them, so the core asks macOS
//! (`Core::permission`), and says what macOS answered (`FromCore::Permission`).
//!
//! Which Python runs the core is setup.rs's `decide`: the release engine NEO's Setup installed
//! (`engines/<id>/bin/python3 -I -u -m neo2`), or scripts/install-app.sh's developer engine
//! (`.venv/bin/python -u -m neo2`, exactly as before 2.2). With neither, the supervisor says "not set
//! up — open NEO Setup" and waits for Setup to say the engine is in (`Core::restart`) instead of
//! looking again every ten minutes.
//!
//! A `sandbox` build (NEO Sandbox, com.arnav.neo.sandbox: scripts/sandbox_build.sh) runs beside the
//! real NEO without touching it: its home is the `home` folder next to the app (its own ~/.neo,
//! Application Support, LaunchAgents and caches — and the core's HOME), and its core listens on
//! 18765 (`NEO_WS_PORT`). A release build can't be one (build.rs, and `lib::run` checks the bundle id).

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// A test build that runs beside the real NEO (see the module's doc).
pub const SANDBOX: bool = cfg!(feature = "sandbox");
/// This build's bundle id: the LaunchAgent's label, what Uninstall resets.
pub const BUNDLE_ID: &str = if SANDBOX { "com.arnav.neo.sandbox" } else { "com.arnav.neo" };
const PORT: u16 = if SANDBOX { 18765 } else { 8765 };
const MAX_RESTARTS: usize = 5; // within RESTART_WINDOW, then give up (a crash loop helps nobody)
const RESTART_WINDOW: Duration = Duration::from_secs(300);
const LOG_ROTATE_BYTES: u64 = 10 * 1024 * 1024;
const GRACE: Duration = Duration::from_secs(12); // the core saves its session summary on SIGTERM
const STARTUP_GRACE: Duration = Duration::from_secs(120); // loading models can take this long
const UNRESPONSIVE: Duration = Duration::from_secs(60); // websocket gone this long → hung
const RETRY_AFTER_GIVING_UP: Duration = Duration::from_secs(600);
/// How the core is started from a developer engine: `python -u -m neo2` (NEO's package was `neo`
/// before 2.1).
const CORE_ARGS: [&str; 3] = ["-u", "-m", "neo2"];
/// …and from a release engine: isolated (`-I`: no PYTHON* variables, no ~/.local site-packages,
/// nothing from the current folder), so only the engine's own packages are imported.
const RELEASE_ARGS: [&str; 4] = ["-I", "-u", "-m", "neo2"];
/// The supervisor's status while there is no engine to run: Setup installs one.
pub const NOT_SET_UP: &str = "not set up — open NEO Setup";
/// …while the bundled neo2 goes into the engine (an update that kept the engine: seconds).
pub const UPDATING: &str = "updating NEO…";
/// …while Setup replaces the engine the core runs from.
pub const HELD: &str = "setting up NEO's engine…";

/// The core's end of the socket pair, by number, in the core's environment.
const LINK_ENV: &str = "NEO_APP_FD";
/// Longer than anything the core says here (a line is under a hundred bytes): dropped.
const LINK_MAX_LINE: usize = 4096;
/// What the shell may ask the core about (`permission`), and the words the core answers with.
pub const PERMISSION_WHATS: [&str; 2] = ["microphone", "camera"];
const PERMISSION_STATUSES: [&str; 5] = ["granted", "denied", "restricted", "not_determined", "unknown"];
/// The supervisor's status while a core started some other way (a developer's `python -m neo2`) is the
/// one on the port: this app has no socket to it (Brains & Keys then shows the terminal commands).
pub const OUTSIDE: &str = "using a core started outside the app";

#[derive(Clone, Default)]
pub struct Core {
    child: Arc<Mutex<Option<Child>>>,
    stopping: Arc<AtomicBool>,
    restart_now: Arc<AtomicBool>,
    supervising: Arc<AtomicBool>,
    pub status: Arc<Mutex<String>>,
    link: Arc<Mutex<Link>>,
    listener: Arc<Mutex<Option<Listener>>>,
    /// Setup is replacing the engine: no core starts until it lets go (`release`).
    hold: Arc<AtomicBool>,
    /// Who hears that a launch needs Setup (lib.rs opens its window).
    on_setup: Arc<Mutex<Option<SetupListener>>>,
}

type SetupListener = Arc<dyn Fn(crate::setup::Decision) + Send + Sync>;

/// How a core is started: which interpreter, with which arguments, from which folder.
#[derive(Debug, Clone, PartialEq)]
pub struct Launch {
    pub python: PathBuf,
    pub args: Vec<&'static str>,
    pub cwd: PathBuf,
}

impl Launch {
    /// A developer engine: `<dir>/.venv/bin/python -u -m neo2`, from `dir`.
    pub fn dev(dir: &std::path::Path) -> Launch {
        Launch { python: dir.join(".venv").join("bin").join("python"), args: CORE_ARGS.to_vec(), cwd: dir.to_path_buf() }
    }
    /// A release engine: `engines/<id>/bin/python3 -I -u -m neo2`, from NEO's folder.
    pub fn release(paths: &crate::setup::Paths, id: &str) -> Launch {
        Launch { python: paths.python(id), args: RELEASE_ARGS.to_vec(), cwd: paths.root.clone() }
    }
    /// What `python -m neo2 fetch …` is run with beside this core (the same interpreter and flags).
    pub fn module(&self) -> Vec<&'static str> {
        self.args.iter().copied().filter(|a| *a != "-u").collect()
    }
}

/// This app's end of the socket it shares with the core it started.
#[derive(Default)]
struct Link {
    stream: Option<UnixStream>,
    /// The core said hello on it: it is reading. Nothing is written before that — a summon made
    /// while the core was still starting must not open the microphone half a minute later.
    ready: bool,
    /// Which core this is (each start has its own pair): an old core's last words don't count.
    generation: u64,
}

type Listener = Arc<dyn Fn(FromCore) + Send + Sync>;

/// What the core says to the app over their socket (shell_link.py lists the lines).
#[derive(Debug, PartialEq)]
pub enum FromCore {
    /// The core is up and reading: where the UI lives, and whether a host is connected.
    Hello { notch: bool, host_attached: bool },
    HostAttached(bool),
    UiHost { notch: bool },
    /// A host asked for one of the menu's actions (the core checked who asked).
    Shell { action: String, on: Option<bool> },
    /// What macOS says of the microphone or the camera (granted | denied | restricted |
    /// not_determined | unknown): the answer to `Core::permission`.
    Permission { what: String, status: String },
    /// The core at the other end is gone.
    Gone,
}

fn parse_from_core(line: &str) -> Option<FromCore> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let notch = |key: &str| match v.get(key)?.as_str()? {
        "notch" => Some(true),
        "window" => Some(false),
        _ => None,
    };
    match v.get("t")?.as_str()? {
        "hello" => Some(FromCore::Hello { notch: notch("ui_host")?, host_attached: v.get("host_attached")?.as_bool()? }),
        "host_attached" => Some(FromCore::HostAttached(v.get("attached")?.as_bool()?)),
        "ui_host" => Some(FromCore::UiHost { notch: notch("host")? }),
        "shell" => Some(FromCore::Shell {
            action: v.get("action")?.as_str()?.to_string(),
            on: v.get("on").and_then(|on| on.as_bool()),
        }),
        "permission" => {
            let what = v.get("what")?.as_str()?;
            let status = v.get("status")?.as_str()?;
            if !PERMISSION_WHATS.contains(&what) || !PERMISSION_STATUSES.contains(&status) {
                return None;
            }
            Some(FromCore::Permission { what: what.to_string(), status: status.to_string() })
        }
        _ => None,
    }
}

/// The line that asks the core about the microphone or the camera: what macOS says now, or (`ask`)
/// its prompt. None for anything else — the core is asked about nothing but those two.
fn permission_line(what: &str, ask: bool) -> Option<serde_json::Value> {
    if !PERMISSION_WHATS.contains(&what) {
        return None;
    }
    Some(serde_json::json!({"t": "permission", "what": what, "op": if ask { "ask" } else { "status" }}))
}

/// The next line of the core's (without its newline), `Some(None)` for one too long to be a
/// message (dropped whole), `None` when the core is gone.
fn read_link_line(reader: &mut impl BufRead) -> Option<Option<String>> {
    let mut buf = Vec::new();
    let n = reader.by_ref().take(LINK_MAX_LINE as u64).read_until(b'\n', &mut buf).ok()?;
    if n == 0 {
        return None;
    }
    if buf.last() != Some(&b'\n') {
        if n < LINK_MAX_LINE {
            return None; // the core went mid-line
        }
        loop {
            buf.clear();
            let n = reader.by_ref().take(LINK_MAX_LINE as u64).read_until(b'\n', &mut buf).ok()?;
            if n == 0 {
                return None;
            }
            if buf.last() == Some(&b'\n') {
                return Some(None);
            }
        }
    }
    buf.pop();
    Some(String::from_utf8(buf).ok())
}

/// A socket pair for the app and the core `cmd` is about to start: this app's end, and the core's
/// (to be dropped here once the core has started — the child has its own copy by then). The pair
/// has no name and no address: nothing can connect to it, a process can only be given it. Both
/// ends are close-on-exec, so no other child of this app (`open`, `ps`) gets one; the core's end
/// is let through only in the core's own fork.
fn link_for(cmd: &mut Command) -> Option<(UnixStream, UnixStream)> {
    let (ours, theirs) = UnixStream::pair().ok()?;
    let fd = theirs.as_raw_fd();
    if fd < 3 {
        return None; // would be taken for the core's stdin/stdout/stderr
    }
    let _ = ours.set_write_timeout(Some(Duration::from_millis(250))); // a core that isn't reading never blocks the app
    cmd.env(LINK_ENV, fd.to_string());
    unsafe {
        // After fork, before exec, in the child only: fcntl is async-signal-safe.
        cmd.pre_exec(move || {
            if libc::fcntl(fd, libc::F_SETFD, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    Some((ours, theirs))
}

/// The user's home — or, in a sandbox build, the `home` folder next to the app: everything NEO
/// Sandbox writes (and its core's HOME) lives there, never in the real one.
pub fn home() -> PathBuf {
    if SANDBOX {
        return sandbox_home();
    }
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
}

fn sandbox_home() -> PathBuf {
    match bundle_path().and_then(|app| app.parent().map(|d| d.join("home"))) {
        Some(h) => h,
        None => std::env::temp_dir().join("neo-sandbox-home"), // unbundled (a test): never the real home
    }
}


pub fn logs_dir() -> PathBuf {
    home().join(".neo").join("logs")
}

fn pidfile() -> PathBuf {
    home().join(".neo").join("core.pid")
}

/// The secret the overlay presents to the core's websocket (~/.neo/ws-token, readable only by
/// this user). Created here if missing, before the core starts, so both sides agree.
pub fn ws_token() -> String {
    let path = home().join(".neo").join("ws-token");
    if let Ok(t) = fs::read_to_string(&path) {
        let t = t.trim().to_string();
        if t.len() >= 32 {
            return t;
        }
    }
    let mut bytes = [0u8; 32];
    if let Ok(mut f) = File::open("/dev/urandom") {
        let _ = f.read_exact(&mut bytes);
    }
    let token: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    let _ = fs::create_dir_all(path.parent().unwrap());
    use std::os::unix::fs::OpenOptionsExt;
    if let Ok(mut f) = OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&path) {
        let _ = f.write_all(token.as_bytes());
    }
    token
}

/// Whether ~/.neo/.env says NEO's UI lives in the notch (`NEO_UI_HOST=notch`, written by the core
/// when NotchHub asks). Read once at launch; after that the core says it (`FromCore::Hello`, `UiHost`).
pub fn ui_host_is_notch() -> bool {
    let path = home().join(".neo").join(".env");
    let Ok(text) = fs::read_to_string(path) else { return false };
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("NEO_UI_HOST="))
        .last()
        .map(|v| v.trim().trim_matches('"').trim_matches('\'') == "notch")
        .unwrap_or(false)
}

/// ~/.neo/.env, where NEO's keys and settings live — created (readable only by this user) with
/// every setting explained if it doesn't exist yet. Never overwritten.
pub fn ensure_env_file() -> std::io::Result<PathBuf> {
    let path = home().join(".neo").join(".env");
    if path.exists() {
        return Ok(path);
    }
    fs::create_dir_all(path.parent().unwrap())?;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path)?;
    f.write_all(ENV_TEMPLATE.as_bytes())?;
    Ok(path)
}

const ENV_TEMPLATE: &str = "\
# Most people never edit this file: use the menu's Brains & Keys… (in the notch: NotchHub's
# Settings → NEO → Brains & Keys…) to pick NEO's brain and paste a key.
#
# NEO's settings, for those who'd rather type them. What you save in Brains & Keys wins over this
# file, and an environment variable wins over both. Save this file, then Restart NEO.

# Keys: set the ones you have. NEO needs at least one AI model to chat and do tasks.
# Google Gemini has a free tier, no card needed: https://aistudio.google.com/apikey
GEMINI_API_KEY=
# Groq, free (quick answers): https://console.groq.com/keys
GROQ_API_KEY=
# Anthropic (Claude), paid: https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY=
# OpenAI, paid: https://platform.openai.com/api-keys
OPENAI_API_KEY=
# OpenRouter, pay as you go (a few models are free): https://openrouter.ai/keys
OPENROUTER_API_KEY=
# Cerebras, paid (https://cloud.cerebras.ai): a very fast brain, tried ahead of Groq.
# NEO spends at most this many dollars a day on it (it tells you at 80%, then skips it).
CEREBRAS_API_KEY=
NEO_CEREBRAS_DAILY_BUDGET=0.50

# Which brain does what. A brain is one of gemini, gemini_lite, groq, groq_vision, cerebras,
# cerebras_vision, claude, openai, openrouter, local (NEO's model on this Mac: python -m neo2 --local)
# or custom:<name> (an endpoint added in Brains & Keys), with a model after \"@\" if you like:
# claude@claude-opus-5-5, openai@gpt-6.1-sol, openrouter@deepseek/deepseek-v4-pro.
# Leave these commented out for NEO's defaults (Gemini for tasks, Groq for quick replies).
# NEO_BRAIN=claude@claude-opus-5-5          tasks (it must call tools)
# NEO_FAST_BRAIN=claude@claude-haiku-5-5    quick replies: chat, summaries, memory
# NEO_EYES_BRAIN=claude@claude-opus-5-5     steps that look at a screenshot or photo
# NEO_BACKUP_BRAINS=groq,gemini             tried in turn when the brain can't answer (at most 4)
# NEO_LIGHT_BRAIN=gemini_lite               one-step commands (empty: the quick-replies brain)
# NEO_CLOUD_EYES=auto                       off: only the brains you picked ever see a screenshot
# NEO_VOICE=auto                            auto (Gemini Live with a Gemini key) | live | local | off
# NEO_DAILY_BUDGETS=anthropic=2,openai=2    dollars a day NEO may spend per paid provider

# Where \"the weather\" means when you haven't told NEO your city, e.g. Dubai.
NEO_HOME_LOCATION=

# How much of what NEO remembers about you goes to the cloud models with each request:
# full (profile, lessons, recent sessions, relevant notes) | profile (only stable facts) | off
NEO_CLOUD_MEMORY=full

# on | off — whether Laya (the on-device reflex) learns from the requests you make.
NEO_LEARN=on
";

/// NEO's installed runtime: its engines (setup.rs), or scripts/install-app.sh's .venv.
/// Outside ~/Documents on purpose: macOS guards that folder, and a background app reading it
/// blocks on a "NEO would like to access your Documents" prompt before Python even starts.
pub fn runtime_dir() -> PathBuf {
    home().join("Library").join("Application Support").join("NEO")
}

/// Where a developer engine runs from. A release build uses only the installed runtime — no
/// environment variable or file can point NEO.app (and the permissions macOS gave it) at other
/// code. A dev build (`tauri dev`) may use $NEO_HOME or the checkout it was built from.
fn core_dir() -> PathBuf {
    if cfg!(debug_assertions) {
        if let Ok(p) = std::env::var("NEO_HOME") {
            return PathBuf::from(p);
        }
        if !runtime_dir().join(".venv").join("bin").join("python").exists() && crate::setup::current_release(&crate::setup::Paths::here()).is_none() {
            let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
            return dir.canonicalize().unwrap_or(dir);
        }
    }
    runtime_dir()
}

/// Which engine this launch runs, and how (None: Setup has to run first — the decision says why).
pub fn launch_for(decision: &crate::setup::Decision) -> Option<Launch> {
    use crate::setup::{Decision, Paths};
    match decision {
        Decision::Dev => Some(Launch::dev(&core_dir())),
        Decision::Release { id } | Decision::UpdateWheel { id } => Some(Launch::release(&Paths::here(), id)),
        _ => None,
    }
}

/// What this launch finds (setup.rs's table), with this build's manifest and developer engine.
pub fn decide_now() -> crate::setup::Decision {
    let dev_ok = core_dir().join(".venv").join("bin").join("python").exists();
    crate::setup::decide(&crate::setup::Paths::here(), crate::setup::manifest(), dev_ok)
}

/// A runtime installed before 2.1 (NEO's package was the top-level `neo`, and its `neo2` had no
/// `__main__`): `-m neo2` can't start there, so say what fixes it instead of crash-looping. Only
/// that pair counts: an editable checkout's venv has neither, and NeuralEnsemble's `neo` next to
/// a current `neo2` has both.
fn runtime_predates_app(dir: &std::path::Path) -> bool {
    let Ok(libs) = fs::read_dir(dir.join(".venv").join("lib")) else {
        return false;
    };
    libs.flatten().any(|e| {
        let site = e.path().join("site-packages");
        let old = site.join("neo").join("__init__.py"); // neo2-codemod: keep (the old package, on purpose)
        old.exists() && !site.join("neo2").join("__main__.py").exists()
    })
}

fn port_in_use() -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], PORT));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

pub fn log(msg: &str) {
    let _ = fs::create_dir_all(logs_dir());
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(logs_dir().join("app.log")) {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(f, "[{secs}] {msg}");
    }
}

fn alive(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}

/// A command line (`ps -o command=`) of NEO's core: `… python -u -m neo2`, or the same with `neo`,
/// the package's name when an older NEO.app started it. Not NEO's training commands
/// (`-m neo2.core.reflex.finetune`), nor anything else.
fn is_core_command(line: &str) -> bool {
    let args: Vec<&str> = line.split_whitespace().collect();
    args.windows(2).any(|w| w[0] == "-m" && (w[1] == "neo2" || w[1] == "neo"))
}

/// A core this app started earlier and lost (its parent is gone, so it was re-parented to
/// launchd): stop it, so this launch owns the core again.
fn reclaim_orphan() {
    let Ok(text) = fs::read_to_string(pidfile()) else { return };
    let Ok(pid) = text.trim().parse::<i32>() else { return };
    if pid <= 1 || !alive(pid) {
        let _ = fs::remove_file(pidfile());
        return;
    }
    let out = Command::new("ps").args(["-o", "ppid=,command=", "-p", &pid.to_string()]).output();
    let Ok(out) = out else { return };
    let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let orphaned = line.split_whitespace().next() == Some("1");
    if orphaned && is_core_command(&line) {
        log(&format!("found an orphaned core (pid {pid}) — stopping it"));
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
        let t0 = Instant::now();
        while alive(pid) && t0.elapsed() < GRACE {
            thread::sleep(Duration::from_millis(200));
        }
        if alive(pid) {
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
        let _ = fs::remove_file(pidfile());
        for _ in 0..30 {
            if !port_in_use() {
                break;
            }
            thread::sleep(Duration::from_millis(200));
        }
    }
}

/// core.log, rotated by size as the core writes to it (its output is piped through the app).
struct RotatingLog {
    path: PathBuf,
    file: Option<File>,
    written: u64,
}

impl RotatingLog {
    fn new(path: PathBuf) -> Self {
        let written = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        Self { path, file: None, written }
    }

    fn line(&mut self, text: &str) {
        if self.written > LOG_ROTATE_BYTES {
            self.file = None;
            let _ = fs::rename(&self.path, self.path.with_extension("log.1"));
            self.written = 0;
        }
        if self.file.is_none() {
            self.file = OpenOptions::new().create(true).append(true).open(&self.path).ok();
        }
        if let Some(f) = self.file.as_mut() {
            if writeln!(f, "{text}").is_ok() {
                self.written += text.len() as u64 + 1;
            }
        }
    }
}

/// The environment a core (and the downloads Setup runs beside it) gets on top of the app's own.
pub fn core_env(cmd: &mut Command) {
    // Apps started from Finder get a bare PATH; the core shells out to Homebrew tools too.
    let path = format!(
        "/opt/homebrew/bin:/usr/local/bin:{}",
        std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin:/usr/sbin:/sbin".into())
    );
    cmd.env("PATH", path).env("PYTHONUNBUFFERED", "1");
    if SANDBOX {
        // NEO Sandbox's core: its own home (keys, memory, models, caches) and its own port.
        cmd.env("HOME", home()).env("NEO_WS_PORT", PORT.to_string());
    }
}

fn spawn(launch: &Launch, sink: Arc<Mutex<RotatingLog>>) -> std::io::Result<(Child, Option<UnixStream>)> {
    let mut cmd = Command::new(&launch.python);
    cmd.args(&launch.args).current_dir(&launch.cwd);
    core_env(&mut cmd);
    cmd.env("NEO_LAUNCHED_BY_APP", "1") // the core exits if this app disappears
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let link = link_for(&mut cmd);
    let mut child = cmd.spawn()?;
    let link = link.map(|(ours, _theirs)| ours); // the core has its end; this app's copy of it closes here
    let _ = fs::write(pidfile(), child.id().to_string());
    if let Some(out) = child.stdout.take() {
        let sink = sink.clone();
        thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                sink.lock().unwrap().line(&line);
            }
        });
    }
    if let Some(err) = child.stderr.take() {
        thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                sink.lock().unwrap().line(&line);
            }
        });
    }
    Ok((child, link))
}

impl Core {
    fn set_status(&self, s: &str) {
        *self.status.lock().unwrap() = s.to_string();
    }

    /// Who hears what the core says over the socket (set once, before `start`).
    pub fn listen(&self, listener: impl Fn(FromCore) + Send + Sync + 'static) {
        *self.listener.lock().unwrap() = Some(Arc::new(listener));
    }

    fn hear(&self, msg: FromCore) {
        let listener = self.listener.lock().unwrap().clone();
        if let Some(listener) = listener {
            listener(msg);
        }
    }

    /// A core just started, with this socket to it (or none): read what it says until it goes.
    fn attach(&self, stream: Option<UnixStream>) {
        let generation = {
            let mut link = self.link.lock().unwrap();
            link.generation += 1;
            link.ready = false;
            link.stream = stream.as_ref().and_then(|s| s.try_clone().ok());
            link.generation
        };
        let Some(stream) = stream else { return };
        let me = self.clone();
        thread::spawn(move || {
            let mut reader = BufReader::new(stream);
            while let Some(line) = read_link_line(&mut reader) {
                let Some(msg) = line.as_deref().and_then(parse_from_core) else { continue };
                {
                    let mut link = me.link.lock().unwrap();
                    if link.generation != generation {
                        return; // a newer core has the app's ear
                    }
                    if matches!(msg, FromCore::Hello { .. }) {
                        link.ready = true;
                    }
                }
                me.hear(msg);
            }
            {
                let mut link = me.link.lock().unwrap();
                if link.generation != generation {
                    return;
                }
                link.stream = None;
                link.ready = false;
            }
            me.hear(FromCore::Gone);
        });
    }

    /// Whether a core of this app's is listening on the socket (it said hello).
    pub fn listening(&self) -> bool {
        self.link.lock().map(|l| l.ready && l.stream.is_some()).unwrap_or(false)
    }

    /// One line to the core over the socket. False when there is no core of this app's that is
    /// listening (still starting, gone, or started outside the app) — the caller decides what then.
    pub fn tell(&self, msg: &serde_json::Value) -> bool {
        let mut link = self.link.lock().unwrap();
        if !link.ready {
            return false;
        }
        let Some(stream) = link.stream.as_mut() else { return false };
        let mut line = msg.to_string();
        line.push('\n');
        stream.write_all(line.as_bytes()).is_ok()
    }

    /// Ask the core about the microphone or the camera (Setup's checklist): what macOS says, or — `ask`
    /// — macOS's prompt, credited to this app. The answer comes back as `FromCore::Permission`. False
    /// when `what` is neither, or no core of this app's is listening.
    pub fn permission(&self, what: &str, ask: bool) -> bool {
        permission_line(what, ask).is_some_and(|line| self.tell(&line))
    }

    /// Start the core (or use one that is already running) and supervise it.
    pub fn start(&self) {
        if self.supervising.swap(true, Ordering::SeqCst) {
            return; // a supervisor is already running
        }
        self.stopping.store(false, Ordering::SeqCst);
        self.restart_now.store(false, Ordering::SeqCst);
        let me = self.clone();
        thread::spawn(move || {
            me.supervise();
            me.supervising.store(false, Ordering::SeqCst);
        });
    }

    fn supervise(&self) {
        let _ = ws_token(); // make sure it exists before the core (or the overlay) needs it
        loop {
            self.run_until_given_up();
            if self.stopping.load(Ordering::SeqCst) {
                return;
            }
            // Gave up (crash loop) or found someone else's core: look again later instead of never —
            // or at once if the user picks Restart NEO. With no engine (or Setup replacing it) there
            // is nothing to look for: wait for Setup to say it's done (`restart`).
            let until = Instant::now() + RETRY_AFTER_GIVING_UP;
            loop {
                if self.stopping.load(Ordering::SeqCst) {
                    return;
                }
                if self.restart_now.swap(false, Ordering::SeqCst) {
                    break;
                }
                let status = self.status.lock().unwrap().clone();
                if status == OUTSIDE && !port_in_use() {
                    break; // that core went away: run our own
                }
                let waits_for_setup = status == NOT_SET_UP || self.hold.load(Ordering::SeqCst);
                if !waits_for_setup && Instant::now() >= until {
                    break;
                }
                thread::sleep(Duration::from_millis(500));
            }
        }
    }

    /// Who hears that this launch needs Setup (set once, before `start`).
    pub fn on_setup(&self, listener: impl Fn(crate::setup::Decision) + Send + Sync + 'static) {
        *self.on_setup.lock().unwrap() = Some(Arc::new(listener));
    }

    fn needs_setup(&self, decision: crate::setup::Decision) {
        self.set_status(NOT_SET_UP);
        let listener = self.on_setup.lock().unwrap().clone();
        if let Some(listener) = listener {
            listener(decision);
        }
    }

    /// Setup is about to replace the engine the core runs from: stop the core, and start none until
    /// `release`.
    pub fn hold(&self) {
        if !self.hold.swap(true, Ordering::SeqCst) {
            log("Setup is replacing NEO's engine: the core stops until it's done");
        }
        self.set_status(HELD);
        self.terminate();
    }

    /// Setup is done (the engine is in, or it gave up): run whatever engine there is now.
    pub fn release(&self) {
        self.hold.store(false, Ordering::SeqCst);
        self.restart();
    }

    fn run_until_given_up(&self) {
        reclaim_orphan();
        if port_in_use() {
            self.set_status(OUTSIDE);
            log(&format!("a core is already listening on {PORT} (not ours) — using it"));
            return;
        }
        if self.hold.load(Ordering::SeqCst) {
            self.set_status(HELD);
            return;
        }
        let decision = decide_now();
        let Some(launch) = launch_for(&decision) else {
            log(&format!("no engine to run ({decision:?}): NEO's Setup installs one"));
            self.needs_setup(decision);
            return;
        };
        if let crate::setup::Decision::UpdateWheel { id } = &decision {
            // The same engine, another neo2 (an app update that kept the engine): offline, seconds.
            self.set_status(UPDATING);
            let (paths, bundle) = (crate::setup::Paths::here(), crate::setup::bundle());
            let done = match (crate::setup::manifest(), bundle.and_then(|b| b.wheel())) {
                (Some(m), Some(wheel)) => crate::setup::update_wheel(&paths, m, &wheel, &crate::setup::Opts::here()),
                _ => Err(crate::setup::Failure { code: "failed", message: "no wheel in this app".into() }),
            };
            if let Err(e) = done {
                log(&format!("updating neo2 in engine {id} failed: {} — NEO's Setup repairs it", e.message));
                self.needs_setup(crate::setup::Decision::Install);
                return;
            }
        }
        if decision == crate::setup::Decision::Dev && runtime_predates_app(&launch.cwd) {
            self.set_status("runtime older than the app — run scripts/install-app.sh");
            log(&format!("the runtime in {} is from before 2.1 (no neo2/__main__.py): run scripts/install-app.sh", launch.cwd.display()));
            return;
        }
        let _ = fs::create_dir_all(logs_dir());
        let sink = Arc::new(Mutex::new(RotatingLog::new(logs_dir().join("core.log"))));
        let mut restarts: Vec<Instant> = Vec::new();
        loop {
            if self.stopping.load(Ordering::SeqCst) || self.hold.load(Ordering::SeqCst) {
                return;
            }
            match spawn(&launch, sink.clone()) {
                Ok((child, link)) => {
                    log(&format!("core started (pid {})", child.id()));
                    self.set_status("running");
                    self.attach(link);
                    *self.child.lock().unwrap() = Some(child);
                }
                Err(e) => {
                    log(&format!("couldn't start the core: {e}"));
                    self.set_status(&format!("couldn't start: {e}"));
                    return;
                }
            }
            // Wait for it to exit (polling, so terminate() can reach the child too) — and restart it
            // if it stops answering: a process that exists isn't the same as a core that works.
            let started = Instant::now();
            let mut last_ok = Instant::now();
            let mut ever_up = false;
            let mut last_probe = Instant::now();
            loop {
                thread::sleep(Duration::from_millis(500));
                if last_probe.elapsed() >= Duration::from_secs(5) {
                    last_probe = Instant::now();
                    if port_in_use() {
                        last_ok = Instant::now();
                        ever_up = true;
                    } else {
                        let hung = if ever_up { last_ok.elapsed() > UNRESPONSIVE } else { started.elapsed() > STARTUP_GRACE };
                        if hung {
                            log("the core stopped answering — restarting it");
                            self.set_status("not responding — restarting");
                            let pid = self.child.lock().unwrap().as_ref().map(|c| c.id());
                            if let Some(pid) = pid {
                                unsafe {
                                    libc::kill(pid as i32, libc::SIGKILL);
                                }
                            }
                            last_ok = Instant::now();
                        }
                    }
                }
                let mut guard = self.child.lock().unwrap();
                match guard.as_mut().map(|c| c.try_wait()) {
                    Some(Ok(Some(code))) => {
                        log(&format!("core exited: {code}"));
                        *guard = None;
                        break;
                    }
                    Some(Ok(None)) => continue,
                    _ => {
                        *guard = None;
                        break;
                    }
                }
            }
            let _ = fs::remove_file(pidfile());
            if self.stopping.load(Ordering::SeqCst) || self.hold.load(Ordering::SeqCst) {
                return;
            }
            if self.restart_now.load(Ordering::SeqCst) {
                return; // asked for (the menu, or Setup): `supervise` starts again at once, and the engine is chosen afresh
            }
            let now = Instant::now();
            restarts.retain(|t| now.duration_since(*t) < RESTART_WINDOW);
            restarts.push(now);
            if restarts.len() > MAX_RESTARTS {
                log("the core keeps crashing — pausing restarts for 10 minutes (see core.log)");
                self.set_status("crashed repeatedly — see Open Logs; retrying in 10 min");
                return;
            }
            self.set_status("restarting");
            // Back off, but wake at once if the user picks Restart NEO.
            let until = Instant::now() + Duration::from_secs(2 * restarts.len() as u64);
            while Instant::now() < until && !self.restart_now.load(Ordering::SeqCst) {
                if self.stopping.load(Ordering::SeqCst) {
                    return;
                }
                thread::sleep(Duration::from_millis(100));
            }
            self.restart_now.store(false, Ordering::SeqCst);
        }
    }

    /// SIGTERM the running core (it saves its session summary) and wait; SIGKILL after GRACE.
    /// Returns whether there was a core to stop.
    fn terminate(&self) -> bool {
        let Some(pid) = self.child.lock().unwrap().as_ref().map(|c| c.id()) else {
            return false;
        };
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }
        let t0 = Instant::now();
        while t0.elapsed() < GRACE {
            thread::sleep(Duration::from_millis(100));
            let mut guard = self.child.lock().unwrap();
            match guard.as_mut() {
                None => return true, // the supervisor saw it exit
                Some(c) => {
                    if let Ok(Some(_)) = c.try_wait() {
                        return true;
                    }
                }
            }
        }
        if let Some(c) = self.child.lock().unwrap().as_mut() {
            let _ = c.kill();
        }
        true
    }

    /// Restart the core — or, if the supervisor already gave up, start supervising again.
    pub fn restart(&self) {
        if !self.supervising.load(Ordering::SeqCst) {
            self.set_status("starting");
            self.start();
            return;
        }
        self.set_status("restarting");
        self.restart_now.store(true, Ordering::SeqCst);
        // Nothing running (between restarts, or waiting to retry): the waiting loop sees
        // restart_now and starts at once.
        self.terminate();
    }

    pub fn stop(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        self.terminate();
    }
}

// ---- launch at login ------------------------------------------------------------------------

/// The login item: ~/Library/LaunchAgents/<bundle id>.plist (a sandbox build's is in its own home,
/// which launchd never reads).
fn agent_plist() -> PathBuf {
    home().join("Library").join("LaunchAgents").join(format!("{BUNDLE_ID}.plist"))
}

/// The NEO.app bundle this binary runs from (None when running unbundled, e.g. `tauri dev`).
pub fn bundle_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    exe.ancestors().find(|p| p.extension().map(|e| e == "app").unwrap_or(false)).map(|p| p.to_path_buf())
}

/// Whether this copy of NEO.app may be its own login item: it runs from /Applications or
/// ~/Applications (never from the disk image, Downloads or a translocated copy — that path goes
/// away). A sandbox build may anywhere it isn't on a disk image: its login item is its own home's.
pub fn may_open_at_login() -> bool {
    let Some(app) = bundle_path() else { return false };
    if SANDBOX {
        return crate::setup::placement_now() == crate::setup::Place::Ok;
    }
    let real_home = PathBuf::from(std::env::var("HOME").unwrap_or_default());
    crate::setup::login_allowed(&app, &real_home)
}

pub fn launch_at_login() -> bool {
    agent_plist().exists()
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn plist_for(exe: &std::path::Path) -> String {
    // launchd runs the app binary itself — not `open -a`, which counts as the user opening it
    // and puts NEO in the Dock's recent apps.
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{BUNDLE_ID}</string>
  <key>ProgramArguments</key>
  <array><string>{}</string></array>
  <key>RunAtLoad</key><true/>
  <key>LimitLoadToSessionType</key><string>Aqua</string>
  <key>ProcessType</key><string>Interactive</string>
</dict>
</plist>
"#,
        xml_escape(&exe.display().to_string())
    )
}

/// Turn the login item on or off. On is written only for a copy that may be one
/// (`may_open_at_login`); off always works.
pub fn set_launch_at_login(on: bool) -> std::io::Result<()> {
    if !on {
        let _ = fs::remove_file(agent_plist());
        return Ok(());
    }
    if !may_open_at_login() {
        log("not writing a login item: this copy of NEO isn't in Applications");
        return Ok(());
    }
    let Ok(exe) = std::env::current_exe() else { return Ok(()) };
    fs::create_dir_all(agent_plist().parent().unwrap())?;
    fs::write(agent_plist(), plist_for(&exe))
}

/// Every bundled launch once setup is done: on the first run, open at login by default; afterwards
/// keep the login item pointing at *this* copy of NEO.app (it may have been moved or rebuilt), if
/// it's on. Nothing at all from a copy that may not be one (`may_open_at_login`), so a first launch
/// from the disk image decides nothing.
pub fn sync_launch_at_login() {
    if !may_open_at_login() {
        return;
    }
    let marker = home().join(".neo").join("app-login-decided");
    if !marker.exists() {
        let _ = set_launch_at_login(true);
        let _ = fs::create_dir_all(marker.parent().unwrap());
        let _ = fs::write(marker, "on");
        return;
    }
    if launch_at_login() {
        if let Ok(exe) = std::env::current_exe() {
            let want = plist_for(&exe);
            if fs::read_to_string(agent_plist()).map(|cur| cur != want).unwrap_or(true) {
                let _ = fs::write(agent_plist(), want);
                log("login item updated to this copy of NEO.app");
            }
        }
    }
}

pub fn remember_login_choice() {
    let marker = home().join(".neo").join("app-login-decided");
    let _ = fs::create_dir_all(marker.parent().unwrap());
    let _ = fs::write(marker, if launch_at_login() { "on" } else { "off" });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn login_item_runs_the_binary_and_escapes_the_path() {
        let p = plist_for(std::path::Path::new("/Applications/NEO & Co.app/Contents/MacOS/neo-ui"));
        assert!(p.contains("<string>/Applications/NEO &amp; Co.app/Contents/MacOS/neo-ui</string>"));
        assert!(!p.contains("/usr/bin/open"));
        assert!(p.contains("LimitLoadToSessionType"));
        assert!(p.contains(&format!("<key>Label</key><string>{BUNDLE_ID}</string>")));
        assert!(agent_plist().ends_with(format!("Library/LaunchAgents/{BUNDLE_ID}.plist")));
    }

    #[test]
    fn a_release_engine_runs_isolated_from_neos_folder_and_a_developer_engine_as_before() {
        let paths = crate::setup::Paths::new("/Users/a/Library/Application Support/NEO");
        let r = Launch::release(&paths, "0123456789abcdef");
        assert_eq!(r.python, PathBuf::from("/Users/a/Library/Application Support/NEO/engines/0123456789abcdef/bin/python3"));
        assert_eq!(r.args, ["-I", "-u", "-m", "neo2"]);
        assert_eq!(r.cwd, paths.root);
        assert_eq!(r.module(), ["-I", "-m", "neo2"]);
        let d = Launch::dev(std::path::Path::new("/Users/a/Library/Application Support/NEO"));
        assert_eq!(d.python, PathBuf::from("/Users/a/Library/Application Support/NEO/.venv/bin/python"));
        assert_eq!(d.args, ["-u", "-m", "neo2"]);
        assert_eq!(d.module(), ["-m", "neo2"]);
        // Only those two shapes: Setup's own cases start nothing.
        use crate::setup::Decision;
        for d in [Decision::Install, Decision::NewEngine { from: "fedcba9876543210".into() }, Decision::NoEngine] {
            assert_eq!(launch_for(&d), None, "{d:?}");
        }
        assert_eq!(launch_for(&Decision::UpdateWheel { id: "0123456789abcdef".into() }).unwrap().args, RELEASE_ARGS);
        if SANDBOX {
            assert_eq!((PORT, BUNDLE_ID), (18765, "com.arnav.neo.sandbox"));
            // Never the real home: unbundled, a test's sandbox home is a scratch folder.
            assert_ne!(home(), PathBuf::from(std::env::var("HOME").unwrap()));
            assert!(home().starts_with(std::env::temp_dir()));
        } else {
            assert_eq!((PORT, BUNDLE_ID), (8765, "com.arnav.neo"));
        }
    }

    #[test]
    fn the_core_is_started_as_neo2_and_an_orphan_is_known_by_either_name() {
        assert_eq!(CORE_ARGS, ["-u", "-m", "neo2"]);
        assert_eq!(RELEASE_ARGS, ["-I", "-u", "-m", "neo2"]);
        let release = "/Users/a/Library/Application Support/NEO/engines/0123456789abcdef/bin/python3";
        assert!(is_core_command(&format!("1 {release} -I -u -m neo2")));
        assert!(!is_core_command(&format!("1 {release} -I -m neo2.core.fetch")));
        let runtime = "/Users/a/Library/Application Support/NEO/.venv/bin/python";
        assert!(is_core_command(&format!("1 {runtime} -u -m neo2")));
        assert!(is_core_command(&format!("1 {runtime} -u -m neo"))); // an older NEO.app's core  neo2-codemod: keep
        assert!(!is_core_command(&format!("1 {runtime} -m neo2.core.reflex.finetune")));
        assert!(!is_core_command(&format!("1 {runtime} -m neo2.mcp --tools web")));
        assert!(!is_core_command("1 /usr/bin/python3 -m neonode"));
        assert!(!is_core_command("1 /Applications/NEO.app/Contents/MacOS/neo-ui"));
    }

    #[test]
    fn a_runtime_from_before_2_1_is_told_apart() {
        let dir = std::env::temp_dir().join(format!("neo-runtime-test-{}", std::process::id()));
        let site = dir.join(".venv").join("lib").join("python3.12").join("site-packages");
        let touch = |rel: &str| {
            let f = site.join(rel);
            fs::create_dir_all(f.parent().unwrap()).unwrap();
            fs::write(f, "").unwrap();
        };
        assert!(!runtime_predates_app(&dir)); // no venv: "not installed" says so
        touch("neo2/__init__.py");
        assert!(!runtime_predates_app(&dir)); // neo2 only (an editable checkout has neither)
        touch("neo/__init__.py"); // neo2-codemod: keep
        assert!(runtime_predates_app(&dir)); // the old `neo` distribution, whose neo2 can't run
        touch("neo2/__main__.py");
        assert!(!runtime_predates_app(&dir)); // current, next to NeuralEnsemble's neo
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn what_the_core_says_over_the_socket() {
        let p = parse_from_core;
        assert_eq!(p(r#"{"t":"hello","ui_host":"notch","host_attached":false}"#), Some(FromCore::Hello { notch: true, host_attached: false }));
        assert_eq!(p(r#"{"t":"hello","ui_host":"window","host_attached":true}"#), Some(FromCore::Hello { notch: false, host_attached: true }));
        assert_eq!(p(r#"{"t":"host_attached","attached":true}"#), Some(FromCore::HostAttached(true)));
        assert_eq!(p(r#"{"t":"ui_host","host":"window"}"#), Some(FromCore::UiHost { notch: false }));
        assert_eq!(p(r#"{"t":"shell","action":"restart"}"#), Some(FromCore::Shell { action: "restart".into(), on: None }));
        assert_eq!(
            p(r#"{"t":"shell","action":"open_at_login","on":false}"#),
            Some(FromCore::Shell { action: "open_at_login".into(), on: Some(false) })
        );
        assert_eq!(
            p(r#"{"t":"permission","what":"microphone","status":"granted"}"#),
            Some(FromCore::Permission { what: "microphone".into(), status: "granted".into() })
        );
        assert_eq!(
            p(r#"{"t":"permission","what":"camera","status":"not_determined"}"#),
            Some(FromCore::Permission { what: "camera".into(), status: "not_determined".into() })
        );
        // Anything else is not a message: no guesses about where the UI lives or who is there.
        for bad in [
            r#"{"t":"permission","what":"screen","status":"granted"}"#,
            r#"{"t":"permission","what":"camera","status":"yes"}"#,
            r#"{"t":"permission","what":"camera"}"#,
            "",
            "quit",
            "[]",
            r#"{"t":"quit"}"#,
            r#"{"action":"quit"}"#,
            r#"{"t":"shell"}"#,
            r#"{"t":"shell","action":7}"#,
            r#"{"t":"hello","ui_host":"desk","host_attached":true}"#,
            r#"{"t":"hello","ui_host":"notch"}"#,
            r#"{"t":"host_attached","attached":"yes"}"#,
            r#"{"t":"ui_host","host":"elsewhere"}"#,
        ] {
            assert_eq!(p(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_core_is_asked_about_the_microphone_and_the_camera_only() {
        assert_eq!(
            permission_line("microphone", false),
            Some(serde_json::json!({"t": "permission", "what": "microphone", "op": "status"}))
        );
        assert_eq!(permission_line("camera", true), Some(serde_json::json!({"t": "permission", "what": "camera", "op": "ask"})));
        assert_eq!(permission_line("accessibility", true), None); // the app's own to ask (perms.rs)
        assert_eq!(permission_line("", false), None);
        assert!(!Core::default().permission("microphone", true)); // no core listening: nothing is sent
    }

    #[test]
    fn a_line_too_long_to_be_a_message_is_dropped_whole() {
        let mut text = String::from("{\"t\":\"host_attached\",\"attached\":true}\n");
        text.push_str(&"x".repeat(LINK_MAX_LINE * 2 + 17));
        text.push_str("\n{\"t\":\"ui_host\",\"host\":\"notch\"}\nhalf a li");
        let mut reader = BufReader::new(text.as_bytes());
        assert_eq!(read_link_line(&mut reader), Some(Some("{\"t\":\"host_attached\",\"attached\":true}".into())));
        assert_eq!(read_link_line(&mut reader), Some(None));
        assert_eq!(read_link_line(&mut reader), Some(Some("{\"t\":\"ui_host\",\"host\":\"notch\"}".into())));
        assert_eq!(read_link_line(&mut reader), None); // the core went mid-line: nothing of it counts
    }

    /// The whole way, with the core's own code at the other end: a process started the way the
    /// core is inherits its end of the pair, claims it with `shell_link.inherited()`, says hello —
    /// and only then does a wake go out; what it answers is heard; when it exits the app knows.
    #[test]
    fn the_app_and_the_core_it_starts_share_a_socket() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let python = root.join(".venv").join("bin").join("python");
        if !python.exists() {
            eprintln!("skipped: no Python environment in the checkout ({})", python.display());
            return;
        }
        const CHILD: &str = r#"
import json, os
from neo2.core.server import shell_link
sock = shell_link.inherited()
assert sock is not None, "no socket inherited"
assert shell_link.ENV not in os.environ and not os.get_inheritable(sock.fileno())
sock.setblocking(True)
f = sock.makefile("rwb")
f.write(b'{"t":"hello","ui_host":"notch","host_attached":true}\n'); f.flush()
msg = json.loads(f.readline())
assert msg == {"t": "wake"}, msg
f.write(b"x" * 9000 + b'\nnot json\n{"t":"shell","action":"open_at_login","on":true}\n'); f.flush()
"#;
        let core = Core::default();
        let (tx, rx) = std::sync::mpsc::channel();
        let tx = Mutex::new(tx);
        core.listen(move |msg| {
            let _ = tx.lock().unwrap().send(msg);
        });
        assert!(!core.tell(&serde_json::json!({"t": "wake"}))); // no core at all

        let mut cmd = Command::new(&python);
        cmd.args(["-c", CHILD]).current_dir(&root).stdin(Stdio::null());
        let (ours, theirs) = link_for(&mut cmd).expect("a socket pair");
        let mut child = cmd.spawn().expect("python starts");
        drop(theirs);
        core.attach(Some(ours));
        // Before the hello nothing is written: this wake is dropped, not kept for later.
        assert!(!core.tell(&serde_json::json!({"t": "wake", "early": true})));
        let wait = Duration::from_secs(20);
        assert_eq!(rx.recv_timeout(wait), Ok(FromCore::Hello { notch: true, host_attached: true }));
        assert!(core.tell(&serde_json::json!({"t": "wake"})));
        assert_eq!(rx.recv_timeout(wait), Ok(FromCore::Shell { action: "open_at_login".into(), on: Some(true) }));
        assert_eq!(rx.recv_timeout(wait), Ok(FromCore::Gone)); // it exited: the socket says so
        assert!(child.wait().expect("python ran").success(), "the core's side of the test failed");
        assert!(!core.tell(&serde_json::json!({"t": "wake"})));

        // No other child of the app gets either end: one started without `link_for` has no such
        // descriptor open, even while a pair exists.
        let mut other = Command::new(&python);
        let (ours, theirs) = link_for(&mut other).expect("a socket pair");
        let probe = format!(
            "import os\nfor fd in ({}, {}):\n    try:\n        os.fstat(fd)\n    except OSError:\n        continue\n    raise SystemExit(f'fd {{fd}} leaked')",
            ours.as_raw_fd(),
            theirs.as_raw_fd()
        );
        let status = Command::new(&python).args(["-c", &probe]).stdin(Stdio::null()).status().expect("python starts");
        assert!(status.success(), "another child inherited the socket");
    }

    /// A new ~/.neo/.env points at Brains & Keys first, marks no key "Required", names every
    /// provider's key, and every setting it names (set or commented out) is one the core reads.
    #[test]
    fn the_settings_file_template_points_at_brains_and_keys() {
        let t = ENV_TEMPLATE;
        assert!(t.starts_with("# Most people never edit this file: use the menu's Brains & Keys…"), "{t}");
        assert!(!t.contains("Required"));
        assert!(!t.contains("menu-bar icon"), "the notch has none");
        for key in ["GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY"] {
            assert!(t.lines().any(|l| l == format!("{key}=")), "{key}");
        }
        // A role left empty would mean "no brain": they are only ever examples, commented out.
        for l in t.lines().filter(|l| !l.starts_with('#')) {
            assert!(!l.starts_with("NEO_") || !l.contains("BRAIN"), "{l}");
        }
        let config = include_str!("../../../neo2/core/config.py");
        for l in t.lines() {
            let name = l.trim_start_matches("# ").split('=').next().unwrap_or("");
            if !l.contains('=') || name.is_empty() || !name.chars().all(|c| c.is_ascii_uppercase() || c == '_') {
                continue;
            }
            let known = match name.strip_prefix("NEO_") {
                Some(field) => config.contains(&format!("\n    {}:", field.to_lowercase())),
                None => config.contains(&format!("validation_alias=\"{name}\"")),
            };
            assert!(known, "{name} isn't a setting NEO reads");
        }
    }

    #[test]
    fn the_log_rotates_by_size_while_the_core_runs() {
        let dir = std::env::temp_dir().join(format!("neo-log-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("core.log");
        let mut log = RotatingLog::new(path.clone());
        log.written = LOG_ROTATE_BYTES + 1; // as if it had grown past the limit
        log.line("after rotation");
        assert!(dir.join("core.log.1").exists() || !path.with_extension("log.1").exists());
        assert_eq!(fs::read_to_string(&path).unwrap(), "after rotation\n");
        let _ = fs::remove_dir_all(dir);
    }
}

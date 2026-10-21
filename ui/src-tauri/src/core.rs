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

const PORT: u16 = 8765;
const MAX_RESTARTS: usize = 5; // within RESTART_WINDOW, then give up (a crash loop helps nobody)
const RESTART_WINDOW: Duration = Duration::from_secs(300);
const LOG_ROTATE_BYTES: u64 = 10 * 1024 * 1024;
const GRACE: Duration = Duration::from_secs(12); // the core saves its session summary on SIGTERM
const STARTUP_GRACE: Duration = Duration::from_secs(120); // loading models can take this long
const UNRESPONSIVE: Duration = Duration::from_secs(60); // websocket gone this long → hung
const RETRY_AFTER_GIVING_UP: Duration = Duration::from_secs(600);
/// How the core is started: `python -u -m neo2` (NEO's package was `neo` before 2.1).
const CORE_ARGS: [&str; 3] = ["-u", "-m", "neo2"];

/// The core's end of the socket pair, by number, in the core's environment.
const LINK_ENV: &str = "NEO_APP_FD";
/// Longer than anything the core says here (a line is under a hundred bytes): dropped.
const LINK_MAX_LINE: usize = 4096;
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
    /// The core is up and reading: where the UI lives, whether a host is connected, and Search
    /// Screen's settings (`None`: a core from before Search Screen — no hold, no overlay).
    Hello { notch: bool, host_attached: bool, lens: Option<LensCfg> },
    HostAttached(bool),
    UiHost { notch: bool },
    /// A host asked for one of the menu's actions (the core checked who asked).
    Shell { action: String, on: Option<bool> },
    /// Close Search Screen's overlay (Erase, Stop, an action that opened something, its time ran out).
    LensClose { why: String },
    /// The core at the other end is gone.
    Gone,
}

/// Search Screen's settings, as the core's hello gives them (NEO_LENS, NEO_LENS_HOLD_MS).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LensCfg {
    pub on: bool,
    pub hold_ms: u32,
}

/// Why the core closes Search Screen, as a word of its own (anything else: "done").
const LENS_CLOSE_WHYS: [&str; 6] = ["erased", "stop", "action", "continue", "idle", "done"];

fn lens_cfg(v: &serde_json::Value) -> Option<LensCfg> {
    let l = v.get("lens")?;
    let hold = l.get("hold_ms")?.as_u64()?;
    Some(LensCfg { on: l.get("on")?.as_bool()?, hold_ms: hold.clamp(600, 3000) as u32 })
}

fn parse_from_core(line: &str) -> Option<FromCore> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let notch = |key: &str| match v.get(key)?.as_str()? {
        "notch" => Some(true),
        "window" => Some(false),
        _ => None,
    };
    match v.get("t")?.as_str()? {
        "hello" => Some(FromCore::Hello {
            notch: notch("ui_host")?,
            host_attached: v.get("host_attached")?.as_bool()?,
            lens: lens_cfg(&v),
        }),
        "lens_close" => {
            let why = v.get("why").and_then(|w| w.as_str()).unwrap_or("");
            let why = LENS_CLOSE_WHYS.iter().find(|w| **w == why).copied().unwrap_or("done");
            Some(FromCore::LensClose { why: why.into() })
        }
        "host_attached" => Some(FromCore::HostAttached(v.get("attached")?.as_bool()?)),
        "ui_host" => Some(FromCore::UiHost { notch: notch("host")? }),
        "shell" => Some(FromCore::Shell {
            action: v.get("action")?.as_str()?.to_string(),
            on: v.get("on").and_then(|on| on.as_bool()),
        }),
        _ => None,
    }
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

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
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

# Search Screen: hold ⌥⌘ for about a second, let go, then circle or drag over anything on screen.
# NEO_LENS=on                  off: holding ⌥⌘ does nothing (a quick tap still summons NEO)
# NEO_LENS_HOLD_MS=1000        how long to hold, 600–3000 (under 1000 also shortens the quick tap)
# NEO_LENS_SEARCH=google       \"Search with…\": google | bing | off, or a site for a text search
# NEO_LENS_ASK=on              off: on-device results only, until you press Ask
";

/// NEO's installed runtime (scripts/install-app.sh puts a Python environment with the core here).
/// Outside ~/Documents on purpose: macOS guards that folder, and a background app reading it
/// blocks on a "NEO would like to access your Documents" prompt before Python even starts.
fn runtime_dir() -> PathBuf {
    home().join("Library").join("Application Support").join("NEO")
}

/// Where the core runs from. A release build uses only the installed runtime — no environment
/// variable or file can point NEO.app (and the permissions macOS gave it) at other code. A dev
/// build (`tauri dev`) may use $NEO_HOME or the checkout it was built from.
fn core_dir() -> PathBuf {
    if cfg!(debug_assertions) {
        if let Ok(p) = std::env::var("NEO_HOME") {
            return PathBuf::from(p);
        }
        if !runtime_dir().join(".venv").join("bin").join("python").exists() {
            let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
            return dir.canonicalize().unwrap_or(dir);
        }
    }
    runtime_dir()
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

fn spawn(dir: &PathBuf, sink: Arc<Mutex<RotatingLog>>) -> std::io::Result<(Child, Option<UnixStream>)> {
    // Apps started from Finder get a bare PATH; the core shells out to Homebrew tools too.
    let path = format!(
        "/opt/homebrew/bin:/usr/local/bin:{}",
        std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin:/usr/sbin:/sbin".into())
    );
    let mut cmd = Command::new(dir.join(".venv").join("bin").join("python"));
    cmd.args(CORE_ARGS)
        .current_dir(dir)
        .env("PATH", path)
        .env("PYTHONUNBUFFERED", "1")
        .env("NEO_LAUNCHED_BY_APP", "1") // the core exits if this app disappears
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

    /// A core of this app's has said hello on the socket and is still there (Search Screen's hold
    /// needs one: the overlay is nothing without the core).
    pub fn linked(&self) -> bool {
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
            // Gave up (crash loop), found someone else's core, or NEO isn't installed: look again
            // later instead of never — or at once if the user picks Restart NEO.
            let until = Instant::now() + RETRY_AFTER_GIVING_UP;
            while Instant::now() < until {
                if self.stopping.load(Ordering::SeqCst) {
                    return;
                }
                if self.restart_now.swap(false, Ordering::SeqCst) {
                    break;
                }
                if *self.status.lock().unwrap() == OUTSIDE && !port_in_use() {
                    break; // that core went away: run our own
                }
                thread::sleep(Duration::from_millis(500));
            }
        }
    }

    fn run_until_given_up(&self) {
        reclaim_orphan();
        if port_in_use() {
            self.set_status(OUTSIDE);
            log("a core is already listening on 8765 (not ours) — using it");
            return;
        }
        let dir = core_dir();
        if !dir.join(".venv").join("bin").join("python").exists() {
            let msg = "not installed — run scripts/install-app.sh".to_string();
            self.set_status(&msg);
            log(&format!("no NEO runtime in {}", dir.display()));
            return;
        }
        if runtime_predates_app(&dir) {
            self.set_status("runtime older than the app — run scripts/install-app.sh");
            log(&format!("the runtime in {} is from before 2.1 (no neo2/__main__.py): run scripts/install-app.sh", dir.display()));
            return;
        }
        let _ = fs::create_dir_all(logs_dir());
        let sink = Arc::new(Mutex::new(RotatingLog::new(logs_dir().join("core.log"))));
        let mut restarts: Vec<Instant> = Vec::new();
        loop {
            if self.stopping.load(Ordering::SeqCst) {
                return;
            }
            match spawn(&dir, sink.clone()) {
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
            if self.stopping.load(Ordering::SeqCst) {
                return;
            }
            if self.restart_now.swap(false, Ordering::SeqCst) {
                restarts.clear(); // asked for from the menu: not a crash
                continue;
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

fn agent_plist() -> PathBuf {
    home().join("Library").join("LaunchAgents").join("com.arnav.neo.plist")
}

/// The NEO.app bundle this binary runs from (None when running unbundled, e.g. `tauri dev`).
fn bundle_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    exe.ancestors().find(|p| p.extension().map(|e| e == "app").unwrap_or(false)).map(|p| p.to_path_buf())
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
  <key>Label</key><string>com.arnav.neo</string>
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

pub fn set_launch_at_login(on: bool) -> std::io::Result<()> {
    if !on {
        let _ = fs::remove_file(agent_plist());
        return Ok(());
    }
    let Some(app) = bundle_path() else {
        return Ok(()); // a dev build has no .app to launch
    };
    let Ok(exe) = std::env::current_exe() else { return Ok(()) };
    let _ = app;
    fs::create_dir_all(agent_plist().parent().unwrap())?;
    fs::write(agent_plist(), plist_for(&exe))
}

/// Every bundled launch: on the first run, open at login by default; afterwards keep the login
/// item pointing at *this* copy of NEO.app (it may have been moved or rebuilt), if it's on.
pub fn sync_launch_at_login() {
    if bundle_path().is_none() {
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
    }

    #[test]
    fn the_core_is_started_as_neo2_and_an_orphan_is_known_by_either_name() {
        assert_eq!(CORE_ARGS, ["-u", "-m", "neo2"]);
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
        assert_eq!(p(r#"{"t":"hello","ui_host":"notch","host_attached":false}"#), Some(FromCore::Hello { notch: true, host_attached: false, lens: None }));
        assert_eq!(p(r#"{"t":"hello","ui_host":"window","host_attached":true}"#), Some(FromCore::Hello { notch: false, host_attached: true, lens: None }));
        assert_eq!(p(r#"{"t":"host_attached","attached":true}"#), Some(FromCore::HostAttached(true)));
        assert_eq!(p(r#"{"t":"ui_host","host":"window"}"#), Some(FromCore::UiHost { notch: false }));
        assert_eq!(p(r#"{"t":"shell","action":"restart"}"#), Some(FromCore::Shell { action: "restart".into(), on: None }));
        assert_eq!(
            p(r#"{"t":"shell","action":"open_at_login","on":false}"#),
            Some(FromCore::Shell { action: "open_at_login".into(), on: Some(false) })
        );
        // Anything else is not a message: no guesses about where the UI lives or who is there.
        for bad in [
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
    fn the_hello_says_whether_search_screen_is_on_and_the_core_can_close_it() {
        let p = parse_from_core;
        let hello = |lens: &str| p(&format!(r#"{{"t":"hello","ui_host":"window","host_attached":false{lens}}}"#));
        let with = |on, hold_ms| Some(FromCore::Hello { notch: false, host_attached: false, lens: Some(LensCfg { on, hold_ms }) });
        assert_eq!(hello(r#","lens":{"on":true,"hold_ms":1000}"#), with(true, 1000));
        assert_eq!(hello(r#","lens":{"on":false,"hold_ms":1200}"#), with(false, 1200));
        assert_eq!(hello(r#","lens":{"on":true,"hold_ms":50}"#), with(true, 600)); // clamped here too
        assert_eq!(hello(r#","lens":{"on":true,"hold_ms":99999}"#), with(true, 3000));
        // An older core says nothing of it; a shape that isn't one counts as nothing said, and the
        // hello itself still counts.
        let none = Some(FromCore::Hello { notch: false, host_attached: false, lens: None });
        for bad in ["", r#","lens":true"#, r#","lens":{"on":"yes","hold_ms":1000}"#, r#","lens":{"on":true}"#, r#","lens":{"on":true,"hold_ms":-5}"#] {
            assert_eq!(hello(bad), none, "{bad}");
        }
        assert_eq!(p(r#"{"t":"lens_close","why":"erased"}"#), Some(FromCore::LensClose { why: "erased".into() }));
        assert_eq!(p(r#"{"t":"lens_close","why":"stop"}"#), Some(FromCore::LensClose { why: "stop".into() }));
        assert_eq!(p(r#"{"t":"lens_close"}"#), Some(FromCore::LensClose { why: "done".into() }));
        assert_eq!(p(r#"{"t":"lens_close","why":"<script>"}"#), Some(FromCore::LensClose { why: "done".into() }));
        assert_eq!(p(r#"{"t":"lens_freeze","ticket":"x"}"#), None); // the shell's to say, never the core's
        // No core, nothing linked.
        assert!(!Core::default().linked());
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
        assert_eq!(rx.recv_timeout(wait), Ok(FromCore::Hello { notch: true, host_attached: true, lens: None }));
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

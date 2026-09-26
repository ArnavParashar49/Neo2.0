//! Runs NEO's Python core as a child of the app and keeps it running.
//!
//! NEO.app is the process macOS asks for permissions (Accessibility, Microphone, Screen
//! Recording): the core it starts inherits them. The core also watches its parent and exits if
//! the app goes away without stopping it (force quit, crash), so a core with the mic open can't
//! be left behind; if one ever is, the next launch recognises it by its pidfile and replaces it.
//! A core started some other way (a developer's `python -m neo`) is used as-is.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpStream};
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

#[derive(Clone, Default)]
pub struct Core {
    child: Arc<Mutex<Option<Child>>>,
    stopping: Arc<AtomicBool>,
    restart_now: Arc<AtomicBool>,
    supervising: Arc<AtomicBool>,
    pub status: Arc<Mutex<String>>,
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

/// Where the NEO repo (with its .venv) lives: $NEO_HOME, else ~/.neo/app.json {"core_dir"},
/// else the checkout this app was built from.
fn core_dir() -> PathBuf {
    if let Ok(p) = std::env::var("NEO_HOME") {
        return PathBuf::from(p);
    }
    if let Ok(text) = fs::read_to_string(home().join(".neo").join("app.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(p) = v.get("core_dir").and_then(|x| x.as_str()) {
                return PathBuf::from(p);
            }
        }
    }
    // ui/src-tauri → the repo root, as it was on the build machine.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    dir.canonicalize().unwrap_or(dir)
}

fn port_in_use() -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], PORT));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn log(msg: &str) {
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
    if orphaned && line.contains("-m neo") {
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

fn spawn(dir: &PathBuf, sink: Arc<Mutex<RotatingLog>>) -> std::io::Result<Child> {
    // Apps started from Finder get a bare PATH; the core shells out to Homebrew tools too.
    let path = format!(
        "/opt/homebrew/bin:/usr/local/bin:{}",
        std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin:/usr/sbin:/sbin".into())
    );
    let mut child = Command::new(dir.join(".venv").join("bin").join("python"))
        .args(["-u", "-m", "neo"])
        .current_dir(dir)
        .env("PATH", path)
        .env("PYTHONUNBUFFERED", "1")
        .env("NEO_LAUNCHED_BY_APP", "1") // the core exits if this app disappears
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
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
    Ok(child)
}

impl Core {
    fn set_status(&self, s: &str) {
        *self.status.lock().unwrap() = s.to_string();
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
        reclaim_orphan();
        if port_in_use() {
            self.set_status("using a core started outside the app");
            log("a core is already listening on 8765 (not ours) — using it");
            return;
        }
        let dir = core_dir();
        if !dir.join(".venv").join("bin").join("python").exists() {
            let msg = format!("can't find NEO's Python environment in {}", dir.display());
            self.set_status(&msg);
            log(&msg);
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
                Ok(child) => {
                    log(&format!("core started (pid {})", child.id()));
                    self.set_status("running");
                    *self.child.lock().unwrap() = Some(child);
                }
                Err(e) => {
                    log(&format!("couldn't start the core: {e}"));
                    self.set_status(&format!("couldn't start: {e}"));
                    return;
                }
            }
            // Wait for it to exit (polling, so terminate() can reach the child too).
            loop {
                thread::sleep(Duration::from_millis(500));
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
                log("the core keeps crashing — stopped restarting it (see core.log)");
                self.set_status("crashed repeatedly — see Open Logs, then Restart NEO");
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
        self.restart_now.store(true, Ordering::SeqCst);
        if !self.terminate() {
            // Nothing running (between restarts): the backoff loop sees restart_now and goes now.
        }
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

//! Runs NEO's Python core as a child of the app and keeps it running.
//!
//! NEO.app is the process macOS asks for permissions (Accessibility, Microphone, Screen
//! Recording): the core it starts inherits them. If a core is already listening on the
//! websocket (a developer running `python -m neo` by hand), the app just connects to it.

use std::fs::{self, OpenOptions};
use std::io::Write;
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

#[derive(Clone, Default)]
pub struct Core {
    child: Arc<Mutex<Option<Child>>>,
    stopping: Arc<AtomicBool>,
    restart_now: Arc<AtomicBool>,
    pub status: Arc<Mutex<String>>,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
}

pub fn logs_dir() -> PathBuf {
    home().join(".neo").join("logs")
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
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn already_running() -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], PORT));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn log(msg: &str) {
    let _ = fs::create_dir_all(logs_dir());
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(logs_dir().join("app.log")) {
        let _ = writeln!(f, "{} {msg}", chrono_now());
    }
}

fn chrono_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("[{secs}]")
}

fn spawn(dir: &PathBuf) -> std::io::Result<Child> {
    let _ = fs::create_dir_all(logs_dir());
    let log_path = logs_dir().join("core.log");
    if fs::metadata(&log_path).map(|m| m.len() > LOG_ROTATE_BYTES).unwrap_or(false) {
        let _ = fs::rename(&log_path, logs_dir().join("core.log.1"));
    }
    let out = OpenOptions::new().create(true).append(true).open(&log_path)?;
    let err = out.try_clone()?;
    // Apps started from Finder get a bare PATH; the core shells out to Homebrew tools too.
    let path = format!(
        "/opt/homebrew/bin:/usr/local/bin:{}",
        std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin:/usr/sbin:/sbin".into())
    );
    Command::new(dir.join(".venv").join("bin").join("python"))
        .args(["-u", "-m", "neo"])
        .current_dir(dir)
        .env("PATH", path)
        .env("PYTHONUNBUFFERED", "1")
        .env("NEO_LAUNCHED_BY_APP", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .spawn()
}

impl Core {
    /// Start the core (unless one is already up) and supervise it on a background thread.
    pub fn start(&self) {
        if already_running() {
            *self.status.lock().unwrap() = "connected to a core that was already running".into();
            log("a core is already listening on 8765 — not starting another");
            return;
        }
        let dir = core_dir();
        if !dir.join(".venv").join("bin").join("python").exists() {
            let msg = format!("can't find NEO's Python environment in {}", dir.display());
            *self.status.lock().unwrap() = msg.clone();
            log(&msg);
            return;
        }
        let me = self.clone();
        thread::spawn(move || me.supervise(dir));
    }

    fn supervise(&self, dir: PathBuf) {
        let mut restarts: Vec<Instant> = Vec::new();
        loop {
            if self.stopping.load(Ordering::SeqCst) {
                return;
            }
            match spawn(&dir) {
                Ok(child) => {
                    log(&format!("core started (pid {})", child.id()));
                    *self.status.lock().unwrap() = "running".into();
                    *self.child.lock().unwrap() = Some(child);
                }
                Err(e) => {
                    log(&format!("couldn't start the core: {e}"));
                    *self.status.lock().unwrap() = format!("couldn't start: {e}");
                    return;
                }
            }
            // Wait for it to exit (polling, so stop() can take the child out of the mutex).
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
                *self.status.lock().unwrap() = "crashed repeatedly — see ~/.neo/logs/core.log".into();
                return;
            }
            *self.status.lock().unwrap() = "restarting".into();
            thread::sleep(Duration::from_secs(2 * restarts.len() as u64));
        }
    }

    /// Ask the running core to stop (SIGTERM, so it can save its session), then make sure.
    fn terminate(&self) {
        let pid = self.child.lock().unwrap().as_ref().map(|c| c.id());
        if let Some(pid) = pid {
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
            for _ in 0..40 {
                thread::sleep(Duration::from_millis(100));
                if self.child.lock().unwrap().is_none() {
                    return; // the supervisor saw it exit
                }
                if let Some(c) = self.child.lock().unwrap().as_mut() {
                    if let Ok(Some(_)) = c.try_wait() {
                        return;
                    }
                }
            }
            if let Some(c) = self.child.lock().unwrap().as_mut() {
                let _ = c.kill();
            }
        }
    }

    pub fn restart(&self) {
        self.restart_now.store(true, Ordering::SeqCst);
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

pub fn set_launch_at_login(on: bool) -> std::io::Result<()> {
    if !on {
        let _ = fs::remove_file(agent_plist());
        return Ok(());
    }
    let Some(app) = bundle_path() else {
        return Ok(()); // a dev build has no .app to launch
    };
    fs::create_dir_all(agent_plist().parent().unwrap())?;
    fs::write(
        agent_plist(),
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.arnav.neo</string>
  <key>ProgramArguments</key>
  <array><string>/usr/bin/open</string><string>-a</string><string>{}</string></array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
"#,
            app.display()
        ),
    )
}

/// First run of the bundled app: start at login unless the user turned that off before.
pub fn default_launch_at_login() {
    let marker = home().join(".neo").join("app-login-decided");
    if marker.exists() || bundle_path().is_none() {
        return;
    }
    let _ = set_launch_at_login(true);
    let _ = fs::create_dir_all(marker.parent().unwrap());
    let _ = fs::write(marker, "on");
}

pub fn remember_login_choice() {
    let marker = home().join(".neo").join("app-login-decided");
    let _ = fs::create_dir_all(marker.parent().unwrap());
    let _ = fs::write(marker, if launch_at_login() { "on" } else { "off" });
}

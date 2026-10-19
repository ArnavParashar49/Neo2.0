//! "Is there a newer NEO?" — a check that installs nothing.
//!
//! A minute after launch and then once a day, NEO.app fetches the latest release's latest.json
//! (`releases/latest/download/latest.json` on GitHub: an asset, so no API rate limit) with
//! `/usr/bin/curl`, ten seconds at most. It sends nothing about the user — a plain GET — and a check
//! that fails is silent. latest.json is the Tauri updater's shape plus a few fields of NEO's own
//! (scripts/build_release.sh writes it):
//!
//! ```json
//! {"version":"2.2.1","pub_date":"…","notes":"…","dmg_url":"…","sha256":"…","min_macos":"14.0",
//!  "platforms":{"darwin-aarch64":{"url":"…","signature":""}}}
//! ```
//!
//! Only `version` is read (X.Y.Z). A newer one shows in the menu as "Update to NEO X…", which opens
//! Setup's Updates page; its button opens that version's release page on GitHub — a page this app
//! builds from the version, never a URL from the file. The user downloads the new DMG and drags it
//! in, and the engine check (setup.rs: cases e and f) does the rest at the next launch.
//! ~/.neo/app-update.json remembers the last check, the newest version seen, one the user skipped,
//! and whether checking is on.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::core;

pub const LATEST_URL: &str = "https://github.com/ArnavParashar49/Neo2.0/releases/latest/download/latest.json";
const RELEASES: &str = "https://github.com/ArnavParashar49/Neo2.0/releases";
/// The newest release's page (Setup's "Releases page" when this NEO's engine is gone from GitHub).
pub const RELEASES_LATEST: &str = "https://github.com/ArnavParashar49/Neo2.0/releases/latest";
pub const FIRST_CHECK: Duration = Duration::from_secs(60);
pub const EVERY: Duration = Duration::from_secs(24 * 3600);
const TIMEOUT_S: &str = "10";
const MAX_BYTES: &str = "65536";

/// What ~/.neo/app-update.json keeps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Remembered {
    #[serde(default = "yes")]
    pub auto: bool,
    #[serde(default)]
    pub last_check: u64,
    #[serde(default)]
    pub latest: Option<String>,
    #[serde(default)]
    pub skipped: Option<String>,
}

fn yes() -> bool {
    true
}

impl Default for Remembered {
    fn default() -> Self {
        Remembered { auto: true, last_check: 0, latest: None, skipped: None }
    }
}

fn file() -> PathBuf {
    core::home().join(".neo").join("app-update.json")
}

pub fn load() -> Remembered {
    fs::read_to_string(file()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

pub fn save(r: &Remembered) {
    let path = file();
    let _ = fs::create_dir_all(path.parent().unwrap());
    let tmp = path.with_extension("json.tmp");
    if fs::write(&tmp, serde_json::to_string(r).unwrap_or_default()).is_ok() {
        let _ = fs::rename(tmp, path);
    }
}

/// X.Y.Z as numbers (None for anything else).
pub fn parse_version(v: &str) -> Option<(u64, u64, u64)> {
    let v = v.trim().strip_prefix('v').unwrap_or(v.trim());
    let mut it = v.split('.');
    let parts = (it.next()?.parse().ok()?, it.next()?.parse().ok()?, it.next()?.parse().ok()?);
    if it.next().is_some() {
        return None;
    }
    Some(parts)
}

pub fn newer(candidate: &str, current: &str) -> bool {
    matches!((parse_version(candidate), parse_version(current)), (Some(a), Some(b)) if a > b)
}

/// The version latest.json names, if it names one NEO can read.
pub fn version_in(json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let s = v.get("version")?.as_str()?;
    let (a, b, c) = parse_version(s)?;
    Some(format!("{a}.{b}.{c}"))
}

/// The release page this app opens for a version (built here, never taken from the file).
pub fn page(version: &str) -> String {
    match parse_version(version) {
        Some((a, b, c)) => format!("{RELEASES}/tag/v{a}.{b}.{c}"),
        None => format!("{RELEASES}/latest"),
    }
}

/// Where this build looks. A sandbox build looks only where its own home says
/// (`<home>/.neo/update-url`, a local test server), and nowhere if it says nothing.
pub fn url() -> Option<String> {
    if core::SANDBOX {
        let u = fs::read_to_string(core::home().join(".neo").join("update-url")).ok()?;
        let u = u.trim().to_string();
        return (u.starts_with("http://127.0.0.1:") || u.starts_with("http://localhost:") || u.starts_with("file:///")).then_some(u);
    }
    Some(LATEST_URL.into())
}

/// One check: the newest version, or why there's none to say.
pub fn fetch(url: &str) -> Result<String, String> {
    let mut cmd = Command::new("/usr/bin/curl");
    cmd.arg("-q")
        .args(["--fail", "--location", "--silent", "--show-error", "--max-time", TIMEOUT_S, "--max-filesize", MAX_BYTES])
        .args(if core::SANDBOX { ["--proto", "=https,http,file"] } else { ["--proto", "=https"] })
        .args(if core::SANDBOX { ["--proto-redir", "=https,http"] } else { ["--proto-redir", "=https"] })
        .args(["--header", "Accept: application/json"])
        .arg(url)
        .stdin(Stdio::null());
    let out = cmd.output().map_err(|e| format!("curl didn't start: {e}"))?;
    if !out.status.success() {
        let why = String::from_utf8_lossy(&out.stderr).trim().chars().take(160).collect::<String>();
        return Err(if why.is_empty() { format!("curl exit {}", out.status.code().unwrap_or(-1)) } else { why });
    }
    version_in(&String::from_utf8_lossy(&out.stdout)).ok_or_else(|| "latest.json names no version".into())
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Check now and remember what was found. Ok(Some(v)): a newer NEO the user hasn't skipped.
pub fn check(current: &str) -> Result<Option<String>, String> {
    let Some(url) = url() else { return Err("this build checks no server".into()) };
    let mut r = load();
    let found = fetch(&url);
    r.last_check = now();
    if let Ok(v) = &found {
        r.latest = Some(v.clone());
    }
    save(&r);
    let v = found?;
    core::log(&format!("update check: the newest NEO is {v} (this is {current})"));
    Ok(offer(&r, current))
}

/// The update the menu offers, from what was remembered: newer than this app, not skipped.
pub fn offer(r: &Remembered, current: &str) -> Option<String> {
    let v = r.latest.as_deref()?;
    (newer(v, current) && r.skipped.as_deref() != Some(v)).then(|| v.to_string())
}

/// Whether a daily check is due (on, and a day since the last).
pub fn due(r: &Remembered, at: u64) -> bool {
    r.auto && at.saturating_sub(r.last_check) >= EVERY.as_secs()
}

pub fn skip(version: &str) {
    let mut r = load();
    r.skipped = Some(version.to_string());
    save(&r);
}

pub fn set_auto(on: bool) {
    let mut r = load();
    r.auto = on;
    save(&r);
}

/// The daily check, in its own thread: `found` hears each newer version (the menu item follows).
pub fn start(current: String, found: impl Fn(Option<String>) + Send + 'static) {
    if url().is_none() {
        return;
    }
    std::thread::spawn(move || {
        found(offer(&load(), &current)); // what an earlier check found, at once
        std::thread::sleep(FIRST_CHECK);
        loop {
            if due(&load(), now()) {
                if let Ok(v) = check(&current) {
                    found(v);
                }
            }
            std::thread::sleep(Duration::from_secs(3600));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_are_compared_as_numbers() {
        assert!(newer("2.2.10", "2.2.9"));
        assert!(newer("3.0.0", "2.9.9"));
        assert!(!newer("2.2.0", "2.2.0"));
        assert!(!newer("2.1.9", "2.2.0"));
        assert!(newer("v2.3.0", "2.2.0"));
        for bad in ["2.2", "2.2.0.1", "2.2.x", "", "latest"] {
            assert_eq!(parse_version(bad), None, "{bad}");
            assert!(!newer(bad, "1.0.0"), "{bad}");
        }
    }

    #[test]
    fn only_latest_jsons_version_is_read_and_the_page_is_neos_own() {
        let json = r#"{"version":"2.2.1","pub_date":"2026-11-01T00:00:00Z","notes":"…","dmg_url":"https://evil.example/x.dmg",
            "platforms":{"darwin-aarch64":{"url":"https://evil.example/x","signature":""}}}"#;
        assert_eq!(version_in(json), Some("2.2.1".into()));
        assert_eq!(version_in(r#"{"version":"2.2.1; rm -rf ~"}"#), None);
        assert_eq!(version_in("<html>"), None);
        assert_eq!(page("2.2.1"), "https://github.com/ArnavParashar49/Neo2.0/releases/tag/v2.2.1");
        assert_eq!(page("../../x"), "https://github.com/ArnavParashar49/Neo2.0/releases/latest");
        assert_eq!(LATEST_URL, "https://github.com/ArnavParashar49/Neo2.0/releases/latest/download/latest.json");
    }

    #[test]
    fn a_skipped_version_is_offered_no_more_and_checks_are_daily() {
        let mut r = Remembered { latest: Some("2.2.1".into()), ..Remembered::default() };
        assert_eq!(offer(&r, "2.2.0"), Some("2.2.1".into()));
        assert_eq!(offer(&r, "2.2.1"), None);
        r.skipped = Some("2.2.1".into());
        assert_eq!(offer(&r, "2.2.0"), None);
        r.latest = Some("2.2.2".into());
        assert_eq!(offer(&r, "2.2.0"), Some("2.2.2".into())); // a newer one than the skipped is offered
        r.last_check = 1_000_000;
        assert!(!due(&r, 1_000_000 + 3600));
        assert!(due(&r, 1_000_000 + 24 * 3600));
        r.auto = false;
        assert!(!due(&r, 9_999_999_999));
        let old: Remembered = serde_json::from_str("{}").unwrap();
        assert!(old.auto, "checking is on unless the user turned it off");
    }

    #[test]
    fn a_check_reads_a_local_latest_json() {
        // (file:// is only what the test can serve; release builds take https only)
        let dir = std::env::temp_dir().join(format!("neo-update-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let f = dir.join("latest.json");
        fs::write(&f, r#"{"version":"9.0.0"}"#).unwrap();
        let mut cmd = Command::new("/usr/bin/curl");
        let out = cmd.args(["-q", "--fail", "--silent", "--max-time", "10"]).arg(format!("file://{}", f.display())).output().unwrap();
        assert_eq!(version_in(&String::from_utf8_lossy(&out.stdout)), Some("9.0.0".into()));
        if !core::SANDBOX {
            // A release build refuses anything but https.
            assert!(fetch(&format!("file://{}", f.display())).is_err());
        }
        let _ = fs::remove_dir_all(dir);
    }
}

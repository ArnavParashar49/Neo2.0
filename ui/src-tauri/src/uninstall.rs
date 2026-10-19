//! Uninstall NEO…: what NEO put on this Mac, each with its size and a tickbox, and all of it to the
//! Trash (native.rs `trash`: NSFileManager, so the user can put anything back, and no Finder
//! automation prompt). Nothing is deleted here, and nothing outside NEO's own:
//!
//! - always: this NEO.app, its login item, its caches and WebKit data;
//! - ticked: Application Support/NEO (the engines; NotchHub's NEO page goes offline with it);
//! - ticked: NEO's own models in the shared Hugging Face cache — only the repos NEO pins, never the
//!   rest of that cache — and ~/.neo/models (the wake word, camera boxes, memory search);
//! - ticked: macOS's permissions for this app (`tccutil reset All <this bundle id>`, nothing else's);
//! - not ticked: Chromium (Playwright's cache is shared with other tools) and "My NEO data"
//!   (~/.neo: keys, memory, logs).
//!
//! ~/Documents/NEO Notes is never touched. Every path is this bundle or under this build's home
//! (`may_trash`); a sandbox build's home is its own folder, so it can only ever empty that.

use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::Serialize;

use crate::core;

/// The repos NEO downloads into the shared Hugging Face cache (neo2/core/models.py, fetch.py).
pub const HF_REPOS: [&str; 6] = [
    "convaiinnovations/laya",
    "sentence-transformers/all-MiniLM-L6-v2",
    "mlx-community/parakeet-tdt-0.6b-v3",
    "mlx-community/Kokoro-82M-bf16",
    "prince-canuma/Kokoro-82M",
    "mlx-community/gemma-4-12B-it-4bit",
];

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Item {
    pub id: &'static str,
    pub label: String,
    pub note: String,
    #[serde(skip)]
    pub paths: Vec<PathBuf>,
    /// Shown to the user (home-relative).
    pub shown: Vec<String>,
    pub bytes: u64,
    /// Ticked to begin with.
    pub default: bool,
    /// Always goes (the app itself).
    pub locked: bool,
}

fn shown(p: &Path, home: &Path) -> String {
    match p.strip_prefix(home) {
        Ok(rest) => format!("~/{}", rest.display()),
        Err(_) => p.display().to_string(),
    }
}

/// Everything Uninstall offers, for this bundle and home (paths that aren't there are left out;
/// an item with none left is too, except the app's own).
pub fn items(app: Option<&Path>, home: &Path, bundle_id: &str) -> Vec<Item> {
    let lib = home.join("Library");
    let hub = home.join(".cache").join("huggingface").join("hub");
    let mut out = Vec::new();
    let mut push = |id: &'static str, label: &str, note: &str, paths: Vec<PathBuf>, default: bool, locked: bool| {
        let paths: Vec<PathBuf> = paths.into_iter().filter(|p| fs::symlink_metadata(p).is_ok()).collect();
        if paths.is_empty() && !locked {
            return;
        }
        let shown = paths.iter().map(|p| shown(p, home)).collect();
        out.push(Item { id, label: label.into(), note: note.into(), paths, shown, bytes: 0, default, locked });
    };
    let mut app_paths: Vec<PathBuf> = app.map(|a| vec![a.to_path_buf()]).unwrap_or_default();
    app_paths.push(lib.join("LaunchAgents").join(format!("{bundle_id}.plist")));
    app_paths.push(lib.join("Caches").join(bundle_id));
    app_paths.push(lib.join("WebKit").join(bundle_id));
    push("app", "NEO.app and its login item", "The app, its login item, caches and web data.", app_paths, true, true);
    push(
        "engine",
        "NEO's engine",
        "Python and NEO's packages (Application Support/NEO). NotchHub's NEO page goes with it.",
        vec![lib.join("Application Support").join("NEO")],
        true,
        false,
    );
    push(
        "models",
        "NEO's models in the Hugging Face cache",
        "Only the models NEO downloaded; the rest of that cache stays.",
        HF_REPOS.iter().map(|r| hub.join(format!("models--{}", r.replace('/', "--")))).collect(),
        true,
        false,
    );
    push(
        "model_files",
        "NEO's other models",
        "The wake word, camera boxes and memory search (~/.neo/models).",
        vec![home.join(".neo").join("models")],
        true,
        false,
    );
    push(
        "chromium",
        "Chromium for NEO's browser",
        "Playwright's browsers — other tools on this Mac may use them too.",
        vec![lib.join("Caches").join("ms-playwright")],
        false,
        false,
    );
    push("data", "My NEO data", "Keys, memory, settings and logs (~/.neo). Leave it to keep them for later.", vec![home.join(".neo")], false, false);
    out
}

/// Whether Uninstall may move `p`: this bundle, or something under `home` (never home itself, and
/// never ~/Documents).
pub fn may_trash(p: &Path, app: Option<&Path>, home: &Path) -> bool {
    if p.components().any(|c| matches!(c, Component::ParentDir | Component::CurDir)) || !p.is_absolute() {
        return false;
    }
    if app.is_some_and(|a| a == p) {
        return p.extension().is_some_and(|e| e == "app");
    }
    p.starts_with(home) && p != home && !p.starts_with(home.join("Documents")) && p.components().count() > home.components().count()
}

/// Bytes under `p` (its files, not following links).
pub fn size(p: &Path) -> u64 {
    let Ok(meta) = fs::symlink_metadata(p) else { return 0 };
    if !meta.is_dir() {
        return meta.len();
    }
    let mut total = 0;
    let mut stack = vec![p.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let Ok(m) = fs::symlink_metadata(e.path()) else { continue };
            if m.is_dir() {
                stack.push(e.path());
            } else if m.is_file() {
                total += m.len();
            }
        }
    }
    total
}

/// `items` with their sizes (this walks every folder: run it off the main thread).
pub fn sized(mut list: Vec<Item>) -> Vec<Item> {
    for it in list.iter_mut() {
        // ~/.neo holds ~/.neo/models too: counted once, under the models.
        it.bytes = it.paths.iter().map(|p| size(p)).sum();
    }
    let models = list.iter().find(|i| i.id == "model_files").map(|i| i.bytes).unwrap_or(0);
    if let Some(d) = list.iter_mut().find(|i| i.id == "data") {
        d.bytes = d.bytes.saturating_sub(models);
    }
    list
}

/// The one command that resets this app's permissions — its own bundle id only.
pub fn reset_command(bundle_id: &str) -> Option<Vec<String>> {
    let ok = bundle_id.starts_with("com.arnav.neo") && bundle_id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.');
    ok.then(|| vec!["/usr/bin/tccutil".into(), "reset".into(), "All".into(), bundle_id.into()])
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Outcome {
    pub moved: Vec<String>,
    pub failed: Vec<String>,
    pub permissions_reset: Option<bool>,
}

/// Move the chosen items to the Trash (the app's own always), then — if asked — reset this app's
/// permissions. `trash` is native.rs's; the tests pass their own.
pub fn run(
    list: &[Item],
    chosen: &[String],
    reset_permissions: bool,
    app: Option<&Path>,
    home: &Path,
    bundle_id: &str,
    trash: &dyn Fn(&Path) -> Result<(), String>,
    exec: &dyn Fn(&[String]) -> bool,
) -> Outcome {
    let mut out = Outcome::default();
    // The longest paths first: ~/.neo/models goes before ~/.neo, if both were chosen.
    let mut paths: Vec<&PathBuf> = list.iter().filter(|i| i.locked || chosen.iter().any(|c| c == i.id)).flat_map(|i| i.paths.iter()).collect();
    paths.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    for p in paths {
        if !may_trash(p, app, home) {
            out.failed.push(format!("{}: not NEO's to move", p.display()));
            continue;
        }
        if fs::symlink_metadata(p).is_err() {
            continue; // gone with its parent, or never there
        }
        match trash(p) {
            Ok(()) => out.moved.push(shown(p, home)),
            Err(e) => out.failed.push(format!("{}: {e}", shown(p, home))),
        }
    }
    if reset_permissions {
        out.permissions_reset = Some(reset_command(bundle_id).is_some_and(|cmd| exec(&cmd)));
    }
    core::log(&format!("uninstall: moved {} to the Trash, {} failed", out.moved.len(), out.failed.len()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn scratch(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("neo-uninstall-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn touch(p: &Path, bytes: usize) {
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, vec![0u8; bytes]).unwrap();
    }

    #[test]
    fn only_neos_own_things_are_listed_ticked_as_the_design_says() {
        let home = scratch("list").join("home");
        let app = home.parent().unwrap().join("Apps/NEO.app");
        fs::create_dir_all(app.join("Contents/MacOS")).unwrap();
        touch(&home.join("Library/Application Support/NEO/engines/0123456789abcdef/bin/python3"), 1000);
        touch(&home.join(".cache/huggingface/hub/models--convaiinnovations--laya/blobs/a"), 5000);
        touch(&home.join(".cache/huggingface/hub/models--someone--else/blobs/b"), 7000);
        std::os::unix::fs::symlink("../../blobs/a", home.join(".cache/huggingface/hub/models--convaiinnovations--laya/x")).unwrap();
        touch(&home.join(".neo/models/vosk/a"), 300);
        touch(&home.join(".neo/memory.sqlite3"), 50);
        touch(&home.join("Library/Caches/ms-playwright/chromium-1/x"), 10);
        touch(&home.join("Documents/NEO Notes/note.md"), 10);
        let list = sized(items(Some(&app), &home, "com.arnav.neo"));
        let ids: Vec<&str> = list.iter().map(|i| i.id).collect();
        assert_eq!(ids, ["app", "engine", "models", "model_files", "chromium", "data"]);
        let by = |id: &str| list.iter().find(|i| i.id == id).unwrap().clone();
        assert!(by("app").locked && by("app").default);
        assert_eq!(by("app").paths, vec![app.clone()]); // no login item or caches here yet
        assert!(by("engine").default && by("models").default && by("model_files").default);
        assert!(!by("chromium").default && !by("data").default);
        assert_eq!(by("models").paths, vec![home.join(".cache/huggingface/hub/models--convaiinnovations--laya")]);
        assert_eq!(by("models").bytes, 5000, "a link is not counted twice");
        assert_eq!(by("data").bytes, 50, "~/.neo/models is counted once, under the models");
        assert_eq!(by("model_files").shown, vec!["~/.neo/models".to_string()]);
        let everything: Vec<&PathBuf> = list.iter().flat_map(|i| i.paths.iter()).collect();
        assert!(!everything.iter().any(|p| p.starts_with(home.join("Documents"))));
        assert!(!everything.iter().any(|p| p.ends_with("models--someone--else")));
        let _ = fs::remove_dir_all(home.parent().unwrap());
    }

    #[test]
    fn nothing_outside_neos_own_is_ever_moved() {
        let home = PathBuf::from("/Users/a");
        let app = PathBuf::from("/Applications/NEO.app");
        assert!(may_trash(&app, Some(&app), &home));
        assert!(may_trash(Path::new("/Users/a/.neo"), Some(&app), &home));
        assert!(may_trash(Path::new("/Users/a/Library/Application Support/NEO"), Some(&app), &home));
        for no in ["/Users/a", "/Users/a/Documents/NEO Notes", "/Users/a/Documents", "/Applications", "/Users/b/.neo", "/Users/a/../b", "relative/.neo", "/"] {
            assert!(!may_trash(Path::new(no), Some(&app), &home), "{no}");
        }
        assert!(!may_trash(Path::new("/Applications/Safari.app"), Some(&app), &home));
    }

    #[test]
    fn the_chosen_items_go_to_the_trash_and_permissions_only_for_this_app() {
        let home = scratch("run").join("home2");
        let app = home.join("Apps/NEO Sandbox.app");
        fs::create_dir_all(&app).unwrap();
        touch(&home.join("Library/Application Support/NEO/engine.json"), 10);
        touch(&home.join(".neo/models/x"), 10);
        touch(&home.join(".neo/keys.json"), 10);
        touch(&home.join("Library/Caches/ms-playwright/x"), 10);
        let list = items(Some(&app), &home, "com.arnav.neo.sandbox");
        let trashed = RefCell::new(Vec::<PathBuf>::new());
        let ran = RefCell::new(Vec::<Vec<String>>::new());
        let chosen: Vec<String> = ["engine", "model_files", "data"].map(String::from).to_vec();
        let out = run(
            &list,
            &chosen,
            true,
            Some(&app),
            &home,
            "com.arnav.neo.sandbox",
            &|p| {
                trashed.borrow_mut().push(p.to_path_buf());
                fs::rename(p, home.parent().unwrap().join(format!("trashed-{}", trashed.borrow().len()))).map_err(|e| e.to_string())
            },
            &|cmd| {
                ran.borrow_mut().push(cmd.to_vec());
                true
            },
        );
        let t = trashed.borrow();
        assert!(t.contains(&app) && t.contains(&home.join("Library/Application Support/NEO")) && t.contains(&home.join(".neo")));
        let models = t.iter().position(|p| p.ends_with(".neo/models")).unwrap();
        let data = t.iter().position(|p| p.ends_with(".neo")).unwrap();
        assert!(models < data, "the inner folder first");
        assert!(!t.iter().any(|p| p.ends_with("ms-playwright")), "not chosen");
        assert_eq!(out.failed, Vec::<String>::new());
        assert_eq!(*ran.borrow(), vec![vec!["/usr/bin/tccutil".to_string(), "reset".into(), "All".into(), "com.arnav.neo.sandbox".into()]]);
        assert_eq!(out.permissions_reset, Some(true));
        assert_eq!(reset_command("com.apple.Safari"), None);
        assert_eq!(reset_command("com.arnav.neo; rm -rf ~"), None);
        let _ = fs::remove_dir_all(home.parent().unwrap());
    }
}

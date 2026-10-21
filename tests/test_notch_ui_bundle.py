"""NEO's UI hosted in the notch (NotchHub's WKWebView, `neo://app/`): every Tauri call goes through
one shim (ui/src/shell.ts), which in notch mode answers from what the host injected and never loads
a Tauri module — so the page works unchanged in NEO's own window and in the notch. The shim's own
behaviour is tested under ui/tests (Node's test runner: `npm test` in ui/)."""

import re
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "ui"
SRC = UI / "src"

# The only modules that may import Tauri: the shim, the window's own frame code, the pop-out
# windows, Brains & Keys and Search Screen's door (which exist only inside NEO.app — the notch has
# none; the keys and lens windows load Tauri lazily, only on their own routes).
TAURI_ALLOWED = {
    "shell.ts", "win.ts", "browser/popout.ts", "camera/popout.ts", "browser/PopoutApp.tsx", "camera/CameraPopoutApp.tsx",
    "keys/wire.ts", "keys/KeysApp.tsx", "lens/door.ts",
}  # fmt: skip


def _sources():
    return {p.relative_to(SRC).as_posix(): p.read_text() for p in SRC.rglob("*") if p.suffix in (".ts", ".tsx")}


def test_tauri_is_imported_only_behind_the_shim():
    for name, src in _sources().items():
        if name in TAURI_ALLOWED:
            continue
        assert "@tauri-apps" not in src and "__TAURI_INTERNALS__" not in src, name


def test_the_shim_loads_tauri_lazily_and_only_in_window_mode():
    shell = (SRC / "shell.ts").read_text()
    assert not re.search(r'^import .* from "@tauri-apps', shell, re.M)  # never at the top
    for line in shell.splitlines():
        if "@tauri-apps" in line and not line.strip().startswith("//"):
            assert "await import(" in line, line
    win = (SRC / "win.ts").read_text()
    assert not re.search(r'^import .* from "@tauri-apps', win, re.M)
    assert 'export const inTauri = !inNotch && "__TAURI_INTERNALS__" in window;' in win


def test_the_host_is_told_once_and_the_page_fills_the_panel():
    host = (SRC / "host.ts").read_text()
    assert 'window.__NEO_HOST__ === "notch"' in host and '"host") === "notch"' in host
    assert 'document.documentElement.dataset.host = "notch"' in host
    shell = (SRC / "shell.ts").read_text()
    assert "window.__NEO_WS_TOKEN__" in shell and "messageHandlers?.neo" in shell and "__neoHostEmit" in shell
    css = (SRC / "App.css").read_text()
    assert 'html[data-host="notch"] .card' in css and "border-radius: 0" in css.split('html[data-host="notch"] .card')[1]
    app = (SRC / "App.tsx").read_text()
    assert "useState(inNotch)" in app  # always the chat, never a pill of its own
    assert 'if (inNotch) { void hideWindow(); return; }' in app  # Esc / collapse fold the host's panel
    assert "if (inNotch) return; // the host times its own collapse" in app


def test_the_shim_has_its_tests():
    tests = (UI / "tests" / "notch-shell.test.ts").read_text()
    assert 'import("../src/shell.ts")' in tests and "__TAURI_INTERNALS__" in tests


def test_the_notch_dress_is_scoped_and_compact():
    """The notch's design (ui/src/notch.css) never leaks into NEO's own window: every rule is under
    html[data-host="notch"]; the chat lays itself out for NotchHub's compact panel; nothing on it
    waits on an entrance animation (WebKit holds the panel's page "hidden" unless the host says
    otherwise, and a frozen fade would leave every row at opacity 0)."""
    app = (SRC / "App.tsx").read_text()
    assert 'import "./notch.css";' in app
    assert "const NOTCH_CARD: Size = { w: 340, h: 300 };" in app  # NeoPanelLayout.standard in NotchHub
    assert "if (inNotch) { void resizeAnchored(target); winSize.current = target; }" in app  # told on load
    css = (SRC / "notch.css").read_text()
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for rule in re.findall(r"([^{}]+)\{", body):
        for selector in rule.split(","):
            selector = selector.strip()
            if not selector or selector.startswith("@"):
                continue
            assert selector.startswith('html[data-host="notch"]'), selector
    assert 'html[data-host="notch"] .msg, html[data-host="notch"] .tools' in body and "animation: none;" in body
    assert "prefers-reduced-motion: reduce" in body


def test_no_page_speaks_for_neo_apps_shell():
    """A host's menu actions, what the menu shows, whether a host is there and the summon keys in
    the notch go between NEO.app's shell and the core on their own socket (core.rs,
    neo2/core/server/shell_link.py) — never through a page. The page of a hidden window is
    suspended by macOS and answers nothing; and a page shows what models and websites wrote, so it
    must have no way to ask the shell for the menu (quit, restart, Open at Login)."""
    ws = (SRC / "ws.ts").read_text()
    shell = (SRC / "shell.ts").read_text()
    for gone in ("shellAction", "shellStatus", "hostAttached", "shell_action", "shell_status", "host_attached", "neo://shell-status"):
        assert gone not in ws and gone not in shell, gone
    assert 'case "shell":' not in ws
    # The shell offers the page none of it either: no such command exists to be invoked…
    lib = (UI / "src-tauri" / "src" / "lib.rs").read_text()
    handlers = lib.split("tauri::generate_handler![")[1].split("]")[0]
    names = {h.strip().split("::")[-1] for h in handlers.split(",") if h.strip()}
    lens_cmds = {"lens_hello", "lens_painted", "lens_close", "lens_peek", "lens_allow_sc", "lens_glow"}
    assert names == {"hide_window", "ws_token", "ui_host_changed", "set_frame", "clipboard_write", "show_quiet", "keys_ticket", "edit_env_file"} | lens_cmds
    # (Brains & Keys' two answer only the window the shell itself created, at #/keys: keys.rs)
    keys = (UI / "src-tauri" / "src" / "keys.rs").read_text()
    for cmd in ("keys_ticket", "edit_env_file"):
        body = keys.split(f"pub fn {cmd}(")[1].split("\n}\n")[0]
        assert "may_get_ticket(window.label(), ours(&app)" in body.split("\n")[2], cmd
    # (and Search Screen's six only its own overlay, at #/lens: lens.rs — none of them a menu action)
    lens = (UI / "src-tauri" / "src" / "lens.rs").read_text()
    assert "may_use_lens(window.label(), ours(app), url.as_ref()" in lens.split("fn check(")[1].split("\n}\n")[0]
    for cmd in lens_cmds:
        body = lens.split(f"pub fn {cmd}(")[1].split("\n}\n")[0]
        assert body.split("\n")[1].strip() == "check(&app, &window)?;", cmd
    for gone in ("fn shell_action", "fn shell_status", "fn host_attached", "neo://shell-status"):
        assert gone not in lib, gone
    # …and in the notch the summon keys go to the core, not to the hidden page.
    show = lib.split("\nfn show(app: &AppHandle) {")[1].split("\n}\n")[0]
    notch, window = show.split("        return;\n", 1)
    assert 'tell(&serde_json::json!({"t": "wake"}))' in notch and "emit(" not in notch
    assert 'emit("neo://wake"' in window  # NEO's own window: the page on screen expands, and wakes
    core = (UI / "src-tauri" / "src" / "core.rs").read_text()
    link = (UI.parent / "neo2" / "core" / "server" / "shell_link.py").read_text()
    assert 'const LINK_ENV: &str = "NEO_APP_FD";' in core and 'ENV = "NEO_APP_FD"' in link

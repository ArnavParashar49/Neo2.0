"""NEO's UI hosted in the notch (NotchHub's WKWebView, `neo://app/`): every Tauri call goes through
one shim (ui/src/shell.ts), which in notch mode answers from what the host injected and never loads
a Tauri module — so the page works unchanged in NEO's own window and in the notch. The shim's own
behaviour is tested under ui/tests (Node's test runner: `npm test` in ui/)."""

import re
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "ui"
SRC = UI / "src"

# The only modules that may import Tauri: the shim, the window's own frame code, and the pop-out
# windows (which exist only inside NEO.app — the notch has none).
TAURI_ALLOWED = {"shell.ts", "win.ts", "browser/popout.ts", "camera/popout.ts", "browser/PopoutApp.tsx", "camera/CameraPopoutApp.tsx"}


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

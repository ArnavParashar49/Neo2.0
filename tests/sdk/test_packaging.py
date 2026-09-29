"""The distribution: a small core, extras for everything else, NEO.app's `app` extra exactly the
dependency list it always had, the licence and links as the owner set them, and the MCP console
script and the typing marker shipped."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.portable  # Linux or macOS, any install (what needs an extra skips): CI's sdk-core job

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
PROJECT = PYPROJECT["project"]
EXTRAS: dict[str, list[str]] = PROJECT["optional-dependencies"]
SELF = re.compile(r"^neo2\[([a-z0-9_,-]+)\]$")

# NEO.app's dependencies before they were split into core + extras (pyproject.toml at 4764b73).
APP_BEFORE = {
    "google-genai>=2.20",
    "openai>=2",
    "anthropic>=1",
    "laya>=0.3",
    "mlx-audio; sys_platform == 'darwin'",
    "mlx-lm; sys_platform == 'darwin'",
    "misaki[en]; sys_platform == 'darwin'",
    "vosk",
    "sounddevice",
    "soundfile",
    "silero-vad",
    "torch",
    "sentence-transformers",
    "playwright",
    "pyobjc-framework-Cocoa; sys_platform == 'darwin'",
    "pyobjc-framework-Quartz; sys_platform == 'darwin'",
    "pyobjc-framework-ApplicationServices; sys_platform == 'darwin'",
    "pyobjc-framework-EventKit>=12.2,<13; sys_platform == 'darwin'",
    "pyobjc-framework-Contacts>=12.2,<13; sys_platform == 'darwin'",
    "pillow",
    "httpx",
    "ddgs",
    "tld",
    "lxml",
    "trafilatura",
    "pypdf",
    "python-dateutil",
    "numpy",
    "pydantic>=2",
    "pydantic-settings>=2.7",
    "python-dotenv",
    "websockets>=13",
}
# What NEO.app gained since (the user's decisions, 2026-09-27): screen capture in NEO's own process
# (ScreenCaptureKit) and on-device text recognition (Vision), the same pyobjc release.
APP_ADDED = {
    "pyobjc-framework-ScreenCaptureKit>=12.2,<13; sys_platform == 'darwin'",
    "pyobjc-framework-Vision>=12.2,<13; sys_platform == 'darwin'",
}
# And the camera (2026-09-29: "what am I holding?"): AVFoundation's capture, and D-FINE's boxes
# through transformers (Laya's already) — the vision extra; then YOLOE-26s's boxes through Core ML
# (the user's choice, the same pyobjc release; nothing new in requirements.lock).
CAMERA_ADDED = {
    "pyobjc-framework-AVFoundation>=12.2,<13; sys_platform == 'darwin'",
    "pyobjc-framework-CoreML>=12.2,<13; sys_platform == 'darwin'",
    "transformers",
    "torchvision",
}
HEAVY = ("torch", "laya", "mlx", "pyobjc", "playwright", "sentence-transformers", "google-genai", "openai", "anthropic")


def _flat(extra: str, seen: tuple[str, ...] = ()) -> set[str]:
    """An extra's requirements, with NEO's own extras (neo2[a,b]) expanded."""
    assert extra in EXTRAS, f"no extra named {extra!r}"
    assert extra not in seen, f"extras include each other in a loop: {seen + (extra,)}"
    out: set[str] = set()
    for req in EXTRAS[extra]:
        m = SELF.match(req)
        if m:
            for e in m.group(1).split(","):
                out |= _flat(e, seen + (extra,))
        else:
            out.add(req)
    return out


def _name(req: str) -> str:
    return re.split(r"[\s;\[<>=!~]", req, maxsplit=1)[0].lower()


def test_neo_app_gets_exactly_the_dependencies_it_always_had():
    assert set(PROJECT["dependencies"]) | _flat("app") == APP_BEFORE | APP_ADDED | CAMERA_ADDED
    assert set(EXTRAS["mac"]) >= APP_ADDED  # macOS frameworks: the mac extra, the camera's its own
    assert set(EXTRAS["vision"]) >= CAMERA_ADDED


def test_the_core_is_small_and_nothing_heavy():
    core = {_name(r) for r in PROJECT["dependencies"]}
    assert core == {"numpy", "httpx", "pydantic", "pydantic-settings", "python-dotenv"}
    assert not [n for n in core if n.startswith(HEAVY)]


def test_the_extras():
    assert _flat("brains") == set(EXTRAS["gemini"]) | set(EXTRAS["openai"]) | set(EXTRAS["claude"])
    assert _flat("all") == _flat("app") | set(EXTRAS["mcp"]) | set(EXTRAS["langchain"])
    assert EXTRAS["mcp"] == ["mcp>=2.2,<3"]  # neo2.mcp uses mcp 2.2's low-level Server and mcp_types
    assert EXTRAS["langchain"] == ["langchain-core>=1,<2"]
    assert EXTRAS["web"] == ["ddgs", "trafilatura", "pypdf"]
    assert EXTRAS["browser"] == ["playwright"]
    # No extra the SDK's own users pick pulls in a model, a macOS framework or a browser.
    for extra in ("web", "brains", "mcp", "langchain"):
        heavy = [r for r in _flat(extra) if _name(r).startswith(("torch", "laya", "mlx", "pyobjc", "playwright"))]
        assert heavy == [], (extra, heavy)
    for extra in EXTRAS:  # every neo2[...] names real extras
        _flat(extra)
    assert set(EXTRAS["dev"]) == {"pytest", "pytest-asyncio", "ruff", "mypy"}
    assert EXTRAS["train"] == ["datasets"]


def test_licence_author_and_links_are_the_owners():
    assert PROJECT["license"] == "MIT" and PROJECT["license-files"] == ["LICENSE"]
    assert PROJECT["authors"] == [{"name": "Arnav Parashar"}]
    assert PROJECT["urls"] == {
        "Homepage": "https://github.com/ArnavParashar49/Neo2.0",
        "Repository": "https://github.com/ArnavParashar49/Neo2.0",
        "GitHub": "https://github.com/ArnavParashar49",
        "LinkedIn": "https://www.linkedin.com/in/arnavparashar49/",
    }
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License\n\nCopyright (c) 2026 Arnav Parashar\n")
    readme = (ROOT / "README.md").read_text()
    assert "## Author" in readme and "[LinkedIn](https://www.linkedin.com/in/arnavparashar49/)" in readme
    assert "[MIT](LICENSE)" in readme


def test_python_version():
    # neo2/core/decide uses Python 3.12 syntax (PEP 695); 3.13 waits for the owner's decision.
    assert PROJECT["requires-python"] == ">=3.12,<3.13"


def test_console_scripts_and_typing_marker():
    import neo2
    import neo2.mcp

    assert PROJECT["scripts"] == {"neo2": "neo2.core.main:main", "neo2-mcp": "neo2.mcp:main"}
    assert callable(neo2.mcp.main)
    # One marker for the whole package, neo2.core too (neo2 re-exports ToolContext, SAFETY_RULES…).
    assert (ROOT / "neo2" / "py.typed").is_file()
    data = PYPROJECT["tool"]["setuptools"]["package-data"]
    assert data == {"neo2": ["py.typed", "core/vision/yoloe_words.txt"]}  # (+ the camera boxes' words)
    assert all((ROOT / "neo2" / f).is_file() for f in data["neo2"])
    assert neo2.__version__ == PROJECT["version"]


def test_the_distribution_is_neo2_and_ships_no_top_level_neo():
    """`neo2` on PyPI (`neo` is NeuralEnsemble's, which installs a top-level package of that name).
    The wheel has neo2 with NEO's own package inside it (neo2.core, `neo` before 2.1); the
    checkout's compatibility shim (a top-level `neo` package) stays out of it."""
    assert PROJECT["name"] == "neo2"
    find = PYPROJECT["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["neo2*"] and set(find["exclude"]) == {"neo", "neo.*"}
    assert (ROOT / "neo" / "__init__.py").is_file()  # the shim is in the repo…  neo2-codemod: keep
    setuptools = pytest.importorskip("setuptools")
    pkgs = set(setuptools.find_packages(str(ROOT), include=find["include"], exclude=find["exclude"]))
    assert {"neo2", "neo2.core", "neo2.core.tools.computer", "neo2.core.decide.kinds", "neo2.sdk", "neo2.mcp"} <= pkgs
    assert not [p for p in pkgs if p == "neo" or p.startswith(("neo.", "tests"))]  # …never in the wheel  neo2-codemod: keep
    # every package NEO has is in the wheel: a folder of neo2/ with an __init__.py
    folders = {".".join(d.relative_to(ROOT).parts) for d in (ROOT / "neo2").rglob("*") if (d / "__init__.py").is_file()}
    assert folders | {"neo2"} == pkgs


def test_the_installers_ship_no_stale_build_and_pin_no_local_path():
    """setuptools builds in the checkout and ships everything under build/lib, so a build/ left by an
    earlier in-tree build (one from before 2.1 holds the old top-level neo package) must go before
    NEO.app's runtime is installed from the checkout. And requirements.lock never pins NEO itself, in
    any form `uv pip freeze` writes it: an absolute file:// path would break the installer elsewhere."""
    lines = [ln.strip() for ln in (ROOT / "scripts" / "install-app.sh").read_text().splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    installs = [i for i, ln in enumerate(lines) if "--reinstall \"$ROOT\"" in ln]
    assert len(installs) == 2
    for i in installs:
        assert lines[i - 1] == 'rm -rf "$ROOT/build"' and lines[i + 1] == 'rm -rf "$ROOT/build"', lines[i - 1 : i + 2]
    pattern = re.search(r'grep -v -E "([^"]+)"', (ROOT / "scripts" / "lock-deps.sh").read_text())[1]
    dropped = re.compile(pattern)
    ours = ["-e file:///Users/a/NEO", "neo2 @ file:///Users/a/NEO", "neo2==2.1.0", "neo @ file:///Users/a/NEO",
            "neo==2.0.0", "pytest==8.4.1", "pytest-asyncio==1.1.0", "ruff==0.12.0", "mypy==1.17.0"]  # fmt: skip
    theirs = ["httpx==0.28.1", "neon==1.0", "neo4j==5.28.1", "laya==0.3.2", "google-genai==2.20.0"]
    assert [x for x in ours if not dropped.match(x)] == [] and [x for x in theirs if dropped.match(x)] == []

def test_model_field_defaults_are_keywords():
    """Type checkers read a pydantic model's field defaults through dataclass_transform, which
    sees only ``Field(default=...)``: with ``Field(24, ge=1)`` every AgentConfig(...) that leaves
    the field out fails type-checking in the host's code."""
    import ast

    bad = []
    for pkg in ("neo2",):  # neo2.core too
        for f in sorted((ROOT / pkg).rglob("*.py")):
            for node in ast.walk(ast.parse(f.read_text(), str(f))):
                call = node.value if isinstance(node, ast.AnnAssign) else None
                if isinstance(call, ast.Call) and getattr(call.func, "id", getattr(call.func, "attr", "")) == "Field" and call.args:
                    bad.append(f"{f.relative_to(ROOT)}:{node.lineno}")
    assert bad == []


def test_the_portable_marker_is_registered():
    markers = PYPROJECT["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("portable:") for m in markers)


def test_the_lock_pins_every_pyobjc_framework_the_app_installs():
    """scripts/install-app.sh builds NEO.app's runtime from requirements.lock: every pyobjc framework
    of the mac and vision extras is pinned there, with what the new ones need (ScreenCaptureKit:
    CoreMedia; Vision: CoreML; AVFoundation: CoreAudio), all of one pyobjc release."""
    lock = {
        name.lower(): version.strip()
        for line in (ROOT / "requirements.lock").read_text().splitlines()
        if "==" in line and not line.startswith("#")
        for name, version in [line.split("==", 1)]
    }
    mac = {
        re.match(r"[A-Za-z0-9_.-]+", r).group(0).lower()
        for r in (*EXTRAS["mac"], *EXTRAS["vision"])
        if r.lower().startswith("pyobjc")
    }
    needed = mac | {f"pyobjc-{n}" for n in ("core", "framework-coremedia", "framework-coreml", "framework-coreaudio")}
    assert needed <= lock.keys(), sorted(needed - lock.keys())
    assert len({v for n, v in lock.items() if n.startswith("pyobjc")}) == 1
